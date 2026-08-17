"""
Extractor classes for extracting spikes from different spike sorting algorithms results files. At the moment
Herdingspikes2, Spyking Circus and Kilosort are supported. Kilosort support is experimental at the moment, based on data
from Schroeder Lab. The Extractor class is a super class that contains the basic methods for extracting the spikes from
the files. The Extractor_HS2, Extractor_SPC and Extractor_KS are the classes that are used to extract the spikes from the
respective files.
@Author: Marvin Seifert 2024
"""

import numpy as np
import h5py
import polars as pl
import pyarrow as pa
from pathlib import Path
from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.shared import probes
import os
import pandas as pd


# Super class for all extractors


class Extractor:
    """
    Extractor class

    This class is the super class for all the extractor classes. It contains the basic methods for extracting the spikes
    from the files. The Extractor_HS2, Extractor_SPC and Extractor_KS are the classes that are used to extract the spikes
    from the respective files.

    Extraction is done in two steps. First the spikes are extracted from the file and saved to an arrow file. This is
    done to avoid memory issues. The second step is to load the spikes from the arrow file and construct a DataFrame
    containing the spikes and save it to a parquet file as polars.DataFrame. The parquet file is then used to access
    the spikes in the analysis.

    The parquet file must match what core.polarspike.spike_loader expects:
    columns cell_index (int64), times (int64) and, when waveforms were
    extracted, a single waveforms column of type list[float32] (not one
    column per sample).
    """

    def __init__(self, file: str, stimulus_df: pd.DataFrame | None = None) -> None:
        """Set up the extractor's file paths and, optionally, its stimulus timing.

        Input:
            file (str): Path to the spike-sorting results file. Subclasses'
                parquet/arrow output is written next to it (same stem).
            stimulus_df (pd.DataFrame | None): Stimulus table with begin_fr,
                end_fr, stimulus_name and sampling_freq columns, used later by
                construct_df. May be added afterwards via add_stimulus_df.

        Output:
            None.
        """
        self.spikes: dict = {}
        self.trigger: np.ndarray = np.array([])
        self.stimulus_names: list[str] = []
        self.file = Path(file)
        self.file_parquet = self.file.with_suffix(".parquet")
        if stimulus_df is not None:
            self.add_stimulus_df(stimulus_df)

    def add_stimulus_df(self, stimulus_df: pd.DataFrame) -> None:
        """Attach stimulus timing so construct_df can align spikes per stimulus.

        Input:
            stimulus_df (pd.DataFrame): One row per stimulus, must contain
                begin_fr, end_fr, stimulus_name and sampling_freq columns.

        Output:
            None. Sets self.trigger, self.stimulus_names and
            self.spikes["sampling"].

        Raises:
            KeyError: if stimulus_df is missing one of the required columns.
        """
        required = {"begin_fr", "end_fr", "stimulus_name", "sampling_freq"}
        missing = required - set(stimulus_df.columns)
        if missing:
            raise KeyError(
                f"stimulus_df is missing required columns: {sorted(missing)}"
            )
        self.trigger = stimulus_df[["begin_fr", "end_fr"]].to_numpy()
        self.stimulus_names = stimulus_df["stimulus_name"].to_list()
        self.spikes["sampling"] = stimulus_df.loc[0, "sampling_freq"].item()

    def to_arrow(self, batch_size: int = 100000) -> None:
        """Write extracted spikes to a chunked Arrow file, to avoid loading them all into memory at once.

        Input:
            batch_size (int): Number of spikes written per chunk.

        Output:
            None. Writes <file>.arrow with columns cell_index (int64), times
            (int64) and, if self.spikes["shapes"] was populated, waveforms
            (list[float32]) built from it.

        Raises:
            KeyError: if self.spikes lacks "cluster_id" or "times", i.e. the
                subclass's extraction step has not run yet.
        """
        if "cluster_id" not in self.spikes or "times" not in self.spikes:
            raise KeyError(
                "to_arrow called before spikes were extracted; run this "
                "extractor's get_spikes() first."
            )
        nr_spikes = self.spikes["times"].shape[0]
        nr_batches = int(np.ceil(nr_spikes / batch_size))
        has_waveforms = "shapes" in self.spikes

        fields = [pa.field("cell_index", pa.int64()), pa.field("times", pa.int64())]
        if has_waveforms:
            fields.append(pa.field("waveforms", pa.list_(pa.float32())))
        schema = pa.schema(fields)

        if has_waveforms:
            # self.spikes["shapes"] is (nr_samples, nr_spikes); transpose to
            # one row of samples per spike.
            shapes = np.asarray(self.spikes["shapes"], dtype=np.float32).T
            nr_samples = shapes.shape[1]

        with pa.OSFile(str(self.file.with_suffix(".arrow")), "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for row in range(nr_batches):
                    start = row * batch_size
                    end = min((row + 1) * batch_size, nr_spikes)
                    columns = [
                        pa.array(self.spikes["cluster_id"][start:end]),
                        pa.array(self.spikes["times"][start:end]),
                    ]
                    if has_waveforms:
                        chunk = shapes[start:end]
                        values = pa.array(chunk.reshape(-1), type=pa.float32())
                        offsets = pa.array(
                            np.arange(
                                0, (chunk.shape[0] + 1) * nr_samples, nr_samples
                            ),
                            type=pa.int32(),
                        )
                        columns.append(pa.ListArray.from_arrays(offsets, values))
                    batch = pa.record_batch(columns, schema=schema)
                    writer.write_batch(batch)

    def to_parquet(self) -> None:
        """Write spikes to <file>.parquet via a chunked intermediate Arrow file.

        Output:
            None. Creates <file>.parquet next to self.file. The intermediate
            <file>.arrow is deleted once the parquet write succeeds.

        Raises:
            KeyError: if spikes have not been extracted yet (see to_arrow).
            pa.ArrowInvalid: if the intermediate Arrow file cannot be read
                back (corrupt or truncated write).
        """
        self.to_arrow()
        arrow_path = self.file.with_suffix(".arrow")
        try:
            with pa.memory_map(str(arrow_path), "rb") as source:
                loaded_array = pa.ipc.open_file(source).read_all()
        except pa.ArrowInvalid as e:
            raise pa.ArrowInvalid(f"could not read back {arrow_path}: {e}") from e
        df = pl.from_arrow(loaded_array)
        df.lazy().sink_parquet(str(self.file_parquet))
        arrow_path.unlink()

    def load(
        self,
        stimulus: bool = True,
        recording_name: str | None = None,
        pandas: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Load this extractor's spikes parquet, optionally summarised per stimulus.

        Input:
            stimulus (bool): If True, return the per-cell/per-stimulus summary
                built by construct_df; if False, return the raw
                cell_index/times spikes.
            recording_name (str | None): Recording name to attach as a
                "recording" column; only used when stimulus is True.
            pandas (bool): Return a pandas.DataFrame if True, else a
                polars.DataFrame.

        Output:
            pd.DataFrame | pl.DataFrame: The summary (stimulus=True) or raw
            spikes (stimulus=False) frame, as pandas or polars per `pandas`.

        Raises:
            FileNotFoundError: if this extractor's parquet file does not
                exist yet (get_spikes() must be run first).
        """
        if not self.file_parquet.exists():
            raise FileNotFoundError(
                f"{self.file_parquet} does not exist; run get_spikes() first."
            )
        df = pl.scan_parquet(str(self.file_parquet)).select(
            pl.col("cell_index", "times")
        )
        if stimulus:
            return self.construct_df(df, recording_name, pandas)
        collected = df.collect()
        return collected.to_pandas() if pandas else collected

    def construct_df(
        self,
        df: pl.LazyFrame,
        recording_name: str | None = None,
        pandas: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Summarise spike counts and locations per cell and stimulus.

        Input:
            df (pl.LazyFrame): Raw spikes with cell_index and times columns
                (e.g. from load(stimulus=False)).
            recording_name (str | None): If given, attached as a "recording"
                column on the result.
            pandas (bool): Return a pandas.DataFrame if True, else a
                polars.DataFrame.

        Output:
            pd.DataFrame | pl.DataFrame: One row per (cell_index, stimulus),
            with stimulus_name, stimulus_index, nr_of_spikes, centres_x,
            centres_y and, if recording_name was given, recording.

        Raises:
            ValueError: if no stimulus_df was added (self.trigger is empty).
            KeyError: if get_spikes() has not populated
                cell_indices/centres yet.
        """
        if self.trigger.size == 0:
            raise ValueError(
                "construct_df requires a stimulus_df; pass one to the "
                "constructor or call add_stimulus_df() first."
            )
        missing = {"cell_indices", "centres"} - self.spikes.keys()
        if missing:
            raise KeyError(
                f"construct_df requires get_spikes() to have been run first "
                f"(missing: {sorted(missing)})."
            )

        dfs = []
        for stimulus in range(self.trigger.shape[0]):
            times = df.filter(
                (pl.col("times") > self.trigger[stimulus, 0])
                & (pl.col("times") <= self.trigger[stimulus, 1])
            )
            times = times.with_columns(
                stimulus_name=pl.lit(self.stimulus_names[stimulus])
            )
            times = times.with_columns(stimulus_index=pl.lit(stimulus))
            df_idx = pl.DataFrame(
                data=self.spikes["cell_indices"],
                schema=[("cell_index", pl.datatypes.Int64)],
            )

            df_temp = (
                times.group_by(pl.col("cell_index", "stimulus_name", "stimulus_index"))
                .count()
                .collect()
            )

            df_idx = df_idx.join(df_temp, on="cell_index", how="left")
            # Fill missing values:
            df_idx = df_idx.with_columns(
                pl.col("stimulus_name").fill_null(self.stimulus_names[stimulus])
            )
            df_idx = df_idx.with_columns(pl.col("count").fill_null(0))
            df_idx = df_idx.with_columns(pl.col("stimulus_index").fill_null(stimulus))
            dfs.append(df_idx)

        # Add the centres
        centres_id = np.hstack(
            [
                self.spikes["cell_indices"].reshape(-1, 1) - 1,
                self.spikes["centres"],
            ]
        )

        df_centre = pl.from_numpy(
            data=centres_id,
            schema=[("cell_index", int), ("centres_x", float), ("centres_y", float)],
        )

        combined_df = pl.concat(dfs).sort("cell_index", descending=False)
        combined_df = combined_df.join(df_centre, on="cell_index", how="left")
        combined_df = combined_df.rename({"count": "nr_of_spikes"})

        if recording_name:
            combined_df = combined_df.with_columns(recording=pl.lit(recording_name))

        return combined_df.to_pandas() if pandas else combined_df


class Extractor_HS2(Extractor):
    """
    Extractor class for HerdingSpikes 2 spikesorting results.
    """

    def __init__(self, file: str, stimulus_df: pd.DataFrame = None):
        super().__init__(file, stimulus_df)

        with h5py.File(file, "r") as f:
            self.spikes["centres"] = np.array(f["/centres"], dtype=float)
            self.spikes["cluster_id"] = np.array(f["/cluster_id"], dtype=int)
            self.spikes["times"] = np.array(f["/times"], dtype=int)
            self.spikes["sampling"] = np.array(f["/Sampling"], dtype=float)
            self.spikes["channels"] = np.array(f["/ch"], dtype=float)
            self.spikes["spike_freq"] = np.array(
                np.unique(self.spikes["cluster_id"], return_counts=True)
            )
            self.spikes["nr_cells"] = np.max(self.spikes["spike_freq"][0, :]) + 1
            self.spikes["cell_indices"] = np.linspace(
                1, self.spikes["nr_cells"], self.spikes["nr_cells"], dtype=int
            )
            self.spikes["max_spikes"] = np.max(self.spikes["spike_freq"][1, :])
            self.spikes["shapes"] = np.array(f["/shapes"], dtype=float)

    def get_spikes(self):
        """
        Get the spikes from the HerdingSpikes 2 results file.
        This is a legacy method that allows consistency in function names between older versions of the code and this version.
        """
        self.to_parquet()
        return


class Extractor_axonsort(Extractor):
    """
    Extractor class for axonsort (kilosort4 + template-matching) results.

    Reads a spikes parquet with cluster_ids/sample_index columns, as written
    by axonsort's template_matching.py. If a waveforms column is also
    present (added by axonsort's export_waveforms.py, which reads the raw
    binary directly since that already depends on kilosort/torch) it is
    carried through to the output parquet via the base class's to_arrow/
    to_parquet, same as Extractor_HS2.

    The source file is already a .parquet (unlike HS2's .hdf5 or SPC/KS's
    sorter-native files), so output is written to a distinct
    <stem>_extracted.parquet rather than <file>.with_suffix(".parquet") —
    otherwise get_spikes() would overwrite its own source, and calling
    load() before get_spikes() would silently read the wrong (pre-
    conversion) columns instead of raising a clear error.
    """

    def __init__(self, file: str, stimulus_df: pd.DataFrame | None = None) -> None:
        """Load spike times, and waveforms if present, from an axonsort spikes parquet.

        Input:
            file (str): Path to spikes.parquet or spikes_with_waveforms.parquet
                (cluster_ids: int, sample_index: int, optionally
                waveforms: list[float], one snippet per spike).
            stimulus_df (pd.DataFrame | None): Stimulus table, see Extractor.

        Output:
            None.

        Raises:
            KeyError: if file is missing cluster_ids or sample_index.
            ValueError: if a waveforms column is present but its snippets
                don't all have the same length.
        """
        super().__init__(file, stimulus_df)
        self.file_parquet = self.file.with_name(self.file.stem + "_extracted.parquet")
        df = pl.read_parquet(file)
        missing = {"cluster_ids", "sample_index"} - set(df.columns)
        if missing:
            raise KeyError(f"{file} is missing required columns: {sorted(missing)}")

        self.spikes["cluster_id"] = df["cluster_ids"].to_numpy()
        self.spikes["times"] = df["sample_index"].to_numpy()
        self.spikes["cell_indices"] = np.unique(self.spikes["cluster_id"])
        self.spikes["centres"] = np.zeros(
            (self.spikes["cell_indices"].shape[0], 2), dtype=float
        )
        if "waveforms" in df.columns:
            try:
                self.spikes["shapes"] = np.stack(df["waveforms"].to_list()).T
            except ValueError as e:
                raise ValueError(
                    f"{file}'s waveforms column has inconsistent snippet "
                    f"lengths: {e}"
                ) from e

    def get_spikes(self) -> None:
        """Write this extractor's spikes to <stem>_extracted.parquet.

        Output:
            None. Delegates to Extractor.to_parquet, producing cell_index/
            times and, if waveforms were loaded, a matching waveforms
            list[float32] column.
        """
        self.to_parquet()

    def construct_df(
        self,
        df: pl.LazyFrame,
        recording_name: str | None = None,
        pandas: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """Summarise spike counts and locations per cell and stimulus.

        Same as Extractor.construct_df, except the centres join uses
        cell_indices as-is: axonsort's cluster ids are sparse (e.g.
        1, 2, 3, 5, 8, ...), not a contiguous 1..N range paired 1:1 with a
        0-indexed centres array as HS2's are, so the base class's
        `cell_indices - 1` offset does not apply here.

        Input:
            df (pl.LazyFrame): Raw spikes with cell_index and times columns
                (e.g. from load(stimulus=False)).
            recording_name (str | None): If given, attached as a "recording"
                column on the result.
            pandas (bool): Return a pandas.DataFrame if True, else a
                polars.DataFrame.

        Output:
            pd.DataFrame | pl.DataFrame: One row per (cell_index, stimulus),
            with stimulus_name, stimulus_index, nr_of_spikes, centres_x,
            centres_y and, if recording_name was given, recording.

        Raises:
            ValueError: if no stimulus_df was added (self.trigger is empty).
            KeyError: if get_spikes() has not populated
                cell_indices/centres yet.
        """
        if self.trigger.size == 0:
            raise ValueError(
                "construct_df requires a stimulus_df; pass one to the "
                "constructor or call add_stimulus_df() first."
            )
        missing = {"cell_indices", "centres"} - self.spikes.keys()
        if missing:
            raise KeyError(
                f"construct_df requires get_spikes() to have been run first "
                f"(missing: {sorted(missing)})."
            )

        dfs = []
        for stimulus in range(self.trigger.shape[0]):
            times = df.filter(
                (pl.col("times") > self.trigger[stimulus, 0])
                & (pl.col("times") <= self.trigger[stimulus, 1])
            )
            times = times.with_columns(
                stimulus_name=pl.lit(self.stimulus_names[stimulus])
            )
            times = times.with_columns(stimulus_index=pl.lit(stimulus))
            df_idx = pl.DataFrame(
                data=self.spikes["cell_indices"],
                schema=[("cell_index", pl.datatypes.Int64)],
            )

            df_temp = (
                times.group_by(pl.col("cell_index", "stimulus_name", "stimulus_index"))
                .count()
                .collect()
            )

            df_idx = df_idx.join(df_temp, on="cell_index", how="left")
            df_idx = df_idx.with_columns(
                pl.col("stimulus_name").fill_null(self.stimulus_names[stimulus])
            )
            df_idx = df_idx.with_columns(pl.col("count").fill_null(0))
            df_idx = df_idx.with_columns(pl.col("stimulus_index").fill_null(stimulus))
            dfs.append(df_idx)

        centres_id = np.hstack(
            [
                self.spikes["cell_indices"].reshape(-1, 1),
                self.spikes["centres"],
            ]
        )
        df_centre = pl.from_numpy(
            data=centres_id,
            schema=[("cell_index", int), ("centres_x", float), ("centres_y", float)],
        )

        combined_df = pl.concat(dfs).sort("cell_index", descending=False)
        combined_df = combined_df.join(df_centre, on="cell_index", how="left")
        combined_df = combined_df.rename({"count": "nr_of_spikes"})

        if recording_name:
            combined_df = combined_df.with_columns(recording=pl.lit(recording_name))

        return combined_df.to_pandas() if pandas else combined_df


class Extractor_SPC(Extractor):
    """
    Extractor class for Spyking Circus spikesorting results.
    Loading results from Spyking-Circus is slightly more complicated than for HerdingSpikes2. This is because we
    need to use the CircusParser and load_data functions from the circus module to access the data (which are written
    in an annoying way).
    """

    def __init__(self, file: str, stimulus_df: pd.DataFrame = None):
        super().__init__(file, stimulus_df)

    def get_spikes(self):
        # This is a work around. The Spyking Circus code needs to change os directory to work, which is stupid.
        # Here I save the directory that the user was in before calling the function, so that I can change the dir back
        # to this after the function has run. Lesson: Never change the directory in a function using OS @Spyking Circus!

        current_dir = os.getcwd()

        os.chdir(self.file.parent)
        # The SpykingCircus module for loding the data is used to access the files:
        params = CircusParser(str(self.file))
        params.get_data_file()
        results = load_data(params, "results")
        spikes = results["spiketimes"]
        result = spikes.items()
        data = list(result)
        array = [data[cell][1] for cell in range(len(data))]
        nr_spikes = [len(data[cell][1]) for cell in range(len(data))]
        array_combined = np.concatenate(array)
        cell_idx = np.arange(0, len(spikes.keys()), 1, dtype=int)
        cell_idx = np.repeat(cell_idx, nr_spikes)

        self.spikes["nr_cells"] = len(spikes.keys())
        self.spikes["cell_indices"] = np.linspace(
            0, self.spikes["nr_cells"], self.spikes["nr_cells"], dtype=int
        )

        self.get_locations(params)

        df = pl.from_numpy(
            np.vstack([cell_idx, array_combined]).T, schema=["cell_index", "times"]
        )
        df.write_parquet(str(self.file.with_suffix(".parquet")))
        os.chdir(current_dir)

    def get_locations(self, params: CircusParser):
        """
        Loactions in Spyking-Circus are not stored the same way as in HerdingSpikes2. This is beacuse the algorithm doesnt
        triangulate the actual location of the spiking signal, but rather the location of the electrode that detected the
        signal strongest.

        Parameters
        ----------
        params: CircusParser
            The CircusParser object that contains the parameters of the spike sorting results.
        """
        probe = probes.read_probe(params)
        results = load_data(params, "clusters")
        electrodes = results["electrodes"]
        electrodes_store = np.zeros((electrodes.shape[0], 2))
        for idx, electrode in enumerate(electrodes):
            electrodes_store[idx, :] = probe["channel_groups"][1]["geometry"][electrode]
        self.spikes["centres"] = electrodes_store


class Extractor_KS(Extractor):
    """
    Extractor class for Kilosort 2 spikesorting results. Written based on data from the Schroeder Lab. Experimental.

    """

    def __init__(self, file: str, stimulus_df: pd.DataFrame = None):
        super().__init__(file, stimulus_df)

    def get_spikes(self):
        working_dir = self.file.parents[0]
        self.spikes["times"] = (
            np.load(self.file, allow_pickle=True).flatten().astype(np.dtypes.Int64DType)
        )
        cell_idx = np.load(working_dir / "spike_clusters.npy", allow_pickle=True)
        cell_idx = cell_idx.astype(np.dtypes.Int64DType)
        self.spikes["cell_indices"] = np.unique(cell_idx)
        df = pl.from_numpy(
            np.vstack([cell_idx, self.spikes["times"]]).T,
            schema=["cell_index", "times"],
        )
        spike_positions = np.load(
            working_dir / "spike_positions.npy", allow_pickle=True
        )
        spike_positions_avg = np.zeros((len(np.unique(cell_idx)), 2), dtype=float)
        for cell in np.unique(cell_idx):
            spike_positions_avg[cell] = np.mean(
                spike_positions[cell_idx == cell], axis=0
            )
        self.spikes["centres"] = spike_positions_avg
        df.write_parquet(str(self.file.with_suffix(".parquet")))

    def construct_df(
        self, df: pl.DataFrame, recording_name: str = None, pandas: bool = True
    ) -> pd.DataFrame | pl.DataFrame:
        dfs = []
        for stimulus in range(self.trigger.shape[0]):
            times = df.filter(
                (pl.col("times") > self.trigger[stimulus, 0])
                & (pl.col("times") <= self.trigger[stimulus, 1])
            )
            times = times.with_columns(
                stimulus_name=pl.lit(self.stimulus_names[stimulus])
            )
            times = times.with_columns(stimulus_index=pl.lit(stimulus))
            # print(times.dtypes, times)
            df_idx = pl.DataFrame(
                data=self.spikes["cell_indices"],
                schema=[("cell_index", pl.datatypes.Int64)],
            )

            df_temp = (
                times.group_by(pl.col("cell_index", "stimulus_name", "stimulus_index"))
                .count()
                .collect()
            )

            df_idx = df_idx.join(df_temp, on="cell_index", how="left")
            # Fill missing values:
            df_idx = df_idx.with_columns(
                pl.col("stimulus_name").fill_null(self.stimulus_names[stimulus])
            )
            df_idx = df_idx.with_columns(pl.col("count").fill_null(0))
            df_idx = df_idx.with_columns(pl.col("stimulus_index").fill_null(stimulus))
            dfs.append(df_idx)

        df = pl.concat(dfs)
        df = df.sort("cell_index", descending=False)
        df = df.rename({"count": "nr_of_spikes"})

        centres_id = np.hstack(
            [
                self.spikes["cell_indices"].reshape(-1, 1),
                self.spikes["centres"],
            ]
        )
        df_centre = pl.from_numpy(
            data=centres_id,
            schema=[("cell_index", int), ("centres_x", float), ("centres_y", float)],
        )
        df = df.sort("cell_index", descending=False)
        df = df.join(df_centre, on="cell_index", how="left")

        # Fill missing values

        if recording_name:
            df = df.with_columns(recording=pl.lit(recording_name))

        if pandas:
            return df.to_pandas()
        else:
            return df
