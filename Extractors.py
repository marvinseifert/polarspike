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
    Spike class

    This class stores methods to read the .hdf5 file which is the output of the spikesorting algorithm Herdingspikes2
    It allows for loading either all the spikes in the file or just a subset either based on number of cells or with a
    maximal nr of spikes threshold.
    """

    spikes = {}
    trigger = np.array([])
    stimulus_names = []

    def __init__(self, file, stimulus_df=None):
        """
        Initialize function. Opens the .hdf5 file that contains results of the
        Herdingspikes2 spikesorting algorithm and assignes them to the object.

        Parameters
        ----------
        file: str: The location of the hdf5 results file

        Returns
        -------

        """
        if stimulus_df is not None:
            self.add_stimulus_df(stimulus_df)
        self.file = Path(file)

    def add_stimulus_df(self, stimulus_df):
        self.trigger = stimulus_df[["begin_fr", "end_fr"]].to_numpy()
        self.stimulus_names = stimulus_df["stimulus_name"].to_list()
        self.spikes["sampling"] = stimulus_df.loc[0, "sampling_freq"].item()

    def to_arrow(self, batch_size=100000):
        nr_batches = int(np.ceil(self.spikes["times"].shape[0] / batch_size))
        print(f"nr of batches: {nr_batches}")
        schema = pa.schema(
            [pa.field("cell_index", pa.int64()), pa.field("times", pa.int64())]
        )
        for i in range(self.spikes["shapes"].shape[0]):
            schema = schema.append(pa.field("w" + str(i), pa.float32()))

        with pa.OSFile(str(self.file.with_suffix(".arrow")), "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for row in range(nr_batches):
                    start = row * batch_size
                    end = (row + 1) * batch_size
                    if end > self.spikes["times"].shape[0]:
                        end = self.spikes["times"].shape[0]
                    data_list = [
                        pa.array(self.spikes["cluster_id"][start:end]),
                        pa.array(self.spikes["times"][start:end]),
                    ]
                    for i in range(self.spikes["shapes"].shape[0]):
                        data_list.append(pa.array(self.spikes["shapes"][i, start:end]))

                    batch = pa.record_batch(data_list, schema=schema)
                    writer.write_batch(batch)

    def to_parquet(self):
        """
        Save the spikes to a parquet file

        Parameters
        ----------
        file: str: The location of the parquet file

        Returns
        -------

        """
        self.to_arrow()
        with pa.memory_map(str(self.file.with_suffix(".arrow")), "rb") as source:
            loaded_array = pa.ipc.open_file(source).read_all()
        df = pl.from_arrow(loaded_array)
        df.lazy().sink_parquet(str(self.file.with_suffix(".parquet")))

    def load(self, stimulus=True, recording_name=None, pandas=True):
        df = pl.scan_parquet(str(self.file.with_suffix(".parquet")))
        df = df.select(pl.col("cell_index", "times"))
        if stimulus:
            return self.construct_df(df, recording_name, pandas)
        if not stimulus:
            return df

    def construct_df(self, df, recording_name=None, pandas=True):
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

        df = pl.concat(dfs)
        df = df.sort("cell_index", descending=False)
        combined_df = df.join(df_centre, on="cell_index", how="left")
        combined_df = combined_df.rename({"count": "nr_of_spikes"})

        # Fill missing values

        if recording_name:
            combined_df = combined_df.with_columns(recording=pl.lit(recording_name))

        if pandas:
            return combined_df.to_pandas()
        else:
            return combined_df


class Extractor_HS2(Extractor):
    def __init__(self, file, stimulus_df=None):
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
        self.to_parquet()
        return


class Extractor_SPC(Extractor):
    spikes = {}

    def __init__(self, file, stimulus_df=None):
        super().__init__(file, stimulus_df)

    def get_spikes(self):
        # This is a work around. The Spyking Circus code needs to change os directory to work, which is stupid.
        # Here I save the directory that the user was in before calling the function, so that I can change the dir back
        # to this after the function has run. NEED to be changed in the future.
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
            1, self.spikes["nr_cells"], self.spikes["nr_cells"], dtype=int
        )

        self.get_locations(params)

        df = pl.from_numpy(
            np.vstack([cell_idx, array_combined]).T, schema=["cell_index", "times"]
        )
        df.write_parquet(str(self.file.with_suffix(".parquet")))
        os.chdir(current_dir)

    def get_locations(self, params):
        probe = probes.read_probe(params)
        results = load_data(params, "clusters")
        electrodes = results["electrodes"]
        electrodes_store = np.zeros((electrodes.shape[0], 2))
        for idx, electrode in enumerate(electrodes):
            electrodes_store[idx, :] = probe["channel_groups"][1]["geometry"][electrode]
        self.spikes["centres"] = electrodes_store
