# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:23:06 2021

@author: Marvin
"""

import contextlib
import pathlib
from importlib.resources import files

import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import traceback
import pickle

import polars as pl


from dataclasses import dataclass, field
from polarspike import (
    stimulus_trace,
    spiketrains,
    stimulus_spikes,
    stimulus_dfs,
    filter_stimulus,
)
from polarspike.grid import Table
from threading import Thread
import warnings
from pathlib import Path, PurePath


def version_control(obj: "Recording") -> "Recording":
    try:
        obj.parquet_path = obj.path
        del obj.path
    except AttributeError:
        pass

    # Need to correct old parquet path to new format. Old format was a windows path unfortunately with two slashes \\
    # which cannot be converted to a pathlib.Path object. Thus, need to do this by hand. The new path should just be
    # the filename without the path.
    if isinstance(obj.parquet_path, str):
        # Check if the path is a windows path with double slashes
        if "\\" in obj.parquet_path and obj.parquet_path.count("\\") > 1:
            # Convert to pathlib.Path object
            obj.parquet_path = Path(obj.parquet_path.split("\\")[-1])
        else:
            # Convert to pathlib.Path object
            obj.parquet_path = Path(obj.parquet_path)

    # Luckily, while the raw_path attribute is also a Windows path, it is not a double slash path. So we can use
    # pathlib.PurePath to get the last part of the path.
    if isinstance(obj.raw_path, str):
        # Convert to pathlib.PurePath object
        obj.raw_path = Path(pathlib.PurePath(obj.raw_path).name)

    try:
        for df in obj.dataframes.values():
            df["recording"] = df["recording"].astype(str)
            df["stimulus_name"] = df["stimulus_name"].astype(str)
            df["stimulus_index"] = df["stimulus_index"].astype(pd.UInt16Dtype())
    except KeyError:
        pass

    return obj


@dataclass
class Recording:
    parquet_path: str | Path  # Path, pointing to the parquet file which stores the spiketimestamps
    raw_path: str | Path  # Path, pointing to the raw file which stores the raw data
    dataframes: dict = field(
        default_factory=lambda: {}, init=True
    )  # Dictionary that stores the dataframes
    sampling_freq: float = field(
        default_factory=lambda: 1.0, init=True
    )  # The sampling frequency of the recording
    views: dict = field(
        default_factory=lambda: {}, init=True
    )  # Dictionary that stores the views of dataframes
    load_path: str | Path = field(
        default_factory=lambda: Path(), init=True
    )  # If a class instance was saved, the path to the saved file is stored here.
    nr_stimuli: int = field(init=False)  # Nr of stimuli in the recording
    nr_cells: int = field(init=False)  # Nr of cells in the recording
    name: str = field(init=False)  # Name of the recording
    analysis: dict = field(default_factory=lambda: {}, init=False)

    def __post_init__(self):
        """
        This function is called after the class is initialized. It updates some of the class attributes
        with the information from inout parameters of the init function.

        Updated parameters:
            - nr_stimuli
            - nr_cells
            - name

        """
        if self.dataframes["spikes_df"] is not None:
            self.nr_stimuli = len(self.dataframes["stimulus_df"])
            self.nr_cells = np.unique(self.dataframes["spikes_df"]["cell_index"]).shape[
                0
            ]
            # Check if a single recording was added:
            self.single_recording_assertion()
            # Define the name of the recording:
            self.name = self.dataframes["spikes_df"]["recording"].unique()[0]
        else:
            self.nr_stimuli = 0
            self.nr_cells = 0

    def single_recording_assertion(self):
        """
        Checks if the loaded dataframes contain only a single recording. If multiple recordings are present, the
        Recording_s class should be used instead.

        """
        assert (
            self.dataframes["spikes_df"]["recording"].unique().shape[0] == 1
        ), "Dataframe contains multiple recordings use Recording_s class instead"

    def get_spikes_triggered(
        self,
        filter_conditions: list[dict],
        time: str = "seconds",
        waveforms: bool = False,
        pandas: bool = True,
        cell_df: str = "spikes_df",
        stimulus_df: str = "stimulus_df",
        carry: list[str] = None,
    ) -> pd.DataFrame | pl.DataFrame:
        # get the filtered spikes_df
        input_df = stimulus_spikes.filter_dataframe_complex(
            self.dataframes[cell_df], filter_conditions
        )
        # convert to polars
        input_df = pl.from_pandas(input_df)
        # get the filter parameters
        filter_dict = stimulus_trace.create_filter_dict(
            input_df, self.dataframes[stimulus_df]
        )
        # Store all the file paths in a dictionary
        if getattr(self, "recordings", None) == None:
            # Need to check if recordings is present, if not, the recording is a single recording
            # if single recording, the self.name is the recording name
            files_dict = {self.name: self.parquet_path}
        else:
            # if multiple recordings are present, the recordings attribute is present
            # each recording's name is stored in the recordings attribute
            files_dict = {
                rec: self.recordings[rec].parquet_path
                for rec in list(filter_dict.keys())
            }
        # get spikes from files
        return stimulus_spikes.get_spikes(
            files_dict,
            filter_dict,
            self.dataframes[cell_df],
            time,
            waveforms,
            pandas,
            carry,
        )

    def _check_spikes_df(
        self,
        df_name: str = "spikes_df",
        return_columns: list[str] = None,
        pandas: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        if return_columns is None:
            return_columns = ["recording", "stimulus_index", "cell_index"]
        try:
            temp_df = self.dataframes[df_name][return_columns]

        except KeyError:
            warnings.warn(
                "The cell dataframe does not contain the required columns. Accidentally provided a stimulus dataframe?"
            )
        if pandas:
            return temp_df
        else:
            return pl.from_pandas(temp_df)

    def get_spikes_df(
        self,
        cell_df: str = "spikes_df",
        stimulus_df: str = "stimulus_df",
        time: str = "seconds",
        waveforms: bool = False,
        pandas: bool = True,
        carry: list[str] = None,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Returns all spikes from all recordings and all stimuli in the choosen "cell_df" and "stimulus_df".
        This is equal to calling get_spikes_triggered with recordings = "all", cells = "all" and stimuli = "all",
        pointing at the same dataframes.

        Parameters
        ----------
        cell_df : str
            Name of the cell dataframe that shall be used.
        stimulus_df : str
            Name of the stimulus dataframe that shall be used.
        time : str
            Defines the time unit of the returned dataframe. Can be "seconds" or "frames".
        waveforms : boolean
            If the waveforms shall be loaded as well. Only possible if the parquet file contains the waveforms.
        pandas : boolean
            If the returned dataframe shall be a pandas dataframe or a polars dataframe.

        """

        # Check if provided dataframe is the right dataframe
        input_df = self._check_spikes_df(df_name=cell_df, pandas=False)
        # Create filter parameters for which cells and stimuli to load
        filter_dict = stimulus_trace.create_filter_dict(
            input_df, self.dataframes[stimulus_df]
        )
        # Store all the file paths in a dictionary
        if getattr(self, "recordings", None) == None:
            files_dict = {self.name: self.parquet_path}
        else:
            files_dict = {
                rec: self.recordings[rec].parquet_path for rec in list(filter_dict.keys())
            }
        # get spikes from files
        return stimulus_spikes.get_spikes(
            files_dict,
            filter_dict,
            self.dataframes[cell_df],
            time,
            waveforms,
            pandas,
            carry,
        )

    def get_stimulus_subset(self, stimulus=None, name=None, dataframes=None):
        """Returns a subset of the dataframe that only contains the spikes that
        were recorded during the presentation of a specific stimulus.

        Parameters
        ----------
        stimulus : int
            Index of the stimulus that shall be used to create the subset.
        name : str
            Name of the subset that is created. If not specified, the name of the stimulus is used.
        dataframes : str
            Name of the dataframe that is used to create the subset.

        Returns
        -------
        subset : pandas.Dataframe
            Subset of the dataframe that only contains the spikes that were recorded during the presentation of a
            specific stimulus.

        """
        if dataframes is None:
            dataframes = ["spikes_df", "stimulus_df"]
        out = []

        for dataframe in dataframes:
            dfs = []
            if name:
                df = (
                    self.dataframes[dataframe]
                    .loc[self.dataframes[dataframe]["stimulus_name"] == name]
                    .copy()
                )
                if stimulus:
                    df = df.loc[df["stimulus_index"] == stimulus]
            else:
                df = (
                    self.dataframes[dataframe]
                    .loc[self.dataframes[dataframe]["stimulus_index"] == stimulus]
                    .copy()
                )

            if dataframe == "spikes_df":
                spikes, _ = self.get_spikes_as_numpy(
                    df["cell_index"].unique().tolist(),
                    df["stimulus_index"].unique().tolist(),
                    time="frames",
                    waveforms=False,
                )
                df = df.set_index(["cell_index"])
                df["spikes"] = spikes
            df = df.reset_index()
            dfs.append(df)
            out.append(pd.concat(dfs))
        return out

    def show_df(
        self,
        name: str = "spikes_df",
        level: str = False,
        condition: str = False,
        viewname: str = None,
    ) -> Table:
        """Main function to interactively view the dataframes.

        Parameters
        ----------
        name : str
            Name of the dataframe that should be shown. Can be "spikes_df" or "stimulus_df" or any dataframe that was
            added manually during the analysis.
        level : str
            Shows a specific column or index named after the string in level.
        condition : str
            Shows a specific subset of the dataframe, for which the condition is true.
            Can only be true if level is true
        viewname : str
            Name of the view that is created. If not specified, the name of the dataframe is used.

        Returns
        -------
        view
            Qgrid class of a Dataframe view.

        """
        # Version control:

        if name == "stimulus":
            name = "stimulus_df"
        if name == "spikes":
            name = "spikes_df"

        if level and not condition:
            try:
                view = Table(self.dataframes[name][level])
            except KeyError:
                view = Table(self.dataframes[name].index.show_level_values(level))

        elif level:
            try:
                view = Table(
                    self.dataframes[name][self.dataframes[name][level] == condition]
                ).show()
            except KeyError:
                view = Table(
                    self.dataframes[name][
                        self.dataframes[name].index.show_level_values(level)
                        == condition
                    ]
                )
        else:
            view = Table(self.dataframes[name])

        # Overwrite the view if it already exists:
        self.views[viewname] = None
        if not viewname:
            viewname = name
        self.views[viewname] = view
        # Keeping track of which cells the user might select:
        return self.views[viewname].show()

    def filtered_df(self, df_name: str) -> pd.DataFrame:
        """
        Returns the filtered dataframe of a specific view.

        Parameters
        ----------
        df_name : str
            Name of the dataframe that shall be returned.
        """
        return self.views[df_name].tabulator.value

    def find_stim_indices(
        self,
        stimulus_names: list[str],
        stimulus_df: str = "stimulus_df",
    ) -> list[list[int]]:
        """Returns the stimulus indices that correspond to the stimulus names.

        Parameters
        ----------
        stimulus_names : list of strings
            List of stimulus names that shall be converted to stimulus indices.
        stimulus_df : str
            Name of the stimulus dataframe that shall be used.
        -------
        stim_indices : list of lists
            List of stimulus indices that correspond to the stimulus names.

        """

        stim_indices = []
        for stimulus in stimulus_names:
            if type(stimulus) is str:
                temp_df = (
                    self.dataframes[stimulus_df]
                    .copy()
                    .set_index(["recording", "stimulus_name"])
                    .sort_index()
                )
                stim_indices.append(
                    temp_df.loc[(recording, stimulus), :]["stimulus_index"].to_list()
                )
            else:
                stim_indices.append(stimulus)
        stim_indices = [[item] for sublist in stim_indices for item in sublist]
        return stim_indices

    def delete_df(self, df_name: str):
        """Deletes a dataframe from the recording object.

        Parameters
        ----------
        df_name : str
            Name of the dataframe that shall be deleted.
        """
        assert df_name != "spikes_df", "Cannot delete spikes_df"
        assert df_name != "stimulus_df", "Cannot delete stimulus_df"
        del self.dataframes[df_name]

    @property
    def spikes_df(self):
        """Returns the spikes_df dataframe. Shortcut version
        Warning: You cannot update this attribute, use self.dataframes["spikes_df"] instead
        """
        return self.dataframes["spikes_df"]

    @property
    def stimulus_df(self):
        """Returns the stimulus_df dataframe. Shortcut version
        Warning: You cannot update this attribute, use self.dataframes["stimulus_df"] instead
        """
        return self.dataframes["stimulus_df"]

    def save(self, filename: str | pathlib.Path):
        """
        Save function that is doing the actual saving after non-picklable attributes have been removed.
        Parameters
        ----------
        filename : str
            The name of the file to save the object to.

        Returns
        -------

        """
        self.views = {}
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save_save(self):
        """
        Save the object to the path that was used to load the object.
        """
        self.views = {}
        with open(self.load_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str | pathlib.Path) -> "Recording":
        """Load a saved object from a file.
        Parameters
        ----------
        filename : str or pathlib.Path
            The name of the file to load.
        Returns
        -------
        obj : object
            The object stored in the file.

        """

        with open(filename, "rb") as f:
            obj = pickle.load(f)
            obj.load_path = Path(filename)

            # Update earlier versions of the recording class:
            obj = version_control(obj)

            # Define the full paths based on the load_path
            obj.parquet_path = obj.load_path.parent / obj.parquet_path
            obj.raw_path = obj.load_path.parent / obj.raw_path

            # %% Check if all paths have the same parent directory
            if not obj.parquet_path.parent == obj.raw_path.parent:
                raise ValueError(
                    "The parquet_path and raw_path must have the same parent directory."
                )
            if not obj.raw_path.parent == obj.parquet_path.parent:
                raise ValueError(
                    "The raw_path and parquet_path must have the same parent directory."
                )

            # Check if the files exist
            if not obj.parquet_path.exists():
                raise FileNotFoundError(f"Parquet file {obj.parquet_path} does not exist.")
            if not obj.raw_path.exists():
                raise FileNotFoundError(f"Raw file {obj.raw_path} does not exist.")

            return obj

    def __delattr__(self, name):
        if name == "dataframes":
            raise AttributeError("Cannot delete dataframes")
        elif name == "views":
            raise AttributeError("Cannot delete views")
        elif name == "nr_stimuli":
            raise AttributeError("Cannot delete nr_stimuli")
        elif name == "nr_cells":
            raise AttributeError("Cannot delete nr_cells")
        elif name == "name":
            raise AttributeError("Cannot delete name")
        elif name == "analysis":
            raise AttributeError("Cannot delete analysis")
        elif name == "parquet_path":
            raise AttributeError("Cannot delete parquet_path")
        elif name == "raw_path":
            raise AttributeError("Cannot delete raw_path")
        elif name == "load_path":
            raise AttributeError("Cannot delete load_path")

        super().__delattr__(name)

    def use_view_as_filter(
        self,
        filter_name="filter",
        filter_values="1,0",
        all_stimuli=True,
        view_name="spikes_df",
        dataframe="spikes_df",
    ):
        """Use the current self.df_view (which is a sub-frame from the complete
        dataframe) as a filter for the complete dataframe. This function will add a new column to the complete dataframe)

        Parameters
        ----------
        filter_name : str
            The name the new introduced filter shall have in
        filter_values : str
            The value the new introduced filter shall have in the new column.
        all_stimuli : boolean
            If the filter shall be applied to all stimuli, independent of the stimuli in the view.
        dataframe : str
            Name of the dataframe that is used as a filter.
        view_name : str
            Name of the view that is used as a filter.
        filter_values : list
            List of length 2, first value is the value that is assigned to the filter,
            second value is the value that is assigned to the rest of the dataframe.

        Returns
        -------
        None

        """
        if type(filter_values) is not list:
            if filter_values == "1,0":
                filter_values = [1, 0]
        else:
            assert len(filter_values) == 2, (
                "filter_values must be a list of length 2, "
                "first value is the value that is assigned to the filter, "
                "second value is the value that is assigned to the rest of the dataframe"
            )

        original_df = (
            self.dataframes[dataframe].copy().set_index(["recording", "cell_index"])
        )
        original_df[filter_name] = filter_values[1]

        if not all_stimuli:
            df_temp = self.views[view_name].tabulator.value.set_index(
                ["recording", "cell_index"]
            )
            df_temp[filter_name] = filter_values[0]

        else:
            pl_df = pl.from_pandas(self.dataframes[dataframe])

            rec_split_dfs = pl_df.partition_by("recording")

            for df in rec_split_dfs:
                cells = df["cell_index"].unique().to_numpy()

                mask = df.select(
                    [pl.col("cell_index").is_in(cells).alias("mask")]
                ).to_numpy()

                df_temp = original_df.loc[mask].copy()
                df_temp[filter_name] = filter_values[0]
                original_df.update(df_temp)

        self.dataframes[dataframe] = original_df.reset_index(drop=False)

    def extract_df_subset(
        self,
        dataframe="spikes_df",
        query_conditions: list[dict] = None,
    ):
        """
        Returns a subset of the dataframe depending on the filter parameters.

        Parameters
        ----------
        dataframe : str
            Name of the dataframe that shall be used to create the subset.
        query_conditions : list of dictionaries
            List of dictionaries that contain the filter parameters.
            Example:
            query_conditions = [{"stimulus_name": "fff", "stimulus_index": 1},

        """

        df = stimulus_spikes.filter_dataframe_complex(query_conditions)
        return df

    def add_column(self, series, dataframe="spikes_df", fill_value=0):
        """
        Updates a dataframe with a new column.

        Parameters
        ----------
        dataframe : str
            Name of the dataframe that shall be updated.
        series : pandas.Series
            Series that shall be added to the dataframe.
        fill_value : any
            Value that is used to fill the empty cells.


        Returns
        -------
        None.

        """
        series_dtype = series.dtype

        self.dataframes[dataframe][series.name] = fill_value
        self.dataframes[dataframe][series.name] = self.dataframes[dataframe][
            series.name
        ].astype(series_dtype)
        if series.index.name is not None:
            self.dataframes[dataframe] = self.dataframes[dataframe].set_index(
                series.index.name
            )
        else:
            self.dataframes[dataframe] = self.dataframes[dataframe].set_index(
                series.index.names
            )
        self.dataframes[dataframe].update(series)
        self.dataframes[dataframe] = self.dataframes[dataframe].reset_index(drop=False)

    def split_triggers(
        self,
        stimulus_df="stimulus_df",
        nr_splits=2,
        stimulus_indices=None,
        recordings=None,
    ):
        """
        Creates new (smaller, calculated) triggers for stimuli. It returns a new dataframe that contains the new triggers.
        The new triggers are calculated by splitting the old triggers in half *nr_splits.
        This can be useful if spikes from specific time intervals of the stimulus are of interest.

        Parameters
        ----------
        stimulus_df : str
            Name of the stimulus dataframe that shall be used.
        nr_splits : int
            Number of times the triggers shall be split.
        stimulus_indices : list of integers
            List of stimulus indices that shall be used to create the new triggers. If not specified, all stimuli are used.

        Returns
        -------
        stimulus_df : pandas.DataFrame
            A dataframe that contains the new triggers.

        """

        if stimulus_indices is not None:
            stimulus_df = self.dataframes[stimulus_df].loc[stimulus_indices]
        else:
            stimulus_df = self.dataframes[stimulus_df]

        nr_stimuli = len(stimulus_df)
        repeat_logic = stimulus_df["stimulus_repeat_logic"].to_numpy()
        old_triggers = np.vstack(stimulus_df["trigger_fr_relative"].to_numpy())
        new_triggers, new_intervals = stimulus_dfs.split_triggers(
            old_triggers, nr_splits
        )
        stimulus_df["trigger_fr_relative"] = np.array_split(
            new_triggers.flatten(), nr_stimuli, axis=0
        )
        stimulus_df["trigger_int"] = np.array_split(
            new_intervals.flatten(), nr_stimuli, axis=0
        )
        stimulus_df["trigger_ends"] = np.array_split(
            new_triggers[:, 1:].flatten(), nr_stimuli, axis=0
        )
        stimulus_df["stimulus_repeat_logic"] = repeat_logic * (2**nr_splits)
        return stimulus_df


class Recording_s(Recording):
    def __init__(self, analysis_path, analysis_name):
        super().__init__(
            "",
            "",
            dataframes={"spikes_df": None, "stimulus_df": None},
        )
        self.store_path = analysis_path
        self.name = analysis_name
        self.recordings = {}
        self.nr_recordings = 0
        self.analysis = {}

    def add_recording(self, recording):
        """
        Adds a recording to the Recording_s object. The recording is added to the recordings dictionary and the
        dataframes are updated.

        Parameters
        ----------
        recording : Recording object
            The recording that shall be added to the Recording_s object.

        Returns
        -------
        self : updated Recording_s object
        """
        self.recordings[recording.name] = recording
        self.nr_recordings += 1
        self.create_combined_dfs()
        self.create_secondary_dfs()
        self.nr_cells = self.nr_cells + recording.nr_cells
        self.nr_stimuli = self.nr_stimuli + recording.nr_stimuli

    def remove_recording(self, recording):
        """
        Removes a recording from the Recording_s object. The recording is removed from the recordings dictionary and the
        dataframes are updated.

        Parameters
        ----------
        recording : str
            Name of the recording that shall be removed from the Recording_s object.
        """
        recording = self.recordings[recording]
        self.synchronize_dataframes()
        self.recordings.pop(recording.name)
        self.nr_recordings -= 1
        self.create_combined_dfs()
        self.create_secondary_dfs()
        self.nr_cells = self.nr_cells - recording.nr_cells
        self.nr_stimuli = self.nr_stimuli - recording.nr_stimuli

    @classmethod
    def load_from_single(cls, analysis_path, analysis_name, recording_path):
        """
        Loads a single recording and creates a Recording_s object.

        Parameters
        ----------
        analysis_path : str
            Path to the analysis folder (standard save folder).
        analysis_name : str
            Name of the analysis.
        recording_path : str
            Path to the recording that shall be loaded.

        Returns
        -------
        recordings : Recording_s object
            The Recording_s object that contains the loaded recording.
        """
        obj = Recording.load(recording_path)
        obj = version_control(obj)
        recordings = Recording_s(analysis_path, analysis_name)
        recordings.add_recording(obj)
        return recordings

    def add_from_saved(self, recording_path):
        """
        Adds a recording to the Recording_s object from a saved recording file.

        Parameters
        ----------
        recording_path : str
            Path to the recording that shall be loaded.
        """
        obj = Recording.load(recording_path)
        obj = version_control(obj)
        self.add_recording(obj)

    def create_combined_dfs(self):
        """
        Creates the combined spikes_df and stimulus_df from all recordings.

        """
        dfs = []
        for recording in self.recordings.values():
            dfs.append(recording.spikes_df)
        self.dataframes["spikes_df"] = pd.concat(dfs).reset_index(drop=True)

        dfs = []
        for recording in self.recordings.values():
            dfs.append(recording.stimulus_df)
        self.dataframes["stimulus_df"] = pd.concat(dfs).reset_index(drop=True)

    def create_secondary_dfs(self):
        """
        Creates the secondary dataframes from all recordings.

        """
        for recording in self.recordings.values():
            dfs = list(recording.dataframes.keys())
            dfs.remove("spikes_df")
            dfs.remove("stimulus_df")
            for df in dfs:
                if df not in self.dataframes.keys():
                    self.dataframes[df] = recording.dataframes[df]
                else:
                    self.dataframes[df] = pd.concat(
                        [self.dataframes[df], recording.dataframes[df]]
                    ).reset_index(drop=True)

    def spikes_for_analysis(self, analysis_name, **kwargs):
        """
        Returns the spikes that are used for a specific analysis.

        Parameters
        ----------
        analysis_name : str
            Name of the analysis that shall be used.

        Returns
        -------
        spikes_df : pandas.DataFrame
            A dataframe that contains the spikes that are used for a specific analysis.
        """
        analysis = self.analysis[analysis_name]
        cell_df = analysis["cell_df"]
        stimulus_df = analysis["stimulus_df"]
        return self.get_spikes_df(cell_df, stimulus_df, **kwargs)

    def get_spikes_chunked(
        self,
        chunk_size,
        chunk_position,
        recordings,
        cells,
        stimuli,
        time="seconds",
        waveforms=False,
        pandas=True,
    ):
        if recordings[0] == "all":
            recordings = self.recordings.keys()

        keys = list(recordings)
        start_index = chunk_size * chunk_position
        end_index = start_index + chunk_size
        recordings = keys[start_index:end_index]
        df = self.get_spikes_triggered(
            recordings, cells, stimuli, time, waveforms, pandas
        )
        return df

    def get_stimuli(self, recording, stimuli, stimulus_df="stimulus_df"):
        return self.dataframes[stimulus_df].query(
            f"recording == {recording} & stimulus_index == {stimuli}"
        )

    def find_stim_indices(self, recordings, stimulus_names, stimulus_df="stimulus_df"):
        self.synchronize_dataframes()
        if recordings[0] == "all":
            recordings = self.recordings.keys()
        indices = []

        for recording in recordings:
            indices.append(
                super().find_stim_indices(stimulus_names, stimulus_df, recording)
            )
        return indices, [[rec] for rec in recordings]

    def synchronize_dataframes(self):
        """
        This function synchronizes the dataframes of all recordings according to changes made to the
        dataframes of the recording_s object.

        """

        for df_name in self.dataframes:
            for recording in self.recordings.values():
                df_temp = self.dataframes[df_name].query("recording == @recording.name")
                self.recordings[recording.name].dataframes[df_name] = df_temp

    def dummy_objects(self, recordings):
        self.synchronize_dataframes()
        paths = []
        for recording_name in recordings:
            recording = self.recordings[recording_name]
            new_rec = Recording(
                recording.parquet_path,
                "",
                recording.dataframes,
                recording.sampling_freq,
            )
            new_rec.save(f"{self.store_path}//{recording.name}")
            paths.append(f"{self.store_path}//{recording.name}")
        return paths

    def delete_df(self, df_name):
        """
        Deletes a dataframe from the recording_s object.

        Parameters
        ----------
        df_name : str
            Name of the dataframe that shall be deleted.
        """
        assert df_name != "spikes_df", "Cannot delete spikes_df"
        assert df_name != "stimulus_df", "Cannot delete stimulus_df"
        for recording in self.recordings.values():
            recording.delete_df(df_name)
            recording.save(recording.load_path)

    def __delattr__(self, name):
        super().__delattr__(name)
        if name == "recordings":
            raise AttributeError("Cannot delete recordings")
        elif name == "nr_recordings":
            raise AttributeError("Cannot delete nr_recordings")
        elif name == "store_path":
            raise AttributeError("Cannot delete store_path")
