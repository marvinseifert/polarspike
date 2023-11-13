# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:23:06 2021

@author: Marvin
"""

import contextlib
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import traceback
import pickle
import stimulus_spikes
import polars as pl
from grid import Table
import df_filter
from dataclasses import dataclass, field
import cells_and_stimuli
import stimulus_trace


def spike_load_worker(args):
    path, cells, stimuli, time, waveforms = args
    obj = Recording.load(path)
    df = obj.get_spikes_triggered(cells, stimuli, time, waveforms, pandas=False)
    df = df.with_columns(recording=pl.lit(obj.name))
    return df


def version_control(obj):
    obj.parquet_path = obj.path
    del obj.path
    return obj


@dataclass
class Recording:
    parquet_path: str  # Path, pointing to the parquet file which stores the spiketimestamps
    name: str  # Recording name
    dataframes: dict = field(
        default_factory=lambda: {}, init=True
    )  # Dictionary that stores the dataframes
    sampling_freq: float = field(
        default_factory=lambda: 1.0, init=True
    )  # The sampling frequency of the recording
    views: dict = field(
        default_factory=lambda: {}, init=True
    )  # Dictionary that stores the views of dataframes
    load_path: str = field(
        default_factory=lambda: "", init=True
    )  # If a class instance was saved, the path to the saved file is stored here.
    nr_stimuli: int = field(init=False)  # Nr of stimuli in the recording
    nr_cells: int = field(init=False)  # Nr of cells in the recording

    def __post_init__(self):
        """
        This function is called after the class is initialized. It updates some of the class attributes
        with the information from inout parameters of the init function.

        Updated parameters:
            - nr_stimuli
            - nr_cells

        """
        if self.dataframes["spikes_df"] is not None:
            self.nr_stimuli = len(self.dataframes["stimulus_df"])
            self.nr_cells = np.unique(self.dataframes["spikes_df"]["cell_index"]).shape[
                0
            ]
        else:
            self.nr_stimuli = 0
            self.nr_cells = 0

    def get_spikes_triggered(
        self,
        cells,
        stimuli,
        time="seconds",
        waveforms=False,
        pandas=True,
        stimulus_df="stimulus_df",
        cell_df="spikes_df",
    ):
        """
        This function returns a dataframe that contains the spikes that were recorded during the presentation
        of a specific stimulus. Spikes are loaded from the connected parquet file.
        len(cells) must be equal to len(stimuli) or 1.
        len(stimuli) must be equal to len(cells) or 1.

        Parameters
        ----------
        cells : list of lists
            List of lists that contains the cell indices that shall be loaded. The first list contains the cell indices
            of the first stimulus, the second list contains the cell indices of the second stimulus and so on.
            If a single list is provided, the same cells are used for all stimuli.
        stimuli : list of lists of strings or integers
            List of lists that contains the stimulus indices, or names that shall be loaded. The first list contains the stimulus
            indices of the first stimulus, the second list contains the stimulus indices of the second stimulus and so
            on. If a single list is provided, the same stimuli are used for all cells.
        time : str
            Defines the time unit of the returned dataframe. Can be "seconds" or "frames".
        waveforms : boolean
            If the waveforms shall be loaded as well. Only possible if the parquet file contains the waveforms.
        pandas : boolean
            If the returned dataframe shall be a pandas dataframe or a polars dataframe.
        stimulus_df : str
            Name of the stimulus dataframe that shall be used.
        cell_df : str
            Name of the cell dataframe that shall be used.

        Returns
        -------
        df : polars.DataFrame or pandas.DataFrame
            A dataframe that contains the spikes (and waveforms) that were recorded during the presentation of a specific
            stimulus and information about trigger and stimulus repeats.
            Carefull: If a single stimulus was provided as string, the returned dataframe may contain multiple
            stimuli, if the same stimulus was presented multiple times.
            Use obj.find_stim_indices([stimulus_str]) to find all indices referring to a single name.

            Carefull, cells, that did not spike during the presentation of a stimulus are not included in the dataframe.
        """

        # Check if the stimuli are strings or integers. If they are strings, they need to be converted to integers.
        stimuli = df_filter.stim_names_to_indices(stimuli, self.dataframes[stimulus_df])
        # Check if len(stimuli) is equal to len(cells). If not, match the stimuli to the cells.
        stimuli, cells = cells_and_stimuli.sort(
            stimuli, cells, self.dataframes[cell_df]["cell_index"].unique()
        )

        dfs = []
        # Loop over all stimuli and all cells and load the spikes (and waveforms):
        for stimulus, cell in zip(stimuli, cells):
            df = self.get_triggered(cell, stimulus, time, waveforms, stimulus_df)
            df = df.with_columns(stimulus_index=pl.lit(stimulus[0]))
            dfs.append(df)
        df = pl.concat(dfs)  # Create one polars dataframe
        if pandas:
            return df.to_pandas()
        else:
            return df

    def get_triggered(
        self,
        cells,
        stimulus,
        time="seconds",
        waveforms=False,
        stimulus_df="stimulus_df",
    ):
        """

        This function returns a dataframe that contains the spikes that were recorded during the presentation
        of a specific stimulus. Spikes are loaded from the connected parquet file.

        Parameters
        ----------
        cells : list of integers
            List of cell indices that shall be loaded.
        stimulus : list of integers
            List of stimulus indices that shall be loaded.
        time : str
            Defines the time unit of the returned dataframe. Can be "seconds" or "frames".
        waveforms : boolean
            If the waveforms shall be loaded as well. Only possible if the parquet file contains the waveforms.
        stimulus_df : str
            Name of the stimulus dataframe that shall be used. (spikes_df is not need, because cell indices are absolute)

        Returns
        -------
        df : polars.DataFrame
            A dataframe that contains the spikes (and waveforms) that were recorded during the presentation of a specific
        """
        # Get the stimulus information for the specific stimulus:
        sub_df = df_filter.find_stimuli(self.dataframes[stimulus_df], stimulus)
        # Extract the necessary information from the stimulus dataframe:
        begin_end, trigger, trigger_end, stim_logic = df_filter.get_stimulus_info(
            sub_df
        )
        # Load the spikes from the parquet file, sorted by trigger and repeats:
        df = stimulus_spikes.load_triggered(
            cells,
            begin_end[0, 0],
            begin_end[0, 1],
            trigger,
            trigger_end,
            stim_logic,
            self.parquet_path,
            waveforms,
        )
        # Convert the times to seconds if necessary:
        if time == "seconds":
            df = df.with_columns(
                pl.col("times").truediv(self.sampling_freq).alias("times")
            )
            df = df.with_columns(
                pl.col("times_relative")
                .truediv(self.sampling_freq)
                .alias("times_relative")
            )
            df = df.with_columns(
                pl.col("times_triggered")
                .truediv(self.sampling_freq)
                .alias("times_triggered")
            )
        return df

    def get_spikes_as_numpy(
        self,
        cells,
        stimulus,
        time="seconds",
        waveforms=False,
        relative=True,
        stimulus_df="stimulus_df",
        cell_df="spikes_df",
    ):
        """
        This function returns the spikes that were recorded during the presentation of a specific stimulus as a numpy array.
        Spikes are loaded from the connected parquet file.
        Only a single stimulus can be loaded at a time. Stimulus names are not valid input.

        Parameters
        ----------
        cells : list of integers
            List of cell indices that shall be loaded.
        stimulus : list, single integer
            The stimulus index that shall be loaded.
        time : str
            Defines the time unit of the returned dataframe. Can be "seconds" or "frames".
        waveforms : boolean
            If the waveforms shall be loaded as well. Only possible if the parquet file contains the waveforms.
        relative : boolean
            If the spikes shall be returned relative to the stimulus onset or absolute (begin of the recording).
        stimulus_df : str
            Name of the stimulus dataframe that shall be used.
        cell_df : str
            Name of the cell dataframe that shall be used, only use case is if cells == "all"

        """

        # The number of cells that are supposed to be loaded, used below to ensure empty cells are returned as empty arrays
        if cells[0] == "all":
            cells = np.unique(self.dataframes[cell_df]["cell_index"]).tolist()
        nr_cells = np.unique(cells)
        if relative:
            col_name = "times_relative"
        else:
            col_name = "times"
        # Load the information about the stimulus:
        sub_df = df_filter.find_stimuli(self.dataframes[stimulus_df], stimulus)
        # Extrat the stimulus information. Since we return spikes only as indices,
        # we only need the beginning and end of the stimulus
        begin_end, _, _, _ = df_filter.get_stimulus_info(sub_df)
        # Load the spikes from the parquet file, sorted by
        df = stimulus_spikes.stimulus_relative(
            cells, begin_end[0, 0], begin_end[0, 1], self.parquet_path, waveforms
        )
        # Create the output array
        data = np.empty_like(nr_cells, dtype=object)
        # Convert time
        if time == "frames":
            series = df.group_by("cell_index").agg(
                pl.col(col_name).apply(lambda x: np.array(x)).alias("result")
            )
        else:
            series = df.group_by("cell_index").agg(
                (pl.col(col_name) / self.sampling_freq)
                .apply(lambda x: np.array(x))
                .alias("result")
            )
        # Convert the polars series to a numpy array (this only contains cells with spikes)
        spiky_cells = series.to_numpy()  # This contains only the cells with spikes

        # Need to correct the index, because the group_by function does not return the index of the original dataframe
        spiky_cells[:, 0] = spiky_cells[:, 0] - df["cell_index"].min()
        # Fill the empty cells with empty arrays:
        data[spiky_cells[:, 0].astype(int)] = spiky_cells[:, 1]
        return data, spiky_cells[:, 0]

    def show_df(self, name="spikes_df", level=False, condition=False, viewname=None):
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

    def filtered_df(self, df_name):
        """
        Returns the filtered dataframe of a specific view.

        Parameters
        ----------
        df_name : str
            Name of the dataframe that shall be returned.
        """
        return self.views[df_name].tabulator.value

    def find_stim_indices(self, stimulus_names):
        """Returns the stimulus indices that correspond to the stimulus names.

        Parameters
        ----------
        stimulus_names : list of strings
            List of stimulus names that shall be converted to stimulus indices.
        Returns
        -------
        stim_indices : list of lists
            List of stimulus indices that correspond to the stimulus names.

        """
        stim_indices = []
        for stimulus in stimulus_names:
            stim_indices.append(
                self.stimulus_df.query(f"stimulus_name=='{stimulus}'")[
                    "stimulus_index"
                ].to_list()
            )
        stim_indices = [[item] for sublist in stim_indices for item in sublist]
        return stim_indices

    @property
    def spikes_df(self):
        """Returns the spikes_df dataframe. Shortcut version
        Warning: You cannot update this dataframe, use self.dataframes["spikes_df"] instead
        """
        return self.dataframes["spikes_df"]

    @property
    def stimulus_df(self):
        """Returns the stimulus_df dataframe. Shortcut version
        Warning: You cannot update this dataframe, use self.dataframes["stimulus_df"] instead
        """
        return self.dataframes["stimulus_df"]

    def save(self, filename):
        """
        Save function that is doing the actual saving after non-picklable attributes have been removed.
        Parameters
        ----------
        filename : str
            The name of the file to save the object to.

        Returns
        -------

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load a saved object from a file.
        Parameters
        ----------
        filename : str
            The name of the file to load.
        Returns
        -------
        obj : object
            The object stored in the file.

        """

        with open(filename, "rb") as f:
            obj = pickle.load(f)
            obj.load_path = filename

            # Update earlier versions of the recording class:
            obj = version_control(obj)

            return obj

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
        filter_value : str
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

        if not all_stimuli:
            df_temp = self.views[view_name].tabulator.value
            df_temp[filter_name] = filter_values[0]

        else:
            cells = (
                self.views[view_name].tabulator.value["cell_index"].unique().tolist()
            )
            pl_df = pl.from_pandas(self.dataframes[dataframe])
            mask = pl_df.select(
                [pl.col("cell_index").is_in(cells).alias("mask")]
            ).to_numpy()
            df_temp = self.dataframes[dataframe].loc[mask].copy()
            df_temp[filter_name] = filter_values[0]

        self.dataframes[dataframe][filter_name] = filter_values[1]
        self.dataframes[dataframe].update(df_temp)

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

    def extract_df_subset(
        self,
        cell_index=None,
        stimulus_index=None,
        stimulus_name=None,
        recording=None,
        dataframe="spikes_df",
    ):
        """
        Returns a subset of the dataframe depending on the filter parameters.

        Parameters
        ----------
        cell_index : list of integers
            List of cell indices that shall be included in the subset.
        stimulus_index : list of integers
            List of stimulus indices that shall be included in the subset.
        stimulus_name : list of strings
            List of stimulus names that shall be included in the subset.
        recording : list of strings
            List of recording names that shall be included in the subset.
        dataframe : str
            Name of the dataframe that shall be used to create the subset.


        """
        if cell_index is None:
            cell_index = self.dataframes[dataframe]["cell_index"].unique().tolist()
        if stimulus_index is None:
            stimulus_index = (
                self.dataframes[dataframe]["stimulus_index"].unique().tolist()
            )
        if stimulus_name is None:
            stimulus_name = (
                self.dataframes[dataframe]["stimulus_name"].unique().tolist()
            )
        if recording is None:
            recording = self.dataframes[dataframe]["recording"].unique().tolist()

        df_temp = pl.from_pandas(self.dataframes[dataframe])
        df_temp = df_temp.filter(
            pl.col("cell_index").is_in(cell_index)
            & pl.col("stimulus_index").is_in(stimulus_index)
            & pl.col("stimulus_name").is_in(stimulus_name)
            & pl.col("recording").is_in(recording)
        )
        return df_temp.to_pandas()

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
        self, stimulus_df="stimulus_df", nr_splits=2, stimulus_indices=None
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
        new_triggers, new_intervals = stimulus_trace.split_triggers(
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
            analysis_path,
            analysis_name,
            dataframes={"spikes_df": None, "stimulus_df": None},
        )
        self.recordings = {}
        self.nr_recordings = 0

    def add_recording(self, recording):
        self.recordings[recording.name] = recording
        self.nr_recordings += 1
        self.create_combined_df()
        self.nr_cells = self.nr_cells + recording.nr_cells
        self.nr_stimuli = self.nr_stimuli + recording.nr_stimuli

    def remove_recording(self, recording):
        self.recordings.pop(recording.name)
        self.nr_recordings -= 1
        self.create_combined_df()
        self.nr_cells = self.nr_cells - recording.nr_cells
        self.nr_stimuli = self.nr_stimuli - recording.nr_stimuli

    def create_combined_df(self):
        dfs = []
        for recording in self.recordings.values():
            dfs.append(recording.spikes_df)
        self.dataframes["spikes_df"] = pd.concat(dfs).reset_index(drop=True)

        dfs = []
        for recording in self.recordings.values():
            dfs.append(recording.stimulus_df)
        self.dataframes["stimulus_df"] = pd.concat(dfs).reset_index(drop=True)

    # @property
    # def spikes_df(self):
    #     """Returns the combined spikes_df of all recordings"""
    #     dfs = []
    #     for recording in self.recordings.values():
    #         dfs.append(recording.spikes_df)
    #     return pd.concat(dfs).reset_index(drop=True)
    #
    # @property
    # def stimulus_df(self):
    #     """Returns the combined stimulus_df of all recordings"""
    #     dfs = []
    #     for recording in self.recordings.values():
    #         dfs.append(recording.stimulus_df)
    #     return pd.concat(dfs).reset_index(drop=True)

    # def get_spikes_triggered(
    #     self, recording, cells, stimuli, time="seconds", waveforms=False, pandas=True
    # ):
    #     df = []
    #     for recording, cells_r, stimuli_r in zip(
    #         self.recordings.keys(), cells, stimuli
    #     ):
    #         temp_df = self.recordings[recording].get_spikes_triggered(
    #             cells_r, stimuli_r, time, waveforms, pandas=False
    #         )
    #         temp_df = temp_df.with_columns(recording=pl.lit(recording))
    #         df.append(temp_df)
    #     df = pl.concat(df)
    #     if pandas:
    #         return df.to_pandas()
    #     else:
    #         return df

    def get_spikes_triggered(
        self, recordings, cells, stimuli, time="seconds", waveforms=False, pandas=True
    ):
        nr_cpus = mp.cpu_count()
        if nr_cpus > len(recordings):
            nr_cpus = len(recordings)

        recordings, cells, stimuli = self.organize_recording_parameters(
            recordings, cells, stimuli
        )

        # Identify the stimulus indices in the recordings if stimulus name was provided.

        stimuli_temp = []
        for idx, rec, stimulus in zip(range(len(recordings)), recordings, stimuli):
            stimulus_rec_temp = []
            for stim_rec in stimulus:
                if type(stim_rec[0]) is str:
                    stimulus_rec_temp.append(
                        self.recordings[rec[0]].find_stim_indices(stim_rec)
                    )
                else:
                    stimulus_rec_temp.append(stim_rec)

            stimuli_temp.append(stimulus_rec_temp)

        stimuli = stimuli_temp

        with mp.Pool(nr_cpus) as pool:
            # Pass all required arguments to the worker function
            dfs = pool.map(
                spike_load_worker,
                [
                    (
                        self.recordings[rec[0]].load_path,
                        c,
                        s,
                        time,
                        waveforms,
                    )
                    for rec, c, s in zip(recordings, cells, stimuli)
                ],
            )

        df = pl.concat(dfs)
        if pandas:
            return df.to_pandas()
        else:
            return df

    def organize_recording_parameters(self, recordings, cells, stimuli):
        recordings_temp = []
        for recording in recordings:
            if recording[0] == "all":
                for key in self.recordings.keys():
                    recordings_temp.append([key])
            else:
                recordings_temp.append([recording])
        recordings = recordings_temp

        if len(cells) != len(recordings):
            cells = cells * len(recordings)
        if len(stimuli) != len(recordings):
            stimuli = stimuli * len(recordings)

        stimuli_temp = []
        cells_temp = []

        for recording, stimulus, cell in zip(recordings, stimuli, cells):
            stimulus_new, cell_new = cells_and_stimuli.sort(
                [stimulus], [cell], self.recordings[recording[0]].nr_cells
            )
            stimuli_temp.append(stimulus_new)
            cells_temp.append(cell_new)
        cells = cells_temp
        stimuli = stimuli_temp

        return recordings, cells, stimuli

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
