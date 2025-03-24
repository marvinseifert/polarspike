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

import stimulus_dfs
import stimulus_spikes
import polars as pl
from grid import Table
import filter_stimulus
from dataclasses import dataclass, field
import cells_and_stimuli, stimulus_trace, recordings_stimuli_cells, spiketrains
from threading import Thread
import warnings
from pathlib import Path


def spike_load_worker(args):
    path, stimuli, cells, time, waveforms, cell_df, stimulus_df = args
    obj = Recording.load(path)
    df = obj.get_spikes_triggered(
        stimuli,
        cells,
        time,
        waveforms,
        stimulus_df=stimulus_df,
        cell_df=cell_df,
        pandas=False,
    )
    df = df.with_columns(recording=pl.lit(obj.name))
    return df


def version_control(obj):
    try:
        obj.parquet_path = obj.path
        del obj.path
    except AttributeError:
        pass

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
    parquet_path: str  # Path, pointing to the parquet file which stores the spiketimestamps
    raw_path: str  # Path, pointing to the raw file which stores the raw data
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

    def get_spikes_triggered(  # Function works on single recording only
        self,
        stimuli,
        cells,
        time="seconds",
        waveforms=False,
        pandas=True,
        stimulus_df="stimulus_df",
        cell_df="spikes_df",
        carry=None,
    ):
        """
        This function returns a dataframe that contains the spikes that were recorded during the presentation
        of a specific stimulus. Spikes are loaded from the connected parquet file.
        len(cells) must be equal to len(stimuli) or 1.
        len(stimuli) must be equal to len(cells) or 1.

        Parameters
        ----------
        stimuli : list of lists of strings or integers
            List of lists that contains the stimulus indices, or names that shall be loaded. The first list contains the stimulus
            indices of the first stimulus, the second list contains the stimulus indices of the second stimulus and so
            on. If a single list is provided, the same stimuli are used for all cells.
        cells : list of lists
            List of lists that contains the cell indices that shall be loaded. The first list contains the cell indices
            of the first stimulus, the second list contains the cell indices of the second stimulus and so on.
            If a single list is provided, the same cells are used for all stimuli.
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
        carry : list
            List of columns that shall be carried over from the cell_df to the final spikes df


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
        stimuli = filter_stimulus.stim_names_to_indices(
            stimuli, self.dataframes[stimulus_df]
        )
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
        if len(df) == 0:
            warnings.warn("No spikes found for the provided parameters.")
        if carry:
            df = df.sort(["cell_index", "stimulus_index"])
            cell_stimuli = df[["cell_index", "stimulus_index"]].to_numpy()
            # Create multiindex for panadas
            index = pd.MultiIndex.from_arrays(
                cell_stimuli.T, names=["cell_index", "stimulus_index"]
            )
            df_temp = self.dataframes[cell_df].set_index("stimulus_index")
            df_temp = df_temp.loc[np.asarray(stimuli).flatten()]
            df_temp = df_temp.reset_index(drop=False).set_index(
                ["cell_index", "stimulus_index"]
            )
            df_add = pl.from_pandas(df_temp.loc[index][carry])
            df = pl.concat([df, df_add], how="horizontal")
        if pandas:
            return df.to_pandas()
        else:
            return df

    def spike_loading_assertions(self, stimuli, cells):
        if len(stimuli[0]) > 1:
            assert (
                len(stimuli[0]) == len(cells[0]) or len(cells[0]) == 1
            ), "If a list with multiple stimuli is provided the len of the cells list must be 1 or equal to len of stimuli."
        if len(cells) > 1:
            assert len(cells) == len(
                stimuli
            ), "If multiple cells are provided, the same amount of stimuli must be provided as well."

    def get_triggered(  # Function works on single recording only
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
        sub_df = filter_stimulus.find_stimuli(self.dataframes[stimulus_df], stimulus[0])
        # Extract the necessary information from the stimulus dataframe:
        begin_end, trigger, trigger_end, stim_logic = filter_stimulus.get_stimulus_info(
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

    def get_spikes_as_numpy(  # Function works with single recording only
        self,
        stimulus,
        cells,
        time="seconds",
        waveforms=False,
        relative=True,
        stimulus_df="stimulus_df",
        cell_df="spikes_df",
    ):
        """
        This function returns the spikes that were recorded during the presentation of a specific stimulus as a numpy array.
        Spikes are loaded from the connected parquet file.
        Only a single stimulus can be loaded at a time.

        Parameters
        ----------
         stimulus : list, single integer
            The stimulus index that shall be loaded.
        cells : list of integers
            List of cell indices that shall be loaded.
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

        Returns
        -------
        spikes_numpy : numpy array
            A numpy array that contains the spikes that were recorded during the presentation of a specific stimulus.
            The array is structured as follows: [cell_index, spikes]

        """

        # The number of cells that are supposed to be loaded, used below to ensure empty cells are returned as empty arrays
        assert (
            len(stimulus) == 1
        ), "Only a single stimulus can be loaded at a time and returned as array."
        assert (
            len(cells) == 1
        ), "Only a single stimulus can be loaded at a time and returned as array."

        spikes = self.get_spikes_triggered(
            stimulus, cells, time, waveforms, False, stimulus_df, cell_df
        )
        spikes_numpy = spiketrains.collect_as_arrays(
            spikes, "cell_index", "times_triggered", "spikes"
        )
        return spikes_numpy["cell_index"].to_numpy(), spikes_numpy["spikes"].to_numpy()

    def get_spikes_df(
        self,
        cell_df="spikes_df",
        stimulus_df="stimulus_df",
        time="seconds",
        waveforms=False,
        pandas=True,
    ):
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

        # Create the inputs to the get_spikes_triggered function:
        try:
            input_df = pl.from_pandas(
                self.dataframes[cell_df][["recording", "stimulus_index", "cell_index"]]
            )
        except KeyError:
            warnings.warn(
                "The cell dataframe does not contain the required columns. Accidentally provided a stimulus dataframe?"
            )
            return

        stimuli = input_df["stimulus_index"].unique().to_list()

        stim_list = [
            [stim_id]
            for stim_id in pl.from_pandas(self.dataframes[stimulus_df])
            .unique("stimulus_index")
            .filter(pl.col("stimulus_index").is_in(stimuli))["stimulus_index"]
            .to_list()
        ]
        input_df = input_df.filter(
            pl.col("stimulus_index").is_in(np.asarray(stim_list).flatten().tolist())
        )
        stim_df = input_df.partition_by("stimulus_index")
        cell_list = [df["cell_index"].to_list() for df in stim_df]

        # Call the get_spikes_triggered function:

        df = self.get_spikes_triggered(
            stim_list,
            cell_list,
            time=time,
            waveforms=waveforms,
            pandas=pandas,
            stimulus_df=stimulus_df,
            cell_df=cell_df,
        )

        return df

    def organize_recording_parameters(
        self, recordings, stimuli, cells, stimulus_df="stimulus_df", all_recordings=None
    ):
        # Get the input in the correct format:
        # Check if the nr of recordings is equal to the nr of stimuli and cells:
        # If either list is a single list, use the same stimuli and cells for all recordings.

        input_new = recordings_stimuli_cells.sort(
            [recordings, stimuli, cells], all_recordings
        )
        recordings, stimuli, cells = input_new
        new_stimuli = []
        new_cells = []
        for recording, stimulus, cell in zip(recordings, stimuli, cells):
            sub_stim_list = []
            sub_cell_list = []
            all_stimuli = [
                [item]
                for item in self.dataframes[stimulus_df].query(
                    "recording == @recording"
                )["stimulus_index"]
            ]
            for sub_stim in stimulus:
                if sub_stim[0] == "all":
                    sub_stim_list.append(all_stimuli)
                    sub_cell_list.append(cell * len(all_stimuli))
                else:
                    sub_stim_list.append(sub_stim)
                    sub_cell_list.append(cell * len(sub_stim))
            new_stimuli.append(sub_stim_list)
            new_cells.append(sub_cell_list[0])
        return recordings, new_stimuli, new_cells

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

    def find_stim_indices(
        self, stimulus_names, stimulus_df="stimulus_df", recording=None
    ):
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
        if recording is None:
            recording = self.name
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

    def delete_df(self, df_name):
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
        self.views = {}
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save_save(self):
        """ """
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
            obj.load_path = filename

            # Update earlier versions of the recording class:
            obj = version_control(obj)

            return obj

    def __delattr__(self, name):
        if name == "dataframes":
            raise AttributeError("Cannot delete dataframes")
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

    def get_spikes_df(
        self,
        cell_df="spikes_df",
        stimulus_df="stimulus_df",
        time="seconds",
        waveforms=False,
        pandas=True,
        carry=None,
    ):
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

        # Create the inputs to the get_spikes_triggered function:
        try:
            input_df = pl.from_pandas(
                self.dataframes[cell_df][["recording", "stimulus_index", "cell_index"]]
            )
        except KeyError:
            warnings.warn(
                "The cell dataframe does not contain the required columns. Accidentally provided a stimulus dataframe?"
            )
            return
        filter_dict = stimulus_trace.create_filter_dict(input_df, self.stimulus_df)
        # Store all the file paths in a dictionary
        files_dict = {
            rec: self.recordings[rec].parquet_path for rec in list(filter_dict.keys())
        }

        # Next, we create a dictionary with all the filters applied to the lazy dataframes
        df = stimulus_spikes.load_triggered_lazy(files_dict, filter_dict, time)

        # Sort data in a meaningful way
        df = df.sort(
            ["cell_index", "repeat", "stimulus_index", "trigger", "times_triggered"]
        )
        # Carry columns if needed
        if carry:
            df = df.sort(["recording", "cell_index", "stimulus_index"])

            df = df.to_pandas().set_index(["recording", "cell_index", "stimulus_index"])

            for column in carry:
                df[column] = 0
                df[column] = df[column].astype(self.dataframes[cell_df][column].dtype)

            temp_df = self.dataframes[cell_df].set_index(
                ["recording", "cell_index", "stimulus_index"]
            )
            df.update(temp_df[carry])

            df = df.reset_index()
            df = pl.from_pandas(df)

        if pandas:
            return df.to_pandas()
        else:
            return df

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

    def get_spikes_triggered(
        self,
        recordings,
        stimuli,
        cells,
        time="seconds",
        waveforms=False,
        pandas=True,
        cell_df="spikes_df",
        stimulus_df="stimulus_df",
    ):
        # self.synchronize_dataframes()

        all_recordings = [[recording] for recording in self.recordings.keys()]
        recordings, stimuli, cells = self.organize_recording_parameters(
            recordings, stimuli, cells, stimulus_df, all_recordings
        )
        nr_cpus = mp.cpu_count()
        if nr_cpus > len(recordings):
            nr_cpus = len(recordings)
        # Now we have lists of list matching the recordings, cells and stimuli. But, we still need to fill in
        # the "all" cells:
        new_cells = []
        for recording, cell_list in zip(recordings, cells):
            stim_cells_new = []
            for stim_cell_list in cell_list:
                cell_sub_list = []
                for cell in stim_cell_list[:1]:
                    if cell == "all":
                        cell_sub_list.append(
                            self.dataframes[cell_df]
                            .query("recording == @recording")["cell_index"]
                            .unique()
                            .tolist()
                        )

                    else:
                        cell_sub_list.append(stim_cell_list)
                stim_cells_new.append(cell_sub_list[0])
            new_cells.append(stim_cells_new)
        cells = new_cells

        # Identify the stimulus indices in the recordings if stimulus name was provided.
        paths = self.dummy_objects([rec[0] for rec in recordings])

        with mp.Pool(nr_cpus) as pool:
            # Pass all required arguments to the worker function
            dfs = pool.map(
                spike_load_worker,
                [
                    (
                        rec,
                        stimulus_list,
                        cell_list,
                        time,
                        waveforms,
                        cell_df,
                        stimulus_df,
                    )
                    for rec, stimulus_list, cell_list in zip(paths, stimuli, cells)
                ],
            )

        df = pl.concat(dfs)
        if len(df) == 0:
            warnings.warn("No spikes found for the provided parameters.")
        if pandas:
            return df.to_pandas()
        else:
            return df

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
