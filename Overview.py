# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:23:06 2021

@author: Marvin
"""

import contextlib
import pandas as pd
import numpy as np
import h5py
import backbone
from importlib import reload

import matplotlib.pyplot as plt
import math
import multiprocessing as mp
from functools import partial
import traceback
import pickle
import plotly
from ipywidgets import widgets
import plotly.graph_objects as go
import stimulus_spikes
import polars as pl

from grid import Table
import df_filter
from dataclasses import dataclass, field




@dataclass
class Recording:

    path: str
    name: str
    dataframes: dict = field(default_factory=lambda: {}, init=True)
    sampling_freq: float = field(default_factory=lambda: 1.0, init=True)
    views: dict = field(default_factory=lambda: {}, init=True)
    nr_stimuli: int = field(init=False)
    nr_cells: int = field(init=False)

    def __post_init__(self):

        self.nr_stimuli = len(self.dataframes["stimulus_df"])
        self.nr_cells = np.unique(self.dataframes["spikes_df"]["cell_index"]).shape[0]

    def get_spikes_triggered(self, cells, stimuli, time="seconds", waveforms=False, pandas=True):
        if cells == "all":
            cells = np.arange(self.nr_cells)
        dfs = []
        for stimulus in stimuli:
            df = self.get_triggered(cells, stimulus, time, waveforms)
            df = df.with_columns(stimulus_index=pl.lit(f"{stimulus}"))
            dfs.append(df)
        df = pl.concat(dfs)
        if pandas:
            return df.to_pandas()
        else:
            return df

    def get_triggered(self, cells, stimuli, time="seconds", waveforms=False):

        sub_df = df_filter.find_stimuli(self.dataframes["stimulus_df"], stimuli)

        begin_end, trigger, trigger_end, stim_logic = df_filter.get_stimulus_info(sub_df)

        df = stimulus_spikes.load_triggered(cells, begin_end[0, 0], begin_end[0, 1], trigger, trigger_end, stim_logic,
                                            self.path, waveforms)

        if time == "seconds":
            df = df.with_columns(pl.col("times").truediv(self.sampling_freq).alias("times"))
            df = df.with_columns(pl.col("times_relative").truediv(self.sampling_freq).alias("times_relative"))
            df = df.with_columns(pl.col("times_triggered").truediv(self.sampling_freq).alias("times_triggered"))
        return df

    def get_spikes_as_numpy(self, cells, stimulus, time="seconds", waveforms=False):
        sub_df = df_filter.find_stimuli(self.dataframes["stimulus_df"], stimulus)

        begin_end, trigger, trigger_end, stim_logic = df_filter.get_stimulus_info(sub_df)

        df = stimulus_spikes.stimulus_relative(cells, begin_end[0, 0], begin_end[0, 1], self.path, waveforms)

        data = np.empty(len(cells), dtype=object)
        cell_index = []
        for idx, cell in enumerate(df.group_by("cell_index")):
            cell_index.append(cell[0])
            data[idx] = cell[1]["times_relative"].to_numpy()
        return data, cell_index

    def show_df(self, name="spikes_df", level=False, condition=False, viewname=None):
        """Main function to interactively view the dataframes. It uses the qgrid package which uses slickgrid.js

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
                view = Table(
                    self.dataframes[name].index.show_level_values(level)
                )

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

    @property
    def spikes_df(self):
        """Returns the spikes_df dataframe. Shortcut version
        Warning: You cannot index into this, use self.dataframes["spikes_df"] instead"""
        return self.dataframes["spikes_df"]

    @property
    def stimulus_df(self):
        """Returns the stimulus_df dataframe. Shortcut version
        Warning: You cannot index into this, use self.dataframes["stimulus_df"] instead"""
        return self.dataframes["stimulus_df"]

    def save(self, filename):
        """
        Backend save function that is doing the actual saving after non-picklable attributes have been removed.
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
            # This is for version control only:
            # End version control
            return obj

    def use_view_as_filter(self, filter_name="filter", filter_value="Yes", all_stimuli=True, view_name="spikes_df", dataframe="spikes_df"):
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

        Returns
        -------
        None

        """

        # Drop the filter column if it already exists:
        if not all_stimuli:
            df_temp = self.views[view_name].tabulator.value
            df_temp[filter_name] = filter_value

        else:
            cells = self.views[view_name].tabulator.value["cell_index"].unique().tolist()
            pl_df = pl.from_pandas(self.dataframes[dataframe])
            mask = pl_df.select([pl.col("cell_index").is_in(cells).alias("mask")]).to_numpy()
            df_temp = self.dataframes[dataframe].loc[mask].copy()
            df_temp[filter_name] = filter_value

        self.dataframes[dataframe] = pd.merge(self.dataframes[dataframe], df_temp,
                                              on=self.dataframes[dataframe].columns.tolist(), how='left')

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
                df = self.dataframes[dataframe].loc[self.dataframes[dataframe]["stimulus_name"] == name].copy()
                if stimulus:
                    df = df.loc[df["stimulus_index"] == stimulus]
            else:
                df = self.dataframes[dataframe].loc[self.dataframes[dataframe]["stimulus_index"] == stimulus].copy()

            if dataframe == "spikes_df":
                spikes, cells = self.get_spikes_as_numpy(df["cell_index"].unique().tolist(),
                                                         df["stimulus_index"].unique().tolist(), time="frames", waveforms=False)
                df = df.set_index(["cell_index"])
                print(len(df))
                df["spikes"] = [np.empty(1, dtype=object)] * len(df)
                df.loc[cells, "spikes"] = spikes
            df = df.reset_index()
            dfs.append(df)
            out.append(pd.concat(dfs))
        return out


class Recording_Overview(Recording):
    def __init__(self, stimulus_path, recording_path, name, sampling_freq):
        dataframes = {"spikes_df": pd.DataFrame(), "stimulus_df": pd.DataFrame()}
        self.stimulus_path = stimulus_path
        super().__init__(recording_path, name, dataframes, sampling_freq)




class Dataframe:
    """Dataframe class: This class allows to load and add multiple recordings to
    a pandas dataframe based on which multiple functions and analysis steps
    can be performed

    Parameters
    ----------
    path : str
        Points to the folder location of the recording that shall be used
        to initiate this class. This folder must contain the spikes_overview and
        stimulus_overview pandas dataframes saved as pickle or csv.
    name : str
        Name of the recording that is initially used as it should appear in the
        dataframe index that will be created

    Attributes
    ----------
    recording_name : list
        Stores the name of the recordings in the order in which they are added
        to the class object and as they will appear in the dataframe index.
    nr_recordings : int
        Keeps track of the number of added recordings.
    spikes_df : pandas.Dataframe
        Central dataframe that stores the information of all added recordings.
        Multiindex with the names: Cell index, Stimulus ID, Recording
    stimulus_df : pandas.Dataframe
        Stores the information about which stimuli were played in each recording.
    nr_stimuli : int
        Number of unique stimuli in the dataframe
    path: list of str
        Contains the path to the folder in which each recording can be found in
        the order in which the recordings were added to the object.


    """

    def __init__(self, path=None, name=None, spikes_df=None, stimulus_df=None):
        """
        Initialize the object
        Parameters
        ----------
        path : str
            Path to the folder in which the recording is located
        name : str
            Name of the recording
        load : bool
            If True, the dataframes are loaded from the folder in the path. If False, empty dataframes are created and
            the data has to be added manually.
        """
        #spikes_df = spikes_df.copy()
        #stimulus_df = stimulus_df.copy()
        #spikes_df.insert(0, "recording", name)
        #stimulus_df.insert(0,"recording",name)

        self.nr_recordings = 1
        self.recordings = {name: {"path": path}}

        self.dataframes = {"all": {"spikes_df": spikes_df, "stimulus_df": stimulus_df}}
        self.dataframes[name] = {}
        self.dataframes[name]["spikes_df"] = self.dataframes["all"]["spikes_df"].lazy().filter(
            pl.col("recording") == name)
        self.dataframes[name]["stimulus_df"] = self.dataframes["all"]["stimulus_df"].lazy().filter(
            pl.col("recording") == name)
        self.nr_stimuli = 0

        self.views = {}
        self.selection_index = None

        # Datale extension further version
        # global_state.set_app_settings(dict(max_column_width=100))


    def split_recordings(self, recording_names=None):
        """
        This function splits the spikes_df and stimulus_df according to recordings and returns them as a list.
        Parameters
        ----------
        recording_names : list of str
            Names of the recordings that should be split into separate objects

        Returns
        -------
        spike_dfs : list of pandas.Dataframe
        stimulus_dfs : list of pandas.Dataframe


        """
        if recording_names is None:
            # All recordings are used
            recording_names = self.recording_name

        # Prepare the spikes_df and stimulus_df for splitting
        spike_dfs = [
            self.dataframes["spikes_df"].loc[
                self.dataframes["spikes_df"].index.get_level_values("Recording") == recording]
            for recording in recording_names]

        stimulus_dfs = [
            self.dataframes["stimulus_df"].loc[
                self.dataframes["stimulus_df"].index.get_level_values("Recording") == recording]
            for recording in recording_names]

        return spike_dfs, stimulus_dfs

    def split_recordings_into_objects(self, recording_names=None):
        """
        This function splits the recordings into separate Dataframe objects and returns them as a list.
        This can be used if only one recording should be analyzed at a time.

        Parameters
        ----------
        recording_names: list of str
            Names of the recordings that should be split into separate objects

        Returns
        -------
        all_objects: list of Dataframe objects

        """
        if recording_names is None:
            # All recordings are used
            recording_names = self.recording_name
        # Get list of split spikes_df and stimulus_df
        spike_dfs, stimulus_dfs = self.split_recordings(recording_names)
        paths = [self.paths[self.recording_name.index(name)] for name in recording_names]
        # Create a list of Dataframe objects
        all_objects = [Dataframe.from_df_subset(path, [name], df, df_s) for path, name, df, df_s in
                       zip(paths, recording_names, spike_dfs, stimulus_dfs)]
        return all_objects

    def prepare_dataframe(self, name):
        """This function prepares the dataframe for the analysis. It adds a
        column with the stimulus ID and a column with the recording name to the
        spikes_df dataframe. It also sets the index of the dataframe to the
        stimulus ID and the recording name.

        Parameters
        ----------
        name : str
            Name of the recording that is initially used as it should appear in the
            dataframe index that will be created

        Returns
        -------
        None.

        """

        name = np.repeat(name, len(self.dataframes["spikes_df"]))
        # spikes_df
        self.dataframes["spikes_df"].loc[:, "recording"] = name.tolist()
        self.dataframes["spikes_df"].loc[:, "idx"] = list(range(len(self.dataframes["spikes_df"])))
        nr_stimuli = len(self.dataframes["spikes_df"].index.unique(1))
        self.dataframes["spikes_df"].reset_index(inplace=True)
        self.dataframes["spikes_df"].set_index(
            ["cell_index", "stimulus_index", "recording"], inplace=True
        )
        # stimulus_df
        self.dataframes["stimulus_df"].loc[:, "recording"] = name[:nr_stimuli].tolist()
        self.dataframes["stimulus_df"].set_index("recording", append=True, inplace=True)

    @property
    def spikes_df(self):
        """Returns the spikes_df dataframe. Shortcut version
        Warning: You cannot index into this, use self.dataframes["spikes_df"] instead"""
        return self.dataframes["all"]["spikes_df"]

    @property
    def stimulus_df(self):
        """Returns the stimulus_df dataframe. Shortcut version
        Warning: You cannot index into this, use self.dataframes["stimulus_df"] instead"""
        return self.dataframes["all"]["stimulus_df"]

    def add_recording(self, path=None, name=None):
        """Adds another recording to the dataframe. Similar to when the object is
        initialized.

        Parameters
        ----------
        path : str
            Points to the folder location of the recording that shall be used
            to initiate this class. This folder must contain the spikes_overview and
            stimulus_overview pandas dataframes saved as pickle or csv.
        name : str
            Name of the recording that is initally used as it should appear in the
            dataframe index that will be created

        Returns
        -------
        Nothing, prints confirmation, that the recording was added.

        """

        self.recording_name.append(name)
        spikes_df, stimulus_df = load_recordings(path)

        name = np.repeat(name, len(spikes_df))
        spikes_df.loc[:, "recording"] = name.tolist()
        nr_stimuli = len(spikes_df.index.unique(1))
        stimulus_df.loc[:, "recording"] = name[:nr_stimuli].tolist()

        stimulus_df.set_index("recording", append=True, inplace=True)

        spikes_df = spikes_df.reset_index()
        spikes_df = spikes_df.set_index(["cell_index", "stimulus_index", "recording"])
        self.dataframes["spikes_df"].drop(columns="idx", inplace=True)
        self.dataframes["spikes_df"] = pd.concat([self.dataframes["spikes_df"], spikes_df])
        self.dataframes["stimulus_df"] = pd.concat([self.dataframes["stimulus_df"], stimulus_df])
        self.dataframes["spikes_df"].loc[:, "idx"] = list(range(len(self.dataframes["spikes_df"])))

        self.nr_recordings = self.nr_recordings + 1
        self.paths.append(path)
        self.nr_stimuli = len(self.dataframes["stimulus_df"]["stimulus_name"].unique())
        self.dataframes["spikes_df"] = self.dataframes["spikes_df"]
        self.dataframes["stimulus_df"] = self.dataframes["stimulus_df"]

    def show_df(self, name="spikes_df", level=False, condition=False, viewname=None):
        """Main function to interactively view the dataframes. It uses the qgrid package which uses slickgrid.js

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
                view = Table(
                    self.dataframes[name].index.show_level_values(level)
                )

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



    ################################################## Possible new version ###############################################
    #
    #
    # def show_df_dtale(self, names=None, level=False, condition=False, show_objects=None):
    #     """Main function to interactively view any dataframe in Jupyter_lab. """
    #
    #     # First we need to kill the old view process:
    #
    #     # Check for conditions or levels and generate the view:
    #
    #     if names is None:
    #         names = ["spikes_df", "stimulus_df"]
    #     if show_objects is None:
    #         show_objects = [False, True]
    #
    #     data_frames = []
    #     for name, show_o in zip(names, show_objects):
    #         if level and not condition:
    #             try:
    #                 df_temp = self.dataframes[name][level]
    #             except KeyError:
    #                 df_temp = self.dataframes[name].index.show_level_values(level)
    #
    #         elif level:
    #             try:
    #                 df_temp = self.dataframes[name][self.dataframes[name][level] == condition]
    #             except KeyError:
    #                 df_temp = self.dataframes[name][
    #                     self.dataframes[name].index.show_level_values(level)
    #                     == condition
    #                     ]
    #
    #         else:
    #             df_temp = self.dataframes[name]
    #
    #         data_frames.append(df_temp)
    #
    #     self.create_views(names, data_frames)
    #     return self.views[names[0]]

    # def create_views(self, names, data_frames: list = None):
    #
    #     for name, frame in zip(names, data_frames):
    #         with contextlib.suppress(KeyError):
    #             global_state.cleanup(self.views[name.replace('_', '')]._data_id)
    #         self.views[name] = dtale.show(frame, name=name.replace('_', ''), allow_cell_edits=False)



    def use_view_as_filter(self, filter_name=False, filter_value="Yes", use_for_all=False, viewname=None):
        """Use the current self.df_view (which is a sub-frame from the complete
        dataframe) as a filter for the complete dataframe. This function will add a new column to the complete dataframe)

        Parameters
        ----------
        filter_name : str
            The name the new introduced filter shall have in
        filter_value : str
            The value the new introduced filter shall have in the new column.
        use_for_all : boolean
            If true, filter will not only apply to the current df_view but also
            to all other entries in the spikes_df with the same cell index and
            recording name indices.
        viewname : str
            Name of the view that is used as a filter.

        Returns
        -------
        None

        """
        # Create a dummy filter name if none is given:
        if not filter_name:
            filter_name = "filter"
        # Drop the filter column if it already exists:
        with contextlib.suppress(KeyError):
            self.dataframes[viewname].drop(columns=filter_name)
        # Get the current view:
        sub_df = self.views[viewname].get_changed_df()
        # Add the filter column to the current view:
        sub_df[filter_name] = filter_value
        # Drop the current view from the complete dataframe and add the new part:
        self.drop_and_fill(sub_df, index_for_drop=sub_df.index)
        # Expand the filter over all stimuli if use_for_all is True:
        if use_for_all:
            self.expand_column_over_all_stim(viewname, column_name=filter_name,
                                             entries=filter_value)
        # Fill the NaN values with "No" and convert the column to string:
        self.dataframes["spikes_df"][filter_name] = self.dataframes["spikes_df"][filter_name].fillna("No")
        self.dataframes["spikes_df"] = self.dataframes["spikes_df"].astype({filter_name: str})

    def expand_column_over_all_stim(self, dataframe="spikes_df", column_name=None, entries=None):
        """Expands values from one column in self.spikes_df over all rows with
        the same cell index and recording name index combination

        Parameters
        ----------
        dataframe : str
            The name of the dataframe which shall be expanded.

        column_name : str
            The name of the column which shall be expanded over all stimuli.

        entries : list
            The entries which shall be expanded over all stimuli. If None, all entries will be expanded.

        Returns
        -------
        None
        """

        unique_entries = entries if type(entries) == np.ndarray else self.dataframes[dataframe][column_name].unique()
        # Iterate over all unique entries in the column:
        for u_idx in unique_entries:
            index_for_filter = (
                self.dataframes[dataframe][column_name]
                .loc[(self.dataframes[dataframe][column_name] == u_idx).to_numpy()]
                .index
            )
            # Iterate over all recordings:
            for recording in index_for_filter.unique("recording"):
                index_cell = index_for_filter.get_level_values("recording") == recording
                # Create a temporary dataframe with the matching entries:
                temp_df = self.dataframes[dataframe].loc[
                          index_for_filter[index_cell].get_level_values("cell_index"),
                          :,
                          index_for_filter[index_cell].get_level_values("recording"),
                          ].copy()
                temp_df[column_name] = self.dataframes[dataframe].loc[index_for_filter[0]][
                    column_name
                ]
                # Drop the old entries and fill the new ones:
                self.drop_and_fill(temp_df, index_for_drop=temp_df.index)

    def find_matching_stimulus(self, dataframe="stimulus_df", index=None):
        """Finds the matching stimulus information from the self.dataframes["stimulus_df"]
        based on the recording and cell index provided.

        Parameters
        ----------
        dataframe : str
            The name of the dataframe which shall be searched for the matching stimulus.

        index : tuple
            A tuple derived from a pd.Index.
            If no index is provided, the function will try to get an index
            from a selected row in the df_view object.


        Returns
        -------
        pandas.Dataframe
            The matching rows from self.dataframes["stimulus_df"]

        """
        if not index:
            index = self.selection_index
        return self.dataframes[dataframe].loc[index[1:]]

    def return_selected_cell_df(self, index=None):
        """Returns a row from the self.spikes_df depending on the index provided.
        Parameters
        ----------
        index : tuple
            tuple derived from a pandas Index. Containg the information about
            which recording, cell and stimulus is selected. Returns the whole
            respective row. If no index is provided, the function will try to
            get an index from a selected row in the df_view object.

        Returns
        -------
        pandas.Dataframe
            The row that was indexed.


        """
        if not isinstance(index, pd.core.indexes.base.Index):
            index = self.selection_index
        return self.dataframes["spikes_df"].loc[index]

    def apply_filter(
            self,
            dataframe="spikes",
            cells=None,
            stimuli=None,
            recordings=None,
            column=None,
            filters=None,
    ):
        """Searches dataframes for specific index combinations and returns a subset
        of the dataframe. This function is a simplification of the pandas loc[()] argument.

        Parameters
        ----------
        dataframe : str
            A string refering to the name of the dataframe that should be searched.
        cells : list of int
            list with cell indices that should be searched for
        stimuli : list of int
            list of stimulus indices that should be searched for
        recordings : list of str
            List with the recording names that shall be searched for
        column : str
            String refering to one additional column that shall be treated as an
            index
        filters : list of str or int
            List that contains the values for which shall be searched in the column.

        Returns
        -------
        df_temp : pandas.DataFrame
            Contains the selected rows from the main dataframe

        """
        # Create empty slices if no arguments are provided:
        if not cells:
            cells = slice(None)
        if not stimuli:
            stimuli = slice(None)
        if not recordings:
            recordings = slice(None)
        # Create a temporary dataframe:
        df_temp = self.dataframes[dataframe]
        # If a column is provided, set the index to the column and search for the filters:
        if column:
            df_temp = df_temp.set_index(column, append=True)
            if "cell_index" in df_temp.index.names:
                df_temp = df_temp.loc[(cells, stimuli, recordings, filters), :]
            else:
                df_temp = df_temp.loc[(stimuli, recordings, filters), :]
        # If no column is provided, search for the filters:
        else:
            if "cell_index" in df_temp.index.names:
                df_temp = df_temp.loc[(cells, stimuli, recordings), :]
            else:
                df_temp = df_temp.loc[(stimuli, recordings), :]
        # Reset the index if a column was provided:
        df_temp = df_temp.reset_index(level=column)
        return df_temp

    def get_single_recording_spikes(self, dataframe="spikes_df", recording=None, recording_idx=None):
        """

        Parameters
        ----------
        dataframe : str
            The name of the dataframe which shall be searched for the matching spikes.
        recording : str
            The name of the recording for which the spikes shall be returned.
        recording_idx : int
            The index of the recording for which the spikes shall be returned. (As in the self.recording_name list)

        Returns
        -------
         pandas.Dataframe

        """
        return self.single_recording(dataframe, recording, recording_idx)

    def single_recording(self, dataframe, recording, recording_idx):
        if recording is not None:
            return self.dataframes[dataframe].loc[
                self.dataframes[dataframe].index.get_level_values("recording") == recording
                ]
        elif recording_idx is not None:
            return self.dataframes[dataframe].loc[
                self.dataframes[dataframe].index.get_level_values("recording")
                == self.recording_name[recording_idx]
                ]

    def get_single_recording_stimulus(self, recording=None, recording_idx=None):
        """

        Parameters
        ----------
        recording : str
            The name of the recording for which the stimulus shall be returned.
        recording_idx: int
            The index of the recording for which the stimulus shall be returned. (As in the self.recording_name list)

        Returns
        -------
        pandas.Dataframe

        """
        return self.single_recording(dataframe="stimulus_df", recording=recording,
                                     recording_idx=recording_idx)

    def get_single_recording(self, recording=None, recording_idx=None, dataframes=None):
        """

        Parameters
        ----------
        recording : str
            The name of the recording for which the stimulus and spikes shall be returned.
        recording_idx: int
            The index of the recording for which the stimulus and spikes shall be returned. (As in the self.recording_name list)
        dataframes : list of str
            A list of strings that contain the names of the dataframes that shall be returned.
        Returns
        -------
        tuple of pandas.Dataframes

        """
        if dataframes is None:
            dataframes = ["spikes_df", "stimulus_df"]
        combined = (
            self.single_recording(dataframe=dataframes[0], recording=recording,
                                  recording_idx=recording_idx),
            self.single_recording(dataframe=dataframes[1], recording=recording,
                                  recording_idx=recording_idx),
        )
        return combined

    def get_cluster_subset(
            self,
            spikes_df_log=None,
            dataframe="spikes_df",
            cluster_nr=1,
            cluster_column="Cluster ID",
            return_logical=False,
    ):
        """
        Returns a subset of the dataframe that contains only the data that belong to a specific cluster (int index).

        Parameters
        ----------
        spikes_df_log : pandas.Series
            A logical series that contains the spikes that shall be used for the subset.
        dataframe : str
            The name of the dataframe that shall be searched for the matching spikes.
        cluster_nr : int
            The cluster number for which the dataframe subset shall be returned.
        cluster_column : str
            The name of the column that contains the cluster numbers.
        return_logical : bool
            If True, the function returns a logical series that can be used to index into the dataframe.

        Returns
        -------

        """
        cluster_df_log = self.dataframes[dataframe][cluster_column] == cluster_nr

        if not type(spikes_df_log) is tuple:
            if return_logical:
                return cluster_df_log
            else:
                return self.dataframes[dataframe].loc[cluster_df_log]
        else:
            combined_df_log = cluster_df_log.multiply(spikes_df_log)
            if return_logical:
                return combined_df_log
            else:
                return self.dataframes[dataframe].loc[combined_df_log]

    def df_by_column_entry(self, dataframes, column, entry):
        """
        Returns a list of dataframes that contain only the entries that match the entry in the column.

        Parameters
        ----------
        dataframes : list of str
            A list of strings that contain the names of the dataframes that shall be filtered by the entry.
        column : str
            The name of the column that shall be used for the filtering.
        entry : str, int or float
            The entry that shall be used for the filtering.

        Returns
        -------
        list of pandas.Dataframes, each filtered by the entry in the column.

        """
        dfs = []
        for dataframe in dataframes:
            if column in self.dataframes[dataframe].columns:
                dfs.append(self.dataframes[dataframe].loc[self.dataframes[dataframe][column] == entry])
            else:
                raise ValueError(f"The column {column} is not in the dataframe {dataframe}.")
        return dfs

    def log_idx_by_column_entry(self, dataframes, column, entry):
        """
        Returns a list of logical arrays which true values refer to the entries that match the entry in the column.

        Parameters
        ----------
        dataframes : list of str
            A list of strings that contain the names of the dataframes that shall be filtered by the entry.
        column : str
            The name of the column that shall be used for the filtering.
        entry : str, int or float
            The entry that shall be used for the filtering.

        Returns
        -------
        list of logical arrays, each referring to the respective dataframe filtered by the entry in the column.

                """
        logs = []
        for dataframe in dataframes:
            if column in self.dataframes[dataframe].columns:
                logs.append(self.dataframes[dataframe] == entry)
            else:
                raise ValueError(f"The column {column} is not in the dataframe {dataframe}.")
        return logs

    def df_by_index_entry(self, dataframes, index, entry):
        """
        Returns a list of dataframes that contain only the entries that match the entry in the index.

        Parameters
        ----------
        dataframes : list of str
            A list of strings that contain the names of the dataframes that shall be filtered by the entry.
        index : str
            The name of the index that shall be used for the filtering.
        entry : str, int or float
            The entry that shall be used for the filtering.

        Returns
        -------
        list of pandas.Dataframes, each filtered by the entry in the index.

        """
        dfs = []
        for dataframe in dataframes:
            if index not in self.dataframes[dataframe].index:
                raise ValueError(f"The column {index} is not in the dataframe {dataframe}.")
            log = self.dataframes[dataframe].index.get_level_values(
                self.dataframes[dataframe].index.names.index(index)) == entry
            dfs.append(self.dataframes[dataframe].loc[log])
        return dfs

    def log_idx_by_index_entry(self, dataframes, index, entry):
        """
        Returns a list of logical arrays referring to entries that match the entry in the index.

        Parameters
        ----------
        dataframes : list of str
            A list of strings that contain the names of the dataframes that shall be filtered by the entry.
        index : str
            The name of the column that shall be used for the filtering.
        entry : str, int or float
            The entry that shall be used for the filtering.

        Returns
        -------
        list of logical arrays.

        """
        logs = []
        for dataframe in dataframes:
            if index in self.dataframes[dataframe].index:

                logs.append(self.dataframes[dataframe].index.get_level_values(
                    self.dataframes[dataframe].index.names.index(index)) == entry)
            else:
                raise ValueError(f"The column {index} is not in the dataframe {dataframe}.")
        return logs

    def get_stimulus_subset(
            self, stimulus=0, name=None, recording_name=None, dataframes=None, return_logical=False
    ):
        """
        Returns a subset of the dataframe that contains only the data that belong to a specific stimulus.
        Stimulus can be specified by the index of the stimulus in the stimulus list or by the name of the stimulus.

        Parameters
        ----------
        stimulus : int
            The index of the stimulus in the stimulus list.
        name : str
            The name of the stimulus.
        recording_name : str
            The name of the recording for which the stimulus and spikes shall be returned.
        dataframes: list of str
            A list of strings that contain the names of the dataframes that shall be returned.
        return_logical : bool
            If True, the function returns a logical series that can be used to index into the dataframe.

        Returns
        -------

        """
        if dataframes is None:
            dataframes = ["spikes_df", "stimulus_df"]
        if recording_name:
            spikes_df, stimulus_df = self.get_single_recording(recording=recording_name)
            if name:
                if return_logical:
                    return self.log_idx_by_column_entry(dataframes, "stimulus_name", name)

                else:
                    return self.df_by_column_entry(dataframes, "stimulus_name", name)
            else:
                if return_logical:
                    spikes_df = spikes_df.index.get_level_values(1) == stimulus
                    stimulus_df = stimulus_df.index.get_level_values(0) == stimulus
                else:
                    spikes_df = spikes_df.loc[
                        spikes_df.index.get_level_values(1) == stimulus
                        ].copy()

                    stimulus_df = stimulus_df.loc[
                        stimulus_df["stimulus_name"] == stimulus
                        ].copy()

        else:
            if name:
                if return_logical:
                    return self.log_idx_by_column_entry(dataframes, "stimulus_name", name)

                else:
                    return self.df_by_column_entry(dataframes, "stimulus_name", name)
            else:
                if return_logical:
                    spikes_df = self.dataframes[dataframes[0]].index.get_level_values(1) == stimulus
                    stimulus_df = self.dataframes[dataframes[1]].index.get_level_values(0) == stimulus
                else:
                    spikes_df = self.dataframes[dataframes[0]].loc[
                        self.dataframes[dataframes[0]].index.get_level_values(1) == stimulus
                        ].copy()

                    stimulus_df = self.dataframes[dataframes[1]].loc[
                        self.dataframes[dataframes[1]]["stimulus_name"] == stimulus
                        ].copy()

        return [spikes_df, stimulus_df]

    def drop(self, column, condition, dataframe="spikes_df"):
        """
        Drops all entries that match the condition in the column.
        Parameters
        ----------
        column : str
            The name of the column that shall be used for the filtering.
        condition : str, int or float
            The entry that shall be used for the filtering.
        dataframe : str
            The name of the dataframe that shall be filtered.

        Returns
        -------
            None
            Updates the dataframe in the object.

        """
        self.dataframes[dataframe].drop(
            self.dataframes[dataframe][self.dataframes[dataframe][column] == condition].index)

    def drop_and_fill(self, add_df, dataframe="spikes_df", column=None, condition=None, index_for_drop=None):
        """
        Drops all entries that match the condition in the column and fills the dataframe with the new dataframe.
        Parameters
        ----------
        add_df : pandas.DataFrame
            The dataframe that shall be added to the dataframe in object.
        dataframe : str
            The name of the dataframe in the object that shall be filled.
        column : str
            The name of the column that shall be used for the filtering. (Only needed if index_for_drop is None)
        condition : str, int or float
            The entry that shall be used for the filtering. (Only needed if index_for_drop is None)
        index_for_drop : pandas.Index
            The index that shall be used for dropping the entries. (Only needed if column and condition are None)

        Returns
        -------
        None
            Updates self.dataframes[dataframe] with the new dataframe.

        """
        # If no index is provided, we construct one from the column and condition.
        # Drops all entries that match the condition in the column and fills the dataframe with the new dataframe.

        if index_for_drop is None:
            try:
                index_for_drop = self.dataframes[dataframe][
                    self.dataframes[dataframe][column] == condition
                    ].index
            except KeyError:
                index_for_drop = self.dataframes[dataframe][
                    self.dataframes[dataframe].index.get_level_values(column) == condition
                    ].index
        # Drops the entries and adds the new dataframe according to the index.
        self.dataframes[dataframe].drop(index_for_drop, inplace=True)
        self.dataframes[dataframe] = pd.concat([self.dataframes[dataframe], add_df])
        self.dataframes[dataframe] = self.dataframes[dataframe].sort_index(level="recording")

    def drop_stimulus_from_recording(self, stimulus, recording, dataframes=None):
        """
        Drops a stimulus from a recording from all specified dataframes.
        Parameters
        ----------
        stimulus : int
            The index of the stimulus in the stimulus list.
        recording : str
            The name of the recording from which the stimulus shall be dropped.
        dataframes : list of str
            A list of strings that contain the names of the dataframes from which to drop the stimuli.

        Returns
        -------
        None
            Updates the dataframes in the object.

        """
        if dataframes is None:
            dataframes = ["spikes_df", "stimulus_df"]

        for df in dataframes:
            self.dataframes[df].drop(
                self.dataframes[df].loc[
                    (self.dataframes[df].index.get_level_values("recording") == recording) &
                    (self.dataframes[df].index.get_level_values("stimulus_index") == stimulus)
                    ].index, inplace=True
            )

    def drop_stimulus_name_from_all_recordings(self, stimulus_name):
        """
        Drops all stimuli with a certain name from all recordings and all dataframes.
        Parameters
        ----------
        stimulus_name : str
            The name of the stimulus that shall be dropped.

        Returns
        -------

        """
        for df in self.dataframes:
            self.dataframes[df].drop(
                self.dataframes[df].loc[
                    self.dataframes[df]["stimulus_name"] == stimulus_name
                    ].index, inplace=True
            )

    def change_stimulus_name(
            self, recording_name=None, stimulus_id=None, new_name=None
    ):
        """
        Changes the name of a stimulus in the all dataframes in the object according to certain filters.
        Parameters
        ----------
        recording_name : str
            The name of the recording for which the stimulus name shall be changed.
        stimulus_id : int
            The stimulus id for which the stimulus name shall be changed.
        new_name  : str
            The new name of the stimulus.

        Returns
        -------
        None
            Updates the stimulus names in all dataframes in the object.

        """

        # If the recording name is None, slice over all recordings
        if recording_name is None:
            for df in self.dataframes:
                df_temp = self.dataframes[df].loc[
                    self.dataframes[df].index.get_level_values("stimulus_index") == stimulus_id]
                # Write the new name into the temporary dataframe
                df_temp["stimulus_name"] = new_name
                # Drop and fill the dataframe
                self.drop_and_fill(df_temp, dataframe=df, index_for_drop=df_temp.index)
            return
        # Else, slice over the provided recording name
        # Names have to be changed in all dataframes
        # Loop over self.dataframes and change the name in all dataframes
        for df in self.dataframes:
            df_temp = self.dataframes[df].xs(
                (stimulus_id, recording_name),
                level=["stimulus_index", "recording"],
                drop_level=False,
            )
            # Write the new name into the temporary dataframe
            df_temp["stimulus_name"] = new_name
            # Drop and fill the dataframe
            self.drop_and_fill(df_temp, dataframe=df, index_for_drop=df_temp.index)

    def change_stimulus_name_globally(self, old_name, new_name):
        """
        Changes the name of a stimulus in the all dataframes in the object (Global name change).
        Parameters
        ----------
        old_name : str
            The old name of the stimulus.
        new_name  : str
            The new name of the stimulus.

        Returns
        -------
        None
            Updates the stimulus names in all dataframes in the object.

        """
        # Loop over all dataframes and change the name in all dataframes
        for df in self.dataframes:
            df_temp = self.dataframes[df].loc[
                self.dataframes[df]["stimulus_name"] == old_name
                ]
            # Write the new name into the temporary dataframe
            df_temp["stimulus_name"] = new_name
            # Drop and fill the dataframe
            self.drop_and_fill(df_temp, dataframe=df, index_for_drop=df_temp.index)

    def get_sorted_cluster_idx(
            self,
            by_stimulus=False,
            stimulus_name=None,
            by_recording=False,
            recording_name=None,
            dataframe="spikes_df",
            cluster_name="Cluster ID",
    ):
        """
        Returns a sorted list of cluster indices according to certain filters.
        Parameters
        ----------
        by_stimulus : bool
            If True, the list is sorted according to the cluster idx for a certain stimulus.
        stimulus_name : str
            The name of the stimulus for which the list shall be sorted.
        by_recording : bool
            If True, the list is sorted according to the cluster idx for a certain recording.
        recording_name : str
            The name of the recording for which the list shall be sorted.
        dataframe: str
            The name of the dataframe from which the cluster indices shall be extracted.
        cluster_name : str
            The name of the column in the dataframe that contains the cluster indices.

        Returns
        -------

        """

        if not by_stimulus and not by_recording:
            return self.dataframes[dataframe].value_counts(subset=cluster_name)
        if by_stimulus:
            temp_df = self.get_stimulus_subset(name=stimulus_name)[0]
            return temp_df.value_counts(subset=cluster_name)

        if by_recording:
            temp_df = self.dataframes[dataframe].loc[:, :, recording_name]
            return temp_df.value_counts(subset=cluster_name)

    def save_backend(self, filename):
        """
        Backend save function that is doing the actual saving after non-picklable attributes have been removed.
        Parameters
        ----------
        filename : str
            The name of the file to save the object to.

        Returns
        -------

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save(self, filename):
        """
        Save the object to a file.
        Parameters
        ----------
        filename : str
            The name of the file to save the object to.

        Returns
        -------

        """
        self.save_backend(filename)

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
            # This is for version control only:
            # End version control
            return obj

    ## Plot stuff




def parallel_df(df, func, *args, **kwargs):
    df_split = np.array_split(df, mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    try:
        par_func = partial(func, *args, **kwargs)
        result_df = pd.concat(pool.map(par_func, df_split))
    except Exception:
        traceback.print_exc()
        pool.close()
    pool.close()
    pool.join()
    return result_df
