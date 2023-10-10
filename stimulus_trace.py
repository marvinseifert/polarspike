# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:23:16 2021
This file contains functions, and classes to deal and analyse the stimulus trace
obtained from 3Brain BioCam MEAs by exporting the trigger channel in .brw or .hdf
file formate.

@author: Marvin
"""
import backbone
import h5py
import pandas as pd
import numpy as np
from ipywidgets import widgets
import matplotlib.pyplot as plt
import scipy.signal as sg
import plotly.graph_objects as go
import pickle
import polars as pl
import re

class Stimulus_Extractor:
    """
    Stimulus trace class
    This class reads and handles the stimulus trace. The stimulus trace is plotted and
    than the user can select stimulus frames based on the plotted trigger signals.


    """

    def __init__(self, stimulus_file, freq=1, stream=1):
        """
        Initialize function. Opens the stimulus file and extracts the important
        data for further analysis. It also initializes object attributes which
        store information about how many stimuli have been defined, and which
        position these stimuli have in the stimulus channel.

        Parameters
        ----------
            stimulus_file: str: The file location of the stimulus trace.



        """

        # Check if file is .brw or .mat
        self.recording_folder = stimulus_file[: stimulus_file.rfind("/") + 1]
        format = backbone.get_file_ending(stimulus_file)
        if format == ".brw":
            with h5py.File(stimulus_file, "r") as f:
                self.channel = pd.DataFrame(
                    np.array([f["/3BData/Raw"]])[0, :], columns=["Voltage"]
                )
                self.max_Voltage = self.channel.Voltage.max(axis=0)
                self.min_Voltage = self.channel.Voltage.min(axis=0)
                self.half_Voltage = (
                        self.min_Voltage + (self.max_Voltage - self.min_Voltage) / 2
                )
                self.Frames = range(0, len(self.channel.index), 1)
                self.sampling_frequency = np.array(
                    f["/3BRecInfo/3BRecVars/SamplingRate"]
                )
                self.Time = self.Frames / self.sampling_frequency
                self.channel["Frame"] = self.Frames
                self.channel["Time_s"] = self.Time
                self.channel.Time_s = pd.to_timedelta(self.channel.Time_s, unit="s")
                self.channel.set_index("Time_s", inplace=True)

        if format == ".h5":
            with h5py.File(stimulus_file, "r") as f:
                self.channel = pd.DataFrame(
                    np.array(f["Data//Recording_0//AnalogStream//Stream_2//ChannelData"][:][stream]),
                    columns=["Voltage"]
                )
                self.max_Voltage = self.channel.Voltage.max(axis=0)
                self.min_Voltage = self.channel.Voltage.min(axis=0)
                self.half_Voltage = (
                        self.min_Voltage + (self.max_Voltage - self.min_Voltage) / 2
                )

                self.Frames = range(0, len(self.channel.index), 1)
                self.sampling_frequency = freq
                self.Time = np.asarray(self.Frames) / self.sampling_frequency
                self.channel["Frame"] = self.Frames
                self.channel["Time_s"] = self.Time
                self.channel.Time_s = pd.to_timedelta(self.channel.Time_s, unit="s")
                self.channel.set_index("Time_s", inplace=True)

        self.switch = 0
        self.begins = []
        self.ends = []
        self.nr_stim = 0
        self.stimuli = pd.DataFrame(
            columns=[
                "stimulus_name",
                "begin_fr",
                "end_fr",
                "trigger_fr_relative",
                "trigger_int",
                "stimulus_index",
                "stimulus_repeat_logic",
                "stimulus_repeat_sublogic",
                "sampling_freq"
            ]
        )
        self.stimuli = self.stimuli.astype({'stimulus_name': 'str', 'begin_fr': 'int32', 'end_fr': 'int32',
                        'trigger_fr_relative': 'object', 'trigger_int': 'object',
                        'stimulus_index': 'int32', 'stimulus_repeat_logic': 'int32',
                        'stimulus_repeat_sublogic': 'int32', 'sampling_freq': 'float'})

    def plot_trigger_channel_new(self, dsf):
        """
        Plots trigger channel using plotly interactive plot and plotly widget.
        This allows for interactive stimulus selection.

        Parameters
        ----------
        dsf: str: example("200ms"). Downsample factor for downsampling the trigger
        channel.

        Returns
        -------
        self.f: The figure object of the plotly plot.

        """
        dsf = int(self.sampling_frequency / convert_time_string_to_frequency(dsf))
        channel = self.channel
        channel = pl.from_pandas(channel)
        channel = channel.sort("Frame")
        df = channel.with_columns((channel['Frame'] // dsf).alias('Group_Key'))
        grouped_df = df.group_by(['Group_Key'], maintain_order=True).agg(pl.col('Voltage').max().alias('Voltage'))
        grouped_df = grouped_df.with_columns((pl.col("Group_Key").mul(dsf)).alias('Frame'))
        channel = grouped_df.to_pandas()
        channel["log"] = channel.Voltage > self.half_Voltage


        convert_time_string_to_frequency
        self.f = go.FigureWidget(
            [
                go.Scattergl(
                    x=channel.Frame,
                    y=channel.log,
                    mode="lines+markers",
                    name="Trigger signal",
                )
            ]
        )
        self.f.update_traces(marker=dict(size=2, line=dict(width=0)))

        self.scatter = self.f.data[0]
        colors = ["#1f77b4"] * len(channel)
        self.scatter.marker.color = colors
        self.scatter.marker.size = [2] * len(channel)
        self.f.layout.hovermode = "closest"
        self.f.update_xaxes(title_text="Frames")
        self.f.update_yaxes(title_text="Trigger")
        self.scatter.on_click(self.update_point)
        self.f.layout.on_change(self.handle_zoom, "mapbox_zoom")
        return self.f

    def handle_zoom(layout, mapbox_zoom):
        print("new mapbox_zoom:", mapbox_zoom)

    # create our callback function
    def update_point(self, trace, points, selector):
        """
        Function to highlight selected points in the trigger channel plot

        Parameters
        ----------
        Not sure, copied from plotly interactive
        TODO: Look what are the input arguments here

        """

        c = list(self.scatter.marker.color)
        s = list(self.scatter.marker.size)

        for i in points.point_inds:
            if s[i] == 10:
                if self.switch == 0:
                    c[i] = "#1f77b4"
                    s[i] = 2
                    del self.ends[-1]
                    self.switch = 1

                else:
                    c[i] = "#1f77b4"
                    s[i] = 2
                    del self.begins[-1]
                    self.switch = 0

            elif self.switch == 0:
                c[i] = "#DFFF00"
                s[i] = 10
                self.begins.append(points.xs[0])
                self.switch = 1
            else:
                c[i] = "#ff2d00"
                s[i] = 10
                self.ends.append(points.xs[0])
                self.switch = 0

            with self.f.batch_update():
                self.scatter.marker.color = c
                self.scatter.marker.size = s

    def trigger_channel_for_selection(self, dsf):
        """
        Function that defines n number of stimuli and creates equal number of
        subplots for the matplot plot, so the user can select one stimulus per
        subplot. Deprecated.

        """

        if self.nr_stim_input.value == 0:
            print("Set number of stimulus to 0, no stimuli can be selected")
            return

        channel = self.channel.resample(dsf).mean()
        self.stim_select_fig, self.stim_select_axs = plt.subplots(
            nrows=self.nr_stim_input.value,
            ncols=1,
            figsize=(9, self.nr_stim_input.value * 2),
        )

        if self.nr_stim_input.value > 1:
            for i in range(self.nr_stim_input.value):
                self.stim_select_axs[i].plot(channel.Frame, channel.Voltage)
                self.stim_select_axs[i].spines["top"].set_visible(False)
                self.stim_select_axs[i].spines["right"].set_visible(False)
        else:
            self.stim_select_axs.plot(channel.Frame, channel.Voltage)

        plt.xlabel("Frames")
        plt.ylabel("Voltage")
        self.stim_select_fig.suptitle("Stimulus channel complete")


    def get_stim_range_new(self):
        """
        Function that calculates the stimulus range and find the trigger times
        based on which stimulus borders were selected by the user in the interactive
        plotly graph beforehand.

        Parameters
        ----------
        Needs a plotted stimulus trigger plotly graph and selected stimulus borders
        for at least 1 stimulus.

        Returns
        -------
        returns to self a dataframe containing the stimulus information
        """

        print(self.nr_stim)
        for i in range(self.nr_stim, len(self.begins)):

            limits_temp = np.array(
                [self.begins[i] - self.sampling_frequency, self.ends[i] + self.sampling_frequency], dtype=int
            )

            if limits_temp[0] < 0:
                limits_temp[0] = 0
            if limits_temp[1] < 0:
                limits_temp[1] = 0
            limits_int = limits_temp.astype(int)
            channel_cut = self.channel[limits_int[0]: limits_int[1]]
            channel_log = channel_cut.Voltage > self.half_Voltage
            peaks = sg.find_peaks(channel_log, height=1, plateau_size=2)

            peaks[0][:] = peaks[0][:] + limits_temp[0]
            peaks_left = peaks[1]["left_edges"] + limits_temp[0]
            stim_begin = int(peaks_left[0])

            trigger_interval = np.diff(peaks_left)
            min_trigger_interval = np.min(trigger_interval)

            stim_end = int(
                peaks_left[-1] + min_trigger_interval
            )  # Adds time after the last trigger, to get the whole stimulus
            # Failsafe: If last trigger was close to the end of the recording make last frame stimulus end
            if len(self.channel) < stim_end:
                stim_end = len(self.channel)

            peaks_left = np.append(peaks_left, peaks_left[-1] + min_trigger_interval)
            trigger_interval = np.diff(peaks_left)
            nr_trigger = len(peaks_left)
            plot_ones = np.ones(nr_trigger, dtype=bool)

            df_temp = pd.DataFrame()

            df_temp["stimulus_name"] = ""
            df_temp["begin_fr"] = [stim_begin]
            df_temp["end_fr"] = [stim_end]
            df_temp["trigger_fr_relative"] = [peaks_left - peaks_left[0]]
            df_temp["trigger_int"] = [trigger_interval]
            df_temp["stimulus_index"] = [i]
            df_temp["stimulus_repeat_logic"] = [0]
            df_temp["stimulus_repeat_sublogic"] = [0]
            df_temp["sampling_freq"] = self.sampling_frequency



            self.stimuli = pd.concat([self.stimuli, df_temp], ignore_index=True)

            self.f.add_trace(
                go.Scattergl(x=peaks_left, y=plot_ones, name="Stimulus " + str(i))
            )
            # Update the figure from stimulus selection with trigger points

        self.nr_stim = len(self.ends)

    def get_stim_start_end(self, limits):
        """
        Function that cuts out one specific limit from the whole trigger channel

        Parameters
        ----------
        limits: np.array() size(2): Array containing the left and right limits
        of the stimulus window in frames

        Returns
        -------
        channel: The cut out trigger channel for one stimulus
        """
        for i in range(0, self.nr_stim_input.value):
            channel = self.channel.Voltage[limits[i, 0]: limits[i, 1]]
            return channel

    def get_changed_names(self):
        """
        Adds information about stimulus names to the previously created plotly
        plot of the trigger channel.

        Parameters
        ----------
        overview: qgrid object: The qgrid overview over the stimulus data frame
        overview: dataframe: The stimulus dataframe can also be provided directly.

        Returns
        -------

        """

        self.stimuli = self.stimuli.set_index("stimulus_index")

        for stimulus in range(len(self.stimuli)):
            self.f.data[stimulus + 1].name = self.stimuli["stimulus_name"].loc[stimulus]
        self.stimuli = find_ends(self.stimuli)

    def load_from_saved(self, savefile, show_plot=False):
        """
        Function that loads a pickled stimulus dataframe

        Parameters
        ----------
        savefile: The pickled stimulus dataframe
        show_plot=False: IF true, the complete trigger channel is plotted with
        respective stimulus ranges and names

        Returns
        -------
        Adds stimulus dataframe and trigger channel figure to self.
        """
        stimulus_df = pd.read_pickle(savefile, compression="zip")
        self.begins = stimulus_df["Begin_Fr"].to_list()
        self.ends = stimulus_df["End_Fr"].to_list()
        self.plot_trigger_channel_new("200ms")
        self.get_stim_range_new()
        self.stimuli = stimulus_df.reset_index(drop=False)
        self.get_changed_names()
        if show_plot:
            self.f.show()

def find_ends(df):
    trigger_ends = []
    for i in range(len(df)):
        if df.loc[i, "stimulus_repeat_logic"] == 0:
            trigger_ends.append(df.loc[i, "trigger_fr_relative"][:-1] + df.loc[i, "trigger_int"])
            continue
        shape = (df.loc[i, "stimulus_repeat_logic"] , int(df.loc[i, "trigger_int"].shape[0] /
                                                                      df.loc[i, "stimulus_repeat_logic"]))
        ints = df.loc[i, "trigger_int"][:np.multiply(shape[0], shape[1])].reshape(shape)
        min_int = np.min(ints, axis=1)
        triggers_sorted = df.loc[i, "trigger_fr_relative"][:np.multiply(shape[0], shape[1])].reshape(shape)
        trigger_ends.append((triggers_sorted + min_int.reshape(-1, 1)).flatten())
    df["trigger_ends"] = trigger_ends
    return df


def convert_time_string_to_frequency(time_str):
    """
    Convert a time string like "150ms" to a frequency (reciprocal of time in seconds).

    Parameters:
        time_str (str): The time string to convert.

    Returns:
        float: The frequency derived from the time string.
    """
    # Use a regular expression to separate the numeric and unit parts
    match = re.match(r"(\d+)(\w+)", time_str)
    if not match:
        raise ValueError(f"Invalid time string: {time_str}")

    # Extract parts
    quantity, unit = match.groups()

    # Convert quantity to float
    quantity = float(quantity)

    # Convert to seconds based on the unit
    if unit == "ms":  # milliseconds
        time_s = quantity / 1000
    elif unit == "s":  # seconds
        time_s = quantity
    elif unit == "us":  # microseconds
        time_s = quantity / 1_000_000
    elif unit == "ns":  # nanoseconds
        time_s = quantity / 1_000_000_000
    else:
        raise ValueError(f"Unrecognized unit: {unit}")

    # Return the reciprocal of time as frequency
    return 1 / time_s


