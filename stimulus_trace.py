# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:23:16 2021
This file contains functions, and classes to deal and analyse the stimulus trace
obtained from 3Brain BioCam MEAs by exporting the trigger channel in .brw or .hdf
file formate.

@author: Marvin
"""

from polarspike import backbone, stimulus_spikes
import h5py
import pandas as pd
import numpy as np
import scipy.signal as sg
import polars as pl
import re


class Stimulus_Extractor:
    """
    Stimulus trace class
    This class reads and handles the stimulus trace. The stimulus trace is plotted and
    than the user can select stimulus frames based on the plotted trigger signals.


    """

    def __init__(self, stimulus_file, freq=1, stream=0):
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
                self.sampling_frequency = np.array(
                    f["/3BRecInfo/3BRecVars/SamplingRate"]
                )[0]

        if format == ".h5":
            with h5py.File(stimulus_file, "r") as f:
                try:
                    self.channel = pd.DataFrame(
                        np.array(
                            f["Data//Recording_0//AnalogStream//Stream_2//ChannelData"][
                                :
                            ][stream]
                        ),
                        columns=["Voltage"],
                    )
                except KeyError:
                    self.channel = pd.DataFrame(
                        np.asarray(
                            f["Data//Recording_0//AnalogStream//Stream_0//ChannelData"][
                                0, :
                            ]
                        ),
                        columns=["Voltage"],
                    )
                self.sampling_frequency = freq

        if format == ".dat":
            self.channel = pd.DataFrame(
                np.fromfile(stimulus_file, dtype=np.int16), columns=["Voltage"]
            )
            self.channel = self.channel[1::2]
            self.sampling_frequency = freq

        self.max_Voltage = self.channel.Voltage.max(axis=0)
        self.min_Voltage = self.channel.Voltage.min(axis=0)
        self.half_Voltage = self.min_Voltage + (self.max_Voltage - self.min_Voltage) / 2
        self.Frames = range(0, len(self.channel.index), 1)
        self.Time = np.asarray(self.Frames) / self.sampling_frequency
        self.channel["Frame"] = self.Frames
        self.channel["Time_s"] = self.Time
        self.channel.Time_s = pd.to_timedelta(self.channel.Time_s, unit="s")
        self.channel.set_index("Time_s", inplace=True)

        self.switch = 0
        self.begins = []
        self.ends = []
        self.stimuli = pd.DataFrame()

    def downsample(self, dsf):
        dsf = int(self.sampling_frequency / convert_time_string_to_frequency(dsf))
        channel = self.channel
        channel = pl.from_pandas(channel)
        channel = channel.sort("Frame")
        df = channel.with_columns((channel["Frame"] // dsf).alias("Group_Key"))
        grouped_df = df.group_by(["Group_Key"], maintain_order=True).agg(
            pl.col("Voltage").max().alias("Voltage")
        )
        grouped_df = grouped_df.with_columns(
            (pl.col("Group_Key").mul(dsf)).alias("Frame")
        )
        channel = grouped_df.to_pandas()
        print(len(channel))
        return channel

    def get_stim_range_new(self, begins, ends):
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
        self.stimuli = create_stim_df()
        self.begins = begins
        self.ends = ends
        channel = self.channel.set_index("Frame")
        for i in range(len(self.begins)):
            limits_temp = np.array(
                [
                    self.begins[i] - self.sampling_frequency,
                    self.ends[i] + self.sampling_frequency,
                ],
                dtype=int,
            )

            if limits_temp[0] < 0:
                limits_temp[0] = 0
            if limits_temp[1] < 0:
                limits_temp[1] = 0
            limits_int = limits_temp.astype(int)
            channel_cut = channel[limits_int[0] : limits_int[1]]
            channel_log = channel_cut.Voltage > self.half_Voltage
            peaks = sg.find_peaks(channel_log, height=1, plateau_size=2)

            peaks[0][:] = peaks[0][:] + limits_temp[0]
            peaks_left = peaks[1]["left_edges"] + limits_temp[0]
            stim_begin = int(peaks_left[0])

            trigger_interval = np.diff(peaks_left)
            min_trigger_interval = np.max(trigger_interval)

            stim_end = int(
                peaks_left[-1] + min_trigger_interval
            )  # Adds time after the last trigger, to get the whole stimulus
            # Failsafe: If last trigger was close to the end of the recording make last frame stimulus end
            if len(channel) < stim_end:
                stim_end = len(channel)

            peaks_left = np.append(peaks_left, peaks_left[-1] + min_trigger_interval)
            trigger_interval = np.diff(peaks_left)

            df_temp = pd.DataFrame()
            df_temp["stimulus_name"] = ""
            df_temp["begin_fr"] = [stim_begin]
            df_temp["end_fr"] = [stim_end]
            df_temp["trigger_fr_relative"] = [peaks_left - peaks_left[0]]
            df_temp["trigger_int"] = [trigger_interval]
            df_temp["stimulus_index"] = [i]
            df_temp["stimulus_repeat_logic"] = [1]
            df_temp["stimulus_repeat_sublogic"] = [1]
            df_temp["sampling_freq"] = self.sampling_frequency

            self.stimuli = pd.concat([self.stimuli, df_temp], ignore_index=True)

        return self.stimuli

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
            channel = self.channel.Voltage[limits[i, 0] : limits[i, 1]]
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
        self.stimuli = get_nr_repeats(self.stimuli)

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

    for row in df.itertuples():
        stimulus_repeat_logic = row.stimulus_repeat_logic
        trigger_int = row.trigger_int
        trigger_fr_relative = row.trigger_fr_relative

        if stimulus_repeat_logic == 0 or stimulus_repeat_logic >= trigger_int.shape[0]:
            trigger_ends.append(trigger_fr_relative[:-1] + trigger_int)
        else:
            shape = (
                stimulus_repeat_logic,
                int(trigger_int.shape[0] / stimulus_repeat_logic),
            )
            ints = trigger_int[: np.multiply(*shape)].reshape(shape)
            min_int = np.min(ints, axis=0)

            triggers_sorted = trigger_fr_relative[: np.multiply(*shape)].reshape(shape)
            trigger_ends.append((triggers_sorted + min_int).flatten())

    df["trigger_ends"] = trigger_ends
    return df


def get_nr_repeats(df):
    repeats = []
    nr_triggers = np.asarray([len(x) - 1 for x in df["trigger_fr_relative"]])
    nr_repeats = nr_triggers // df["stimulus_repeat_logic"]
    df["nr_repeats"] = nr_repeats
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


def create_stim_df():
    stimuli_df = pd.DataFrame(
        columns=[
            "stimulus_name",
            "begin_fr",
            "end_fr",
            "trigger_fr_relative",
            "trigger_int",
            "stimulus_index",
            "stimulus_repeat_logic",
            "stimulus_repeat_sublogic",
            "sampling_freq",
        ]
    )
    stimuli_df = stimuli_df.astype(
        {
            "stimulus_name": "str",
            "begin_fr": "int32",
            "end_fr": "int32",
            "trigger_fr_relative": "object",
            "trigger_int": "object",
            "stimulus_index": "int32",
            "stimulus_repeat_logic": "int32",
            "stimulus_repeat_sublogic": "int32",
            "sampling_freq": "float",
        }
    )
    return stimuli_df


# def correct_last_trigger(df):
#     sampling_freq = df["sampling_freq"].values[0]
#     trigger_fr_relative = []
#     for row in df.itertuples():
#         stimulus_index = row.stimulus_index
#         mean_trigger_times = stimulus_spikes.mean_trigger_times(
#             df, [stimulus_index], time="frames"
#         )
#
#
#
#         # Check if the last trigger matches the mean trigger times
#         last_trigger_diff = row.trigger_int[-1]
#         if last_trigger_diff > mean_trigger_times[-1]:
#             # Correct the last trigger
#             new_trigger_fr_relative = row.trigger_fr_relative.copy()
#             new_trigger_fr_relative[-1] = (
#                 new_trigger_fr_relative[-2] + mean_trigger_times[-1]
#             )
#             trigger_fr_relative.append(new_trigger_fr_relative)
#         else:
#             trigger_fr_relative.append(row.trigger_fr_relative)
#
#     df["trigger_fr_relative"] = trigger_fr_relative
#     return df
