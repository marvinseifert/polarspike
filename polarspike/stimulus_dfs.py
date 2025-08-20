import pandas as pd
import numpy as np
from polarspike import stimulus_trace


class Stimulus_df_schroeder:
    def __init__(self, recording_name, sampling_freq):
        self.recording_name = recording_name
        self.stimulus_df = pd.DataFrame()
        self.stimulus_idx = 0
        self.sampling_freq = sampling_freq

    def add_stimulus(
            self,
            stimulus_name,
            starts,
            ends,
            stimulus_repeat_logic=1,
            stimulus_repeat_sublogic=1,
            random_ids=None,
    ):
        # Establish the trigger by interleaving the start and end times
        trigger_store = np.empty((1), dtype=object)
        trigger_store[0] = self.create_trigger(starts, ends, self.sampling_freq)
        # Create a DataFrame with the triggers
        stimulus_df = pd.DataFrame()
        stimulus_df["stimulus_name"] = [stimulus_name]
        stimulus_df["begin_fr"] = [trigger_store[0][0]]
        stimulus_df["end_fr"] = [trigger_store[0][-1]]
        stimulus_df["trigger_fr_relative"] = trigger_store
        trigger_int = np.diff(trigger_store[0])
        trigger_store[0] = trigger_int
        stimulus_df["trigger_int"] = trigger_store
        stimulus_df["stimulus_index"] = self.stimulus_idx
        stimulus_df["stimulus_repeat_logic"] = stimulus_repeat_logic
        stimulus_df["stimulus_repeat_sublogic"] = stimulus_repeat_sublogic
        stimulus_df["sampling_freq"] = self.sampling_freq
        stimulus_df["recording"] = self.recording_name
        stimulus_df = stimulus_trace.find_ends(stimulus_df)
        stimulus_df = stimulus_trace.get_nr_repeats(stimulus_df)
        # In case of a random stimulus, add the random ids
        if random_ids is not None:
            id_store = np.empty((1), dtype=object)
            id_store[0] = random_ids
            stimulus_df["random_ids"] = id_store
        self.stimulus_df = pd.concat([self.stimulus_df, stimulus_df], ignore_index=True)
        self.stimulus_idx += 1

    @staticmethod
    def create_trigger(starts, ends, sampling_freq):
        triggers = np.zeros((starts.shape[0] * 2))
        triggers[::2] = starts
        triggers[1::2] = ends
        mean_trigger_diff = np.mean(np.diff(triggers))
        triggers = np.hstack([triggers, triggers[-1] + mean_trigger_diff])
        triggers = (triggers * sampling_freq).astype(int)
        return triggers


def split_triggers(old_triggers, nr_splits=1):
    # Add dimension if flat array is provided
    new_triggers = np.concatenate(
        (old_triggers, old_triggers[:, :-1] + np.diff(old_triggers, axis=1) / 2),
        axis=1,
    ).astype(int)
    new_triggers = np.sort(new_triggers, axis=1)
    for _ in range(1, nr_splits):
        old_triggers = new_triggers.copy()
        new_triggers = np.concatenate(
            (
                old_triggers,
                old_triggers[:, :-1] + np.diff(old_triggers, axis=1) / 2,
            ),
            axis=1,
        ).astype(int)
        new_triggers = np.sort(new_triggers, axis=1)
    new_intervals = np.diff(new_triggers, axis=1)
    return new_triggers, new_intervals


def split_triggers_df(
        df: pd.DataFrame, stimulus_id: list, nr_splits: int = 1
) -> pd.DataFrame:
    """
    Splits the triggers of a stimulus dataframe for a given stimulus id or a number of stimulus ids.
    :param df : The stimulus dataframe
    :param stimulus_id: The stimulus id or ids that should be split
    :param nr_splits: The number of splits
    :return:
    """

    array_of_triggers = np.vstack(df.loc[stimulus_id, "trigger_fr_relative"].values)
    new_triggers, new_intervals = split_triggers(array_of_triggers, nr_splits)


def add_triggers(df, stimulus_id, new_trigger, new_logic, new_sublogic):
    if type(stimulus_id[0]) is str and stimulus_id[0] == "all":
        stimulus_id = np.arange(len(df)).tolist()
    else:
        stimulus_id = df.query("stimulus_index in @stimulus_id").index.tolist()
    sub_df = df.loc[stimulus_id]
    sub_df["trigger_fr_relative"] = new_trigger
    sub_df["trigger_int"] = np.diff(new_trigger)
    sub_df["stimulus_repeat_logic"] = new_logic
    sub_df["stimulus_repeat_sublogic"] = new_sublogic

    return df


def add_trigger_int(df, stimulus_id, interval, new_logic, new_sublogic):
    """
    Adds a trigger interval to the stimulus dataframe.
    :param df : pd.DataFrame : The stimulus dataframe
    :param stimulus_id : list : The stimulus indices to which the trigger interval will be added
    :param interval : float : The interval in seconds
    :param new_logic : int : The new logic for the stimulus
    :param new_sublogic : int : The new sublogic for the stimulus
    :return:
    """
    if type(stimulus_id[0]) is str and stimulus_id[0] == "all":
        stimulus_id = np.arange(len(df)).tolist()
    else:
        stimulus_id = df.query("stimulus_index in @stimulus_id").index.tolist()

    df = df.copy()
    begin_fr = df["begin_fr"].values[stimulus_id]
    end_fr = df["end_fr"].values[stimulus_id]
    sampling_freq = df["sampling_freq"].values[stimulus_id]
    new_trigger_all = []
    for idx in range(len(stimulus_id)):
        new_trigger = (
                np.arange(
                    begin_fr[idx],
                    end_fr[idx]
                    + interval * sampling_freq[idx]
                    - (interval * sampling_freq[idx]),
                    interval * sampling_freq[idx],
                )
                - begin_fr[idx]
        ).astype(int)
        new_trigger_all.append(new_trigger)
    new_trigger_array = np.empty(len(new_trigger_all), dtype=object)
    new_trigger_array[:] = new_trigger_all
    df = add_triggers(df, stimulus_id, new_trigger_array, new_logic, new_sublogic)
    return df
