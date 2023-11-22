import pandas as pd
import numpy as np
import stimulus_trace


class Stimulus_df_schroeder:

    def __init__(self, recording_name, sampling_freq):
        self.recording_name = recording_name
        self.stimulus_df = pd.DataFrame()
        self.stimulus_idx = 0
        self.sampling_freq = sampling_freq

    def add_stimulus(self, stimulus_name, starts, ends, stimulus_repeat_logic=1, stimulus_repeat_sublogic=1,
                     random_ids=None):
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
