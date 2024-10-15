from polarspike import Overview, stimulus_trace
import numpy as np

# %%
recording = Overview.Recording.load(r"A:\Marvin\chicken_30_08_2024\Phase_00\overview")

# %%
triggers = recording.stimulus_df.loc[0]["trigger_fr_relative"]

mean_diff = np.mean(np.diff(triggers))

nr_frames = 24000
frame_duration = (1 / 60) * 20000

new_frames = np.arange(0, nr_frames + 1) * frame_duration
new_frames = np.ceil(new_frames).astype(int)

# %%
all_triggers = recording.stimulus_df["trigger_fr_relative"].to_numpy()
all_triggers[0] = new_frames
recording.stimulus_df["trigger_fr_relative"] = all_triggers
all_trigger_ints = recording.stimulus_df["trigger_int"].to_numpy()
all_trigger_ints[0] = np.diff(new_frames)
recording.stimulus_df["trigger_int"] = all_trigger_ints

# %%
recording.stimulus_df.loc[0, "stimulus_repeat_logic"] = 2400
new_df = stimulus_trace.get_nr_repeats(recording.stimulus_df)

recording.save_save()
