from polarspike import Overview
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# %%
recording = Overview.Recording.load(
    r"/mnt/nas_b/Marvin/chicken_29_05_2026/Phase_01/overview"
)

recording.stimulus_df.loc[0, "trigger_fr_relative"].shape

# %%
expected = 15 * 60 * 60 * 5 + 1
missing = expected - recording.stimulus_df.loc[0, "trigger_fr_relative"].shape[0]

# %%
trigger = np.arange(
    0,
    missing * (1 / 15) * recording.sampling_freq,
    (1 / 15) * recording.sampling_freq,
    dtype=int,
) + np.max(recording.stimulus_df.loc[0, "trigger_fr_relative"])

# %%
np.mean(np.diff(trigger))
np.mean(np.diff(recording.stimulus_df.loc[0, "trigger_fr_relative"]))
# %%
recording.dataframes["stimulus_df"].at[0, "trigger_fr_relative"] = np.concatenate(
    [recording.stimulus_df.loc[0, "trigger_fr_relative"], trigger]
)
# %%
recording.dataframes["stimulus_df"].at[
    0, "trigger_fr_relative"
] = recording.stimulus_df.loc[0, "trigger_fr_relative"][:-1]
# %%
recording.save_save()
