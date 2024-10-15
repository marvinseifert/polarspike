import Overview, histograms
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
project_root = Path(r"A:\Marvin\chicken_19_07_2024\Phase_00")
recording = Overview.Recording.load(project_root / "overview")

# %%
all_spikes = pl.read_parquet(recording.parquet_path)

# %%
recording_duration_min = all_spikes["times"].max() / recording.sampling_freq / 60
# %%
psth, bins = np.histogram(
    all_spikes["times"], bins=int(np.ceil(recording_duration_min)) * 10
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(all_spikes["times"], bins=258 * 10, histtype="step", cumulative=True)
# change axis to log
fig.show()


# %%
fig, ax = plt.subplots(figsize=(10, 10))
psth_z = (psth - psth.mean()) / psth.std()
ax.plot(bins[:-2], np.diff(psth_z))
ax.hlines(-5, 0, bins[-1], color="red")
fig.show()


# %%
wave_bins = np.where(psth_z > 5)[0]
# %% Revert back to frames
frames_per_bin = all_spikes["times"].max() / len(psth) / 6
wave_frames = wave_bins * frames_per_bin * 6

# %% 1 minute gap before and after wave

wave_begins = wave_frames - frames_per_bin
wave_ends = wave_frames + frames_per_bin


# %%
np.save(project_root / "wave_begins.npy", wave_begins)
np.save(project_root / "wave_ends.npy", wave_ends)
# %% zscore the psth
psth_z = (psth - psth.mean()) / psth.std()

fig, ax = plt.subplots()
ax.plot(bins[:-1], psth_z)
fig.show()

# %% Detrend the psth
from scipy.signal import detrend

psth_detrend = detrend(psth, bp=np.arange(0, 2580, 10))
psth_z = (psth_detrend - psth_detrend.mean()) / psth_detrend.std()
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(bins[:-1], psth_z)
fig.show()


np.where(psth_z > 5)
