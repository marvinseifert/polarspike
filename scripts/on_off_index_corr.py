from polarspike import (
    Overview,
    histograms,
    Opsins,
    colour_template,
    stimulus_spikes,
)
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# %%
recording = Overview.Recording.load(
    r"B:\Marvin\chicken_02_13_2024_2nd\Phase_01\overview"
)
stimulus_id = 0
bin_size = 0.05
# %%
mean_trigger = stimulus_spikes.mean_trigger_times(recording.stimulus_df, [stimulus_id])
spikes = recording.get_spikes_triggered([[stimulus_id]], [["all"]], pandas=False)

# %%
trace, bins = histograms.psth(spikes, bin_size=bin_size, end=np.sum(mean_trigger))

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(bins[:-1], trace, c="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Number of spikes")
fig.show()

# %%
# We create a signal trace which is zero everywhere except for the stimulus onset
# Create the signal trace
average_stim_length = np.mean(mean_trigger)
average_nr_bins = int(np.round(average_stim_length / bin_size))
signal_trace_on = np.zeros(trace.shape[0] * 2)
signal_trace_on[::average_nr_bins] = 1


# %%
# Cross correlate
cross_corr_on = np.correlate(trace, signal_trace_on, mode="same")


# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(cross_corr_on, c="blue", label="On")
ax.legend()
fig.show()

# %%

cross_corr_on_off = cross_corr_on_off[
    int(np.floor(len(cross_corr_on_off) / 2))
    + 1 : int(np.floor(len(cross_corr_on_off) / 2))
    + average_nr_bins * 2
]
cross_corr_on = cross_corr_on[
    int(np.floor(len(cross_corr_on) / 2))
    + 1 : int(np.floor(len(cross_corr_on) / 2))
    + average_nr_bins * 2
]
cross_corr_off = cross_corr_off[
    int(np.floor(len(cross_corr_off) / 2))
    + 1 : int(np.floor(len(cross_corr_off) / 2))
    + average_nr_bins * 2
]

# %%
# Plot the cross correlation
x_lag = np.arange(0, average_nr_bins * 2 * bin_size, bin_size)[:-1]
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(x_lag, cross_corr_on_off, c="red", label="On-Off")
ax.plot(x_lag, cross_corr_on, c="blue", label="On")
ax.plot(x_lag, cross_corr_off, c="green", label="Off")
ax.legend()
ax.set_xlabel("Lag")
ax.set_ylabel("Cross correlation")
fig.show()

# %%
sum_on = np.sum(cross_corr_on)
sum_off = np.sum(cross_corr_off)
sum_on_off = np.sum(cross_corr_on_off)

# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(["On", "Off", "On-Off"], [sum_on, sum_off, sum_on_off])
ax.set_ylabel("Sum of cross correlation")
fig.show()
# %% Need to come up with an index that shows how far away the cells are from perfect OnOff responses
# In case a cell is perfectly On Off, the relation between sum_on_off, sum_on and sum_off is 2:1:1
# The index should be 0 in this case
# The index should be 1 in case the cell is perfectly On
# The index should be -1 in case the cell is perfectly Off
# The index should be 0.5 in case the cell is 50% On and 50% Off
# The index should be -0.5 in case the cell is 50% Off and 50% On

# %%
# Calculate the index
index = sum_on / sum_off - 1
