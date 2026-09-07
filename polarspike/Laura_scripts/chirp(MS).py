# Script to process chirp spikes

# Import dependencies
from polarspike import Overview, histograms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from matplotlib.ticker import FixedLocator
from skimage.feature import peak_local_max, canny
from sklearn.cluster import HDBSCAN
from plotly.subplots import make_subplots
from scipy.signal import butter


# %% Create helper functions for normalisation and averaging of data
def normalize_0_1(data):
    return (data - data.min()) / (data.max() - data.min())


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# %% Load chirp spikes
# Select the recording
recordings = Overview.Recording.load(r"F:\Laura\zebrafish_05_11_2025\Phase_01\overview")
# Select stimulus that are chirps
stimulus_id = [1,7,9]
# Create new "chirps" pandas dataframe, stored within the recording.dataframes dictionary
recordings.dataframes["chirps"] = recordings.spikes_df.query(
    "stimulus_index in @stimulus_id"
).copy()
# Get just the spikes from the chirp dataframe, and also carries across the stimulus name column, to create new dataframe called spikes
spikes = recordings.get_spikes_df("chirps", carry=["stimulus_name"])


# %%
# Function to create an exponential frequency chirp, from start freq to end freq over length duration.
# It then converts this chirp into an LED brightness/PWM waveform (0 to max power)
# It pads a 30s chirp into a 35s total, to match how the snippets are padded
# PWM is pulse width modulation (method to control average power delivered to an LED by rapidly switching it on and off). This is what is used to control brightness.
def generate_chirp_data(start_freq, end_freq, duration, refresh_rate, max_power=4095):
    """
    Translates the C++ freq_chirp_template logic into Python.
    Returns time, brightness values, and instantaneous frequency arrays.
    """
    # 1. Setup constants (matching C++ logic)
    num_cycles = int(duration * refresh_rate)

    # Calculate beta (growth rate)
    # C++: log(end / start) / duration
    beta = np.log(end_freq / start_freq) / duration

    # 2. Generate Time Array
    # C++: i / refresh_rate
    t = np.linspace(0, duration, num_cycles, endpoint=False)

    # 3. Calculate Exponential Term for all t
    # C++: exp_bt = exp(beta * t)
    exp_bt = np.exp(beta * t)

    # 4. Calculate Frequency at time t
    # C++: freq = start_freq * exp_bt
    freqs = start_freq * exp_bt

    # 5. Calculate Correct Instantaneous Phase
    # C++: phase = 2 * PI * (start / beta) * (exp_bt - 1)
    phase = 2 * np.pi * (start_freq / beta) * (exp_bt - 1)

    # 6. Calculate Sine Value
    # C++: value = sin(phase)
    raw_sine = np.sin(phase)

    # 7. Scale to LED Brightness (0 to max_power)
    # C++: (value + 1.0f) * max_power / 2.0f
    chirp_signal = (raw_sine + 1.0) * max_power / 2.0
    chirp_signal = chirp_signal.astype(int)

    return t, chirp_signal, freqs


# --- Configuration (Matching your snippets) ---
start_freq = 1  # Hz
end_freq = 30  # Hz
duration = 30  # Seconds
refresh_rate = 300  # Hz

# --- Generate Data ---
time, chirp_real, freq_actual = generate_chirp_data(
    start_freq, end_freq, duration, refresh_rate
)

# --- Output Formatting (Matching your second snippet) ---
# Your snippet padded the data into a larger array.
# Replicating that structure here:

total_length_sec = 35
total_samples = int(total_length_sec * refresh_rate) - 1  # Based on your snippet length

# Create the full container arrays filled with zeros
chirp_complete = np.zeros(total_samples)
c_freqs = np.zeros(total_samples)
full_time_axis = np.arange(0, total_length_sec - 1 / refresh_rate, 1 / refresh_rate)

# Define offset (Your snippet used 3 seconds padding: 300 * 3)
offset = 300 * 3
end_idx = offset + len(chirp_real)

# Insert the generated chirp into the complete arrays
# Ensure we don't overflow if the calculated lengths differ slightly due to rounding
limit = min(len(chirp_complete), end_idx)
insert_len = limit - offset

if insert_len > 0:
    chirp_complete[offset:limit] = chirp_real[:insert_len]
    c_freqs[offset:limit] = freq_actual[:insert_len]

# --- Verification Plot (Optional) ---
# Plots LED brightness (PWM value) vs time, across the chirp
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(full_time_axis, chirp_complete)
plt.title("LED Brightness (Chirp Signal)")
plt.ylabel("PWM Value")

# Plots chirp frequency versus time (you can see it increases exponentially)
plt.subplot(2, 1, 2)
plt.plot(full_time_axis, c_freqs, color="orange")
plt.title("Instantaneous Frequency (Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()

# # %%
# # We want to compare our actual chirp stimulus ("chirp_complete", generated in the generate_chirp_data function)
# # to the "intended_chirp" (a perfect representation of a chirp), to see how close they are and that our chirp isn't distorted etc/only changing in frequency not amplitude/brightness.
# # To do this, we the Hilbert Transform can be applied to each chirp separately and the results compared.
# # The Hilbert transform extracts properties from the chirp signals. The "envelope" is like a boundary line around the overall pattern of the chirp signal.
# # The actual chirp stimulus envelope is not flat since the PWM is not centred around zero - this is fine for us I think, as brightness cannot go below 0. We could adjust for baseline to centre it and then it would be flat.
# from scipy.signal import hilbert, chirp
#
# intended_chirp = chirp(time, f0=1, f1=30, t1=30, method="logarithmic") # create "perfect" intended chirp
# time_intended = np.linspace(3, 33, intended_chirp.shape[0]) # make a shifted time axis for intended chirp to match our chirp data
# transform_intended = hilbert(intended_chirp) # apply hilbert transform to intended chirp
# transform = hilbert(chirp_complete) # apply hilbert transform to actual chirp
# envelope_intended = np.abs(transform_intended) # get envelope from hilbert transform of intended chirp
# envelope = np.abs(transform) # get envelope from hilbert transform of actual chirp
#
# # Plot actual chirp and envelope and intended chirp and envelope for visual check
# fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
# fig.add_scatter(x=time, y=chirp_complete, row=1, col=1, name="chirp actual")
# fig.add_scatter(x=time, y=envelope, row=2, col=1, name="hilbert actual")
# fig.add_scatter(x=time_intended, y=intended_chirp, row=3, col=1, name="chirp intended")
# fig.add_scatter(x=time_intended, y=envelope_intended, row=4, col=1, name="hilbert intended")
# fig.show(renderer="browser")

# %%
# Load an example cell fom spikes dataframe, creating new pandas dataframe called cell_spikes
cell_spikes = spikes.query(f"cell_index == 270 & stimulus_index == 1")

# %%
# Generate peristimulus time histogram (PSTH) - one for each repeat of the chirp (x3)
psth, bins, repeat = histograms.psth_by_index(
    cell_spikes, index=["repeat"], bin_size=0.01 / 3, window_end=35, return_idx=True
)

# # plot PSTH for each repeat, and then chirp stimulus below
# fig, axs = plt.subplots(4, 1, figsize=(10, 10))
# for idx, trace in enumerate(psth):
#     axs[idx].plot(bins[:-1], trace)
# axs[-1].plot(bins[:-1], chirp_complete)
# fig.show()
#
# # Plots PSTH for repeat one of the chirp, then the hilbert envelope of the psth, then the actual chirp, then the hilbert envelope
# psth_hilbert = np.abs(hilbert(psth))
# fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
# fig.add_scatter(x=bins[:-1], y=psth[0], row=1, col=1, name="psth")
# fig.add_scatter(x=bins[:-1], y=psth_hilbert[0], row=2, col=1, name="psth_hilbert")
# fig.add_scatter(x=time, y=chirp_complete, row=3, col=1, name="chirp actual")
# fig.add_scatter(x=time, y=envelope, row=4, col=1, name="hilbert actual")
# fig.show(renderer="browser")


# %% Wavelet transform
# Unlike the Hilbert transform, which extracts one dominant oscillation, wavelets looks at many frequencies simultaneously

# Select a wavelet to use - CMW (complex morlet wavelet)
wavelet = "cmor3.0-1.5"

# Wavelets do not directly use frequency, they use scale (smaller scale = higher frequency and vice versa, as you stretch and compress the wavelet).
# You then shift the wavelet along your signal, changing the scale as you go (?)
# A logarithmic scale is chosen since biological systems respond on log frequency scales, as suggested by Torrence and Compo:
widths = np.geomspace(1, 1024, num=300)  # geomspace is logarithmic
sampling_period = 0.01 / 3  # matches the PSTH bin sizes (3.33ms bins)

# Perform CMW
cwtmatr, freqs = pywt.cwt(
    psth[2] - psth[2].mean(),  # uses third chirp repeat
    widths,
    wavelet,
    sampling_period=sampling_period,
)

cwtmatr = np.abs(cwtmatr[:-1, :-1])
# cwtmatr is a 2D matrix where rows are wavelet frequency bins and columns are time points

# %%
# find closest position to actual frequency (i.e. find the wavelet frequency bin which is closest to the chirp frequency at every time point)
# create new array (c_freqs_positions) that stores “Which wavelet frequency bin is closest to the real chirp frequency at each time point?”
c_freqs_positions = np.zeros(
    c_freqs.shape[0] - 1, dtype=int
)  # c_freq is the actual chirp frequency at every time point (ground truth)
for i, freq in enumerate(c_freqs[:-1]):
    c_freqs_positions[i] = np.argmin(np.abs(freq - freqs[:-1]))

# %% find the most powerful power trace
# We already have “best-guess” path through the time–frequency plane (c_freqs_positions, one frequency-bin index per time bin).
# This section searches around the stimulus-frequency track to find the frequency trajectory through the CWT that captures the most total power.
# i.e. asking: # “What if the neuron’s response is systematically shifted a bit higher or lower in frequency than the stimulus? Which shifted path gives the strongest total wavelet power?”
max_power = np.zeros(cwtmatr.shape[1])
max_positions = np.zeros(cwtmatr.shape[1], dtype=int)
power = -np.inf
for i in range(150):
    temp_positions = c_freqs_positions + i
    temp_positions[temp_positions >= 299] = 0
    power_trace = cwtmatr[temp_positions, np.arange(cwtmatr.shape[1])]
    if np.sum(power_trace) > power:
        max_power = power_trace
        power = np.sum(power_trace)
        max_positions = temp_positions
    temp_positions = c_freqs_positions - i
    temp_positions[temp_positions < 0] = 0
    power_trace = cwtmatr[temp_positions, np.arange(cwtmatr.shape[1])]
    if np.sum(power_trace) > power:
        max_power = power_trace
        power = np.sum(power_trace)
        max_positions = temp_positions
# # %%
# # plot result using matplotlib's pcolormesh (image with annoted axes)
# # each pixel tells us how strongly the neuron "oscillates" at frequency f at time t
# # black line is chirp stimuli
# fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# pcm = axs.pcolormesh(bins[:-1], freqs, cwtmatr)
# axs.set_yscale("log")
# axs.set_xlabel("Time (s)")
# axs.set_ylabel("Frequency (Hz)")
# axs.set_title("Continuous Wavelet Transform (Scaleogram)")
# # axs[1].plot(bins[:-1], psth[0])
# # fig.colorbar(pcm, ax=axs[0])
# # change y ticks to show frequency in Hz
# log_ticks = [0.1, 1, 10, 30, 100]
# axs.yaxis.set_major_locator(FixedLocator(log_ticks))
#
# # Set the labels to match the log ticks
# axs.set_yticklabels(log_ticks)
# # plot red horizontal line at 1, 10 and 100 Hz
# # for freq in log_ticks:
# #     axs[0].axhline(freq, color="red", linestyle="--", linewidth=0.5)
# axs.plot(time, freq_actual, c="k")
# # axs.plot(bins[:-1], c_freqs_positions, c="r")
# plt.show()

# # %%
# # plot again with best fit overlaid
# # We are visualising the frequency trajectory that captures the most total neural power.
# # If the neuron follows the chirp then the red line/dots will lie along the black line.
max_power[c_freqs[:-1] == 0] = 0
# fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# pcm = axs.pcolormesh(bins[:-1], np.arange(300), cwtmatr)
# axs.scatter(bins[:-2], max_positions, c="r", s=10)
# fig.show()
# # %%
# # Plots
# # simple plot - raw max power trace only
# fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# ax.plot(bins[:-2], np.abs(max_power))
# fig.show()

# extra plot with 10% thresholds and smoothed data
# sos = butter(10, 0.5, "low", fs=300, output="sos") # low pass filter created here but not used
filtered_signal = moving_average(
    np.abs(max_power), 400
)  # smooth max power trace (along red line)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(bins[:-2], np.abs(max_power))  # plot raw max power
ax.hlines(
    np.max(np.abs(max_power) * 0.1), 0, 35, color="r"
)  # Plot a red horizontal threshold at 10% of the raw max_power peak
ax.plot(bins[:-401], moving_average(np.abs(max_power), 400))  # plot smoothed av power
ax.hlines(
    np.max(filtered_signal) * 0.10, 0, 35, color="k"
)  # Plot a black horizontal threshold at 10% of the smoothed peak
fig.show()

# %% Extract statistics from max power trace
# find location where signal is 5% of max on the right side of the power trace

power_max = np.max(
    np.abs(max_power)
)  # the highest value of your stimulus-following power trace.
power_max_position = np.argmax(
    np.abs(max_power)
)  # the time index where that maximum occurs.
power_max_frequency = c_freqs[power_max_position]  # converted to frequency of stimulus
threshold = power_max * 0.10  # define a cutoff at 10% of peak tracking strength.
threshold_position = np.where(np.abs(filtered_signal) > threshold)[0][
    -1
]  # find time index
threshold_frequency = c_freqs[threshold_position]  # convert to frequency of stimulus
threshold_power = np.sum(
    np.abs(max_power)[threshold_position:]
)  # find all power after this cut off threshold
power_ratio = (
    np.sum(np.abs(max_power)[:threshold_position]) / threshold_power
) / np.sum(np.abs(max_power))

print(f"max frequency: {power_max_frequency}, power_ratio: {power_ratio}")
print(f"frequency threshold at {threshold_frequency} Hz")

# # %% find peaks
# peaks = peak_local_max(cwtmatr, min_distance=4, threshold_rel=0.4)
# # plot result using matplotlib's pcolormesh (image with annoted axes)
# fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# pcm = axs.pcolormesh(bins[:-1], freqs, cwtmatr)
# # add peaks
# # axs[0].scatter(bins[peaks[:, 1]], freqs[peaks[:, 0]], c="r", s=10)
# axs.set_yscale("log")
# axs.set_xlabel("Time (s)")
# axs.set_ylabel("Frequency (Hz)")
# axs.set_title("Continuous Wavelet Transform (Scaleogram)")
# fig.show()
#
# # %%
# peaks = peak_local_max(cwtmatr, min_distance=4, threshold_rel=0.4)
#
# fig, ax = plt.subplots(figsize=(20, 10))
#
# im = ax.imshow(
#     cwtmatr,
#     aspect="auto",
#     origin="lower",
#     extent=[
#         bins[0],      # left time
#         bins[-2],     # right time
#         freqs[0],     # bottom freq
#         freqs[-1]     # top freq
#     ]
# )
#
# # overlay peaks in real coordinates
# ax.scatter(
#     bins[:-1][peaks[:, 1]],     # time positions
#     freqs[peaks[:, 0]],         # frequency positions
#     c="r",
#     s=10
# )
#
# ax.set_yscale("log")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Frequency (Hz)")
# ax.set_title("Continuous Wavelet Transform (Scaleogram)")
#
# plt.colorbar(im, ax=ax)
# plt.show()
#
# # %% find edges
# # edges = canny(cwtmatr, sigma=3)
# # fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# # pcm = axs.pcolormesh(bins[:-1], freqs, edges, cmap="gray")
# # axs.set_yscale("log")
# # axs.set_xlabel("Time (s)")
# # fig.show()
#
# # find edges
# edges = canny(cwtmatr, sigma=3)
#
# fig, ax = plt.subplots(figsize=(20, 10))
#
# im = ax.imshow(
#     edges,
#     aspect="auto",
#     origin="lower",
#     cmap="gray",
#     extent=[
#         bins[0],      # left (time)
#         bins[-2],     # right (time)
#         freqs[0],     # bottom (freq)
#         freqs[-1]     # top (freq)
#     ]
# )
#
# ax.set_yscale("log")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Frequency (Hz)")
# ax.set_title("CWT Edge Detection")
#
# plt.show()
#
# # %%
# edge_y, edge_x = np.where(edges)
# # edge_y = np.log10(edge_y[::-1])
# # %% plot as scatter
# clustering = HDBSCAN(min_cluster_size=10, min_samples=5)
# clustering.fit(np.stack([edge_x, edge_y], axis=1))
#
# fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# axs.scatter(bins[edge_x], freqs[edge_y], c=clustering.labels_, s=10)
# fig.show()
# # %% plot only biggest cluster
# biggest_cluster = np.argmax(np.bincount(clustering.labels_))
# fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# axs.scatter(
#     bins[edge_x][clustering.labels_ == biggest_cluster],
#     freqs[edge_y][clustering.labels_ == biggest_cluster],
#     s=10,
# )
# fig.show()
#
# # %%
#
# # %%
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
#
# # %% fit spline
# spline = make_pipeline(PolynomialFeatures(3), Ridge())
# spline.fit(
#     bins[edge_x][clustering.labels_ == biggest_cluster].reshape(-1, 1),
#     freqs[edge_y][clustering.labels_ == biggest_cluster],
# )
# # %% plot spline
# fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
# axs.scatter(
#     bins[edge_x][clustering.labels_ == biggest_cluster],
#     freqs[edge_y][clustering.labels_ == biggest_cluster],
#     s=10,
# )
# x = np.linspace(
#     bins[edge_x][clustering.labels_ == biggest_cluster].min(),
#     bins[edge_x][clustering.labels_ == biggest_cluster].max(),
#     1000,
# )
# y = spline.predict(x.reshape(-1, 1))
# axs.plot(x, y)
# fig.show()
