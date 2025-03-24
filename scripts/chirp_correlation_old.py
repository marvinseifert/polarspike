from polarspike import Overview, spiketrain_plots, colour_template, histograms, chirps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from matplotlib.ticker import FixedLocator
from skimage.feature import peak_local_max, canny
from sklearn.cluster import HDBSCAN
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfilt


# %%
def normalize_0_1(data):
    return (data - data.min()) / (data.max() - data.min())


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
recordings.dataframes["chirps"] = recordings.spikes_df.query(
    "stimulus_name == 'chirp'"
    "| stimulus_name == 'chirp413'|stimulus_name == 'chirp460'|stimulus_name == 'chirp535'|stimulus_name == 'chirp610'"
).copy()

# %% import labels
# labels_df = pd.read_csv(r"A:\Marvin\fff_clustering\labels.csv")
labels_df = pd.read_pickle(r"A:\Marvin\fff_clustering\coeff_label_df")
# labels_df = labels_df.set_index(["recording", "cell_index"])

recordings.dataframes["chirps"]["labels"] = -1

recordings.dataframes["chirps"] = recordings.dataframes["chirps"].set_index(
    ["recording", "cell_index"]
)
recordings.dataframes["chirps"].update(labels_df)

recordings.dataframes["chirps"] = recordings.dataframes["chirps"].reset_index()

spikes = recordings.get_spikes_df("chirps", carry=["stimulus_name", "labels"])
# %%
# Define chirp
start_freq = 1  # Start frequency in Hz
end_freq = 30  # End frequency in Hz
duration = 30  # Duration in seconds
refresh_rate = 300  # Refresh rate in Hz
num_cycles = duration * refresh_rate

chirp_real = []
time = []
freq = []
for i in range(num_cycles):
    t = i / refresh_rate
    f = np.power(end_freq / start_freq, t / duration) * start_freq
    # Intended instantaneous frequency (exponential chirp)
    k = np.log(end_freq / start_freq) / duration
    f_intended = start_freq * np.exp(k * t)

    # Actual instantaneous frequency due to incorrect phase
    f_actual = f_intended * (1 + k * t)
    value = np.sin(2 * np.pi * f * t)
    chirp_real.append(int((value + 1) * 4095 / 2))
    time.append(t)
    freq.append(f_actual)

chirp_real = np.asarray(chirp_real)

chirp_real = normalize_0_1(chirp_real)
chirp_complete = np.zeros(35 * 300 - 1)
chirp_complete[300 * 3 : 3300 * 3] = chirp_real
c_freqs = np.zeros_like(chirp_complete)
c_freqs[300 * 3 : 3300 * 3] = np.asarray(freq)
time = np.arange(0, 35 - 1 / 300, 1 / 300)

# %%
from scipy.signal import hilbert, chirp

intended_chirp = chirp(time, f0=1, f1=30, t1=30, method="logarithmic")
time_intended = np.linspace(3, 33, intended_chirp.shape[0])
transform_intended = hilbert(intended_chirp)
transform = hilbert(chirp_complete)
envelope_intended = np.abs(transform_intended)
envelope = np.abs(transform)
# %%
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
fig.add_scatter(x=time, y=chirp_complete, row=1, col=1)
fig.add_scatter(x=time, y=envelope, row=2, col=1)
fig.add_scatter(x=time_intended, y=intended_chirp, row=3, col=1)
fig.add_scatter(x=time_intended, y=envelope_intended, row=4, col=1)
fig.show(renderer="browser")
# %% Example cell
idx = 5
cell_spikes = spikes.query(
    f"recording == 'chicken_19_07_2024_p0' & cell_index == 454 & stimulus_name == 'chirp'"
)
# %% PSTH
psth, bins, repeat = histograms.psth_by_index(
    cell_spikes, index=["repeat"], bin_size=0.01 / 3, window_end=35, return_idx=True
)

# %%
psth_hilbert = np.abs(hilbert(psth[3]))
fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
fig.add_scatter(x=bins[:-1], y=psth[3], row=1, col=1)
fig.add_scatter(x=bins[:-1], y=psth_hilbert, row=2, col=1)
fig.add_scatter(x=time, y=chirp_complete, row=3, col=1)
fig.add_scatter(x=time, y=envelope, row=4, col=1)
fig.show(renderer="browser")

# %% plot
fig, axs = plt.subplots(5, 1, figsize=(10, 10))
for idx, trace in enumerate(psth):
    axs[idx].plot(bins[:-1], trace)
axs[-1].plot(bins[:-1], chirp_complete)
fig.show()

# %% wavelet transform
# perform CWT
wavelet = "cmor3.0-1.5"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=300)
sampling_period = 0.01 / 3
cwtmatr, freqs = pywt.cwt(
    psth[2] - psth[2].mean(),
    widths,
    wavelet,
    sampling_period=sampling_period,
)

cwtmatr = np.abs(cwtmatr[:-1, :-1])

# %%

# find closest position to actual frequency
c_freqs_positions = np.zeros(c_freqs.shape[0] - 1, dtype=int)
for i, freq in enumerate(c_freqs[:-1]):
    c_freqs_positions[i] = np.argmin(np.abs(freq - freqs[:-1]))

# %% find the most powerfull power trace
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
# %%

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
pcm = axs.pcolormesh(bins[:-1], freqs, cwtmatr)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Frequency (Hz)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
# axs[1].plot(bins[:-1], psth[0])
# fig.colorbar(pcm, ax=axs[0])
# change y ticks to show frequency in Hz
log_ticks = [0.1, 1, 30, 100]
axs.yaxis.set_major_locator(FixedLocator(log_ticks))

# Set the labels to match the log ticks
axs.set_yticklabels(log_ticks)
# plot red horizontal line at 1, 10 and 100 Hz
# for freq in log_ticks:
#     axs[0].axhline(freq, color="red", linestyle="--", linewidth=0.5)
axs.plot(time, c_freqs, c="k")
# axs.plot(bins[:-1], c_freqs_positions, c="r")
fig.show()


# %%
fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
pcm = axs.pcolormesh(bins[:-1], np.arange(300), cwtmatr)
axs.scatter(bins[:-2], max_positions, c="r", s=10)
fig.show()
# %%
sos = butter(10, 0.5, "low", fs=300, output="sos")
filtered_signal = moving_average(np.abs(max_power), 400)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(bins[:-2], np.abs(max_power))
ax.hlines(np.max(np.abs(max_power) * 0.1), 0, 35, color="r")
ax.plot(bins[:-401], moving_average(np.abs(max_power), 400))
ax.hlines(np.max(filtered_signal) * 0.10, 0, 35, color="k")
fig.show()


# %% downsample
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(bins[:-2], np.abs(max_power))
fig.show()
# %% plot spectrogram
# find location where signal is 5% of max on the right side of the power trace
power_max = np.max(np.abs(max_power))
power_max_position = np.argmax(np.abs(max_power))
power_max_frequency = c_freqs[power_max_position]
threshold = power_max * 0.10
threshold_position = np.where(np.abs(max_power) > threshold)[0][-1]
threshold_power = np.sum(np.abs(max_power)[threshold_position:])
power_ratio = (
    np.sum(np.abs(max_power)[:threshold_position]) / threshold_power
) / np.sum(np.abs(max_power))

print(f"max frequency: {power_max_frequency}, power_ratio: {power_ratio}")
# %% find peaks
peaks = peak_local_max(cwtmatr, min_distance=4, threshold_rel=0.4)
# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
pcm = axs.pcolormesh(bins[:-1], freqs, cwtmatr)
# add peaks
# axs[0].scatter(bins[peaks[:, 1]], freqs[peaks[:, 0]], c="r", s=10)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Frequency (Hz)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
fig.show()
# %% find edges
edges = canny(cwtmatr, sigma=3)
fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
pcm = axs.pcolormesh(bins[:-1], freqs, edges, cmap="gray")
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
fig.show()
# %%
edge_y, edge_x = np.where(edges)
# edge_y = np.log10(edge_y[::-1])
# %% plot as scatter
clustering = HDBSCAN(min_cluster_size=10, min_samples=5)
clustering.fit(np.stack([edge_x, edge_y], axis=1))

fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
axs.scatter(bins[edge_x], freqs[edge_y], c=clustering.labels_, s=10)
fig.show()
# %% plot only biggest cluster
biggest_cluster = np.argmax(np.bincount(clustering.labels_))
fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
axs.scatter(
    bins[edge_x][clustering.labels_ == biggest_cluster],
    freqs[edge_y][clustering.labels_ == biggest_cluster],
    s=10,
)
fig.show()

# %%

# %%
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# %% fit spline
spline = make_pipeline(PolynomialFeatures(3), Ridge())
spline.fit(
    bins[edge_x][clustering.labels_ == biggest_cluster].reshape(-1, 1),
    freqs[edge_y][clustering.labels_ == biggest_cluster],
)
# %% plot spline
fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
axs.scatter(
    bins[edge_x][clustering.labels_ == biggest_cluster],
    freqs[edge_y][clustering.labels_ == biggest_cluster],
    s=10,
)
x = np.linspace(
    bins[edge_x][clustering.labels_ == biggest_cluster].min(),
    bins[edge_x][clustering.labels_ == biggest_cluster].max(),
    1000,
)
y = spline.predict(x.reshape(-1, 1))
axs.plot(x, y)
fig.show()
