from polarspike import Overview, spiketrain_plots, colour_template, histograms, chirps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from matplotlib.ticker import LogLocator
import pywt
from matplotlib.ticker import FixedLocator
from sklearn.decomposition import PCA
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from sklearn.cluster import HDBSCAN
from scipy.interpolate import interp1d
from sklearn.linear_model import SGDClassifier
import polars as pl

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
colours = {
    "chirp": "gray",
    "chirp413": CT.colours[8],
    "chirp460": CT.colours[6],
    "chirp535": CT.colours[4],
    "chirp610": CT.colours[0],
}
# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

# %%
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
# %%
recordings.dataframes["chirps"]["labels"].max()
# %%
spikes = recordings.get_spikes_df("chirps", carry=["stimulus_name", "labels"])
# %%
psths, bins, c_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=35,
    bin_size=0.01,
)
# %%
fig, ax = plt.subplots(
    np.max(spikes["labels"]) + 1, 5, figsize=(20, 60), sharex=True, sharey=True
)
chirp_position = {
    "chirp": 0,
    "chirp413": 1,
    "chirp460": 2,
    "chirp535": 3,
    "chirp610": 4,
}
for cluster in spikes["labels"].unique():
    spikes_temp = spikes.query(f"labels == {cluster}")
    psths, bins, c_index = histograms.psth_by_index(
        spikes_temp,
        index=["stimulus_name"],
        window_end=35,
        bin_size=0.01,
        return_idx=True,
    )
    psths /= len(spikes_temp.set_index(["recording", "cell_index"]).index.unique())

    for idx, psth in enumerate(psths):
        ax[cluster, chirp_position[c_index[idx][0]]].plot(
            bins[:-1], psth, c=colours[c_index[idx][0]]
        )
    ax[cluster, 0].set_title(f"Cluster {cluster}")
fig.show()


# %% Look at fourrier transform
example_cluster = 28
spikes_temp = spikes.query(f"labels == {cluster}")
psths, bins, c_index = histograms.psth_by_index(
    spikes_temp,
    index=["stimulus_name"],
    window_end=35,
    bin_size=0.005,
    return_idx=True,
)
sampling_rate = 100
fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=5)
for c_idx, chirp_name in enumerate(list(colours.keys())):
    psth = psths[np.where(c_index == chirp_name)[0][0]]
    # %% welch
    f, Pxx = welch(psth, fs=sampling_rate, nperseg=1024)
    axs[0, c_idx].semilogx(f, Pxx, c=colours[chirp_name])
    axs[1, c_idx].plot(bins[:-1], psth, c=colours[chirp_name])
ax.set_xlabel("Frequency [Hz]")
fig.show()

# %% Wavelet transform
chirp_x, chirp = chirps.expo_chirp(3, 33, 1, 30)
# extend chirp by adding zeros at start and finish
chirp_binsize = 30 / chirp.shape[0]
nr_front_zeros = int(3 / chirp_binsize)
nr_back_zeros = int(2 / chirp_binsize)
chirp_complete = np.zeros(chirp.shape[0] + nr_front_zeros + nr_back_zeros)
chirp_complete[nr_front_zeros : chirp.shape[0] + nr_front_zeros] = chirp

# %% plot chirp
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(chirp_complete)
fig.show()
# %%

# perform CWT
wavelet = "cmor3.0-1.5"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
sampling_period = 0.01
cwtmatr, freqs = pywt.cwt(
    psth - psth.mean(), widths, wavelet, sampling_period=sampling_period
)
# map chirp to have same length as cwtmatr by extrapolating
func = interp1d(
    np.linspace(0, 35, chirp_complete.shape[0]), chirp_complete, kind="linear"
)
x_new = np.linspace(0, 35, cwtmatr.shape[1])
chirp_bined = func(x_new)

# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

# cut to relevant frequencies 1Hz - 120Hz
lower_idx = np.where(freqs > 0.9)[0][-1]
upper_idx = np.where(freqs > 120)[0][0]

cwtmatr = cwtmatr[upper_idx : lower_idx - 1, :]
freqs = freqs[upper_idx:lower_idx]


# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
pcm = axs[0].pcolormesh(bins[:-1], freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
axs[1].plot(bins[:-1], psth)
# fig.colorbar(pcm, ax=axs[0])
# change y ticks to show frequency in Hz
log_ticks = [0.1, 1, 10, 100]
axs[0].yaxis.set_major_locator(FixedLocator(log_ticks))

# Set the labels to match the log ticks
axs[0].set_yticklabels(log_ticks)
# plot red horizontal line at 1, 10 and 100 Hz
for freq in log_ticks:
    axs[0].axhline(freq, color="red", linestyle="--", linewidth=0.5)
axs[0].plot(x_new, chirp_bined, c="k")
fig.show()
# %% find peaks
peaks = peak_local_max(cwtmatr, min_distance=4, threshold_rel=0.4)
# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
pcm = axs[0].pcolormesh(bins[:-1], freqs, cwtmatr)
# add peaks
axs[0].scatter(bins[peaks[:, 1]], freqs[peaks[:, 0]], c="r", s=10)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
axs[1].plot(bins[:-1], psth)
fig.show()
# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(x=chirp_bined[peaks[:, 1]], y=freqs[peaks[:, 0]][::-1], c="k")
ax.set_xlabel("chirp_freq")
ax.set_ylabel("peak_freq")
# plot linear increase
ax.plot(np.linspace(1, 3, 10), np.linspace(1, 3, 10), c="r")

fig.show()
# %% Linear regression
lr = LinearRegression()
lr.fit(np.linspace(1, 3, 10).reshape(-1, 1), np.linspace(1, 3, 10))

# %% SGDClassifier
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)
clf.fit(chirp_bined[peaks[:, 1]], freqs[peaks[:, 0]][::-1])

# %%
data_for_pca = np.stack([chirp_bined[peaks[:, 1]], freqs[peaks[:, 0]][::-1]]).T
# %%
pca = PCA(n_components=2)
pca.fit(data_for_pca)
pca_transform = pca.transform(data_for_pca)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(pca_transform[:, 0])
fig.show()
# %%
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(data_for_pca)
labels = gmm.predict(data_for_pca)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(x=chirp_bined[peaks[:, 1]], y=freqs[peaks[:, 0]][::-1], c=labels)
ax.set_xlabel("chirp_freq")
ax.set_ylabel("peak_freq")
# plot linear increase
ax.plot(np.linspace(1, 3, 10), np.linspace(1, 3, 10), c="r")

fig.show()
# %%
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
pcm = axs[0].pcolormesh(bins[:-1], freqs, cwtmatr)
# add peaks
axs[0].scatter(bins[peaks[:, 1]], freqs[peaks[:, 0]], c=labels, s=10)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
axs[1].plot(bins[:-1], psth)
fig.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(peaks[:, 1], np.log(peaks[:, 0]))
fig.show()

# %%
data_combined = np.stack([peaks[:, 1], np.log(peaks[:, 0])[::-1]]).T
from sklearn.preprocessing import StandardScaler

data_combined_scaled = StandardScaler().fit_transform(data_combined)
# %%
bics = []
for i in range(1, 10):
    gmm = GaussianMixture(n_components=i, covariance_type="tied")
    gmm.fit(data_combined_scaled)
    bics.append(gmm.bic(data_combined_scaled))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(range(1, 10), bics)
fig.show()
gmm = GaussianMixture(n_components=np.argmin(bics) + 1, covariance_type="tied")
labels = gmm.fit_predict(data_combined_scaled)

# %%
from sklearn.cluster import HDBSCAN, AffinityPropagation
from sklearn.mixture import BayesianGaussianMixture

# %%
clustering = BayesianGaussianMixture(n_components=2, covariance_type="tied")
labels = clustering.fit_predict(data_combined_scaled)
# %%
clustering = HDBSCAN(metric="euclidean", min_cluster_size=10)
labels = clustering.fit_predict(data_combined_scaled)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(peaks[:, 1], np.log(peaks[:, 0]), c=labels)
fig.show()
# %%
from sklearn.linear_model import RANSACRegressor

cluster_sizes = np.unique(labels, return_counts=True)
# sort by cluster size
cluster_sizes = cluster_sizes[0][np.argsort(cluster_sizes[1])]
cluster_sizes = cluster_sizes[cluster_sizes != -1][::-1]

X_new = data_combined[np.where(labels == cluster_sizes[0])[0]]
# Fit first line
ransac1 = RANSACRegressor()
ransac1.fit(X_new[:, 0].reshape(-1, 1), X_new[:, 1])
inlier_mask1 = ransac1.inlier_mask_
cluster_1 = X_new[inlier_mask1]
cluster_2 = X_new[~inlier_mask1]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(cluster_1[:, 0], cluster_1[:, 1])
ax.scatter(cluster_2[:, 0], cluster_2[:, 1])
ax.plot(
    np.linspace(0, np.max(data_combined[:, 0]), 10),
    ransac1.predict(np.linspace(0, np.max(data_combined[:, 0]), 10).reshape(-1, 1)),
)
fig.show()
# %%
# Optionally, fit second line to cluster_2
X_new = data_combined[np.where(labels == 1)[0]]
ransac2 = RANSACRegressor()
ransac2.fit(X_new[:, 0].reshape(-1, 1), X_new[:, 1])
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(data_combined[:, 0], data_combined[:, 1], c=labels)
ax.plot(
    np.linspace(0, np.max(data_combined[:, 0]), 10),
    ransac1.predict(np.linspace(0, np.max(data_combined[:, 0]), 10).reshape(-1, 1)),
)
ax.plot(
    np.linspace(0, np.max(data_combined[:, 0]), 10),
    ransac2.predict(np.linspace(0, np.max(data_combined[:, 0]), 10).reshape(-1, 1)),
)
fig.show()
# %% Get gradient
grad1 = 10 ** ransac1.estimator_.coef_[0]
grad2 = 10 ** ransac2.estimator_.coef_[0]
print(grad1, grad2)
# %% get intercept
intercept1 = ransac1.estimator_.intercept_
intercept2 = ransac2.estimator_.intercept_
print(intercept1, intercept2)
# %% write out function
print(f"y = {grad1}x + {intercept1}")

# %% plot function into original plot
# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
pcm = axs[0].pcolormesh(bins[:-1], freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
axs[1].plot(bins[:-1], psth)
# fig.colorbar(pcm, ax=axs[0])
# change y ticks to show frequency in Hz
log_ticks = [0.1, 1, 10, 100]
axs[0].yaxis.set_major_locator(FixedLocator(log_ticks))

# Set the labels to match the log ticks
axs[0].set_yticklabels(log_ticks)
# plot red horizontal line at 1, 10 and 100 Hz
for freq in log_ticks:
    axs[0].axhline(freq, color="red", linestyle="--", linewidth=0.5)
axs[0].plot(x_new, chirp_bined, c="k")
# plot chirp function
x = bins[:-1]
y = np.log(grad1 * x + 1)
axs[0].plot(x, y, c="r")
fig.show()


# %%
def normalize_0_1(data):
    return (data - data.min()) / (data.max() - data.min())


# %%
start_freq = 1  # Start frequency in Hz
end_freq = 30  # End frequency in Hz
duration = 30  # Duration in seconds
refresh_rate = 100  # Refresh rate in Hz
num_cycles = duration * refresh_rate

chirp_real = []
time = []
freq = []
for i in range(num_cycles):
    t = i / refresh_rate
    f = np.power(end_freq / start_freq, t / duration) * start_freq
    value = np.sin(2 * np.pi * f * t)
    chirp_real.append(int((value + 1) * 4095 / 2))
    time.append(t)
    freq.append(f)

chirp_real = np.asarray(chirp_real)
time = np.asarray(time)
chirp_real = normalize_0_1(chirp_real)
chirp_complete = np.zeros(3499)
chirp_complete[300:3300] = chirp_real
freqs = np.zeros_like(chirp_complete)
freqs[300:3300] = np.asarray(freq)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.plot(np.linspace(0, 30, 3000), chirp_signal, c="k")
ax.plot(time, chirp_real, c="k")
ax.plot(bins[:-1], psths[3], c="r")
fig.show()
# %%
idx = 5
cell_spikes = spikes.query(
    f"recording == 'chicken_30_08_2024_p0' & cell_index == 239 & stimulus_name == 'chirp610'"
)
# %%
psth, bins = histograms.psth_by_index(
    cell_spikes, index=["repeat"], bin_size=0.01, window_end=35
)
# psth = psth.astype(bool)
# %% plot
fig, axs = plt.subplots(5, 1, figsize=(10, 10))
for idx, trace in enumerate(psth):
    axs[idx].plot(bins[:-1], trace)
axs[-1].plot(bins[:-1], chirp_complete)
fig.show()
# %%
from statsmodels.tsa.stattools import grangercausalitytests


# %%
def normalize_0_1(data):
    return (data - data.min()) / (data.max() - data.min())


# %%

data = np.stack([normalize_0_1(psth[0][:1200]), normalize_0_1(chirp_complete[:1200])]).T
# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=bins[np.where(data[:, 0])[0]],
        y=np.ones(np.where(data[:, 0])[0].shape[0]),
        mode="markers",
        marker=dict(color="red"),
    )
)
fig.add_trace(
    go.Scatter(
        x=bins[:-1],
        y=chirp_complete,
        mode="lines",
        line=dict(color="black"),
    )
)
fig.add_trace(go.Scatter(x=bins[:-1], y=psth[0], mode="lines", line=dict(color="red")))

fig.show(renderer="browser")
# %%
t_r = grangercausalitytests(data, maxlag=4)
# %5
from statsmodels.tsa.stattools import adfuller

test = adfuller(data[:, 0])

# %%
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=bins[:-1],
        y=data[:, 1],
        mode="lines",
        line=dict(color="black"),
    )
)
fig.add_trace(
    go.Scatter(
        x=bins[1:-1], y=normalize_0_1(psth[0][1:]), mode="lines", line=dict(color="red")
    )
)

fig.show(renderer="browser")
# %%
from scipy.signal import correlate

# Perform CWT using PyWavelets
wavelet = "cmor3.0-1.5"
scales = np.arange(1, 128)
coeffs1, _ = pywt.cwt(chirp_complete, scales, wavelet)
coeffs2, _ = pywt.cwt(psth[0], scales, wavelet)

# Cross-correlate wavelet coefficients at each scale
cross_corr = []
for i in range(len(scales)):
    corr = correlate(coeffs1[i, :], coeffs2[i, :], mode="full")
    cross_corr.append(corr)

# %%
# perform CWT
wavelet = "cmor3.0-1.5"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
sampling_period = 0.01
cwtmatr, freqs = pywt.cwt(
    chirp_complete - chirp_complete.mean(),
    widths,
    wavelet,
    sampling_period=sampling_period,
)
# map chirp to have same length as cwtmatr by extrapolating
func = interp1d(np.linspace(0, 35, freq.shape[0]), freq, kind="linear")
x_new = np.linspace(0, 35, cwtmatr.shape[1])
chirp_bined = func(x_new)

# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

# cut to relevant frequencies 1Hz - 120Hz
lower_idx = np.where(freqs > 0.9)[0][-1]
upper_idx = np.where(freqs > 120)[0][0]

cwtmatr = cwtmatr[upper_idx : lower_idx - 1, :]
freqs = freqs[upper_idx:lower_idx]


# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
pcm = axs[0].pcolormesh(bins[:-1], freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
axs[1].plot(bins[:-1], psth[0])
# fig.colorbar(pcm, ax=axs[0])
# change y ticks to show frequency in Hz
log_ticks = [0.1, 1, 10, 100]
axs[0].yaxis.set_major_locator(FixedLocator(log_ticks))

# Set the labels to match the log ticks
axs[0].set_yticklabels(log_ticks)
# plot red horizontal line at 1, 10 and 100 Hz
for freq in log_ticks:
    axs[0].axhline(freq, color="red", linestyle="--", linewidth=0.5)
axs[0].plot(x_new, chirp_bined, c="k")
fig.show()
