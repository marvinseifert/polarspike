"""
This script is used to calculate the tuning curves for responses. Two different methods are used:
1. Summing the responses over a window of time.
2. Finding the peaks in the response.

The script will calculate the correlation coefficient between the tuning curves and a diagonal line which indicates
the ideal linear response.

Differences between the methods:
- The first method is more sensitive to the overall response of the cell. Particularly, if cells have sustained responses
this method might more accurately reflect the response.
- The second method is more sensitive to the peaks in the response. Particularly, if cells have transient responses
this method might more accurately reflect the response.

"""
import pandas as pd
import spiketrain_plots
from polarspike import (
    Overview,
    histograms,
    Opsins,
    colour_template,
    stimulus_spikes,
    spiketrain_plots,
)
from polarspike.analysis import response_peaks, count_spikes
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import holoviews as hv
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import AffinityPropagation, KMeans

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
OT = Opsins.Opsin_template()

double_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_double")
single_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_oil")
# %%
# Input parameters
recordings = Overview.Recording_s.load_from_single(
    r"D:/combined_analysis",
    "zebrafish_combined",
    r"B:\Evelyn\Zebrafish_30_04_2024\Phase_00\overview",
)

recordings.add_from_saved(r"A:\Marvin\zebrafish_05_07_2024\Phase_00\overview")
recordings.add_from_saved(r"B:\Marvin\Zebrafish_04_23_2024\Phase_01\overview")
recordings.add_from_saved(r"B:\Marvin\Zebrafish_04_22_2024\Phase_01\overview")
recordings.add_from_saved(r"B:\Marvin\Zebrafish_19_04_2024\Phase_01\overview")
recordings.add_from_saved(r"A:\Marvin\zebrafish_08_07_2024\Phase_00\overview")

# %%
recordings = Overview.Recording_s.load_from_single(
    r"D:/combined_analysis",
    "chicken_combined",
    r"B:\Marvin\chicken_02_13_2024\Phase_00\overview",
)
recordings.add_from_saved(r"B:\Marvin\chicken_02_13_2024_2nd\Phase_01\overview")
recordings.add_from_saved(r"B:\Marvin\chicken_02_12_2024\Phase_00\overview")

# %%
recordings = Overview.Recording_s.load(r"D:\fff_clustering\records")
recordings.add_from_saved(r"A:\Marvin\chicken_11_09_2024\Phase_00\overview_top")
# %%

window = (
    0.5  # This is the window size in seconds over which the response will be summed
)
stimulus_name = "fff_top"
use_opsin = True  # If contrast steps = False, if chromatic stimuli = True
colour_name = "FFF_6_MC"  # Which colour template to use
animal_name = ["Chicken"]
bin_size = 0.05  # Binsize for findpeaks method

# %%
# Get the stimulus id
fff_stimulus = recordings.stimulus_df.query("stimulus_name == @stimulus_name")
recordings.dataframes["fff_stimulus"] = fff_stimulus
# %%
mean_trigger = stimulus_spikes.mean_trigger_times(fff_stimulus, ["all"])

# %%
recordings.dataframes[stimulus_name] = recordings.spikes_df.query(
    "stimulus_name==@stimulus_name"
)
recordings.dataframes[f"{stimulus_name}_filtered"] = recordings.dataframes[
    stimulus_name
].query("qi>0.3")
# %%
spikes = recordings.get_spikes_df(
    cell_df=f"{stimulus_name}_filtered", stimulus_df="fff_stimulus", pandas=False
)


# %%
# Create a list of possible windows
spikes_summed = count_spikes.sum_spikes(
    spikes, mean_trigger, window=window, group_by="stimulus_index"
)[0]

# %%
spikes_rec = spikes.partition_by("recording")

# %%
spikes_summed_all = []
for recording in spikes_rec:
    spikes_summed = count_spikes.sum_spikes(
        recording, mean_trigger, window=window, group_by="cell_index"
    )[0]
    spikes_summed_all.append(spikes_summed)


# %%
spikes_summed = np.concatenate(spikes_summed_all, axis=0)


# %%
# Normalize 0 - 1 along 1st axis
spikes_summed_norm = spikes_summed / np.max(spikes_summed, axis=1)[:, None]


# %% PLot everything on top of each other
fig, ax = plt.subplots(figsize=(20, 10))
for curve in spikes_summed_norm:
    ax.plot(CT.wavelengths[::2], curve[1::2], c="k", alpha=0.1)
plt.show()
# %%

ts_df = pd.DataFrame(spikes_summed_norm[:, ::2])
wavelengths = CT.wavelengths[::2]
points = 6
time = np.asarray(wavelengths)
# normalize wavelengths 01
wave_norm = (wavelengths - np.min(wavelengths)) / (
    np.max(wavelengths) - np.min(wavelengths)
)
width = 400

cvs = ds.Canvas(plot_height=400, plot_width=width)
agg = cvs.line(
    ts_df,
    x=time,
    y=list(range(points)),
    agg=ds.count(),
    axis=1,
    antialias=True,
)
img = tf.shade(agg, how="log")
img_pil = img.to_pil()

# Display with Matplotlib
fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(img_pil)
ax.set_xticks(wave_norm * width)
ax.set_xticklabels(wavelengths)
# set title
plt.title("Tuning curves ON")
fig.show()


# %%
ts_df = pd.DataFrame(spikes_summed_norm[:, 1::2])
wavelengths = CT.wavelengths[::2]
points = 6
time = np.asarray(wavelengths)
# normalize wavelengths 01
wave_norm = (wavelengths - np.min(wavelengths)) / (
    np.max(wavelengths) - np.min(wavelengths)
)
width = 400

cvs = ds.Canvas(plot_height=400, plot_width=width)
agg = cvs.line(
    ts_df,
    x=time,
    y=list(range(points)),
    agg=ds.count(),
    axis=1,
    antialias=True,
)
img = tf.shade(agg, how="log")
img_pil = img.to_pil()

# Display with Matplotlib
fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(img_pil)
ax.set_xticks(wave_norm * width)
ax.set_xticklabels(wavelengths)
plt.title("Tuning curves OFF")
fig.show()
# %%
opts = hv.opts.RGB(width=600, height=300)
ndoverlay = hv.NdOverlay(
    {c: hv.Curve((dfm["Time"], dfm[c]), vdims=["Time"]) for c in cols}
)
datashade(ndoverlay, cnorm="linear", aggregator=ds.count(), line_width=3).opts(opts)

# %%
pca = PCA(n_components=6)
transformed = pca.fit_transform(spikes_summed_norm[:, ::2])
# %%
fig, ax = plt.subplots()
ax.plot(transformed[:, 0], transformed[:, 1], "o", alpha=0.2)
# add zero lines
ax.axhline(0, c="k", linestyle="--")
ax.axvline(0, c="k", linestyle="--")
fig.show()

# %%
ica = FastICA(n_components=6)
transformed = ica.fit_transform(spikes_summed[:, ::2])

fig, axs = plt.subplots(6, 1, figsize=(15, 10))
for i in range(6):
    axs[i].plot(wavelengths, ica.components_[i])
    axs[i].set_xticks([])
axs[i].set_xticks(wavelengths)
fig.show()

# %% Plot loadings of PCA
fig, axs = plt.subplots(6, 1, figsize=(15, 10))
for i in range(6):
    axs[i].plot(wavelengths, pca.components_[i])
    axs[i].set_xticks([])
axs[i].set_xticks(wavelengths)
fig.show()

# %%
clustering = AffinityPropagation(random_state=5, max_iter=1000).fit(
    spikes_summed_norm[:, 1::2]
)
labels = clustering.labels_
print(np.max(labels) + 1)

# %%
clustering = KMeans(n_clusters=2)
labels = clustering.fit_predict(spikes_summed_norm[:, ::2])


# %% plot the clusters
fig, axs = plt.subplots(nrows=np.max(labels) + 1, ncols=1, figsize=(5, 10))
for cluster in range(np.max(labels) + 1):
    mean = np.mean(spikes_summed_norm[labels == cluster, ::2], axis=0)
    mean = mean / np.max(mean)
    axs[cluster].plot(
        wavelengths, mean, "o-", label=f"Cluster {cluster}", alpha=0.5, c="k"
    )
    for cone in single_ab_df["cone"].unique():
        opsin_trace = single_ab_df.query("cone==@cone")["absorption"]
        opsin_wavelength = single_ab_df.query("cone==@cone")["wavelength"]
        axs[cluster].plot(
            opsin_wavelength, opsin_trace, c=OT.cone_colours[cone], label=cone
        )
    for cone in double_ab_df["cone"].unique():
        cones = []
        opsin_trace = double_ab_df.query("cone==@cone")["absorption"]
        cones.append(opsin_trace)
        opsin_wavelength = double_ab_df.query("cone==@cone")["wavelength"]
        axs[cluster].plot(opsin_wavelength, opsin_trace, "--", c="orange")

    axs[cluster].plot(
        opsin_wavelength,
        np.mean(np.stack(cones), axis=0) / np.nanmax(np.mean(np.stack(cones), axis=0)),
        c="orange",
    )
    axs[cluster].set_xticks([])
    axs[cluster].set_title(f"Cluster {cluster}, n={np.sum(labels == cluster)}")
axs[cluster].set_xticks(wavelengths)
fig.show()
# %% Findpeaks method

all_heights, peak_locations = response_peaks.find_peaks(
    spikes, mean_trigger, bin_size=bin_size
)

spikes_heights_norm = all_heights / np.max(all_heights)
# %% Plot the results
CT = colour_template.Colour_template()
CT.pick_stimulus(colour_name)
if use_opsin:
    Ops = Opsins.Opsin_template()
    fig = Ops.plot_overview(animal_name)
else:
    fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[::2],
        y=spikes_summed_norm[::2],
        mode="lines+markers",
        line=dict(color="grey"),
        name="Summed spikes ON",
    )
)
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[1::2],
        y=spikes_summed_norm[1::2],
        mode="lines+markers",
        line=dict(color="black"),
        name="Summed spikes OFF",
    )
)
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[::2],
        y=spikes_heights_norm[::2],
        mode="lines+markers",
        line=dict(color="grey", dash="dash"),
        name="Peaks ON",
    )
)
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[1::2],
        y=spikes_heights_norm[1::2],
        mode="lines+markers",
        line=dict(color="black", dash="dash"),
        name="Peaks OFF",
    )
)
# Add diagonal line
fig.add_trace(
    go.Scatter(
        x=[CT.wavelengths[-1], CT.wavelengths[0]],
        y=[0, 1],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Diagonal",
    )
)
fig.update_layout(height=700, width=700)
fig.show(renderer="browser")
# %%

fig, ax = spiketrain_plots.whole_stimulus(spikes, indices=["stimulus_index", "repeat"])
fig = CT.add_stimulus_to_plot(fig, mean_trigger)
fig.show()
# %% Calculate the euclidean distance between the two lines and the diagonal

diagonal = np.arange(0, 1, 0.1)[::-1]
correlation_coeff_summed_on = np.corrcoef(spikes_summed_norm[::2], diagonal)[1, 0]
correlation_coeff_summed_off = np.corrcoef(spikes_summed_norm[1::2], diagonal)[1, 0]
correlation_coeff_peaks_on = np.corrcoef(spikes_heights_norm[::2], diagonal)[1, 0]
correlation_coeff_peaks_off = np.corrcoef(spikes_heights_norm[1::2], diagonal)[1, 0]

print(
    f"Correlation coefficient summed ON: {correlation_coeff_summed_on}\n"
    f"Correlation coefficient summed OFF: {correlation_coeff_summed_off}\n"
    f"Correlation coefficient peaks ON: {correlation_coeff_peaks_on}\n"
    f"Correlation coefficient peaks OFF: {correlation_coeff_peaks_off}"
)
