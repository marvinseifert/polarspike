from polarspike import (
    Overview,
    colour_template,
    spiketrain_plots,
    Opsins,
    spiketrains,
    histograms,
)
from group_pca import GroupPCA
import panel as pn

pn.extension("tabulator")
from bokeh.io import show
from importlib import reload
import numpy as np
import polars as pl
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
from pathlib import Path
import pickle
from sklearn.cluster import AffinityPropagation

# Import normalizer
from sklearn.preprocessing import StandardScaler

# %%
# recordings = Overview.Recording_s.load_from_single(
#     r"D:/combined_analysis",
#     "chicken_combined",
#     r"A:\Marvin\chicken_30_08_2024\Phase_00\overview",
# )
recordings = Overview.Recording_s.load(r"D:\fff_clustering\clustering")
nr_cells = {}
for rec in recordings.recordings.keys():
    nr_cells[rec] = recordings.recordings[rec].nr_cells
#
# recordings.add_from_saved(r"A:\Marvin\zebrafish_05_07_2024\Phase_00\overview")
# recordings.add_from_saved(r"B:\Marvin\Zebrafish_04_23_2024\Phase_01\overview")
# recordings.add_from_saved(r"B:\Marvin\Zebrafish_04_22_2024\Phase_01\overview")
# recordings.add_from_saved(r"B:\Marvin\Zebrafish_19_04_2024\Phase_01\overview")
# recordings.add_from_saved(r"A:\Marvin\zebrafish_08_07_2024\Phase_00\overview")

# recordings = Overview.Recording_s.load(r"D:\zebrafish_combined\recordings_combined")
# %%
stimuli = ["fff", "chirp", "csteps"]
episodes = [6, 1, 10]
# stim_end = [24, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
stim_end = [24, 35, 40]
bin_size = 0.05
nr_bins = [int(end / bin_size) for end in stim_end]
store = []
store_scaled = []
cell_dfs = []
for stim_idx, stimulus in enumerate(stimuli):
    recordings.dataframes[stimulus] = recordings.spikes_df.query(
        "stimulus_name==@stimulus"
    )
    # recordings.dataframes[stimulus] = recordings.dataframes[stimulus].query("qi>0.2")
    cell_dfs.append(recordings.dataframes[stimulus][["recording", "cell_index"]])

cell_dfs = pd.concat(cell_dfs)
cell_dfs = cell_dfs.set_index(["recording", "cell_index"])
unique_indices = cell_dfs.index.unique()

# %%
store_df = pd.DataFrame(index=unique_indices)
for stimulus, s_end in zip(stimuli, stim_end):
    store_df[stimulus] = np.split(
        np.zeros((len(unique_indices), int(s_end / bin_size))),
        len(unique_indices),
        axis=0,
    )
# %%
for stimulus_name, s_end in zip(stimuli, stim_end):
    spikes = recordings.get_spikes_df(stimulus_name, pandas=False)

    psths, bins, cells = histograms.psth_by_index(
        spikes,
        index=["recording", "cell_index"],
        bin_size=bin_size,
        return_idx=True,
        window_end=s_end + bin_size,
    )
    new_index = pd.MultiIndex.from_arrays(cells.T, names=["recording", "cell_index"])
    new_df = pd.DataFrame(index=new_index)
    new_df[stimulus_name] = np.split(psths, psths.shape[0], axis=0)
    store_df.update(new_df)


# %% fill all zeros with medians

for stimulus in stimuli:
    median_values = np.mean(np.vstack(store_df[stimulus]), axis=0)
    store_df[stimulus] = store_df[stimulus].apply(
        lambda x: np.atleast_2d(median_values) if np.all(x == 0) else x
    )

# %%
channels = ["630", "560"]
rf_keys = ["center_outline", "in_out_outline", "sta_single", "surround_outline"]
all_columns = [f"{key}_{channel}" for key in rf_keys for channel in channels]
# %%
for rec in recordings.recordings.keys():
    rec_list = []
    with open(
        Path(recordings.recordings[rec].load_path).parent / "rf_dict.pkl", "rb"
    ) as f:
        rf_dict = pickle.load(f)
        rf_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                ((rec, cell) for cell in range(0, nr_cells[rec])),
                names=["recording", "cell_index"],
            ),
        )
        for channel in channels:
            for key in rf_keys:
                rf_df[f"{key}_{channel}"] = np.vsplit(
                    rf_dict[channel][key], len(rf_dict[channel][key])
                )


# %%
store_scaled = []
store = []
scaler = StandardScaler(with_mean=False)

for idx, stimulus in enumerate(stimuli):
    cut_length = int(nr_bins[idx] / episodes[idx])
    for ep in range(episodes[idx]):
        store_scaled.append(
            scaler.fit_transform(
                np.vstack(store_df[stimulus])[
                    :, cut_length * ep : cut_length * ep + cut_length
                ]
            )
        )
    store.append(np.vstack(store_df[stimulus]))

# %%
cells = np.vstack(store_df.index)
# %%
for column in all_columns:
    store_scaled.append(scaler.fit_transform(np.vstack(rf_df[column].values)))

# %%
for column in all_columns:
    store.append(np.vstack(rf_df[column].values))
# %%
psths = np.hstack(store)

# %%
group_pca = GroupPCA(n_components=20)
reduced_data = group_pca.fit_transform(store_scaled)


# %% Normal PCA
pca = PCA(n_components=20)
reduced_data = pca.fit_transform(np.hstack(store_scaled))


# %% Direction selective
ds_vals = np.zeros((ds_df["cell_index"].max() + 1, 2))
ds_vals[ds_df["cell_index"], 0] = ds_df["mean_deg"]
ds_vals[ds_df["cell_index"], 1] = ds_df["z_val"]
# %%
ds_vals = ds_vals[rf_cells]

# %%
reduced_data = np.hstack((reduced_data, ds_vals))
# %%
# Plot the first two pcs
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c="black")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()

# %% Plot the first two pcs coloured by the recording
fig, ax = plt.subplots(figsize=(10, 10))
for recording in np.unique(cells[:, 0]):
    ax.scatter(
        reduced_data[cells[:, 0] == recording, 0],
        reduced_data[cells[:, 0] == recording, 1],
        label=recording,
    )
ax.legend()
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()

# %%
bic = []
for i in range(1, 30):
    gmm = GaussianMixture(n_components=i)
    gmm.fit(reduced_data[:, :10])
    bic.append(gmm.bic(reduced_data[:, :10]))

# Plot the BIC
fig, ax = plt.subplots()
ax.plot(range(1, 30), bic, c="black")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("BIC")
fig.show()

# %%
n_clusters = np.argmin(bic) + 1
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(reduced_data[:, :10])
labels = gmm.predict(reduced_data[:, :10])
# %%
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(n_clusters):
    ax.scatter(
        reduced_data[labels == i, 0], reduced_data[labels == i, 1], label=f"Cluster {i}"
    )
ax.legend()
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()


# %%
clustering = AffinityPropagation(random_state=5, max_iter=1000).fit(psths)
labels = clustering.labels_
print(np.max(labels) + 1)
n_clusters = np.max(labels) + 1
# %%
rfs = np.load(r"A:\Marvin\chicken_19_07_2024\Phase_00\rfs.npy")

# %% Plot the traces in subplots. The first column is the original PSTH, the second column is RF. The second column
# should be much smaller (10 %) than the first column
# Share the x-axis
# CT = colour_template.Colour_template()
# CT.pick_stimulus("FFF_6_MC")
fig, ax = plt.subplots(
    n_clusters + 1,
    2,
    figsize=(48, 27),
    gridspec_kw={"width_ratios": [10, 1]},
)
for i in range(n_clusters):
    trace = np.mean(psths[labels == i].T, axis=1) / np.max(
        np.mean(psths[labels == i].T, axis=1)
    )
    ax[i, 0].plot(trace, color="black")
    # Add the number of cells in the cluster
    ax[i, 0].text(
        0.1,
        0.9,
        f"Cluster={i}; n={np.sum(labels == i)}",
        transform=ax[i, 0].transAxes,
    )
    ax[i, 0].set_ylim(0, 1.2)
    # Plot the RF
    # rf_mean = np.mean(rfs[labels == i], axis=0)
    # ax[i, 1].imshow(rf_mean, cmap="seismic")
ax[4, 0].set_xlabel("Bins")
ax[4, 0].set_ylabel("Mean firing rate")
# Remove top y box and right x box
for i in range(n_clusters):
    ax[i, 0].spines["top"].set_visible(False)
    ax[i, 0].spines["right"].set_visible(False)
# fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %% Plot all traces from a single cluster
# Create two subplots, one for the PSTH and one for the stimulus trace. The lower subplot shall be much smaller
cluster = 0
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    gridspec_kw={"height_ratios": [10, 1]},
    sharex=True,
)

for trace in psths[labels == cluster]:
    ax[0].plot(trace, alpha=0.1)
ax[0].plot(np.mean(psths[labels == cluster], axis=0), color="black")
ax[1].set_xlabel("Bins")
ax[1].set_ylabel("Mean firing rate")
# fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %%
# plot the percentage of single recordings contributing to the clusters
cluster_rec_counts = np.zeros((n_clusters, len(np.unique(cells[:, 0]))))
rec_arr = np.unique(cells[:, 0])
for cluster in range(n_clusters):
    recs, counts = np.unique(cells[labels == cluster, 0], return_counts=True)
    for rec, count in zip(recs, counts):
        cluster_rec_counts[cluster, np.where(rec_arr == rec)[0][0]] = count

# %% PLot the percentage of single recordings contributing to the clusters
fig, ax = plt.subplots(figsize=(20, 10))
for cluster in range(n_clusters):
    ax.bar(rec_arr, cluster_rec_counts[cluster, :], label=f"Cluster {cluster}")
ax.legend()
fig.show()
