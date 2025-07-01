from polarspike import (
    Overview,
    histograms,
)
import panel as pn

pn.extension("tabulator")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import normalizer
from sklearn.preprocessing import StandardScaler

# %%
# recordings = Overview.Recording_s.load_from_single(
#     r"D:/combined_analysis",
#     "zebrafish_combined",
#     r"B:\Evelyn\Zebrafish_30_04_2024\Phase_00\overview",
# )
#
# recordings.add_from_saved(r"A:\Marvin\zebrafish_05_07_2024\Phase_00\overview")
# recordings.add_from_saved(r"B:\Marvin\Zebrafish_04_23_2024\Phase_01\overview")
# recordings.add_from_saved(r"B:\Marvin\Zebrafish_04_22_2024\Phase_01\overview")
# recordings.add_from_saved(r"B:\Marvin\Zebrafish_19_04_2024\Phase_01\overview")
# recordings.add_from_saved(r"A:\Marvin\zebrafish_08_07_2024\Phase_00\overview")

recordings = Overview.Recording_s.load(r"D:\zebrafish_combined\recordings_combined")
# %%
stimuli = ["fff", "chirp"]
episodes = [6, 10]
stim_end = [24, 35]
bin_size = 0.05
nr_bins = [int(end / bin_size) for end in stim_end]
store = []
store_scaled = []
cell_dfs = []
for stim_idx, stimulus in enumerate(stimuli):
    recordings.dataframes[stimulus] = recordings.spikes_df.query(
        "stimulus_name==@stimulus"
    )
    recordings.dataframes[stimulus] = recordings.dataframes[stimulus].query("qi>0.2")
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
store_scaled = []
store = []
scaler = StandardScaler()

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
psths = np.hstack(store)

# %%
from cdb_clustering import clustering

# %%
Cl = clustering.CompressionBasedDissimilarity(psths, np.arange(0, 189, 30), 22)
# Cl = Cl.normalize_01("all")
Cl.add_noise()

# %%
Cl.pair_wise_distances()

# %%
fig, ax = Cl.dendrogram(color_threshold=1.5)
fig.show()

# %%
labels = Cl.get_clusters(1.5)
labels = labels - 1
n_clusters = np.unique(labels).shape[0]
# %% Plot the traces in subplots. The first column is the original PSTH, the second column is RF. The second column
# should be much smaller (10 %) than the first column
# Share the x-axis
# CT = colour_template.Colour_template()
# CT.pick_stimulus("FFF_6_MC")
fig, ax = plt.subplots(
    n_clusters + 1,
    2,
    figsize=(40, 10),
    gridspec_kw={"width_ratios": [10, 1]},
    sharex=True,
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
    # ax[i, 1].imshow(
    #     np.load(rf"D:\combined_analysis\clusters\cluster_{i}.npy"), cmap="seismic"
    # )
ax[-1, 0].set_xlabel("Bins")
ax[-1, 0].set_ylabel("Mean firing rate")
# Remove top y box and right x box
for i in range(n_clusters):
    ax[i, 0].spines["top"].set_visible(False)
    ax[i, 0].spines["right"].set_visible(False)
# fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %% Plot all traces from a single cluster
# Create two subplots, one for the PSTH and one for the stimulus trace. The lower subplot shall be much smaller
cluster = 6
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    gridspec_kw={"height_ratios": [10, 1]},
    sharex=True,
)

for trace in psths[labels == cluster]:
    ax[0].plot(bins[:-1], trace, alpha=0.1)
ax[0].plot(bins[:-1], np.mean(psths[labels == cluster], axis=0), color="black")
ax[1].set_xlabel("Bins")
ax[1].set_ylabel("Mean firing rate")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
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
fig, ax = plt.subplots(figsize=(10, 10))
for cluster in range(n_clusters):
    ax.bar(rec_arr, cluster_rec_counts[cluster, :], label=f"Cluster {cluster}")
ax.legend()
fig.show()
