from polarspike import (
    Overview,
    colour_template,
    spiketrain_plots,
    Opsins,
    spiketrains,
    histograms,
)
from polarspike.clustering import CompressionBasedDissimilarity as CBD
from polarspike import clustering
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

# Import normalizer
from sklearn.preprocessing import StandardScaler


recording = Overview.Recording.load(r"B:\Marvin\Zebrafish_04_23_2024\Phase_01\overview")
interesting_stimuli = [11]
nr_cells = recording.nr_cells
good_cells = recording.dataframes["good_cells_df"]["cell_index"].unique()
good_cells = np.load(r"C:\Users\Marvin\github_packages\sta_covariance\good_cells.npy")
# good_cells = np.concatenate([good_cells, np.setdiff1d(good_cells_cr, good_cells)])
scaler = StandardScaler()


# %% PSTH
spikes = recording.get_spikes_triggered(
    [[1]],
    [["all"]],
    cell_df="good_cells_df",
    pandas=False,
)

psths, bins, cell_index = histograms.psth_by_index(
    spikes,
    bin_size=0.05,
    to_bin="times_triggered",
    return_idx=True,
)
new_psth = np.zeros((nr_cells, psths.shape[1]))
new_psth[cell_index.astype(int), :] = psths

# %%
psths = new_psth[good_cells, :]
# %%
episodes = 10
store = np.empty(episodes, dtype=object)
store_scaled = np.empty(episodes, dtype=object)
cut_length = psths.shape[1] // episodes
for i in range(episodes):
    # realign the psths
    store[i] = new_psth[good_cells, cut_length * i : cut_length * i + cut_length]
    store_scaled[i] = scaler.fit_transform(
        new_psth[good_cells, cut_length * i : cut_length * i + cut_length]
    )

# %%
group_pca = GroupPCA(n_components=10)
reduced_data = group_pca.fit_transform(store_scaled)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c="black")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()

# %%
bic = []
for i in range(1, 20):
    gmm = GaussianMixture(n_components=i)
    gmm.fit(reduced_data[:, :10])
    bic.append(gmm.bic(reduced_data[:, :10]))

# %%
# Plot the BIC
fig, ax = plt.subplots()
ax.plot(range(1, 20), bic, c="black")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("BIC")
fig.show()

# %%
n_clusters = np.argmin(bic) + 1
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(reduced_data[:, :10])
labels = gmm.predict(reduced_data[:, :10])

# %% Plot the traces in subplots
fig, ax = plt.subplots(n_clusters, 1, figsize=(20, 10))
for i in range(n_clusters):
    ax[i].plot(np.mean(psths[labels == i].T, axis=1), c="black")
    # Add the number of cells in the cluster
    ax[i].text(
        0.1,
        0.9,
        f"Cluster={i}; n={np.sum(labels == i)}",
        transform=ax[i].transAxes,
    )
ax[4].set_xlabel("Bins")
ax[4].set_ylabel("Mean firing rate")
# Remove top y box and right x box
for i in range(n_clusters):
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)
fig.show()
