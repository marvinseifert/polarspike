from polarspike import (
    Overview,
    spiketrain_plots,
    moving_bars,
    quality_tests,
    stimulus_spikes,
    binarizer,
    colour_template,
    histograms,
)
from bokeh.io import show
import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, HDBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from tslearn.metrics import cdist_dtw
from cdb_clustering.clustering import CompressionBasedDissimilarity as CBD

CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%
recordings = Overview.Recording_s.load_from_single(
    r"D:\fff_clustering",
    "chicken_fff",
    "" r"A:\Marvin\chicken_30_08_2024\Phase_00\overview",
)
recordings.add_from_saved(r"A:\Marvin\chicken_04_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_05_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_09_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_17_07_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_18_07_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_19_07_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_11_09_2024\Phase_00\overview")
# %%

recordings.dataframes["fff"] = recordings.spikes_df.query("stimulus_name == 'fff'")
# %%
spikes = recordings.get_spikes_df("fff")

# %%
qis = quality_tests.spiketrain_qi(spikes, max_window=24, max_repeat=10)
# %%
recordings.dataframes["fff"] = recordings.dataframes["fff"].set_index(
    ["cell_index", "recording"]
)
# %%
recordings.dataframes["fff"].update(qis)
recordings.dataframes["fff"] = recordings.dataframes["fff"].reset_index()
# %% plot qi as histogram
fig = px.histogram(recordings.dataframes["fff"], x="qi", nbins=500, color="recording")
fig.show(renderer="browser")

# %%
recordings.dataframes["fff_filtered"] = recordings.dataframes["fff"].query("qi>0.3")
# %%
spikes = recordings.get_spikes_df("fff_filtered")
# %%
psth, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=24,
    bin_size=0.1,
)

# %%
psth_sorted = psth[np.argsort(cell_index[:, 0])]
sorted_recordings = cell_index[np.argsort(cell_index[:, 0]), 0]
nr_rec = np.unique(sorted_recordings, return_counts=True)


# %% Saving
np.save(r"D:\fff_clustering\psth_sorted.npy", psth_sorted)
np.save(r"D:\fff_clustering\sorted_recordings.npy", sorted_recordings)

# %% Clustering
scaler = StandardScaler()
psth_scaled = scaler.fit_transform(psth)

# %%
distance_matrix_euc = pairwise_distances(psth, metric="euclidean", n_jobs=-1)

# %%
distance_matrix = cdist_dtw(psth, n_jobs=-1)

# %%
distance_matrix_scaled = cdist_dtw(psth_scaled, n_jobs=-1)
# %% Plot distance matrix
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(distance_matrix_euc)
# Add labels for unique recordings
for i, n in enumerate(nr_rec[1]):
    ax.axvline(np.sum(nr_rec[1][:i]), color="r", linewidth=1)
    ax.axhline(np.sum(nr_rec[1][:i]), color="k", linewidth=1)
fig.colorbar(cax)
fig.show()
# %% PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(psth_scaled)
pca_transform = pca.transform(psth_scaled)
fig, ax = plt.subplots()
ax.scatter(pca_transform[:, 0], pca_transform[:, 1])
fig.show()
# %%
s_scores = []
for c_n in range(2, 20):
    clustering = AffinityPropagation(random_state=5, max_iter=1000).fit(psth)
    labels = clustering.labels_
    print(np.max(labels) + 1)

# %%
s_scores = []
for c_n in range(2, 20):
    clustering = AgglomerativeClustering(
        n_clusters=c_n,  # Adjust the number of clusters as needed
        metric="precomputed",
        linkage="average",
        compute_full_tree=True,  # Needed for dendrogram
    )
    labels = clustering.fit_predict(distance_matrix_scaled)
    s_scores.append(silhouette_score(psth_scaled, labels))

# %%
clustering = AgglomerativeClustering(
    n_clusters=20,  # Adjust the number of clusters as needed
    metric="precomputed",
    linkage="average",
    compute_full_tree=True,  # Needed for dendrogram
)
labels = clustering.fit_predict(distance_matrix_scaled)


# %%
clustering = CBD(psth_scaled, bins=np.linspace(0, np.max(psth_scaled), 100))
clustering.add_noise()
clustering.pair_wise_distances()
fig, ax = clustering.dendrogram()
# %%
fig.show()
# %%
labels = clustering.get_clusters(cut=1)
labels = labels - 1
print(np.max(labels) + 1)
# %%
clustering = HDBSCAN(metric="precomputed", min_cluster_size=3)
labels = clustering.fit_predict(distance_matrix_euc)
print(np.max(labels) + 1)
# %% Plot silhouette scores
fig, ax = plt.subplots()
ax.plot(range(2, 20), s_scores)
fig.show()
# %%
sdtw_km = TimeSeriesKMeans(
    n_clusters=15,
    metric="dtw",
    # metric_params={"gamma": 0.01},
    verbose=True,
    random_state=5,
    n_jobs=-1,
)
y_pred = sdtw_km.fit_predict(psth)
# %%
labels = sdtw_km.labels_
print(np.max(labels) + 1)

# %% Calculate silhouette score
silhouette_avg = silhouette_score(psth, labels)
print(silhouette_avg)
# %%
fig, ax = plt.subplots(nrows=np.max(labels) + 2, ncols=1, figsize=(10, 20), sharex=True)
for cluster in range(np.max(labels) + 1):
    mean_psth = np.mean(psth[labels == cluster], axis=0)
    ax[cluster].plot(bins[:-1], mean_psth, c="k")
    ax[cluster].set_title(
        f"Cluster {cluster}, n={np.sum(labels == cluster)}", position=(1, 0.8)
    )
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()


# %% Plot all cells from one cluster
cluster = 23
cells = cell_index[labels == cluster, 1]
random_pick = np.random.choice(cells, 10, replace=False)
traces = psth[random_pick.astype(int)]
fig, axs = plt.subplots(nrows=10, ncols=1, figsize=(10, 10))
for idx, trace in enumerate(traces):
    axs[idx].plot(bins[:-1], trace, c="k", alpha=1)
fig.show()
# %%
cluster_df = pd.DataFrame(
    {"cluster": labels, "recording": cell_index[:, 0], "cell_index": cell_index[:, 1]}
)
cluster_df = cluster_df.set_index(["recording", "cell_index"])

# %%
recordings.dataframes["fff_filtered"]["cluster"] = 0
recordings.dataframes["fff_filtered"] = recordings.dataframes["fff_filtered"].set_index(
    ["recording", "cell_index"]
)
recordings.dataframes["fff_filtered"].update(cluster_df)
recordings.dataframes["fff_filtered"] = recordings.dataframes[
    "fff_filtered"
].reset_index()

# %%
recordings.save(r"D:\fff_clustering\clustering")
