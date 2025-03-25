import sys
sys.path.extend(['/home/marvin-seifert/PycharmProjects'])
from polarspike import Overview, spiketrain_plots, colour_template, histograms, chirps
import numpy as np
import pandas as pd
from pathlib import Path
from tslearn.metrics import dtw_limited_warping_length
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# %%
root = Path(r"A:\Marvin\fff_clustering_zf")
# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering_zf\records")
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%

spikes = recordings.get_spikes_df("fff_filtered", carry=["stimulus_name", "qi"])
# %%
psths, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=24.1,
    bin_size=0.1,
)
# %%
all_distances = np.zeros((psths.shape[0], psths.shape[0]))
for idx, trace in enumerate(psths):
    for idx2, trace2 in enumerate(psths):
        all_distances[idx, idx2] = dtw_limited_warping_length(
            trace.reshape(1, -1), trace2.reshape(1, -1), 5
        )
# %%
all_distances = pairwise_distances(psths, metric="cosine")
# %%
drop_columns = [
    "OFF_Linear_LWS",
    "OFF_Linear_SWS1",
    "OFF_Linear_SWS2",
    "OFF_Linear_MWS",
    "ON_Linear_LWS",
    "ON_Linear_SWS1",
    "ON_Linear_SWS2",
    "ON_Linear_MWS",
    "pca_1",
    "pca_2",
    "tsne_1",
    "tsne_2",
    "labels",
]
feauture_df = feauture_df.drop(drop_columns, axis=1)

# %%

lasso_columns = [
    "ON_Lasso_LWS",
    "ON_Lasso_SWS1",
    "ON_Lasso_SWS2",
    "ON_Lasso_MWS",
    "OFF_Lasso_LWS",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_SWS2",
    "OFF_Lasso_MWS",
]
# %%
all_distances_f = pairwise_distances(
    feauture_df.loc[index][lasso_columns].values, metric="cosine"
)
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(all_distances_f)
fig.colorbar(cax)
fig.show()
# %% plot similarity matrix
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(all_distances)
fig.colorbar(cax)
fig.show()
# %%
combine_matrix = all_distances / np.max(all_distances) + all_distances_f / np.max(
    all_distances_f
)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(combine_matrix)
fig.colorbar(cax)
fig.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
dendrogram(linkage(combine_matrix, method="complete"), ax=ax, color_threshold=5)
fig.show()
# %%
clustering = fcluster(
    linkage(combine_matrix, method="complete"), criterion="distance", t=6
)
clustering = clustering - 1
print(np.max(clustering))

# %%
index = pd.MultiIndex.from_arrays(
    [cell_index[:, 0], cell_index[:, 1]], names=["recording", "cell_index"]
)
cluster_df = pd.DataFrame(data={"labels": clustering}, index=index, columns=["labels"])

# %%
recordings.dataframes["fff_cluster"] = recordings.dataframes["fff_filtered"].copy()
recordings.dataframes["fff_cluster"] = recordings.dataframes["fff_cluster"].set_index(
    ["recording", "cell_index"]
)
recordings.dataframes["fff_cluster"]["cluster"] = 0
recordings.dataframes["fff_cluster"].update(cluster_df)
recordings.dataframes["fff_cluster"] = recordings.dataframes[
    "fff_cluster"
].reset_index()
# %%
spikes = recordings.get_spikes_df("fff_cluster", carry=["stimulus_name", "cluster"])
# %%
# add unique index per recording and cell_index
spikes["unique"] = spikes.groupby(["recording", "cell_index", "cluster"]).ngroup()
fig, ax = spiketrain_plots.whole_stimulus(
    spikes,
    indices=["stimulus_name", "unique"],
    width=20,
    bin_size=0.05,
    cmap="Greys",
    norm="eq_hist",
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.savefig(r"A:\Marvin\fff_clustering\spiketrain_whole_stimulus.png")
# %% plot cluster PSTHs
fig, ax = plt.subplots(np.max(clustering), 1, figsize=(10, 30))
for i in range(np.max(clustering)):
    ax[i].plot(np.nanmean(psths[clustering == i], axis=0))
    ax[i].set_xticks([])
    ax[i].set_title(f"Cluster {i}, n={np.sum(clustering == i)}")
ax[-1].set_xticks(range(psths.shape[1]))
fig.show()
# %% load feature df
root = Path(r"A:\Marvin\fff_clustering_zf")
feauture_df = pd.read_pickle(root / "features_coeffs_pca")
# %%
feauture_df.update(cluster_df)
# %%
feauture_df.to_pickle(root / "features_coeffs_pca_clustered")
