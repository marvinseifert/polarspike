import numpy as np
import matplotlib.pyplot as plt

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.metrics import dtw_limited_warping_length
from polarspike import Overview, histograms, colour_template, spiketrain_plots
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from pathlib import Path
import pandas as pd

# %%
root = Path(r"A:\Marvin\fff_clustering_zf")
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
feature_df = pd.read_pickle(root / "features_coeffs_pca")
# %%
recordings = Overview.Recording_s.load(root / "records")
spikes = recordings.get_spikes_df("fff_filtered", pandas=False)
# %%
psth, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=24.1,
    bin_size=0.1,
)

# %%
noise = np.random.normal(0, 0.1, psth.shape[1])
psth_add_noise = psth + noise
# %% load example
example_psth = np.load(root / "ON_Lasso_principal_example_psths.npy")
# example_psth = example_psth + noise
# %%  euclidean distance between example and all
all_distances = cdist(example_psth[1], psth, metric="cosine")
# %%
psth_distance_sorted = psth[np.argsort(all_distances, axis=1)].squeeze()

# %%
fig, ax = plt.subplots(11, 1, figsize=(10, 10), sharex=True)
for idx, trace in enumerate(psth_distance_sorted[:10]):
    ax[idx].plot(bins[:-1], trace, color="black")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %%
all_distances = np.zeros((1, psth.shape[0]))
for idx, trace in enumerate(psth):
    all_distances[0, idx] = dtw_limited_warping_length(
        example_psth[1].reshape(1, -1), trace.reshape(1, -1), 5
    )
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(all_distances.squeeze(), bins=100)
fig.show()
# %%
indices = cell_index[np.argsort(all_distances, axis=1).squeeze()[:10]]
m_index = pd.MultiIndex.from_arrays(
    [indices[:, 0], indices[:, 1]], names=["recording", "cell_index"]
)
# %%

selected_features = feature_df.loc[m_index][
    [
        "OFF_Lasso_principal",
        "OFF_Lasso_accessory",
        "OFF_Lasso_LWS",
        "OFF_Lasso_MWS",
        "OFF_Lasso_SWS1",
        "OFF_Lasso_SWS2",
        "ON_Lasso_principal",
        "ON_Lasso_accessory",
        "ON_Lasso_LWS",
        "ON_Lasso_MWS",
        "ON_Lasso_SWS1",
        "ON_Lasso_SWS2",
    ]
].values.T
# %% plot as heatmap
fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(selected_features, aspect="auto", vmin=-1, vmax=1)
cm = fig.colorbar(im)
# add colorbar legend
cm.set_label("cone contributions")
# add xticks
ax.set_yticks(np.arange(selected_features.shape[0]))
ax.set_yticklabels(
    [
        "OFF_Lasso_principal",
        "OFF_Lasso_accessory",
        "OFF_Lasso_LWS",
        "OFF_Lasso_MWS",
        "OFF_Lasso_SWS1",
        "OFF_Lasso_SWS2",
        "ON_Lasso_principal",
        "ON_Lasso_accessory",
        "ON_Lasso_LWS",
        "ON_Lasso_MWS",
        "ON_Lasso_SWS1",
        "ON_Lasso_SWS2",
    ],
)
fig.show()
# %%
recordings.dataframes["fff_coeffs"] = (
    recordings.dataframes["spikes_df"].query("stimulus_name=='chirp'").copy()
)
recordings.dataframes["fff_coeffs"]["OFF_Lasso_principal"] = 0
recordings.dataframes["fff_coeffs"]["OFF_Lasso_accessory"] = 0
recordings.dataframes["fff_coeffs"]["OFF_Lasso_LWS"] = 0
recordings.dataframes["fff_coeffs"]["OFF_Lasso_MWS"] = 0
recordings.dataframes["fff_coeffs"]["OFF_Lasso_SWS1"] = 0
recordings.dataframes["fff_coeffs"]["OFF_Lasso_SWS2"] = 0
recordings.dataframes["fff_coeffs"].set_index(["recording", "cell_index"], inplace=True)
recordings.dataframes["fff_coeffs"].update(feature_df)
# %%
recordings.dataframes["fff_coeffs"] = (
    recordings.dataframes["fff_coeffs"]
    .sort_values(
        [
            "OFF_Lasso_principal",
            "OFF_Lasso_accessory",
            "OFF_Lasso_LWS",
            "OFF_Lasso_MWS",
            "OFF_Lasso_SWS1",
            "OFF_Lasso_SWS2",
        ],
        ascending=False,
    )
    .reset_index()
)
# %%
new_index = np.linspace(
    0,
    len(recordings.dataframes["fff_coeffs"]) / 20,
    len(recordings.dataframes["fff_coeffs"]),
    dtype=int,
)
recordings.dataframes["fff_coeffs"]["new_index"] = new_index
# %%
spikes_sorted = recordings.get_spikes_df(
    "fff_coeffs", pandas=False, carry=["new_index", "stimulus_name"]
)
# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_sorted,
    indices=["new_index", "stimulus_name"],
    width=20,
)

fig.show()
