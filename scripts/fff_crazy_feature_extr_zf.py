from polarspike import (
    Overview,
    spiketrain_plots,
    moving_bars,
    quality_tests,
    stimulus_spikes,
    binarizer,
    colour_template,
    histograms,
    Opsins,
)
from pathlib import Path

from sklearn.cluster import HDBSCAN, MeanShift

from sklearn.preprocessing import StandardScaler

import pandas as pd

from sklearn.decomposition import PCA

from scipy.stats import zscore
import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances
import matplotlib.pyplot as plt
from polarspike.analysis import response_peaks, count_spikes
from matplotlib.patches import Rectangle
import polars as pl
from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging
from tslearn.neighbors import KNeighborsTimeSeries
import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler

# %%
root = Path(r"A:\Marvin\fff_clustering_zf")
recordings = Overview.Recording_s.load(root / "records")

# %%
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
feature_df = pd.DataFrame(
    index=pd.MultiIndex.from_arrays(
        [cell_index[:, 0], cell_index[:, 1]], names=["recording", "cell_index"]
    )
)
# %%
psth_z = psth.copy()  # zscore(psth, axis=1)

# %% PLot 10 random examples
fig, ax = plt.subplots(10, 1, figsize=(10, 20))
indices = np.random.choice(psth_z.shape[0], 10, replace=False)
for i in range(10):
    ax[i].plot(psth_z[indices[i]])
    ax[i].set_title(f"Cell {cell_index[indices[i]]}")
fig.show()
# %% find maxima
psth_argmax = np.argmax(psth_z, axis=1)
psth_max = np.max(psth_z, axis=1)
# %%
feature_df["maxima"] = psth_max
# feature_df["argmax"] = psth_argmax
# %% Plot maxima
fig, ax = plt.subplots(10, 1, figsize=(10, 10))
for i in range(10):
    ax[i].plot(psth_z[indices[i]])
    ax[i].scatter(psth_argmax[indices[i]], psth_max[indices[i]], c="r")
    ax[i].set_title(f"Cell {cell_index[indices[i]]}")
fig.show()

# %%
sparsity = np.sum(psth_z > 0, axis=1) / psth_z.shape[1]

# %%
feature_df["sparsity"] = sparsity
# feature_df["mean_firing_rate"] = np.mean(psth_z, axis=1)
feature_df["medium_firing_rate"] = np.median(psth_z, axis=1)

# %% features per epoch
epoch_lenght = 20
epoch_max = np.zeros((psth_z.shape[0], 12))
epoch_argmax = np.zeros_like(epoch_max)
epoch_sparsity = np.zeros_like(epoch_max)
epoch_mean_firing_rate = np.zeros_like(epoch_max)
epoch_medium_firing_rate = np.zeros_like(epoch_max)
epoch_sum = np.zeros_like(epoch_max)
epoch_sustained = np.zeros_like(epoch_max)

for i in range(12):
    epoch_max[:, i] = np.max(
        psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
    )
    epoch_argmax[:, i] = np.argmax(
        psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
    )
    epoch_sparsity[:, i] = (
        np.sum(psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght] > 0, axis=1)
        / epoch_lenght
    )
    # epoch_mean_firing_rate[:, i] = np.mean(
    #     psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
    # )
    epoch_medium_firing_rate[:, i] = np.median(
        psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
    )
    epoch_sum[:, i] = np.sum(
        psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
    )
    transient_sum = np.sum(psth_z[:, i * epoch_lenght : (i + 2) * epoch_lenght], axis=1)
    sustained_sum = np.sum(
        psth_z[:, i * epoch_lenght + 2 : (i + 18) * epoch_lenght], axis=1
    )
    epoch_sustained[:, i] = (transient_sum - sustained_sum) / (
        transient_sum + sustained_sum
    )


epoch_sustained[np.isnan(epoch_sustained)] = 0

# %% Check for any NaN values
print(
    f"""NaN values detected: {np.any(
    np.isnan(
        np.concatenate(
            [
                epoch_max,
                epoch_argmax,
                epoch_sparsity,
                epoch_medium_firing_rate,
                epoch_sum,
            ],
            axis=1,
        )
    )
)}"""
)

# %% Add to dataframe
feature_names = [
    "610",
    "610_off",
    "560",
    "560_off",
    "535",
    "535_off",
    "460",
    "460_off",
    "413",
    "413_off",
    "365",
    "365_off",
]
for i, name in enumerate(feature_names):
    feature_df[f"{name}_max"] = epoch_max[:, i]
    feature_df[f"{name}_argmax"] = epoch_argmax[:, i]
    feature_df[f"{name}_sparsity"] = epoch_sparsity[:, i]
    # feature_df[f"{name}_mean_firing_rate"] = epoch_mean_firing_rate[:, i]
    feature_df[f"{name}_medium_firing_rate"] = epoch_medium_firing_rate[:, i]
    feature_df[f"{name}_sum"] = epoch_sum[:, i]
    feature_df[f"{name}_tr_sus_index"] = epoch_sustained[:, i]


# %% save df
feature_df.to_pickle(root / "feature_df")
