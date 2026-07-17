from polarspike import Overview
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ruptures as rpt
from polarspike import histograms
from polarspike.Overview import Recording
from polarspike import colour_template
import pandas as pd
import umato
import umap
from tensorly.decomposition import non_negative_tucker
from polarspike import spiketrain_plots
from pathlib import Path
import plotly.express as px

# %%
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
# %%
colours = ct.colours[::2]
colours = np.repeat(colours, 80, axis=0)

# %%
recordings = Overview.Recording_s(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis",
    "fff_analysis",
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_11_02_2026/Phase_01/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_12_02_2026/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_13_11_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_17_11_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_18_11_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_01_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_05_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_07_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_14_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_20_04_2026/Phase_00/overview"
)

# %%
spikes_fff = recordings.get_spikes_triggered(
    [{"stimulus_name": "fff"}], pandas=False, carry=["stimulus_name", "qi"]
)

# %%
df_fff = pl.DataFrame(
    {
        "recordings": spikes_fff["recording"],
        "times_relative": spikes_fff["times_relative"],
        "times_triggered": spikes_fff["times_triggered"],
        "trigger": spikes_fff["trigger"],
        "stimulus_name": spikes_fff["stimulus_name"],
        "cell_index": spikes_fff["cell_index"],
        "repeat": spikes_fff["repeat"],
        "qi": spikes_fff["qi"],
    }
).lazy()

# %%
df_fff = df_fff.sort(
    ["recordings", "stimulus_name", "cell_index", "times_relative"], descending=False
)
df_fff = df_fff.with_columns(
    pl.col("times_relative").diff(null_behavior="ignore").alias("isi")
)

df_fff = df_fff.filter((pl.col("isi") > 0) & (pl.col("isi") > 0.005)).collect()
# %%
df_fff_search = df_fff.group_by(
    ["recordings", "cell_index", "repeat"], maintain_order=True
).agg("isi", "times_triggered", "trigger")


def sliding_window_search(ts, query):
    n = len(ts)
    m = len(query)
    distances = []

    # Slide the window across the sequence
    for i in range(n - m + 1):
        window = ts[i: i + m]
        # Calculate Euclidean distance
        dist = np.linalg.norm(window - query)
        distances.append(dist)

    return np.array(distances)


# %% Load motives
all_motives = []
root = Path(r"/mnt/mea_nas_25/Marvin/combined_analysis")
leds = [365, 413, 460, 500, 535, 560, 610]
for led in leds:
    all_motives.append(np.load(root / f"led_{led}_motifs.npy"))
# %%
all_motives = np.concatenate(all_motives)

# %%
from numpy.lib.stride_tricks import sliding_window_view

# 1. Pre-process motifs into a single Matrix
# Shape: (num_motifs, motif_len)
M = np.array(all_motives)
M_sq_norms = np.sum(M ** 2, axis=1)  # Shape: (num_motifs,)
motif_len = M.shape[1]


# def batch_matrix_search(ts):
#     """
#     Computes distances for ALL motifs at once using Matrix Multiplication.
#     Returns a dictionary of lists (one for each motif).
#     """
#     if ts is None or len(ts) < motif_len:
#         # Return empty lists for each motif to maintain schema
#         return {f"distance{i}": [] for i in range(len(M))}
#
#     # Create sliding windows: Shape (num_windows, motif_len)
#     W = sliding_window_view(ts, window_shape=motif_len)
#
#     # 1. Compute window squared norms: ||w||^2
#     W_sq_norms = np.sum(W**2, axis=1)  # Shape: (num_windows,)
#
#     # 2. Compute dot products: (w · q)
#     # Resulting shape: (num_windows, num_motifs)
#     dot_products = W @ M.T
#
#     # 3. Combine using the expansion: ||w||^2 + ||q||^2 - 2(w · q)
#     # Using broadcasting: (W_norm[:, None]) + (M_norm[None, :]) - 2 * dot
#     sq_distances = (
#         W_sq_norms[:, np.newaxis] + M_sq_norms[np.newaxis, :] - 2 * dot_products
#     )
#
#     # Handle potential negative values from floating point precision
#     distances = np.sqrt(np.maximum(sq_distances, 0))
#
#     # Return as a dictionary so Polars can unnest into columns
#     return {f"distance{i}": distances[:, i].tolist() for i in range(len(M))}
# %%
def batch_matrix_search(ts):
    if ts is None or len(ts) < motif_len:
        return {f"distance{i}": [] for i in range(len(M))}

    W = sliding_window_view(ts, window_shape=motif_len)

    # L1: sum of absolute differences per window per motif
    # W[:, np.newaxis, :] -> (num_windows, 1, motif_len)
    # M[np.newaxis, :, :] -> (1, num_motifs, motif_len)
    distances = np.sum(np.abs(W[:, np.newaxis, :] - M[np.newaxis, :, :]), axis=-1)
    # Shape: (num_windows, num_motifs)

    return {f"distance{i}": distances[:, i].tolist() for i in range(len(M))}


# 2. Apply to Polars
# Define the output schema for the Struct
output_schema = pl.Struct(
    [pl.Field(f"distance{i}", pl.List(pl.Float64)) for i in range(len(M))]
)

df_fff_search = (
    df_fff.group_by(["recordings", "cell_index", "repeat"], maintain_order=True)
    .agg(pl.col("isi"), pl.col("times_triggered"), pl.col("trigger"))
    .with_columns(
        pl.col("isi")
        .map_elements(batch_matrix_search, return_dtype=output_schema)
        .alias("results")
    )
    .unnest("results")  # This expands the dict into distance0, distance1, etc.
)
# %%
trigger_extraction_exprs = [
    pl.col("trigger")
    .list.get(pl.col(f"distance{i}").list.arg_min())
    .alias(f"best_trigger_{i}")
    for i in range(all_motives.shape[0])
]
df_fff_search = df_fff_search.with_columns(trigger_extraction_exprs)
# %%
df_fff_search_summary = df_fff_search.with_columns(
    [
        pl.col(f"distance{i}").list.min().alias(f"distance{i}_min")
        for i in range(all_motives.shape[0])
    ],
)
# %%
df_fff_search_summary = df_fff_search_summary.drop_nulls()
# %%
distance_cols = [f"distance{i}_min" for i in range(all_motives.shape[0])]
# %%
stats_df = pl.DataFrame(
    {
        "condition": distance_cols,
        "skew": [
            df_fff_search_summary.select(pl.col(c).skew()).item() for c in distance_cols
        ],
        "contrast": [
            df_fff_search_summary.select(
                (pl.col(c).quantile(0.01) / pl.col(c).median())
            ).item()
            for c in distance_cols
        ],
        "cv": [
            df_fff_search_summary.select(pl.col(c).std() / pl.col(c).mean()).item()
            for c in distance_cols
        ],
    }
)

# %%
fig = px.scatter_3d(
    stats_df.to_pandas(), x="cv", y="contrast", z="skew", hover_data="condition"
)
fig.show(renderer="browser")
# %%
fig, ax = plt.subplots()
ax.hist(df_fff_search_summary["best_trigger_155"], 8)
fig.show()
# %%
X = np.vstack(
    [
        stats_df["cv"].to_numpy(),
        stats_df["contrast"].to_numpy(),
        # stats_df["skew"].to_numpy(),
    ]
).T
# %%
from scipy import stats
from sklearn.covariance import MinCovDet

# Fit a robust covariance (resistant to outliers skewing the fit)
mcd = MinCovDet(support_fraction=0.75).fit(X)

# Mahalanobis distance for every point
mahal = mcd.mahalanobis(X)  # squared distances

# Chi-squared threshold: 2 DOF (2D data), pick confidence level
threshold = stats.chi2.ppf(0.975, df=2)  # 97.5% = ~5.02

outliers = X[mahal > threshold]
inliers = X[mahal <= threshold]
# %%
fig, ax = plt.subplots()
ax.scatter(
    stats_df["cv"].to_numpy(),
    stats_df["contrast"].to_numpy(),
    c=mahal > threshold,
    cmap="tab10",
)
ax.set_ylabel("contrast")
ax.set_xlabel("cv")
fig.show()
# %%
outlier_idx = np.where(mahal > threshold)[0]

# %%
outlier_df = stats_df[outlier_idx]
# %%
import re

motif_indices = [
    int(x) for x in re.findall(r"\d+", " ".join(outlier_df["condition"].to_list()))
]
# %%
outlier_motifs = all_motives[motif_indices]
spiketrains = np.cumsum(
    np.hstack([np.zeros((outlier_motifs.shape[0], 1)), outlier_motifs]), axis=1
)
# %%
fig, ax = plt.subplots(figsize=(20, 10))

for spk_idx, spiketrain in enumerate(spiketrains):
    ax.scatter(
        spiketrain, np.ones(spiketrain.shape[0]) * spk_idx, marker="|", c="black"
    )
ax.vlines(np.arange(0, np.max(spiketrains), 2), 0, spiketrains.shape[0])
ax.set_xlim((0, 5))
fig.show()

# %%
fig, ax = plt.subplots()
ax.hist(np.median(outlier_motifs, axis=1), bins=np.arange(0, 0.6, 0.005))
fig.show()

# %% distance histograms
for mot_idx, motif in enumerate(motif_indices):
    fig = plt.figure(figsize=(20, 10))

    # 2. Define a 2x2 grid
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 1])

    # 3. Assign the subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top Left
    ax2 = fig.add_subplot(gs[0, 1])  # Top Right
    ax3 = fig.add_subplot(gs[1, :])  # Bottom - Spans all columns (":")
    ax1.hist(
        df_fff_search_summary[f"distance{motif}_min"].to_numpy(),
        bins=np.arange(0, 2, 0.1),
    )
    ax2.hist(
        df_fff_search_summary[f"best_trigger_{motif}"].to_numpy(),
        bins=np.arange(0, 9, 1),
    )
    ax3.scatter(
        spiketrains[mot_idx],
        np.ones(spiketrains[mot_idx].shape[0]),
        marker="|",
        c="black",
    )
    fig.savefig(
        rf"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis/distance_hist/motif_{motif}.svg"
    )
    plt.close(fig)
# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.hist(
    df_fff_search_summary.select(pl.col("isi").explode()).to_numpy(),
    bins=np.arange(0, 0.1, 0.001),
)
ax.set_xscale("log")
fig.show()

# %%
for mot_idx, motif in enumerate(motif_indices):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        df_fff_search_summary[f"distance{motif}_min"].to_numpy(),
        df_fff_search_summary[f"repeat"].to_numpy(),
    )
    fig.savefig(
        rf"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis/distance_repeat/motif_{motif}.svg"
    )
    plt.close(fig)

# %%
new_x = np.arange(0, 2 * 16, 0.05)


def interpolate_df(row):
    interpolated = np.interp(
        new_x, row[4][:-4], row[52], right=np.quantile(row[52], 0.80)
    )
    return (interpolated.tolist(),)


interpolated_df = df_fff_search_summary.map_rows(
    interpolate_df, return_dtype=pl.List(pl.Float64)
)

# %%
all_interpolated = np.stack(interpolated_df["column_0"].to_numpy())

# %%
fig, ax = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [10, 1]}, sharex=True)
ax[0].plot(new_x, np.median(all_interpolated, axis=0))
# ax[0].set_yscale("log")
fig = ct.add_stimulus_to_plot(fig, [2] * 16)
fig.show()
# %%
from scipy.interpolate import interp1d

for p in range(200):
    try:
        extract_df = df_fff_search_summary.sample(1)
        single_cell_distance = df_fff_search_summary.filter(
            (pl.col("recordings") == extract_df["recordings"])
            & (pl.col("cell_index") == extract_df["cell_index"])
        )
        new_x = np.arange(0, 2 * 16, 0.05)
        interpolated = []
        for i in range(single_cell_distance["repeat"].unique().max() + 1):
            interpolated.append(
                np.interp(
                    new_x,
                    single_cell_distance.filter(pl.col("repeat") == i)[
                        "times_triggered"
                    ].to_numpy()[0][:-4],
                    single_cell_distance.filter(pl.col("repeat") == i)[
                        "distance46"
                    ].to_numpy()[0],
                ),
            )

        fig, ax = plt.subplots(
            figsize=(20, 10),
            nrows=4,
            gridspec_kw={"height_ratios": [2, 5, 5, 1]},
            sharex=True,
        )
        psth_cell, bins_cell = histograms.psth(
            spikes_fff.filter(
                (pl.col("recording") == extract_df["recordings"])
                & (pl.col("cell_index") == extract_df["cell_index"])
            )
        )
        ax[0].plot(bins_cell[:-1], psth_cell)
        ax[1].plot(new_x, np.median(interpolated, axis=0))
        ax[1].set_yscale("log")
        for i in range(single_cell_distance["repeat"].unique().max() + 1):
            times_triggered = single_cell_distance.filter(pl.col("repeat") == i)[
                "times_triggered"
            ].to_numpy()[0]
            ax[2].scatter(
                times_triggered,
                np.ones(times_triggered.shape[0]) * i,
                marker="|",
                c="black",
            )
        fig = ct.add_stimulus_to_plot(fig, [2] * 16)
        ax[1].set_ylim((10 ** -2.5, 10))
        fig.savefig(
            rf"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis/distance_fff/fff_{p}.svg"
        )
        plt.close(fig)
    except IndexError:
        pass
# %%
