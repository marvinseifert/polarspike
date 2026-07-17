from polarspike import (
    Overview,
    colour_template,
    spike_loader,
    histograms,
    spiketrain_plots,
)

from polarspike.analysis import count_spikes
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LassoCV, LinearRegression
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
import warnings
from scipy.ndimage import gaussian_filter1d

# %%
from scipy.ndimage import convolve1d


def half_gaussian_kernel(sigma, truncate=3.0):
    """
    Build a normalized half-Gaussian kernel (causal — past only).
    Index 0 = current sample, higher indices = further into the past.
    """
    half = int(truncate * sigma + 0.5)
    n = np.arange(0, half + 1)
    kernel = np.exp(-(n ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()  # normalize so weights sum to 1
    return kernel


def half_gaussian_filter(signal, sigma, truncate=3.0):
    """
    Apply a causal half-Gaussian filter to a 1D signal.
    Each output point is a weighted sum of the current and past samples only.
    """
    kernel = half_gaussian_kernel(sigma, truncate)
    # origin=0 aligns kernel[0] with the current sample,
    # kernel[1..] reach back into the past — no future leakage
    return convolve1d(signal, kernel, mode="nearest", origin=-(len(kernel) // 2))


# %%
stimulus_name = "fff"
window = 0.2
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
colours = CT.colours[::2]
cone_colours = [
    colours[5],
    colours[4],
    colours[2],
    colours[0],
    "orange",
    "darkgoldenrod",
    "saddlebrown",
]
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
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
# %%
# spikes_df = recordings.get_spikes_triggered(
#     [{"stimulus_name": ["scf_06nd", "scf(0.6)"]}],
#     pandas=False,
#     carry=["stimulus_name", "qi"],
# )
spikes_df = recordings.get_spikes_triggered(
    [{"stimulus_name": ["fff", "scf(0.0ND)"]}],
    pandas=False,
    carry=["stimulus_name", "qi"],
)
spikes_df = spikes_df.filter(pl.col("qi") >= 0.3)
spikes_df = spikes_df.with_columns(dummy=pl.lit(1))

# %%
mean_of_all, bins = histograms.psth(spikes_df)
mean_of_all = mean_of_all / np.max(mean_of_all)
# %%
cell_psth, bins = histograms.psth_by_index(spikes_df, index=["recording", "cell_index"])

# %%
cell = 901
fig, ax = plt.subplots(
    figsize=(20, 10), nrows=3, gridspec_kw={"height_ratios": [10, 10, 1]}, sharex=True
)
ax[0].plot(bins[:-1], cell_psth[cell, :] / np.max(cell_psth[cell, :]))
ax[1].plot(
    bins[:-1],
    half_gaussian_filter(cell_psth[cell, :] / np.max(cell_psth[cell, :]), 4)
    - half_gaussian_filter(mean_of_all, 4),
)
ax[1].hlines(0, 0, 2 * 16)
fig, ct.add_stimulus_to_plot(fig, [2] * 16)
fig.show()
# %%
fig, ax = plt.subplots()
ax.plot(mean_of_all)
fig.show()
# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_df, indices=["dummy", "cell_index"], width=20
)
fig = ct.add_stimulus_to_plot(fig, [2] * 16)
fig.show()
# %%
fff_stimulus = recordings.stimulus_df.query("stimulus_name == @stimulus_name")
recordings.dataframes["fff_stimulus"] = fff_stimulus

# %%
mean_trigger = spike_loader.mean_trigger_times(fff_stimulus, ["all"])

# %%
spikes_rec = spikes_df.partition_by("recording")

# %%
spikes_summed, cell_indices = count_spikes.sum_spikes(
    spikes_df,
    mean_trigger,
    window=window,
    group_by=["recording", "cell_index", "repeat"],
)
# %% transfer to polars dataframe
summed_df = pl.DataFrame(
    {
        "recordings": cell_indices[:, 0].tolist(),
        "cell_index": cell_indices[:, 1].astype(int),
        "repeat": cell_indices[:, 2].astype(int),
        "spikes_summed": spikes_summed,
    },
    schema={
        "recordings": pl.String,
        "cell_index": pl.UInt32,
        "repeat": pl.UInt8,
        "spikes_summed": pl.List,
    },
)
unique_pairs = summed_df.select(["recordings", "cell_index"]).unique()
all_repeats = pl.DataFrame({"repeat": pl.Series(range(5), dtype=pl.UInt8)})

# Cross join to get all expected combinations
expected = unique_pairs.join(all_repeats, how="cross")

# Left join back to the actual data, filling missing with zeros
zero_array = [0.0] * 16

summed_df = (
    expected.join(summed_df, on=["recordings", "cell_index", "repeat"], how="left")
    .sort(["recordings", "cell_index", "repeat"])
    .with_columns(
        pl.col("spikes_summed").fill_null(
            pl.lit(zero_array, dtype=pl.Array(pl.Float64, 16))
        )
    )
)
# %%
summed_df = summed_df.with_columns(
    pl.col("spikes_summed").list.gather_every(2).alias("spikes_summed_on")
)
summed_df = summed_df.with_columns(
    pl.col("spikes_summed").list.gather_every(2, 1).alias("spikes_summed_off")
)

# %% Import sensitivity curves
double_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_double")
single_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_oil")

# %%
double_ab_df.loc[double_ab_df["absorption"] < 0.01, "absorption"] = np.nan
single_ab_df.loc[single_ab_df["absorption"] < 0.01, "absorption"] = np.nan

single_ab_df["absorption"] = np.log(single_ab_df["absorption"])
double_ab_df["absorption"] = np.log(double_ab_df["absorption"])

# normalize the absorption values to 0-1 per cone
for cone in single_ab_df["cone"].unique():
    single_ab_df.loc[single_ab_df["cone"] == cone, "absorption"] = (
                                                                           single_ab_df.loc[single_ab_df[
                                                                                                "cone"] == cone, "absorption"]
                                                                           - np.nanmin(single_ab_df["absorption"])
                                                                   ) / (np.nanmax(
        single_ab_df["absorption"]) - np.nanmin(single_ab_df["absorption"]))
for cone in double_ab_df["cone"].unique():
    double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"] = (
                                                                           double_ab_df.loc[double_ab_df[
                                                                                                "cone"] == cone, "absorption"]
                                                                           - np.nanmin(double_ab_df["absorption"])
                                                                   ) / (np.nanmax(
        double_ab_df["absorption"]) - np.nanmin(double_ab_df["absorption"]))

# %% Extract individual opsins
wavelengths = [365, 416, 460, 500, 535, 560, 610, 660]
wavelengths = np.flip(wavelengths)
# need to extract the absorption values for each wavelength
# for each cone type
cone_names = []
X = np.zeros((6, 8))
for i, cone in enumerate(single_ab_df["cone"].unique()):
    cone_names.append(cone)
    for j, wavelength in enumerate(wavelengths):
        print(wavelength)
        X[i, j] = single_ab_df.query("cone == @cone & wavelength == @wavelength")[
            "absorption"
        ]
# add double cone
for i, d_cone in enumerate(double_ab_df["cone"].unique()):
    cone_names.append(d_cone)
    for j, wavelength in enumerate(wavelengths):
        X[i + 4, j] = double_ab_df.query("cone == @d_cone & wavelength == @wavelength")[
            "absorption"
        ]
# X[-1, :] = X[4, :] + X[5, :]
# cone_names.append("both_double")
# Max normalize X per cone
X = X / np.nanmax(X, axis=1)[:, None]
X[np.isnan(X)] = 0.001
# %% expand for repeats
nr_repeats = len(all_repeats)

X_tiled = np.tile(X.T, (nr_repeats, 1))
X_tiled[X_tiled <= 0.001] = 0
# %%
spikes_summed_flattened = summed_df.group_by(
    ["recordings", "cell_index"], maintain_order=True
).agg(pl.col("spikes_summed_on").flatten(), pl.col("spikes_summed_off").flatten())

spikes_per_cell = (
    spikes_df.group_by(["recording", "cell_index", "repeat"], maintain_order=True)
    .agg(pl.col("times_triggered").count().alias("spike_counts"))
    .group_by(pl.col(["recording", "cell_index"]), maintain_order=True)
    .agg(pl.col("spike_counts").mean().alias("mean_spike_count"))
    .with_columns(pl.col("mean_spike_count") / (1 / window * np.sum(mean_trigger)))
)
spikes_per_cell = spikes_per_cell.rename({"recording": "recordings"})
spikes_summed_flattened = spikes_summed_flattened.join(
    spikes_per_cell, on=["recordings", "cell_index"]
)

# )
#
spikes_summed_flattened = (
    spikes_summed_flattened.explode(["spikes_summed_on", "spikes_summed_off"])
    .with_columns(
        [
            (pl.col("spikes_summed_on") / pl.col("mean_spike_count"))
            .over(["recordings", "cell_index"])
            .alias("spikes_summed_on"),
            (pl.col("spikes_summed_off") / pl.col("mean_spike_count"))
            .over(["recordings", "cell_index"])
            .alias("spikes_summed_off"),
        ]
    )
    .fill_nan(0)
    .group_by(["recordings", "cell_index"], maintain_order=True)
    .agg(
        [
            pl.col("spikes_summed_on"),
            pl.col("spikes_summed_off"),
        ]
    )
)

# %%
n_splits = 5
custom_splits = []

for _ in range(n_splits):
    test_repeat = np.random.choice(nr_repeats)
    train_repeats = [r for r in range(nr_repeats) if r != test_repeat]

    test_mask = np.repeat(np.arange(nr_repeats), len(wavelengths)) == test_repeat
    custom_splits.append((np.where(~test_mask)[0], np.where(test_mask)[0]))

# def apply_lasso(array):
#     # groups = np.tile(np.arange(len(wavelengths)), nr_repeats)
#     # cv_splits = list(
#     #     GroupKFold(n_splits=len(wavelengths)).split(X_tiled, array, groups)
#     # )
#     # groups = np.repeat(np.arange(nr_repeats), len(wavelengths))
#     # cv_splits = list(GroupKFold(n_splits=nr_repeats).split(X_tiled, array, groups))
#     lasso = LassoCV(
#         cv=custom_splits,
#         n_jobs=-1,
#         alphas=np.logspace(-4, 0, 50),
#         max_iter=100000,
#         tol=1e-5,
#         fit_intercept=False,
#     )
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=ConvergenceWarning)
#         lasso.fit(X_tiled, array)
#     return {
#         "coefs": lasso.coef_.tolist(),
#         "r2": lasso.score(X_tiled, array),
#         "intercept": lasso.intercept_,
#     }
# %%
from sklearn.linear_model import Lasso


def apply_lasso(array):
    array = array.to_numpy()
    alphas = np.logspace(-4, 0, 50)
    alpha_scores = []

    # Cross-validate over alphas, clipping predictions in each fold
    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in custom_splits:
            X_train, X_val = X_tiled[train_idx], X_tiled[val_idx]
            y_train, y_val = array[train_idx], array[val_idx]

            lasso = Lasso(
                alpha=alpha,
                max_iter=100000,
                tol=1e-5,
                fit_intercept=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                lasso.fit(X_train, y_train)

            predictions = np.maximum(lasso.predict(X_val), 0)

            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            if ss_tot > 0:
                ss_res = np.sum((y_val - predictions) ** 2)
                fold_scores.append(1 - ss_res / ss_tot)
            else:
                fold_scores.append(0.0)

        alpha_scores.append(np.mean(fold_scores))

    # Fit final model on all data with best alpha
    best_alpha = alphas[np.argmax(alpha_scores)]
    lasso = Lasso(
        alpha=best_alpha,
        max_iter=100000,
        tol=1e-5,
        fit_intercept=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso.fit(X_tiled, array)

    predictions = np.maximum(lasso.predict(X_tiled), 0)
    ss_res = np.sum((array - predictions) ** 2)
    ss_tot = np.sum((array - np.mean(array)) ** 2)
    r2_corrected = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coefs": lasso.coef_.tolist(),
        "r2": r2_corrected,
        "intercept": lasso.intercept_,
    }


# %%
from sklearn.linear_model import RidgeCV


def apply_ridge(array):
    ridge = RidgeCV(
        cv=custom_splits, alphas=np.logspace(-4, 0, 50), fit_intercept=False
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        ridge.fit(X_tiled, array)
    return {
        "coefs": ridge.coef_.tolist(),
        "r2": ridge.score(X_tiled, array),
        "intercept": ridge.intercept_,
    }


# %%
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ShuffleSplit

# treat each repeat-worth of data as a unit
n_splits = 10
custom_splits = []

for _ in range(n_splits):
    test_repeat = np.random.choice(nr_repeats)
    train_repeats = [r for r in range(nr_repeats) if r != test_repeat]

    test_mask = np.repeat(np.arange(nr_repeats), len(wavelengths)) == test_repeat
    custom_splits.append((np.where(~test_mask)[0], np.where(test_mask)[0]))

# %%
from sklearn.linear_model import ElasticNet


def apply_elastic_net(array):
    array = array.to_numpy()
    alphas = np.logspace(-4, 0, 50)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    best_score = -np.inf
    best_alpha = None
    best_l1_ratio = None

    # Cross-validate over alphas and l1_ratios, clipping predictions in each fold
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_scores = []
            for train_idx, val_idx in custom_splits:
                X_train, X_val = X_tiled[train_idx], X_tiled[val_idx]
                y_train, y_val = array[train_idx], array[val_idx]

                en = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=100000,
                    tol=1e-5,
                    fit_intercept=False,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    en.fit(X_train, y_train)

                predictions = np.maximum(en.predict(X_val), 0)

                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                if ss_tot > 0:
                    ss_res = np.sum((y_val - predictions) ** 2)
                    fold_scores.append(1 - ss_res / ss_tot)
                else:
                    fold_scores.append(0.0)

            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    # Fit final model on all data with best alpha and l1_ratio
    en = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        max_iter=100000,
        tol=1e-5,
        fit_intercept=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        en.fit(X_tiled, array)

    predictions = np.maximum(en.predict(X_tiled), 0)
    ss_res = np.sum((array - predictions) ** 2)
    ss_tot = np.sum((array - np.mean(array)) ** 2)
    r2_corrected = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coefs": en.coef_.tolist(),
        "r2": r2_corrected,
        # "best_alpha": best_alpha,
        # "best_l1_ratio": best_l1_ratio,
        "intercept": en.intercept_,
    }


# %%
# def apply_elasticnet(array):
#     trial_matrix = array.to_numpy().reshape(nr_repeats, len(wavelengths))
#     mean_response = trial_matrix.mean(axis=0)
#
#     groups = np.tile(np.arange(len(wavelengths)), nr_repeats)
#
#     # Custom CV scoring against mean rather than noisy trials
#     best_score = -np.inf
#     best_alpha = None
#     best_l1 = None
#
#     for l1_ratio in [0.5, 0.7, 0.9, 0.95, 0.99]:
#         for alpha in np.logspace(-4, 0, 40):
#             fold_scores = []
#             gkf = GroupKFold(n_splits=len(wavelengths))
#
#             for train_idx, test_idx in gkf.split(X_tiled, array, groups):
#                 en = ElasticNet(
#                     alpha=alpha, l1_ratio=l1_ratio, positive=True, max_iter=100000
#                 )
#                 en.fit(X_tiled[train_idx], array[train_idx])
#
#                 # score against mean at held-out wavelengths
#                 wl_idx = test_idx[
#                     : len(test_idx) // nr_repeats
#                 ]  # one index per wavelength
#                 y_true = mean_response[groups[wl_idx]]
#                 y_pred = en.predict(X_tiled[wl_idx])
#
#                 ss_res = np.sum((y_true - y_pred) ** 2)
#                 ss_tot = np.sum((y_true - y_true.mean()) ** 2)
#                 fold_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
#
#             score = np.mean(fold_scores)
#             if score > best_score:
#                 best_score = score
#                 best_alpha = alpha
#                 best_l1 = l1_ratio
#
#     # Final fit on all tiled data with best params
#     en_final = ElasticNet(
#         alpha=best_alpha, l1_ratio=best_l1, positive=True, max_iter=100000
#     )
#     en_final.fit(X_tiled, array)
#     return {
#         "coefs": en_final.coef_.tolist(),
#         "r2": en_final.score(X_tiled, array),
#         "intercept": en_final.intercept_,
#     }


# %%
results = spikes_summed_flattened.with_columns(
    [
        pl.col("spikes_summed_on")
        .map_elements(
            apply_elastic_net,
            return_dtype=pl.Struct(
                [
                    pl.Field("coefs_on", dtype=pl.List(pl.Float64)),
                    pl.Field("r2_on", pl.Float64),
                    pl.Field("intercept", pl.Float64),
                ]
            ),
            strategy="threading",
        )
        .alias("results_on"),
        pl.col("spikes_summed_off")
        .map_elements(
            apply_elastic_net,
            return_dtype=pl.Struct(
                [
                    pl.Field("coefs_off", dtype=pl.List(pl.Float64)),
                    pl.Field("r2_off", pl.Float64),
                    pl.Field("intercept", pl.Float64),
                ]
            ),
            strategy="threading",
        )
        .alias("results_off"),
    ]
)

# %%
results = results.unnest(
    "results_on",
)
results = results.rename(
    {"coefs": "coef_on", "r2": "r2_on", "intercept": "intercept_on"}
)
results = results.unnest(
    "results_off",
)
results = results.rename(
    {"coefs": "coef_off", "r2": "r2_off", "intercept": "intercept_off"}
)
# %%
coef_on = np.stack(results["coef_on"].to_numpy())
coef_on = coef_on / np.max(np.abs(coef_on), axis=1, keepdims=True)
coef_on[np.isnan(coef_on)] = 0
coef_off = np.stack(results["coef_off"].to_numpy())
coef_off = coef_off / np.max(np.abs(coef_off), axis=1, keepdims=True)
coef_off[np.isnan(coef_off)] = 0
fig, ax = plt.subplots(figsize=(15, 20))
ax.imshow(
    coef_on,
    cmap="PuOr_r",
    aspect="auto",
    interpolation="nearest",
    vmin=-np.nanmax(np.abs(coef_on)),
    vmax=np.nanmax(np.abs(coef_on)),
)
fig.show()

# %%
from sklearn.metrics import r2_score

r2_final_on = []
r2_final_off = []
for entry in range(len(results)):
    example = results[entry]  # or filter by a specific cell_index

    coefs_on = np.array(example["coef_on"].to_list()[0])
    coefs_off = np.array(example["coef_off"].to_list()[0])
    spikes_on = np.array(example["spikes_summed_on"].to_list()[0])
    spikes_off = np.array(example["spikes_summed_off"].to_list()[0])
    intercept_on = np.array(example["intercept_on"].to_list()[0])
    intercept_off = np.array(example["intercept_off"].to_list()[0])

    # predicted is X_tiled @ coefs, giving one prediction per repeat*wavelength
    predicted_on = X_tiled @ coefs_on + intercept_on
    predicted_off = X_tiled @ coefs_off + intercept_off
    actual_on_mean = spikes_on.reshape(nr_repeats, len(wavelengths)).mean(axis=0)
    actual_off_mean = spikes_off.reshape(nr_repeats, len(wavelengths)).mean(axis=0)
    predicted_on_mean = predicted_on.reshape(nr_repeats, len(wavelengths)).mean(axis=0)
    predicted_off_mean = predicted_off.reshape(nr_repeats, len(wavelengths)).mean(
        axis=0
    )
    predicted_on_mean[predicted_on_mean < 0] = 0
    predicted_off_mean[predicted_on_mean < 0] = 0
    r2_final_on.append(r2_score(actual_on_mean, predicted_on_mean))
    r2_final_off.append(r2_score(actual_off_mean, predicted_off_mean))
# %%
results = results.with_columns(
    r2_final_on=np.array(r2_final_on), r2_final_off=np.array(r2_final_off)
)
# %%
fig, ax = plt.subplots()
ax.scatter(r2_final_on, r2_final_off)
ax.set_xlim((0, np.max(r2_final_on) + 0.5))
ax.set_ylim(0, np.max(r2_final_off) + 0.5)
fig.show()
# %%
# example = results.filter(
#     (pl.col("recordings") == "chicken_14_05_2025_p0") & (pl.col("cell_index") == 220)
# )  # or filter by a specific cell_index
example = results[20]
coefs_on = np.array(example["coef_on"].to_list()[0])
coefs_off = np.array(example["coef_off"].to_list()[0])
spikes_on = np.array(example["spikes_summed_on"].to_list()[0])
spikes_off = np.array(example["spikes_summed_off"].to_list()[0])
intercept_on = np.array(example["intercept_on"].to_list()[0])
intercept_off = np.array(example["intercept_off"].to_list()[0])

# predicted is X_tiled @ coefs, giving one prediction per repeat*wavelength
predicted_on = X_tiled @ coefs_on + intercept_on
predicted_off = X_tiled @ coefs_off + intercept_off

# average over repeats to get one value per wavelength for plotting
actual_on_mean = spikes_on.reshape(nr_repeats, len(wavelengths)).mean(axis=0)
actual_off_mean = spikes_off.reshape(nr_repeats, len(wavelengths)).mean(axis=0)
predicted_on_mean = predicted_on.reshape(nr_repeats, len(wavelengths)).mean(axis=0)
predicted_off_mean = predicted_off.reshape(nr_repeats, len(wavelengths)).mean(axis=0)

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, actual, predicted, title in zip(
        axes,
        [actual_on_mean, actual_off_mean],
        [predicted_on_mean, predicted_off_mean],
        ["ON", "OFF"],
):
    ax.plot(wavelengths, actual, "o-", label="actual", color="black")
    ax.plot(wavelengths, predicted, "o--", label="predicted", color="red")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised spikes")
    ax.set_title(f"{title} | r2={example[f'r2_{title.lower()}'].item():.2f}")
    ax.legend()

fig.suptitle(
    f"Cell {example['cell_index'].item()}, Recording {example['recordings'].item()}"
)
fig.tight_layout()
fig.show()
print(f"coef ons: {coefs_on}, coef off: {coefs_off}")
# %%
fig, ax = plt.subplots(nrows=5, sharey=True)
psth, bins = histograms.psth_by_index(
    spikes_df.filter(
        (pl.col("recording") == example["recordings"])
        & (pl.col("cell_index") == example["cell_index"])
    ),
    index=["repeat"],
)
for i in range(5):
    ax[i].plot(psth[i, :])
fig.show()
# %%

coef_on = np.stack(results["coef_on"].to_numpy())
coef_on = coef_on / np.max(np.abs(coef_on), axis=1, keepdims=True)
coef_on[np.isnan(coef_on)] = 0
coef_off = np.stack(results["coef_off"].to_numpy())
coef_off = coef_off / np.max(np.abs(coef_off), axis=1, keepdims=True)
coef_off[np.isnan(coef_off)] = 0
# %%
fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
ax[0].bar(cone_names, np.mean(coef_on, axis=0), color=cone_colours)
ax[0].set_title("ON")
ax[1].bar(cone_names, np.mean(coef_off, axis=0), color=cone_colours)
ax[1].set_title("OFF")
ax[0].set_ylabel("cone contributions")
fig.show()
# %%


# %%
nonzero_coeff_on = np.sum(coef_on, axis=1).astype(int) != 0
nonzero_coeff_off = np.sum(coef_off, axis=1).astype(int) != 0
# %%
clustering = AgglomerativeClustering(n_clusters=30, linkage="average", metric="cosine")
labels_on = clustering.fit_predict(
    coef_on[nonzero_coeff_on]
    * results["r2_final_on"].to_numpy()[nonzero_coeff_on, np.newaxis]
)
all_linear_coefs_sorted_on = coef_on[nonzero_coeff_on][np.argsort(labels_on)]
labels_on_complete = np.ones(coef_on.shape[0]) * -1
labels_on_complete[nonzero_coeff_on] = labels_on
np.unique(labels_on_complete, return_counts=True)
# %%
clustering = AgglomerativeClustering(n_clusters=30, linkage="average", metric="cosine")
labels_off = clustering.fit_predict(
    coef_off[nonzero_coeff_off]
    * results["r2_final_off"].to_numpy()[nonzero_coeff_off, np.newaxis]
)
all_linear_coefs_sorted_off = coef_off[nonzero_coeff_off][np.argsort(labels_off)]
np.unique(labels_off, return_counts=True)
labels_off_complete = np.ones(coef_off.shape[0]) * -1
labels_off_complete[nonzero_coeff_off] = labels_off
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(results["r2_final_on"].to_numpy(), results["r2_final_off"].to_numpy())
fig.show()
# %%
# coef_on = coef_on/np.max(np.abs(coef_on), axis=1, keepdims=True)
# coef_on[np.isnan(coef_on)] = 0
# coef_off = coef_off/np.max(np.abs(coef_off), axis=1, keepdims=True)
# coef_off[np.isnan(coef_off)] = 0
# %%
import umato

embedder = umato.UMATO(
    n_neighbors=100,
    min_dist=0,
    n_components=2,
    metric="cosine",
)
embedding_on = embedder.fit_transform(coef_on)
embedding_off = embedder.fit_transform(coef_off)
# %%
from sklearn.cluster import HDBSCAN

clusterer_on = HDBSCAN(
    metric="euclidean", store_centers="centroid", min_cluster_size=20
)
labels_on = clusterer_on.fit_predict(embedding_on[nonzero_coeff_on])
labels_on_complete = np.ones(coef_on.shape[0]) * -1
labels_on_complete[nonzero_coeff_on] = labels_on
np.unique(labels_on_complete, return_counts=True)
# %%
clusterer_off = HDBSCAN(
    metric="euclidean", store_centers="centroid", min_cluster_size=20
)
labels_off = clusterer_off.fit_predict(embedding_off[nonzero_coeff_off])
labels_off_complete = np.ones(coef_off.shape[0]) * -1
labels_off_complete[nonzero_coeff_off] = labels_off
# %%
fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
ax[0].scatter(
    embedding_on[nonzero_coeff_on][:, 0],
    embedding_on[nonzero_coeff_on][:, 1],
    c=labels_on,
    cmap="tab10",
)
ax[1].scatter(
    embedding_off[nonzero_coeff_off][:, 0],
    embedding_off[nonzero_coeff_off][:, 1],
    c=labels_off,
    cmap="tab10",
)
fig.show()
# %%

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
cax = ax[0].matshow(
    all_linear_coefs_sorted_on,
    aspect="auto",
    cmap="PuOr_r",
    vmin=-np.max(np.abs(coef_on)),
    vmax=np.max(np.abs(coef_on)),
)
ax[0].set_xlabel("Cone type")
ax[0].set_xticks(range(6), cone_names)
ax[0].set_ylabel("Cell index")

cax = ax[1].matshow(
    all_linear_coefs_sorted_off,
    aspect="auto",
    cmap="PuOr_r",
    vmin=-np.max(np.abs(coef_off)),
    vmax=np.max(np.abs(coef_off)),
)
ax[1].set_xlabel("Cone type")
ax[1].set_xticks(range(6), cone_names)
ax[1].set_ylabel("Cell index")
fig.show()
# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].matshow(
    clusterer_on.centroids_,
    aspect="auto",
    cmap="PuOr_r",
    vmin=-np.max(np.abs(clusterer_off.centroids_)),
    vmax=np.max(np.abs(clusterer_off.centroids_)),
)
ax[1].matshow(
    clusterer_off.centroids_,
    aspect="auto",
    cmap="PuOr_r",
    vmin=-np.max(np.abs(clusterer_on.centroids_)),
    vmax=np.max(np.abs(clusterer_on.centroids_)),
)
fig.show()
# %%
results = results.with_columns(labels_on=labels_on_complete)
results = results.with_columns(labels_off=labels_off_complete)

# %%
fig, ax = plt.subplots(
    nrows=30,
    ncols=4,
    figsize=(20, 20),
    gridspec_kw={"width_ratios": [1, 10, 1, 10]},
    sharey="col",
)
for l in range(len(np.unique(labels_on_complete)) - 1):
    subset_df = results.filter((pl.col("labels_on") == l))
    if subset_df["r2_final_on"].mean() > 0:
        subset_fff = spikes_df.filter(
            (pl.col("recording").is_in(subset_df["recordings"]))
            & (pl.col("cell_index").is_in(subset_df["cell_index"]))
        )
        psth, bins = histograms.psth(subset_fff, end=np.sum(mean_trigger))
        ax[l, 1].plot(bins[:-1], psth / np.max(psth), c="black")
        ax[l, 0].bar(
            cone_names,
            # clusterer_on.centroids_[l] / np.max(np.abs(clusterer_on.centroids_[l])),
            np.stack(coef_on)[l] / np.max(np.abs(np.stack(coef_on)[l])),
            color=cone_colours,
        )
        ax[l, 0].hlines(0, cone_names[1], cone_names[-1])
        ax[l, 0].spines["top"].set_visible(False)
        ax[l, 0].spines["right"].set_visible(False)
        ax[l, 0].spines["bottom"].set_visible(False)
        ax[l, 0].spines["left"].set_visible(False)
        ax[l, 0].set_axis_off()
        ax[l, 0].hlines(0, cone_names[1], cone_names[-1])
        ax[l, 1].spines["top"].set_visible(False)
        ax[l, 1].spines["right"].set_visible(False)

        ax[l, 1].spines["left"].set_visible(False)
    else:
        ax[l, 0].set_axis_off()
        ax[l, 1].set_axis_off()

for l in range(len(np.unique(labels_off_complete)) - 1):
    subset_df = results.filter((pl.col("labels_off") == l))
    if subset_df["r2_final_off"].mean() > 0:
        subset_fff = spikes_df.filter(
            (pl.col("recording").is_in(subset_df["recordings"]))
            & (pl.col("cell_index").is_in(subset_df["cell_index"]))
        )
        psth, bins = histograms.psth(subset_fff, end=np.sum(mean_trigger))
        ax[l, 3].plot(bins[:-1], psth / np.max(psth), c="black")
        ax[l, 2].bar(
            cone_names,
            # clusterer_off.centroids_[l] / np.max(np.abs(clusterer_off.centroids_[l])),
            np.stack(coef_off)[l] / np.max(np.abs(np.stack(coef_off)[l])),
            color=cone_colours,
        )
        ax[l, 2].hlines(0, cone_names[1], cone_names[-1])
        # remove spines
        ax[l, 2].spines["top"].set_visible(False)
        ax[l, 2].spines["right"].set_visible(False)
        ax[l, 2].spines["bottom"].set_visible(False)
        ax[l, 2].spines["left"].set_visible(False)
        ax[l, 3].spines["top"].set_visible(False)
        ax[l, 3].spines["right"].set_visible(False)
        ax[l, 3].spines["left"].set_visible(False)
        ax[l, 2].set_axis_off()
    else:
        ax[l, 2].set_axis_off()
        ax[l, 3].set_axis_off()
fig.show()
fig.savefig(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis/cluster_overview_elastic_06nd.svg"
)
# %%
fig, ax = plt.subplots(figsize=(20, 10))
subset_df = results.filter((pl.col("labels_on") == 2))
subset_fff = spikes_df.filter(
    (pl.col("recording").is_in(subset_df["recordings"]))
    & (pl.col("cell_index").is_in(subset_df["cell_index"]))
)
psth, bins = histograms.psth_by_index(
    subset_fff, window_end=np.sum(mean_trigger), index=["recording", "cell_index"]
)

ax.imshow(psth, cmap="Greys")
fig.show()
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embedding = pca.fit_transform(coef_off)

fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_off, cmap="tab20")
fig.show()
# %% add results to feature_df
modes = ["Lasso", "Linear"]
feature_df = pd.read_pickle(root / "feature_df")
feature_df = feature_df.loc[multi_index_1]
for mode in modes:
    all_coefs = np.hstack(
        [
            results_dict["ON"][mode]["coefs"],
            results_dict["OFF"][mode]["coefs"],
        ]
    )

    cone_coeff_columns_on = [f"ON_{mode}_{item}" for item in cone_names]
    cone_coeff_columns_off = [f"OFF_{mode}_{item}" for item in cone_names]

    feature_df[cone_coeff_columns_on] = results_dict["ON"][mode]["coefs"]
    feature_df[cone_coeff_columns_off] = results_dict["OFF"][mode]["coefs"]

# %% add pca and tsne
features = feature_df[cone_coeff_columns_on + cone_coeff_columns_off].values
features_scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
pca.fit_transform(features_scaled.T)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(pca.components_[0], pca.components_[1])
fig.show()
print(pca.explained_variance_ratio_)

# %%
tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features_scaled)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(projections[:, 0], projections[:, 1])
fig.show()
# %%
feature_df["tsne_1"] = projections[:, 0]
feature_df["tsne_2"] = projections[:, 1]
feature_df["pca_1"] = pca.components_[0]
feature_df["pca_2"] = pca.components_[1]
# %% clustering
clustering = AgglomerativeClustering(n_clusters=30, linkage="ward")
labels = clustering.fit_predict(features)
max_label = np.max(labels) + 1

feature_df["labels"] = labels
# %% save df
feature_df.to_pickle(root / "features_coeffs_pca")

# %% plot results in tsne and pca space
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].scatter(
    projections[:, 0],
    projections[:, 1],
    c=labels,
    cmap="tab20",
)
axs[1].scatter(
    pca.components_[0],
    pca.components_[1],
    c=labels,
    cmap="tab20",
)
# add titles
axs[0].set_title("t-SNE")
axs[1].set_title("PCA")
fig.show()
# %%
# double_ab_df.loc[double_ab_df["absorption"] < 0.01, "absorption"] = np.nan
# single_ab_df.loc[single_ab_df["absorption"] < 0.01, "absorption"] = np.nan
# %%
# log transform the absorption values
# single_ab_df["absorption"] = np.log(single_ab_df["absorption"])
# double_ab_df["absorption"] = np.log(double_ab_df["absorption"])

# normalize the absorption values to 0-1 per cone
for cone in single_ab_df["cone"].unique():
    single_ab_df.loc[single_ab_df["cone"] == cone, "absorption"] = (
                                                                           single_ab_df.loc[single_ab_df[
                                                                                                "cone"] == cone, "absorption"]
                                                                           - np.nanmin(single_ab_df["absorption"])
                                                                   ) / (np.nanmax(
        single_ab_df["absorption"]) - np.nanmin(single_ab_df["absorption"]))
for cone in double_ab_df["cone"].unique():
    double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"] = (
                                                                           double_ab_df.loc[double_ab_df[
                                                                                                "cone"] == cone, "absorption"]
                                                                           - np.nanmin(double_ab_df["absorption"])
                                                                   ) / (np.nanmax(
        double_ab_df["absorption"]) - np.nanmin(double_ab_df["absorption"]))
# %%
fig = px.line(
    single_ab_df,
    x="wavelength",
    y="absorption",
    color="cone",
    color_discrete_sequence=cone_colours,
)
fig.show(renderer="browser")

# %% Extract individual opsins
wavelengths = [365, 416, 460, 535, 560, 610]
wavelengths = np.flip(wavelengths)
# need to extract the absorption values for each wavelength
# for each cone type
cone_names = []
X = np.zeros((6, 6))
for i, cone in enumerate(single_ab_df["cone"].unique()):
    cone_names.append(cone)
    for j, wavelength in enumerate(wavelengths):
        print(wavelength)
        X[i, j] = single_ab_df.query("cone == @cone & wavelength == @wavelength")[
            "absorption"
        ]
# add double cone
for i, d_cone in enumerate(double_ab_df["cone"].unique()):
    cone_names.append(d_cone)
    for j, wavelength in enumerate(wavelengths):
        X[i + 4, j] = double_ab_df.query("cone == @d_cone & wavelength == @wavelength")[
            "absorption"
        ]
# X[-1, :] = X[4, :] + X[5, :]
# cone_names.append("both_double")
# Max normalize X per cone
X = X / np.nanmax(X, axis=1)[:, None]
# X[np.isnan(X)] = 0.001

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for i in range(6):
    ax.plot(wavelengths, X[i, :], c=cone_colours[i], label=f"cone {i}")
fig.show()

# %%
nr_cells = spikes_summed_all_on.shape[0]
results_dict = {}
results_dict["ON"] = {}
results_dict["OFF"] = {}
results_dict["ON"]["Lasso"] = {}
results_dict["ON"]["Linear"] = {}
results_dict["OFF"]["Lasso"] = {}
results_dict["OFF"]["Linear"] = {}
results_dict["ON"]["Lasso"]["coefs"] = np.zeros((nr_cells, X.shape[1]))
results_dict["ON"]["Lasso"]["goodness_of_fit"] = np.zeros(nr_cells)
results_dict["ON"]["Linear"]["coefs"] = np.zeros((nr_cells, X.shape[1]))
results_dict["ON"]["Linear"]["goodness_of_fit"] = np.zeros(nr_cells)
results_dict["OFF"]["Lasso"]["coefs"] = np.zeros((nr_cells, X.shape[1]))
results_dict["OFF"]["Lasso"]["goodness_of_fit"] = np.zeros(nr_cells)
results_dict["OFF"]["Linear"]["coefs"] = np.zeros((nr_cells, X.shape[1]))
results_dict["OFF"]["Linear"]["goodness_of_fit"] = np.zeros(nr_cells)

# %%
for cell in range(spikes_summed_all_on.shape[0]):
    lasso = LassoCV(cv=5, random_state=0, n_jobs=-1)
    linear = LinearRegression()
    # ON
    lasso.fit(X.T, spikes_summed_all_on[cell, :])
    linear.fit(X.T, spikes_summed_all_on[cell, :])
    results_dict["ON"]["Lasso"]["coefs"][cell, :] = lasso.coef_
    results_dict["ON"]["Lasso"]["goodness_of_fit"][cell] = lasso.score(
        X.T, spikes_summed_all_on[cell, :]
    )
    results_dict["ON"]["Linear"]["coefs"][cell, :] = linear.coef_
    results_dict["ON"]["Linear"]["goodness_of_fit"][cell] = linear.score(
        X.T, spikes_summed_all_on[cell, :]
    )
    # OFF
    lasso.fit(X.T, spikes_summed_all_off[cell, :])
    linear.fit(X.T, spikes_summed_all_off[cell, :])
    results_dict["OFF"]["Lasso"]["coefs"][cell, :] = lasso.coef_
    results_dict["OFF"]["Lasso"]["goodness_of_fit"][cell] = lasso.score(
        X.T, spikes_summed_all_off[cell, :]
    )
    results_dict["OFF"]["Linear"]["coefs"][cell, :] = linear.coef_
    results_dict["OFF"]["Linear"]["goodness_of_fit"][cell] = linear.score(
        X.T, spikes_summed_all_off[cell, :]
    )
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.hist(
    results_dict["OFF"]["Lasso"]["goodness_of_fit"], bins=100, label="goodness of fit"
)
fig.show()
# %% create array of on and off coefficients lasso
all_coefs = np.hstack(
    [
        results_dict["ON"]["Lasso"]["coefs"],
        results_dict["OFF"]["Lasso"]["coefs"],
    ]
)
# %% load feature_df
feature_df = pd.read_pickle(root / "feature_df_corrected")
feature_df = feature_df.loc[multi_index_1]
# %%
cone_coeff_columns_on = [f"on_Lasso_{item}" for item in cone_names]
cone_coeff_columns_off = [f"off_Lasso_{item}" for item in cone_names]
# %%
feature_df[cone_coeff_columns_on] = results_dict["ON"]["Lasso"]["coefs"]
feature_df[cone_coeff_columns_off] = results_dict["OFF"]["Lasso"]["coefs"]

# %%
features = feature_df.values
features[features == np.inf] = 0
features[features == -np.inf] = 0
feaures_scaled = StandardScaler().fit_transform(features)
# %%
pca = PCA(n_components=2)
pca.fit_transform(feaures_scaled.T)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(pca.components_[0], pca.components_[1])
fig.show()
print(pca.explained_variance_ratio_)

# %%
tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(projections[:, 0], projections[:, 1])
fig.show()
# %%
feature_df["tsne_1"] = projections[:, 0]
feature_df["tsne_2"] = projections[:, 1]
feature_df["pca_1"] = pca.components_[0]
feature_df["pca_2"] = pca.components_[1]
# %%
clustering = AgglomerativeClustering(n_clusters=30, linkage="ward")
labels = clustering.fit_predict(features)
max_label = np.max(labels) + 1
# %%
clustering = HDBSCAN(min_cluster_size=5)
labels = clustering.fit_predict(projections)
print(np.max(labels) + 1)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(projections[:, 0], projections[:, 1], c=labels, cmap="tab20")
fig.show()
# %%
feature_df["labels"] = labels
feature_df.to_pickle(root / "features_coeffs_pca")

# %%
feature_df["labels"] = labels

pd.to_pickle(feature_df["labels"], root / "coeff_label_df")

# %% or external labels
label_df = pd.read_csv(root / "labels.csv")
label_df = label_df.set_index(["recording", "cell_index"])
labels = label_df.loc[multi_index_1]["labels_combined"].values
max_label = np.max(labels) + 1
# %% add label sums
label_sums = []
previous = 0
for i in range(max_label):
    label_sums.append(previous + np.sum(labels == i))
    previous = label_sums[-1]

# %%
sub_index = psths_df.index.intersection(spikes_summed_df.index)
psths_df = psths_df.loc[sub_index]
# %% PLot ON OFF Lasso (figure has 30 rows, but the matshows span all rows)
mode = "Lasso"
fig = plt.figure(figsize=(10, max_label))
gs = gridspec.GridSpec(
    max_label + 1,
    3,
)
ax_on_mat = fig.add_subplot(gs[:, 0])
ax_on_mat.matshow(
    results_dict["ON"][mode]["coefs"][np.argsort(labels), :],
    aspect="auto",
    cmap="PuOr",
    vmin=-np.max(np.abs(all_coefs)),
    vmax=np.max(np.abs(all_coefs)),
)
# add red lines to separate clusters


ax_off_mat = fig.add_subplot(gs[:, 1])
ax_off_mat.matshow(
    results_dict["OFF"][mode]["coefs"][np.argsort(labels), :],
    aspect="auto",
    cmap="PuOr",
    vmin=-np.max(np.abs(all_coefs)),
    vmax=np.max(np.abs(all_coefs)),
)
# add subplot titles
ax_on_mat.set_title(f"ON {mode}")
ax_off_mat.set_title(f"OFF {mode}")
# add red lines to separate clusters
for i in range(max_label):
    ax_on_mat.axhline(label_sums[i], color="r", linewidth=0.5)
    ax_off_mat.axhline(label_sums[i], color="r", linewidth=0.5)
    ax_on_mat.text(-1, label_sums[i] - 0.5, f"C {i}", color="r")
    ax_off_mat.text(-1, label_sums[i] - 0.5, f"C {i}", color="r")
    ax = fig.add_subplot(gs[i, 2])
    # get the cell index
    sum_indices = spikes_summed_df.loc[labels == i].index
    # get the psth index
    psth_indices = psths_df.loc[labels == i]["psth_index"].values
    psth_mean = np.mean(psths[psth_indices], axis=0)
    ax.plot(bins[:-1], psth_mean, c="k")
    ax.set_title(f"Cluster {i}, n={len(sum_indices)}", y=0.5, x=1)
    ax.set_xticks([])
    ax.set_yticks([])
    # remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

ax = fig.add_subplot(gs[max_label, 2])
duration = 2
for i in range(12):
    ax.fill_between(
        [i * duration, (i + 1) * duration],
        0,
        1,
        color=CT.colours[i],
        alpha=0.5,
        label=f"{i}",
    )
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.show()
fig.savefig(root / f"{mode}_coeffs.png")
# %% PLot ON OFF Linear
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].matshow(
    results_dict["ON"]["Linear"]["coefs"][np.argsort(labels), :],
    aspect="auto",
    cmap="PuOr",
    vmin=-np.max(np.abs(all_coefs)),
    vmax=np.max(np.abs(all_coefs)),
)
ax[1].matshow(
    results_dict["OFF"]["Linear"]["coefs"][np.argsort(labels), :],
    aspect="auto",
    cmap="PuOr",
    vmin=-np.max(np.abs(all_coefs)),
    vmax=np.max(np.abs(all_coefs)),
)
# add subplot titles
ax[0].set_title("ON linear")
ax[1].set_title("OFF linear")
fig.show()
# %%
clustering = AgglomerativeClustering(n_clusters=20, linkage="ward")
labels = clustering.fit_predict(linear_coefs)
all_linear_coefs_sorted = linear_coefs[np.argsort(labels), :]
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 10))
cax = ax.matshow(
    all_linear_coefs_sorted,
    aspect="auto",
    cmap="PuOr",
    vmin=-np.max(np.abs(all_coefs)),
    vmax=np.max(np.abs(all_coefs)),
)
fig.colorbar(cax)
ax.set_xlabel("Cone type")
ax.set_xticks(range(6), cone_names)
ax.set_ylabel("Cell index")
fig.show()
