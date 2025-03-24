import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from polarspike import colour_template, Overview, histograms
from tslearn.neighbors import KNeighborsTimeSeries
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN, MeanShift
from sklearn.metrics import pairwise_distances
from sklearn import linear_model
from quality_tests import spiketrain_qi

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%
root = Path(r"A:\Marvin\fff_clustering")
feauture_df = pd.read_pickle(root / "features_coeffs_pca")
# %%
drop_columns = [
    "OFF_Linear_LWS",
    "OFF_Linear_SWS1",
    "OFF_Linear_SWS2",
    "OFF_Linear_MWS",
    "OFF_Linear_principal",
    "OFF_Linear_accessory",
    "ON_Linear_LWS",
    "ON_Linear_SWS1",
    "ON_Linear_SWS2",
    "ON_Linear_MWS",
    "ON_Linear_principal",
    "ON_Linear_accessory",
    "pca_1",
    "pca_2",
    "tsne_1",
    "tsne_2",
    "labels",
]
feauture_df = feauture_df.drop(drop_columns, axis=1)


# %% add chirp data
chirp_features = pd.read_pickle(r"A:\Marvin\chirps\chirp_analysis_df_mean.pkl")
for column in chirp_features.columns:
    if column != "stimulus_name":
        feauture_df[column] = 0.0
    else:
        continue


# %%
# add chirp data
feauture_df.update(chirp_features.query("stimulus_name=='chirp'"))
# %% normalize ON and OFF coefficients
on_lasso = [
    "ON_Lasso_LWS",
    "ON_Lasso_MWS",
    "ON_Lasso_SWS2",
    "ON_Lasso_SWS1",
    "ON_Lasso_principal",
    "ON_Lasso_accessory",
]
off_lasso = [
    "OFF_Lasso_LWS",
    "OFF_Lasso_MWS",
    "OFF_Lasso_SWS2",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_principal",
    "OFF_Lasso_accessory",
]
# %%
feauture_df[on_lasso] = feauture_df[on_lasso].div(
    feauture_df[on_lasso].max(axis=1), axis=0
)
feauture_df[off_lasso] = feauture_df[off_lasso].div(
    feauture_df[off_lasso].max(axis=1), axis=0
)
# %% fill nans with 0
feauture_df = feauture_df.fillna(0)
# fill infs with 1
feauture_df = feauture_df.replace([np.inf], 1)
feauture_df = feauture_df.replace([-np.inf], -1)
# %% Standardize the data
scaler = StandardScaler()
feautures = scaler.fit_transform(feauture_df.values)

# %% Clustering purely on Lasso coefficients
tsne = TSNE(n_components=2, perplexity=10, n_iter=10000)
tsne_results = tsne.fit_transform(feautures)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(tsne_results[:, 0], tsne_results[:, 1])
fig.show()
# %%
clustering = HDBSCAN(min_cluster_size=3, min_samples=5)
labels = clustering.fit_predict(tsne_results)
# %%
cluster_labels = np.unique(labels[labels != -1])
cluster_centers = np.array(
    [feautures[labels == label].mean(axis=0) for label in cluster_labels]
)

# Step 2: Assign outliers to nearest cluster center
outliers = feautures[labels == -1]
distances = pairwise_distances(outliers, cluster_centers)
nearest_clusters = cluster_labels[distances.argmin(axis=1)]

# Step 3: Update outlier labels
labels[labels == -1] = nearest_clusters
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="tab20")
fig.show()
# %% add labels to df
feauture_df["labels"] = labels
pd.to_pickle(feauture_df, root / "features_coeffs_labels_15")
# %% Generate some more features
cones = ["SWS1", "SWS2", "MWS", "LWS", "principal", "accessory"]
conditions = ["ON", "OFF"]
model = ["Lasso", "Linear"]
leds = ["610", "560", "535", "460", "413", "365"]
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
cone_colours = CT.colours[[0, 4, 6, 8]].tolist() + ["orange", "darkorange"]


# %% Check how spectrally narrow the cells are
def narrow_tuning(row):
    columns_on = [f"{conditions[0]}_{model[0]}_{cone}" for cone in cones]
    all_values_on = row[columns_on].values
    columns_off = [f"{conditions[1]}_{model[0]}_{cone}" for cone in cones]
    all_values_off = row[columns_off].values
    all_values = np.abs(all_values_on) + np.abs(all_values_off)
    # Divide the second largest value by the largest value
    all_values = np.sort(all_values)
    all_values = np.round(all_values, 2)
    if np.sum(all_values[:-1]) == 0:
        return 0
    else:
        return (np.sum(all_values[:-1]) - all_values[-1]) / (
            np.sum(all_values[:-1]) + all_values[-1]
        )


def on_off_index(row):
    for led in leds:
        if (
            row[f"{led}_max"] + row[f"{led}_off_max"]
        ) == 0:  # Avoid division by zero error
            row[f"{led}_on_off_index"] = 0
            continue
        polarity = (row[f"{led}_max"] - row[f"{led}_off_max"]) / (
            row[f"{led}_max"] + row[f"{led}_off_max"]
        )
        row[f"{led}_on_off_index"] = polarity


# %%
feauture_df["tuning_narrowness"] = feauture_df.apply(narrow_tuning, axis=1)
feauture_df = feauture_df.sort_values("tuning_narrowness", ascending=True)
# %%
fff_df = pd.read_pickle(root / "fff_df.pkl")
narrow_cells = np.stack(
    fff_df.loc[feauture_df.loc[feauture_df["tuning_narrowness"] > 40].index].values,
)
fig, axs = plt.subplots(nrows=11, figsize=(20, 10), sharex=True)
for i in range(10):
    axs[i].plot(np.arange(0, 23.95, 0.05), narrow_cells[i, :][0])
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %%
for led in leds:
    polarity = (feauture_df[f"{led}_max"] - feauture_df[f"{led}_off_max"]) / (
        feauture_df[f"{led}_max"] + feauture_df[f"{led}_off_max"]
    )
    feauture_df[f"{led}_on_off_index"] = polarity
# %% Check how spectrally broad the cells are
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(feauture_df["tuning_narrowness"], bins=50)
ax.set_xlabel("Tuning Narrowness")
ax.set_ylabel("Number of Cells")
ax.vlines(np.mean(feauture_df["tuning_narrowness"]), 0, 100, color="red")
fig.show()

# %%
all_corr = feauture_df.corr("spearman").stack().reset_index(name="correlation")
g = sns.relplot(
    height=20,
    data=all_corr,
    x="level_0",
    y="level_1",
    hue="correlation",
    size="correlation",
    palette="Spectral_r",
    hue_norm=(-1, 1),
    edgecolor=".7",
    sizes=(50, 250),
    size_norm=(-0.2, 0.8),
)
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(0.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
g.fig.show()
g.fig.savefig(root / "correlation_matrix.png")

# %%
for cluster in feauture_df["labels"].unique():
    sub_df = feauture_df.query(f"labels =={cluster}")
    all_corr = sub_df.corr().stack().reset_index(name="correlation")
    g = sns.relplot(
        height=20,
        data=all_corr,
        x="level_0",
        y="level_1",
        hue="correlation",
        size="correlation",
        palette="Spectral_r",
        hue_norm=(-1, 1),
        edgecolor=".7",
        sizes=(50, 250),
        size_norm=(-0.2, 0.8),
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(0.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    # g.fig.show()
    g.fig.savefig(root / f"cluster_correlations/correlation_matrix_{cluster}.png")
# %% take a subset of columns
sub_columns = ["OFF_Lasso_principal"]
sub_corr = all_corr.query("level_0 in @sub_columns")
# drop self correlation
sub_corr = sub_corr.query("level_0 != level_1")
# g = sns.relplot(
#     height=30,
#     data=sub_corr,
#     x="level_0",
#     y="level_1",
#     hue="correlation",
#     size="correlation",
#     palette="Spectral_r",
#     hue_norm=(-1, 1),
#     edgecolor=".7",
#     sizes=(50, 250),
#     size_norm=(-0.2, 0.8),
# )
# g.set(xlabel="", ylabel="", aspect="equal")
# g.despine(left=True, bottom=True)
# g.ax.margins(0.02)
# for label in g.ax.get_xticklabels():
#     label.set_rotation(90)
# g.fig.show()
fig, ax = plt.subplots(figsize=(30, 10))
ax.bar(sub_corr["level_1"], sub_corr["correlation"])
for label in ax.get_xticklabels():
    label.set_rotation(90)
# expand margins
fig.subplots_adjust(bottom=0.3)
fig.suptitle(f"Correlation {sub_columns} Cell Coefficients")
fig.show()

# %%
fig, ax = plt.subplots(ncols=2, figsize=(10, 30))
for idx, column in enumerate(on_lasso):
    sub_corr = all_corr.query("level_0 in @column")
    # drop self correlation
    sub_corr = sub_corr.query("level_0 != level_1")

    ax[0].scatter(
        sub_corr["correlation"], sub_corr["level_1"], c=cone_colours[idx], label=column
    )
    ax[0].set_title("ON Lasso")
    ax[0].set_xlabel("Correlation")
    ax[0].set_ylabel("indices")

for idx, column in enumerate(off_lasso):
    sub_corr = all_corr.query("level_0 in @column")
    # drop self correlation
    sub_corr = sub_corr.query("level_0 != level_1")

    ax[1].scatter(
        sub_corr["correlation"], sub_corr["level_1"], c=cone_colours[idx], label=column
    )
    ax[1].set_title("ON Lasso")
    ax[1].set_xlabel("Correlation")
    ax[1].set_ylabel("indices")
# # turn x labels
# for label in ax[0].get_xticklabels():
#     label.set_rotation(90)
# for label in ax[1].get_xticklabels():
#     label.set_rotation(90)
# # show legend
# hide ticks in second plot
ax[1].set_yticks([])
# set x limits to -1, 1
ax[0].set_xlim(-1, 1)
ax[1].set_xlim(-1, 1)
# add zero line
ax[0].axvline(0, color="black", linestyle="-")
ax[1].axvline(0, color="black", linestyle="-")
# add line at 0.5 and -.5
ax[0].axvline(0.5, color="black", linestyle="--")
ax[1].axvline(0.5, color="black", linestyle="--")
ax[0].axvline(-0.5, color="black", linestyle="--")
ax[1].axvline(-0.5, color="black", linestyle="--")
# add legend outside of plot
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
# expand margins
fig.subplots_adjust(left=0.3)
fig.subplots_adjust(right=0.7)

fig.show()


# %%
cluster_subset = feauture_df.query("labels == 9")
all_corr = cluster_subset.corr("spearman").stack().reset_index(name="correlation")
# %%
fig, ax = plt.subplots(ncols=2, figsize=(10, 30))
for idx, column in enumerate(on_lasso):
    sub_corr = all_corr.query("level_0 in @column")
    # drop self correlation
    sub_corr = sub_corr.query("level_0 != level_1")

    ax[0].scatter(
        sub_corr["correlation"], sub_corr["level_1"], c=cone_colours[idx], label=column
    )
    ax[0].set_title("ON Lasso")
    ax[0].set_xlabel("Correlation")
    ax[0].set_ylabel("indices")

for idx, column in enumerate(off_lasso):
    sub_corr = all_corr.query("level_0 in @column")
    # drop self correlation
    sub_corr = sub_corr.query("level_0 != level_1")

    ax[1].scatter(
        sub_corr["correlation"], sub_corr["level_1"], c=cone_colours[idx], label=column
    )
    ax[1].set_title("ON Lasso")
    ax[1].set_xlabel("Correlation")
    ax[1].set_ylabel("indices")
# # turn x labels
# for label in ax[0].get_xticklabels():
#     label.set_rotation(90)
# for label in ax[1].get_xticklabels():
#     label.set_rotation(90)
# # show legend
# hide ticks in second plot
ax[1].set_yticks([])
# set x limits to -1, 1
ax[0].set_xlim(-1, 1)
ax[1].set_xlim(-1, 1)
# add zero line
ax[0].axvline(0, color="black", linestyle="-")
ax[1].axvline(0, color="black", linestyle="-")
# add line at 0.5 and -.5
ax[0].axvline(0.5, color="black", linestyle="--")
ax[1].axvline(0.5, color="black", linestyle="--")
ax[0].axvline(-0.5, color="black", linestyle="--")
ax[1].axvline(-0.5, color="black", linestyle="--")
# add legend outside of plot
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
# expand margins
fig.subplots_adjust(left=0.3)
fig.subplots_adjust(right=0.7)

fig.show()

# %%
sub_df = feauture_df.query("threshold_freq > 0")
to_test = "threshold_freq"
sub_df = sub_df.query(f"{to_test} > 0")
g = sns.lmplot(
    data=sub_df,
    x=to_test,
    y="610_tr_sus_index",
    height=10,
    aspect=1,
    scatter_kws={"s": 5, "color": ".15"},
    order=1,
)
g.fig.show()
# %% subset of columns
lasso_columns = [
    "ON_Lasso_LWS",
    "ON_Lasso_SWS1",
    "ON_Lasso_SWS2",
    "ON_Lasso_MWS",
    "ON_Lasso_principal",
    "ON_Lasso_accessory",
    "OFF_Lasso_LWS",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_SWS2",
    "OFF_Lasso_MWS",
    "OFF_Lasso_principal",
    "OFF_Lasso_accessory",
]
linear_columns = [
    "ON_Linear_LWS",
    "ON_Linear_SWS1",
    "ON_Linear_SWS2",
    "ON_Linear_MWS",
    "ON_Linear_principal",
    "ON_Linear_accessory",
    "OFF_Linear_LWS",
    "OFF_Linear_SWS1",
    "OFF_Linear_SWS2",
    "OFF_Linear_MWS",
    "OFF_Linear_principal",
    "OFF_Linear_accessory",
]
# %%
principal_colulmns = [
    "ON_Lasso_principal",
    "OFF_Lasso_principal",
    "sparsity",
    "maxima",
    "medium_firing_rate",
    "610_tr_sus_index",
    "560_tr_sus_index",
    "535_tr_sus_index",
    "460_tr_sus_index",
    "413_tr_sus_index",
    "365_tr_sus_index",
    "threshold_freq",
    "threshold_power",
    "power_max",
    "frequency",
    "tuning_narrowness",
]
test_columns = [
    "sparsity",
    "maxima",
    "medium_firing_rate",
    "610_tr_sus_index",
    "560_tr_sus_index",
    "535_tr_sus_index",
    "460_tr_sus_index",
    "413_tr_sus_index",
    "365_tr_sus_index",
    "threshold_freq",
    "threshold_power",
    "power_max",
    "frequency",
    "tuning_narrowness",
]
mws_colulmns = [
    "ON_Lasso_MWS",
    "ON_Lasso_MWS",
    "OFF_Lasso_MWS",
    "OFF_Lasso_MWS",
    "sparsity",
    "maxima",
    "medium_firing_rate",
    "610_tr_sus_index",
    "560_tr_sus_index",
    "535_tr_sus_index",
    "460_tr_sus_index",
    "413_tr_sus_index",
    "365_tr_sus_index",
]

all_corr = (
    feauture_df[principal_colulmns].corr().stack().reset_index(name="correlation")
)
g = sns.relplot(
    height=15,
    data=all_corr,
    x="level_0",
    y="level_1",
    hue="correlation",
    size="correlation",
    palette="Spectral_r",
    hue_norm=(-1, 1),
    edgecolor=".7",
    sizes=(50, 250),
    size_norm=(-0.2, 0.8),
)
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(0.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
g.fig.show()
# g.fig.savefig(root / "correlation_matrix.png")
# %%
columns = test_columns
for to_test in lasso_columns:
    # Create x*x lmplots for all combinations
    all_combinations = []
    for i in range(len(columns)):
        all_combinations.append((columns[i], to_test))

    save_dir = Path(rf"A:\Marvin\fff_clustering\cone_correlations_ransac\{to_test}")
    save_dir.mkdir(exist_ok=True)
    figs = []
    for combination in all_combinations:
        if combination[0] == combination[1]:
            continue
        sub_df = feauture_df[[combination[0], combination[1]]].copy()
        sub_df = sub_df.query(f"{combination[1]}!=0")
        if combination[0] == "frequency":
            sub_df = sub_df.query(f"{combination[0]}>0")
        if combination[0] == "threshold_freq":
            sub_df = sub_df.query(f"{combination[0]}>0")
        if combination[0] == "threshold_power":
            sub_df = sub_df.query(f"{combination[0]}>0")
        if combination[0] == "power_max":
            sub_df = sub_df.query(f"{combination[0]}>0")
        max_x = sub_df[combination[0]].max()
        max_y = sub_df[combination[1]].max()
        min_x = sub_df[combination[0]].min()
        min_y = sub_df[combination[1]].min()
        # fill 0 values with nan and drop them
        # sub_df = sub_df.replace(0, np.nan).dropna()
        # fitt RANSAC regression
        ransac = linear_model.RANSACRegressor()
        ransac.fit(
            sub_df[combination[0]].values.reshape(-1, 1), sub_df[combination[1]].values
        )
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_X = np.linspace(min_x, max_x, 100)
        line_y_ransac = ransac.predict(line_X.reshape(-1, 1))
        fig, axs = plt.subplots(ncols=2, figsize=(20, 10), sharex=True, sharey=True)
        axs[0].scatter(
            sub_df[combination[0]],
            sub_df[combination[1]],
            c="black",
            label="Data",
            s=5,
        )
        axs[0].plot(
            line_X,
            line_y_ransac,
            color="blue",
            label="RANSAC regressor",
            linewidth=2,
        )
        # Only plot the inliers
        axs[1].scatter(
            sub_df[combination[0]][inlier_mask],
            sub_df[combination[1]][inlier_mask],
            c="blue",
            label="Inliers",
            s=5,
        )
        axs[1].scatter(
            sub_df[combination[0]][outlier_mask],
            sub_df[combination[1]][outlier_mask],
            c="black",
            label="Outliers",
            s=5,
        )
        axs[1].plot(
            line_X,
            line_y_ransac,
            color="blue",
            label="RANSAC regressor",
            linewidth=2,
        )
        # print the ransac score
        axs[1].text(
            0.05,
            0.95,
            f"RANSAC score: {ransac.score(sub_df[combination[0]].values.reshape(-1, 1), sub_df[combination[1]].values)}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axs[1].transAxes,
        )
        # print the score only for the inliers
        axs[1].text(
            0.05,
            0.9,
            f"RANSAC score inliers: {ransac.score(sub_df[combination[0]][inlier_mask].values.reshape(-1, 1), sub_df[combination[1]][inlier_mask].values)}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axs[1].transAxes,
        )

        axs[0].set_xlabel(combination[0])
        axs[0].set_ylabel(combination[1])
        axs[0].set_title(f"{combination[0]} vs {combination[1]}")
        axs[0].legend()
        fig.savefig(save_dir / f"{combination[0]}_{combination[1]}.png")

        # g = sns.lmplot(
        #     data=sub_df,
        #     x=combination[0],
        #     y=combination[1],
        #     height=10,
        #     aspect=1,
        #     scatter_kws={"s": 5, "color": ".15"},
        # )

    # g.fig.savefig(save_dir / f"{combination[0]}_{combination[1]}.png")

# %%
r_squares = np.zeros(len(all_combinations))
ps = np.zeros_like(r_squares)
for idx, combination in enumerate(all_combinations):
    if combination[0] == combination[1]:
        continue
    try:
        X = sm.add_constant(feauture_df[combination[0]])  # Add intercept
        model = sm.OLS(feauture_df[combination[1]], X).fit()
        r_squares[idx] = model.rsquared
        ps[idx] = model.pvalues[combination[0]]
    except statsmodels.tools.sm_exceptions.MissingDataError:
        continue

# %% Create df
r_square_df = pd.DataFrame(all_combinations, columns=["x", "y"])
r_square_df["r_squared"] = r_squares
r_square_df["p_value"] = ps
r_square_df = r_square_df.sort_values("r_squared", ascending=False)
# drop every second row
r_square_df = r_square_df.iloc[::2]
# %% plot as matrix. need to reshape the data
r_significant = r_square_df.query("p_value < 0.005")

# %%
r_sig_lasso = r_significant.query("x in @lasso_columns")
r_sig_linear = r_significant.query("x in @linear_columns")
# %% add the figures to a single subplot
figs[4].fig.show()
# %% compare to single feature
# %%
to_test = "medium_firing_rate"
figs = []
for column in columns:
    sub_df = feauture_df[[column, to_test]].copy()
    # fill 0 values with nan and drop them
    # sub_df = sub_df.replace(0, np.nan).dropna()
    g = sns.lmplot(
        data=sub_df,
        x=column,
        y=to_test,
        height=10,
        aspect=1,
        scatter_kws={"s": 5, "color": ".15"},
    )
    g.fig.savefig(root / f"cone_correlations/{column}_{to_test}.png")

# %% Check if I can find any cells with only OFF_Lasso_principal contributions

columns = [
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
test_condition = "ON_Lasso_principal"

sub_df = feauture_df[columns].copy()
columns.remove(test_condition)
sub_df["sum"] = sub_df[columns].abs().sum(axis=1)
# %%
low_df = sub_df.query(f"sum < 0.1 & {test_condition} > 0.1")
print(len(low_df))
# %%
recordings = Overview.Recording_s.load(root / "records")
recordings.dataframes["low_df"] = (
    recordings.dataframes["spikes_df"]
    .query("stimulus_name=='fff'")
    .set_index(["recording", "cell_index"])
    .loc[low_df.index]
    .reset_index()
)
# %%
spikes = recordings.get_spikes_df("low_df", pandas=False)
psth, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=24.1,
    bin_size=0.1,
)
# %%
fig, axs = plt.subplots(psth.shape[0] + 1, 1, figsize=(20, 10), sharex=True)
for idx, trace in enumerate(psth):
    axs[idx].plot(bins[:-1], trace, c="black")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
# add title
fig.suptitle(f"Cells with high {test_condition} and low other cone contributions")
fig.show()

# %% save psth
np.save(root / f"{test_condition}_example_psths.npy", psth)

# %% test for cells with equal contributions
columns = [
    "OFF_Lasso_principal",
    "OFF_Lasso_accessory",
    "OFF_Lasso_LWS",
    "OFF_Lasso_MWS",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_SWS2",
]
sub_df = feauture_df[columns].copy()
values = sub_df.values
# %% normalize 0-1 and check which rows have a sum equal to nr of columns
values = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
sums = np.sum(values, axis=1)
# %%
equal_df = sub_df.loc[sums > 2.8]
print(len(equal_df))
# %%
recordings.dataframes["equal_df"] = (
    recordings.dataframes["spikes_df"]
    .query("stimulus_name=='fff'")
    .set_index(["recording", "cell_index"])
    .loc[equal_df.index]
    .reset_index()
)
spikes = recordings.get_spikes_df("equal_df", pandas=False)
psth, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=24.1,
    bin_size=0.1,
)
# %%
fig, axs = plt.subplots(psth.shape[0] + 1, 1, figsize=(20, 10), sharex=True)
for idx, trace in enumerate(psth):
    axs[idx].plot(bins[:-1], trace, c="black")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
# add title
fig.suptitle(f"Cells with high {test_condition} and low other cone contributions")
fig.show()
