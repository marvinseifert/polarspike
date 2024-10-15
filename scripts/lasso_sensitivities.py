from polarspike import (
    Overview,
    colour_template,
    stimulus_spikes,
    histograms,
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

# %%
stimulus_name = "fff"
window = 0.5
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
# %%
root = Path(r"A:\Marvin\fff_clustering")
recordings = Overview.Recording_s.load(root / "records")
# %%
spikes_df = recordings.get_spikes_df("fff_filtered", pandas=False)
# %%
fff_stimulus = recordings.stimulus_df.query("stimulus_name == @stimulus_name")
recordings.dataframes["fff_stimulus"] = fff_stimulus

# %%
mean_trigger = stimulus_spikes.mean_trigger_times(fff_stimulus, ["all"])

# %%
spikes_rec = spikes_df.partition_by("recording")

# %%
spikes_summed_all = []
cell_indices_all = []
recordings = []
recordings_psth = []
psths = []
psths_cells = []
for recording in spikes_rec:
    spikes_summed, cell_indices = count_spikes.sum_spikes(
        recording, mean_trigger, window=window, group_by="cell_index"
    )
    spikes_summed_all.append(spikes_summed)
    recordings = recordings + [recording["recording"][0]] * len(cell_indices)
    cell_indices_all = cell_indices_all + cell_indices.flatten().tolist()
    psth, bins, cell_index = histograms.psth_by_index(
        recording, index=["cell_index"], return_idx=True, window_end=24
    )
    psths.append(psth)
    psths_cells = psths_cells + cell_index.flatten().tolist()
    recordings_psth = recordings_psth + [recording["recording"][0]] * len(cell_index)

# %%
spikes_summed_indices = np.vstack([recordings, cell_indices_all])
psths_indices = np.vstack([recordings_psth, psths_cells])
# %%
psths = np.vstack(psths)
# %%
multi_index_1 = pd.MultiIndex.from_arrays(
    [recordings, np.asarray(cell_indices_all, dtype=int)],
    names=("recording", "cell_index"),
)
multi_index_2 = pd.MultiIndex.from_arrays(
    [recordings_psth, np.asarray(psths_cells)], names=("recording", "cell_index")
)
# store indices in a dataframe
spikes_summed_df = pd.DataFrame(
    np.arange(spikes_summed_indices.shape[1]),
    index=multi_index_1,
    columns=["sum_index"],
)
psths_df = pd.DataFrame(
    np.arange(psths_indices.shape[1]), index=multi_index_2, columns=["psth_index"]
)


# %%
spikes_summed_all = np.vstack(spikes_summed_all)
spikes_summed_all_on = np.vstack(spikes_summed_all[:, ::2])
spikes_summed_all_off = np.vstack(spikes_summed_all[:, 1::2])
# %% normalize each trace to the maximum
spikes_summed_all_on = spikes_summed_all_on / spikes_summed_all_on.max(axis=1)[:, None]
spikes_summed_all_off = (
    spikes_summed_all_off / spikes_summed_all_off.max(axis=1)[:, None]
)
spikes_summed_all_on[np.isnan(spikes_summed_all_on)] = 0
spikes_summed_all_off[np.isnan(spikes_summed_all_off)] = 0
# %% Import sensitivity curves
double_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_double")
single_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_oil")

# %%
double_ab_df.loc[double_ab_df["absorption"] < 0.01, "absorption"] = np.nan
single_ab_df.loc[single_ab_df["absorption"] < 0.01, "absorption"] = np.nan

single_ab_df["absorption"] = np.log(single_ab_df["absorption"])
double_ab_df["absorption"] = np.log(double_ab_df["absorption"])

# normalize the absorption values to 0-1 per cone
for cone in single_ab_df["cone"].unique():
    single_ab_df.loc[single_ab_df["cone"] == cone, "absorption"] = (
        single_ab_df.loc[single_ab_df["cone"] == cone, "absorption"]
        - np.nanmin(single_ab_df["absorption"])
    ) / (np.nanmax(single_ab_df["absorption"]) - np.nanmin(single_ab_df["absorption"]))
for cone in double_ab_df["cone"].unique():
    double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"] = (
        double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"]
        - np.nanmin(double_ab_df["absorption"])
    ) / (np.nanmax(double_ab_df["absorption"]) - np.nanmin(double_ab_df["absorption"]))

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
X[np.isnan(X)] = 0.001

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
    results_dict["OFF"]["Lasso"]["goodness_of_fit"], bins=100, label="OFF", alpha=0.5
)
ax.hist(results_dict["ON"]["Lasso"]["goodness_of_fit"], bins=100, label="ON", alpha=0.5)
ax.set_xlabel("Goodness of fit")
ax.set_ylabel("Number of cells")
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
        single_ab_df.loc[single_ab_df["cone"] == cone, "absorption"]
        - np.nanmin(single_ab_df["absorption"])
    ) / (np.nanmax(single_ab_df["absorption"]) - np.nanmin(single_ab_df["absorption"]))
for cone in double_ab_df["cone"].unique():
    double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"] = (
        double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"]
        - np.nanmin(double_ab_df["absorption"])
    ) / (np.nanmax(double_ab_df["absorption"]) - np.nanmin(double_ab_df["absorption"]))
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
