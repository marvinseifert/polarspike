import pandas as pd
import spiketrain_plots
from holoviews.plotting.bokeh.styles import marker
from polarspike import (
    Overview,
    histograms,
    Opsins,
    colour_template,
    stimulus_spikes,
    spiketrain_plots,
)
from polarspike.analysis import response_peaks, count_spikes
import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances
import matplotlib.pyplot as plt
from scipy.signal import correlate
import polars as pl
from bokeh.io.export import export_svgs
import panel as pn
from scipy.stats import zscore

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
OT = Opsins.Opsin_template()

# double_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_double")
# single_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_oil")

# %%
recording = Overview.Recording.load(
    r"/home/mawa/nas_a/Marvin/chicken_14_05_2025/Phase_00/overview"
)

recording.dataframes["oil_no_oil"] = recording.spikes_df.query(
    "stimulus_name=='fff' | stimulus_name=='fff_top'"
)
# %%
spikes_oil = recording.get_spikes_triggered(
    [{"stimulus_name": "fff"}], carry=["qi"], pandas=False
)
spikes_no_oil = recording.get_spikes_triggered(
    [{"stimulus_name": "fff_top"}], carry=["qi"], pandas=False
)

# %%
matching_trigger = [1, 2, 3, 5, 6, 7]
missing_trigger = [0, 4]
spikes_oil = spikes_oil.filter(pl.col("trigger").is_in(matching_trigger))
# %% we need to subtract 2 seconds from "times_triggered" column for all triggers smaller than every entry in "missing_trigger"
spikes_oil = spikes_oil.lazy()
for trigger in missing_trigger:
    spikes_oil = spikes_oil.with_columns(
        pl.when(pl.col("trigger") > trigger)
        .then(pl.col("times_triggered") - 2)
        .otherwise(pl.col("times_triggered"))
        .alias("times_triggered")
    )
spikes_oil = spikes_oil.collect()
# %%

spikes_all = pl.concat([spikes_oil, spikes_no_oil], how="vertical")

# %% check for good cells
good_cells = (
    spikes_all.filter(pl.col("qi") > 0.4).select("cell_index").unique().to_numpy()
)
# %%
spikes_good = spikes_all.filter(pl.col("cell_index").is_in(good_cells.flatten()))
spikes_oil = spikes_oil.filter(pl.col("cell_index").is_in(good_cells.flatten()))
spikes_no_oil = spikes_no_oil.filter(pl.col("cell_index").is_in(good_cells.flatten()))
# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_all,
    indices=["stimulus_index", "cell_index"],
    single_psth=False,
    cmap=["Greys", "Reds"],
    width=30,
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_good,
    indices=["stimulus_index", "cell_index"],
    single_psth=False,
    cmap=["Greys", "Reds"],
    width=30,
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()
# %% check individual cells

cell_index = 294
cell_df = spikes_all.filter(pl.col("cell_index") == cell_index)

fig = spiketrain_plots.spikes_and_trace(
    cell_df,
    indices=["stimulus_index", "repeat"],
    single_psth=False,
    line_colour=["black", "red"],
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)

pn.extension()
pn.panel(fig).show()

# %% calculate the histograms to subtract
histograms_oil = histograms.psth(spikes_oil, end=24)
histograms_no_oil = histograms.psth(spikes_no_oil, end=24)

# %% diff
diff_raw = histograms_oil[0] - histograms_no_oil[0]
diff_centered = diff_raw - np.mean(diff_raw)  # remove overall bias
max_abs = np.max(np.abs(diff_centered))
diff_normalized = diff_centered / max_abs  # in [-1, 1]
# %% plot the difference, 2 subplots with the second row being small for stimulus
fig, ax = plt.subplots(
    nrows=2, gridspec_kw={"height_ratios": [3, 1]}, figsize=(10, 5), sharex=True
)
ax[0].plot(histograms_oil[1][:-1], diff_normalized, color="black")
# replace the y ticks, -1 indicates no oil is stronger, 1 indicates oil is stronger
ax[0].set_yticks([-1, 0, 1])
ax[0].set_yticklabels(["No oil", "Equal", "Oil"])
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %% Bootstrap the difference
histogram_oil_complete = np.zeros(
    ((spikes_all["cell_index"].unique().max() + 1), 480),
)
histogram_no_oil_complete = np.zeros(
    ((spikes_all["cell_index"].unique().max() + 1), 480),
)
histograms_oil, _, cell_idx_oil = histograms.psth_by_index(
    spikes_oil, index=["cell_index"], window_end=24.05, return_idx=True
)
histograms_no_oil, bins, cell_idx_no_oil = histograms.psth_by_index(
    spikes_no_oil, index=["cell_index"], window_end=24.05, return_idx=True
)
# %% fill the complete histograms
histogram_oil_complete[cell_idx_oil.flatten().astype(int), :] = histograms_oil
histogram_no_oil_complete[cell_idx_no_oil.flatten().astype(int), :] = histograms_no_oil

histogram_oil_complete = histogram_oil_complete[good_cells.flatten(), :]
histogram_no_oil_complete = histogram_no_oil_complete[good_cells.flatten(), :]
# %%
histogram_oil_complete = (
        histogram_oil_complete
        / np.maximum(np.max(np.abs(histogram_oil_complete), axis=1), 1e-10)[:, np.newaxis]
)
histogram_no_oil_complete = (
        histogram_no_oil_complete
        / np.maximum(np.max(np.abs(histogram_no_oil_complete), axis=1), 1e-10)[
          :, np.newaxis
          ]
)

# %% normalize to 0-1


# %%
histogram_oil_complete = zscore(histogram_oil_complete, axis=1)
histogram_no_oil_complete = zscore(histogram_no_oil_complete, axis=1)
# %% fill nans with 0
histogram_oil_complete[np.isnan(histogram_oil_complete)] = 0
histogram_no_oil_complete[np.isnan(histogram_no_oil_complete)] = 0
# %%
mean_oil = histogram_oil_complete.mean(axis=0)
mean_no_oil = histogram_no_oil_complete.mean(axis=0)


# %%


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    N, T = data.shape
    boot_means = np.zeros((n_bootstrap, T))
    for i in range(n_bootstrap):
        sample_idx = np.random.choice(N, size=N, replace=True)
        boot_sample = data[sample_idx]
        boot_means[i] = boot_sample.mean(axis=0)
    lower = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)
    return lower, upper


ci_oil_lower, ci_oil_upper = bootstrap_ci(histogram_oil_complete)
ci_no_oil_lower, ci_no_oil_upper = bootstrap_ci(histogram_no_oil_complete)
# %%
significant = (ci_oil_lower > ci_no_oil_upper) | (ci_oil_upper < ci_no_oil_lower)
print(np.sum(significant))
# %% plot the difference with confidence intervals
fig, ax = plt.subplots(
    nrows=3, gridspec_kw={"height_ratios": [5, 5, 1]}, figsize=(15, 7), sharex=True
)
ax[0].plot(bins[:-1], mean_oil, color="black", label="Oil")
ax[0].fill_between(
    bins[:-1],
    ci_oil_lower,
    ci_oil_upper,
    color="grey",
    alpha=0.5,
)
# plot the no oil histogram
ax[0].plot(bins[:-1], mean_no_oil, color="red", label="No Oil")
ax[0].fill_between(
    bins[:-1],
    ci_no_oil_lower,
    ci_no_oil_upper,
    color="red",
    alpha=0.5,
)
# plot markers where the difference is significant
y_max = max(np.max(ci_oil_upper), np.max(ci_no_oil_upper)) * 1.1
sig_x = bins[:-1][significant]

if len(sig_x) > 0:
    ax[0].scatter(
        sig_x,
        np.ones_like(sig_x) * y_max,
        marker="|",
        color="blue",
        s=50,
        label="Significant",
    )

ax[0].legend()
# add legend for confidence intervals in grey and red
ax[0].legend(
    handles=[
        plt.Line2D([0], [0], color="grey", lw=4, alpha=0.5, label="Oil CI"),
        plt.Line2D([0], [0], color="red", lw=4, alpha=0.5, label="No Oil CI"),
        plt.Line2D(
            [0],
            [0],
            marker="|",
            color="blue",
            linestyle="None",
            markersize=10,
            label="Significant",
        ),
        plt.Line2D([0], [0], color="black", lw=2, label="Oil"),
        plt.Line2D([0], [0], color="red", lw=2, label="No Oil"),
    ],
    loc="upper right",
)
# add the difference on axis 1
diff_mean = mean_oil - mean_no_oil
ax[1].plot(bins[:-1], diff_mean, color="black")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)

# remove top and right spines
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[1].set_ylabel("Difference (Oil - No Oil)")
ax[2].set_xlabel("Time (s)")
ax[0].set_ylabel("Z-Score (Spike Count)")

fig.show()

# %% Project all histograms into PC space
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
histograms_all = np.vstack([histogram_oil_complete, histogram_no_oil_complete])

transformed = pca.fit_transform(histograms_all)

# %% plot the first pc loadings
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(bins[:-1], pca.components_[0], color="black", label="PC1")
fig.show()
# %% color_array
color_array = np.array(
    ["blue"] * histogram_oil_complete.shape[0]
    + ["red"] * histogram_no_oil_complete.shape[0]
)
# %%
fig, ax = plt.subplots(figsize=(12, 8))

# Split the data by color/condition
oil_indices = color_array == "blue"
no_oil_indices = color_array == "red"

# Create 2D histograms for each condition
h_oil = ax.hist2d(
    transformed[oil_indices, 0],
    transformed[oil_indices, 1],
    bins=np.arange(np.min(transformed), np.max(transformed), 1),
    cmap="Reds",
    alpha=1,
    label="Oil",
)

h_no_oil = ax.hist2d(
    transformed[no_oil_indices, 0],
    transformed[no_oil_indices, 1],
    bins=np.arange(np.min(transformed), np.max(transformed), 1),
    cmap="Greens",
    alpha=0.5,
    label="No Oil",
)

# Add colorbar
plt.colorbar(h_oil[3], ax=ax, label="Count")
plt.colorbar(h_no_oil[3], ax=ax, label="Count")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA projection of histograms")
fig.show()
# %%
pca = PCA(n_components=2)
pca.fit(histogram_oil_complete)
transformed_oil = pca.transform(histogram_oil_complete)
transformed_no_oil = pca.transform(histogram_no_oil_complete)

# %%
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate centroids
oil_centroid = np.mean(transformed_oil, axis=0)
no_oil_centroid = np.mean(transformed_no_oil, axis=0)

# Draw arrow between centroids
ax.arrow(
    oil_centroid[0],
    oil_centroid[1],
    no_oil_centroid[0] - oil_centroid[0],
    no_oil_centroid[1] - oil_centroid[1],
    head_width=0.5,
    head_length=0.7,
    fc="black",
    ec="black",
    length_includes_head=True,
)

# Add centroid markers
ax.scatter(
    oil_centroid[0], oil_centroid[1], color="blue", s=100, marker="X", edgecolor="black"
)
ax.scatter(
    no_oil_centroid[0],
    no_oil_centroid[1],
    color="red",
    s=100,
    marker="X",
    edgecolor="black",
)

# Calculate average shift distance
shift_distance = np.linalg.norm(oil_centroid - no_oil_centroid)
ax.text(
    0.05,
    0.95,
    f"Shift distance: {shift_distance:.2f}",
    transform=ax.transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
)

# Identify subgroups using k-means clustering (assuming 3 subgroups)
from sklearn.cluster import KMeans

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(transformed_oil)
clusters = kmeans.labels_

# Plot individual shift vectors for a subset of paired points
# Assuming points are paired 1:1 between oil and no-oil conditions
colors = ["green", "purple", "orange", "black", "cyan"]
for cluster_id in range(n_clusters):
    mask = clusters == cluster_id
    cluster_oil = transformed_oil[mask]
    cluster_no_oil = transformed_no_oil[mask]

    # Calculate this cluster's centroid
    cluster_oil_centroid = np.mean(cluster_oil, axis=0)
    cluster_no_oil_centroid = np.mean(cluster_no_oil, axis=0)

    # Draw arrow for this cluster's shift
    ax.arrow(
        cluster_oil_centroid[0],
        cluster_oil_centroid[1],
        cluster_no_oil_centroid[0] - cluster_oil_centroid[0],
        cluster_no_oil_centroid[1] - cluster_oil_centroid[1],
        head_width=0.4,
        head_length=0.6,
        fc=colors[cluster_id],
        ec=colors[cluster_id],
        length_includes_head=True,
        linestyle="-",
        linewidth=2,
    )

    # Mark cluster centroid
    ax.scatter(
        cluster_oil_centroid[0],
        cluster_oil_centroid[1],
        color=colors[cluster_id],
        s=150,
        marker="o",
        edgecolor="black",
    )
    ax.scatter(
        cluster_no_oil_centroid[0],
        cluster_no_oil_centroid[1],
        color=colors[cluster_id],
        s=150,
        marker="*",
        edgecolor="black",
    )

    # Calculate this cluster's shift distance
    cluster_shift = np.linalg.norm(cluster_oil_centroid - cluster_no_oil_centroid)
    ax.text(
        cluster_oil_centroid[0],
        cluster_oil_centroid[1] + 0.5,
        f"Cluster {cluster_id + 1}\nShift: {cluster_shift:.2f}",
        ha="center",
        color=colors[cluster_id],
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7),
    )

for cluster_id in range(n_clusters):
    mask = clusters == cluster_id
    ax.scatter(
        transformed_oil[mask, 0],
        transformed_oil[mask, 1],
        color=colors[cluster_id],
        label=f"Cluster {cluster_id + 1} (Oil)"
        if cluster_id == 0
        else f"Cluster {cluster_id + 1} (Oil)",
        alpha=0.7,
        marker="o",
    )
    ax.scatter(
        transformed_no_oil[mask, 0],
        transformed_no_oil[mask, 1],
        color=colors[cluster_id],
        label=f"Cluster {cluster_id + 1} (No Oil)"
        if cluster_id == 0
        else f"Cluster {cluster_id + 1} (No Oil)",
        alpha=0.7,
        marker="*",
    )

ax.legend()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA projection showing subgroup shifts between Oil and No Oil conditions")

fig.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Fit PCA on pooled data
all_hist = np.vstack([histogram_oil_complete, histogram_no_oil_complete])
pca = PCA(n_components=2)
pca.fit(all_hist)

# 2. Project each condition
trans_oil = pca.transform(histogram_oil_complete)
trans_no_oil = pca.transform(histogram_no_oil_complete)

# 3. Compute per-cell shift vectors
#    (no_oil minus oil for each cell)
diff_vecs = trans_no_oil - trans_oil  # shape (N_cells, 2)
max_vec = np.abs(diff_vecs).max() + 2  # max shift in each PC
# 4. (Optional) Standardize the shift vectors
scaler = StandardScaler()
diff_std = scaler.fit_transform(diff_vecs)

# 5. Cluster the shift vectors
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(diff_std)
centers = kmeans.cluster_centers_

# 6. Visualize
fig, ax = plt.subplots(figsize=(8, 8))
# scatter each cell's shift
ax.scatter(diff_vecs[:, 0], diff_vecs[:, 1], c=labels, cmap="tab10", alpha=0.7)
# plot cluster centroids (back-transform if you standardized)
ctr_orig = scaler.inverse_transform(centers)
ax.scatter(
    ctr_orig[:, 0], ctr_orig[:, 1], marker="X", s=200, c="k", label="Cluster centers"
)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Shift on PC1")
ax.set_ylabel("Shift on PC2")
ax.set_title("K-means clustering of oil→no-oil shift vectors")
ax.legend()
ax.set_ylim(-max_vec, max_vec)
ax.set_xlim(-max_vec, max_vec)
fig.show()

# %% plot means per cluster
fig, ax = plt.subplots(
    nrows=n_clusters * 2 + 1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [5, 3] * n_clusters + [1]},
)
spike_traces = np.arange(0, n_clusters * 2 + 1, 2)
diff_traces = np.arange(1, n_clusters * 2 + 1, 2)
for cluster_id in range(n_clusters):
    mask = labels == cluster_id
    mean_trace_oil = histogram_oil_complete[mask].mean(axis=0)
    mean_trace_no_oil = histogram_no_oil_complete[mask].mean(axis=0)
    # for better comparison normalize to max value
    mean_trace_oil /= np.max(np.abs(mean_trace_oil)) + 1e-10
    mean_trace_no_oil /= np.max(np.abs(mean_trace_no_oil)) + 1e-10
    ax[spike_traces[cluster_id]].plot(
        bins[:-1], mean_trace_oil, color="black", label="Oil"
    )
    ax[spike_traces[cluster_id]].plot(
        bins[:-1], mean_trace_no_oil, color="red", label="No Oil"
    )
    ax[spike_traces[cluster_id]].set_title(
        f"Cluster {cluster_id + 1}, n={np.sum(mask)}"
    )
    diff = mean_trace_no_oil - mean_trace_oil
    ax[diff_traces[cluster_id]].plot(bins[:-1], diff, color="black")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()
# %% single cell example

cell_index = 305

# %%
psth_oil, bins = histograms.psth(
    spikes_oil.filter(pl.col("cell_index") == cell_index), end=24.05
)
psth_no_oil, bins = histograms.psth(
    spikes_no_oil.filter(pl.col("cell_index") == cell_index), end=24.05
)
# %%
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(bins[:-1], psth_oil / np.max(psth_oil), color="black", label="Oil")
ax.plot(bins[:-1], psth_no_oil / np.max(psth_no_oil), color="red", label="No Oil")
fig.show()

# %%
from sklearn.decomposition import NMF

# Option B: Decompose difference curves
X = histogram_no_oil_complete - histogram_oil_complete  # shape (N, T)
X -= X.min()

# %%
nmf = NMF(n_components=20, init="nndsvd", max_iter=100000, random_state=0)
W = nmf.fit_transform(X)  # coefficients per time series
H = nmf.components_  # basis vectors (motifs)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(3):
    ax.plot(bins[:-1], H[i], label=f"Motif {i + 1}")
fig.show()

# %% gaussian mixture model using BIC to find the optimal number of components
from sklearn.cluster import HDBSCAN

clusters = HDBSCAN(
    min_samples=5,  # minimum number of samples in a cluster
    cluster_selection_epsilon=0.1,  # distance threshold for clustering
).fit_predict(W)

np.max(clusters) + 1  # number of clusters found
np.unique(clusters, return_counts=True)  # cluster labels, -1 indicates noise

# %% plot the clusters
fig, ax = plt.subplots(nrows=np.max(clusters) + 2, figsize=(12, 8))
for cluster_id in range(-1, np.max(clusters) + 1):
    mask = clusters == cluster_id
    if np.sum(mask) > 0:
        ax[cluster_id + 1].plot(
            bins[:-1],
            X[mask].mean(axis=0),
            label=f"Cluster {cluster_id + 1} (n={np.sum(mask)})",
        )
fig.show()

# %% distance matrix
dist_matrix = pairwise_distances(X, metric="euclidean")
# %% plot the distance matrix
fig, ax = plt.subplots(figsize=(30, 10))
ax.imshow(histograms_all)
fig.show()

# %%
N = histogram_oil_complete.shape[0]  # number of cells
y = np.array([0] * N + [1] * N)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(histograms_all, y)
# %%
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(clf.feature_importances_)
fig.show()
# %% Interpretation of the results
from imodels import RuleFitClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %%
X_train, X_test, y_train, y_test = train_test_split(
    histograms_all, y, stratify=y, random_state=42, test_size=0.2
)

# Fit the RuleFit model
model = RuleFitClassifier(n_estimators=4)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
y_true = y_test
# Get indices of true class 1
true_1_indices = np.where(y_true == 1)[0]

# Separate TP and FN among them
tp_indices = [i for i in true_1_indices if y_pred[i] == 1]
fn_indices = [i for i in true_1_indices if y_pred[i] == 0]

# %% plot correctly predicted class 1 traces
fig, ax = plt.subplots(figsize=(12, 8))
for i in tp_indices:
    ax.plot(X_test[i], color="black", alpha=0.1)
ax.plot(X_test[tp_indices].mean(axis=0), color="red", label="TP mean")
ax.vlines(82, ymin=-1, ymax=1, color="blue", linestyle="--", label="Stimulus onset")
fig.show()
# %% plot incorrectly predicted class 1 traces
fig, ax = plt.subplots(figsize=(12, 8))
for i in fn_indices:
    ax.plot(bins[:-1], X_test[i], color="red", alpha=0.1)
ax.plot(bins[:-1], X_test[fn_indices].mean(axis=0), color="black", label="TP mean")
fig.show()

# %%
import pandas as pd

rules = model._get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
pd.set_option("display.max_colwidth", 120)
print(rules[["rule", "support", "coef", "importance"]])

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

feautre_names = np.repeat(CT.names, 40)
# Train surrogate
fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
plot_tree(
    tree,
    filled=True,
    ax=ax,
    class_names=["Oil", "No Oil"],
    max_depth=4,
    feature_names=feautre_names,
)

# Create a custom legend to show color mapping
import matplotlib.patches as mpatches

# In sklearn's plot_tree, class 0 is typically colored in a blue shade and class 1 in orange
oil_patch = mpatches.Patch(color="#729ECE", label="Oil (class 0)")
no_oil_patch = mpatches.Patch(color="#FF9E4A", label="No Oil (class 1)")
ax.legend(handles=[oil_patch, no_oil_patch], loc="lower right")

fig.show()

# %%
from sklearn.metrics import accuracy_score

surrogate_preds = tree.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train, surrogate_preds)
print(f"Surrogate model accuracy: {accuracy:.2f}")
