from sympy.printing.pretty.pretty_symbology import line_width

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
fig, ax = plt.subplots(figsize=(20, 10))
for trigger in range(8):
    trigger_subset = spikes_fff.filter(pl.col("trigger") == trigger)
    trigger_subset = trigger_subset.with_columns(
        times_triggered=pl.col("times_triggered") - 4 * trigger
    )
    psth_fff, bins, cells = histograms.psth_by_index(
        trigger_subset,
        index=["stimulus_index", "recording", "cell_index"],
        bin_size=0.01,
        window_end=4,
        return_idx=True,
    )

    ax.plot(
        bins[:-1],
        np.mean(psth_fff, axis=0),
        c=ct.colours[::2][trigger],
        alpha=0.9,
        lw=2,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of spikes")
fig.show()
# %%
trigger = 7
trigger_subset = spikes_fff.filter(pl.col("trigger") == trigger)
trigger_subset = trigger_subset.with_columns(
    times_triggered=pl.col("times_triggered") - 4 * trigger
)
psth_fff, bins, cells = histograms.psth_by_index(
    trigger_subset,
    index=["stimulus_index", "recording", "cell_index"],
    bin_size=0.01,
    window_end=4,
    return_idx=True,
)
# %%
embedder = umato.UMATO(
    n_neighbors=15,
    min_dist=0.01,
    n_components=2,
    metric="jaccard",
)
embedding = embedder.fit_transform(psth_fff)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=10)
ax.set_xlabel("umato 1")
ax.set_ylabel("umato 2")
fig.show()

# %%
trigger_subset = df_fff.filter(pl.col("trigger") == trigger)
window_size = 5

window_df = (
    trigger_subset.sort("cell_index")  # ensure consistent ordering
    .group_by("recordings", "cell_index", "repeat", maintain_order=True)
    .agg(pl.col("isi"))
    .with_columns(
        pl.col("isi")
        .map_elements(
            lambda vals: [
                vals[i : i + window_size] for i in range(len(vals) - window_size + 1)
            ],
            return_dtype=pl.List(pl.List(pl.Float64)),
        )
        .alias("windows")
    )
    .explode("windows")
    .drop("isi")
    .drop_nulls()
)

# %%
unique_motifs = window_df.group_by(["recordings", "cell_index", "repeat"]).agg(
    pl.col("windows").sample(1).explode()
)
# %% add noise
all_windows = np.vstack(unique_motifs["windows"].to_numpy())
all_windows = np.round(all_windows, 3)
# %%
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, metric="euclidean")
clusters = clusterer.fit_predict(all_windows)
unique_motifs = unique_motifs.with_columns(pl.Series(clusters).alias("cluster"))
# %%

embedder = umato.UMATO(
    n_neighbors=200,
    min_dist=0,
    n_components=2,
    metric="euclidean",
)
embedding = embedder.fit_transform(all_windows)
# %%
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, metric="euclidean")
clusters = clusterer.fit_predict(embedding)
unique_motifs = unique_motifs.with_columns(pl.Series(clusters).alias("cluster"))
# %%
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    s=10,
    c=clusterer.labels_,
    cmap="tab20",
    alpha=0.7,
)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.show()
# %%
for cluster in np.unique(clusters):
    cluster_windows = all_windows[clusters == cluster]
    mean_window = np.var(cluster_windows, axis=0)
    print(f"Cluster {cluster}: mean window = {mean_window}")
print(f"Number of clusters: {np.unique(clusters, return_counts=True)}")

# %%
mask = clusterer.labels_ != 1
embedder = umato.UMATO(
    n_neighbors=200,
    min_dist=0,
    n_components=2,
    metric="euclidean",
)
embedding_clean = embedder.fit_transform(all_windows[mask])
# %%
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    embedding_clean[:, 0],
    embedding_clean[:, 1],
    s=10,
    c=clusterer.labels_[mask],
    cmap="tab20",
    alpha=0.7,
)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.show()

# %%
from scipy.spatial.distance import cdist


def run_single_iteration(
    window_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One sampling iteration. Returns:
    - all_windows: raw ISI vectors sampled
    - labels: HDBSCAN cluster labels
    - centroids: mean ISI vector per cluster (excluding noise)
    """
    # Sample one random motif per cell per repeat
    unique_motifs = window_df.group_by(["recordings", "cell_index", "repeat"]).agg(
        pl.col("windows").sample(1).explode()
    )

    all_windows = np.vstack(unique_motifs["windows"].to_numpy())
    all_windows = np.round(all_windows, 3)

    # Embed
    embedder = umato.UMATO(
        n_neighbors=200,
        hub_num=all_windows.shape[0] // 10,
        min_dist=0,
        n_components=2,
        metric="euclidean",
    )
    embedding = embedder.fit_transform(all_windows)

    # Cluster on embedding
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, metric="euclidean")
    labels = clusterer.fit_predict(embedding)

    # Remove dominant background cluster (largest cluster by count)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    background_label = unique_labels[np.argmax(counts)]
    mask = labels != background_label
    if np.sum(mask) < 10:
        return np.empty(0), np.empty(0), np.empty(0)
    # Re-embed and re-cluster without background
    embedding_clean = umato.UMATO(
        n_neighbors=min(200, all_windows[mask].shape[0] - 1),
        hub_num=all_windows[mask].shape[0] // 10,
        min_dist=0,
        n_components=2,
        metric="euclidean",
    ).fit_transform(all_windows[mask])

    clusterer_clean = hdbscan.HDBSCAN(min_cluster_size=30, metric="euclidean")
    labels_clean = clusterer_clean.fit_predict(embedding_clean)

    # Compute centroids in raw ISI space (not embedding space)
    centroids = {}
    for label in np.unique(labels_clean):
        if label == -1:
            continue
        centroids[label] = np.mean(all_windows[mask][labels_clean == label], axis=0)

    return all_windows[mask], labels_clean, centroids


def match_centroids(
    reference: dict, candidate: dict, distance_threshold: float = 0.1
) -> dict:
    """
    Match clusters from one iteration to a reference set of centroids
    using nearest centroid in ISI space. Returns mapping candidate_label -> reference_label.
    """
    if not reference or not candidate:
        return {}

    ref_labels = list(reference.keys())
    cand_labels = list(candidate.keys())

    ref_matrix = np.stack([reference[l] for l in ref_labels])
    cand_matrix = np.stack([candidate[l] for l in cand_labels])

    distances = cdist(cand_matrix, ref_matrix, metric="euclidean")
    best_matches = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(cand_labels)), best_matches]

    mapping = {}
    for i, cand_label in enumerate(cand_labels):
        if best_distances[i] < distance_threshold:
            mapping[cand_label] = ref_labels[best_matches[i]]
        else:
            mapping[cand_label] = f"new_{cand_label}_iter"  # unmatched — new motif

    return mapping


# %%
N_ITERATIONS = 50
centroid_accumulator: dict[str, list[np.ndarray]] = {}
reference_centroids: dict = {}  # built from first iteration

for iteration in range(N_ITERATIONS):
    print(f"Iteration {iteration + 1}/{N_ITERATIONS}")

    windows, labels, centroids = run_single_iteration(window_df)

    if iteration == 0:
        # First iteration defines the reference
        reference_centroids = {str(k): v for k, v in centroids.items()}
        for k, v in reference_centroids.items():
            centroid_accumulator[k] = [v]
        continue

    # Match this iteration's clusters to the reference
    str_reference = {str(k): v for k, v in reference_centroids.items()}
    mapping = match_centroids(str_reference, centroids, distance_threshold=0.15)

    for cand_label, ref_label in mapping.items():
        canonical = str(ref_label)
        if canonical not in centroid_accumulator:
            centroid_accumulator[canonical] = []
        centroid_accumulator[canonical].append(centroids[cand_label])
# %%
print("\n--- Motif Stability Report ---")
all_motifs = []
for label, centroid_list in centroid_accumulator.items():
    rediscovery_rate = len(centroid_list) / N_ITERATIONS
    mean_centroid = np.mean(centroid_list, axis=0)
    all_motifs.append(mean_centroid)
    std_centroid = np.std(centroid_list, axis=0)
    print(f"\nMotif {label}:")
    print(
        f"  Rediscovery rate: {rediscovery_rate:.2f} ({len(centroid_list)}/{N_ITERATIONS})"
    )
    print(f"  Mean ISI:  {np.round(mean_centroid, 3)}")
    print(f"  Std  ISI:  {np.round(std_centroid, 3)}")
# %%
all_motifs = np.array(all_motifs)
np.save(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis/led_365_motifs.npy",
    all_motifs,
)
# %%
fig, ax = plt.subplots(figsize=(20, 10))
for idx, motif in enumerate(all_motifs[:5, :]):
    ax.scatter(
        np.cumsum(np.concatenate([[0], motif])),
        np.ones(motif.shape[0] + 1) * idx,
        marker="|",
        c="black",
    )
fig.show()

# %%
isi_array = trigger_subset.group_by(
    ["recordings", "cell_index", "repeat"], maintain_order=True
).agg("isi", "times_triggered")


def sliding_window_search(ts, query):
    n = len(ts)
    m = len(query)
    distances = []

    # Slide the window across the sequence
    for i in range(n - m + 1):
        window = ts[i : i + m]
        # Calculate Euclidean distance
        dist = np.linalg.norm(window - query)
        distances.append(dist)

    return np.array(distances)


# %%
from functools import partial

apply_window_search = partial(sliding_window_search, query=all_motifs[3])

# %%
isi_array = isi_array.with_columns(
    pl.col("isi")
    .map_elements(apply_window_search, return_dtype=pl.List(pl.Float64))
    .alias("distance"),
    pl.col("times_triggered"),
)
# %%
isi_array = isi_array.with_columns(pl.col("distance").list.len().alias("distance_len"))
isi_array = isi_array.with_columns(pl.col("distance").list.min().alias("min_distance"))
# %%
isi_array = isi_array.fill_null(1000)
# %%
times_triggered = isi_array.sort("min_distance")["times_triggered"].to_list()
distances = isi_array.sort("min_distance")["distance"].to_list()
# %%
fig, ax = plt.subplots(figsize=(20, 10))
for times, dist in zip(times_triggered, distances):
    ax.scatter(times[:-1], dist, s=0.5, alpha=0.5)
fig.show()
# %%
cell = 5
fig, ax = plt.subplots()
ax.plot(times_triggered[cell][:-4], distances[cell])
fig.show()
# %%
isi_array = isi_array.with_columns(
    pl.col("isi")
    .map_elements(lambda x: np.linalg.norm(x - all_motifs, axis=1))
    .alias("distance")
)
# %%
distances = np.linalg.norm(all_windows - all_motifs[0], axis=1)

# %%
fig, ax = plt.subplots()
ax.hist(distances, 100)
fig.show()
# %%
for cluster in np.unique(clusters):
    cluster_windows = all_windows[clusters == cluster]
    mean_window = np.var(cluster_windows, axis=0)
    print(f"Cluster {cluster}: mean window = {mean_window}")
print(f"Number of clusters: {np.unique(clusters, return_counts=True)}")
# %%
cells_subset = spikes_fff.filter(
    pl.col("cell_index").is_in(
        window_df.filter(pl.col("cluster") == 1)["cell_index"].to_list()
    )
)

# %%
fig, ax = spiketrain_plots.whole_stimulus(
    cells_subset, indices=["cell_index", "repeat"]
)
fig = ct.add_stimulus_to_plot(fig, [2] * 16, names=False)
fig.show()
# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(bins[:-1], psth_cells.T, c="black", alpha=0.1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Number of spikes")
fig.show()
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_embedding = pca.fit_transform(all_windows)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(pca_embedding[:, 0], pca_embedding[:, 1], s=10)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
fig.show()
# %%
core, factors = non_negative_tucker(
    all_windows, rank=[100, 5], init="svd", random_state=0
)
# %%
