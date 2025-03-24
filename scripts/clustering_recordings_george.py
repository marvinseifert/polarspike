from group_pca import GroupPCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Import normalizer
from sklearn.preprocessing import StandardScaler


"""
How it works.
I cut off the original top of this script, as the data gets loaded from my polarspike project. You need to load 
your data on your own. 
The way the data should be organized is as follows:
psths: np.array
    The PSTHs of the cells. The first axis should be the cells (from all recordings), the second binned responses of all
    stimuli combined of a cell.
bins: np.array
    The bins of the PSTHs. The length should be the same as the second axis of psths.
cells: np.array
    The cells array. The first column should be the name of the recording, the second the cell index in that recording.
    This is just to keep track of the different recordings and cells.
store_scaled: list
    This is a list containing psths of single stimuli or even single episodes of a stimulus. For example, you could
    split based on individual colours. So the psths of the green, blue, uv,  flashs get put into the list as a separate array.
    Or you just split based on stimuli. 
    Importantly, these psths should be scaled using the scaler object. This data will be feed into the PCA. PC analysis 
    will be performed on each element independently and then concatenated.
    
    Here is an example code that seperates the psths into 12 episodes and scales them using the StandardScaler object.:
    episodes = 12
    scaler = StandardScaler()
    store = []
    store_scaled = []
    cut_length = psths.shape[1] // episodes
    for i in range(episodes):
        # realign the psths
        store.append(psths[:, cut_length * i : cut_length * i + cut_length])
        store_scaled.append(
            scaler.fit_transform(psths[:, cut_length * i : cut_length * i + cut_length])
    )

"""

# %%

scaler = StandardScaler()


# %%
group_pca = GroupPCA(n_components=20)
reduced_data = group_pca.fit_transform(store_scaled)

# %%
# Plot the first two pcs
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c="black")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()

# %% Plot the first two pcs coloured by the recording
# Look for batch effects.
fig, ax = plt.subplots(figsize=(10, 10))
for recording in np.unique(cells[:, 0]):
    ax.scatter(
        reduced_data[cells[:, 0] == recording, 0],
        reduced_data[cells[:, 0] == recording, 1],
        label=recording,
    )
ax.legend()
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()

# %% Estimating how many clusters are required
upper_cluster_limit = 30
bic = []
for i in range(1, upper_cluster_limit):
    gmm = GaussianMixture(n_components=i)
    gmm.fit(reduced_data[:, :10])
    bic.append(gmm.bic(reduced_data[:, :10]))

# Plot the BIC
fig, ax = plt.subplots()
ax.plot(range(1, upper_cluster_limit), bic, c="black")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("BIC")
fig.show()

# %%
n_clusters = (
    np.argmin(bic) + 1
)  # Set to integer of choice if you dont want to use the bic for the number of clusters
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(reduced_data[:, :10])
labels = gmm.predict(reduced_data[:, :10])
# %% Plot the clusters in pc space
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(n_clusters):
    ax.scatter(
        reduced_data[labels == i, 0], reduced_data[labels == i, 1], label=f"Cluster {i}"
    )
ax.legend()
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.show()


# %% Plot the traces in subplots. The first column is the original PSTH, the second column is RF. The second column
# should be much smaller (10 %) than the first column
# Share the x-axis
fig, ax = plt.subplots(
    n_clusters + 1,
    2,
    figsize=(20, 10),
    gridspec_kw={"width_ratios": [10, 1]},
    sharex=True,
)
for i in range(n_clusters):
    trace = np.mean(psths[labels == i].T, axis=1) / np.max(
        np.mean(psths[labels == i].T, axis=1)
    )
    ax[i, 0].plot(bins[:-1], trace, color="black")
    # Add the number of cells in the cluster
    ax[i, 0].text(
        0.1,
        0.9,
        f"Cluster={i}; n={np.sum(labels == i)}",
        transform=ax[i, 0].transAxes,
    )
    ax[i, 0].set_ylim(0, 1.2)
    # Plot the RF
    # ax[i, 1].imshow(
    #     np.load(rf"D:\combined_analysis\clusters\cluster_{i}.npy"), cmap="seismic"
    # )
ax[4, 0].set_xlabel("Bins")
ax[4, 0].set_ylabel("Mean firing rate")
# Remove top y box and right x box
for i in range(n_clusters):
    ax[i, 0].spines["top"].set_visible(False)
    ax[i, 0].spines["right"].set_visible(False)

fig.show()


# %% Plot all traces from a single cluster
cluster = 1
fig, ax = plt.subplots(figsize=(10, 10))
for trace in psths[labels == cluster]:
    ax.plot(bins[:-1], trace, alpha=0.1)
ax.set_xlabel("Bins")
ax.set_ylabel("Mean firing rate")
fig.show()

# %%
# plot the percentage of single recordings contributing to the clusters
cluster_rec_counts = np.zeros((n_clusters, len(np.unique(cells[:, 0]))))
rec_arr = np.unique(cells[:, 0])
for cluster in range(n_clusters):
    recs, counts = np.unique(cells[labels == cluster, 0], return_counts=True)
    for rec, count in zip(recs, counts):
        cluster_rec_counts[cluster, np.where(rec_arr == rec)[0][0]] = count

fig, ax = plt.subplots(figsize=(10, 10))
for cluster in range(n_clusters):
    ax.bar(rec_arr, cluster_rec_counts[cluster, :], label=f"Cluster {cluster}")
ax.legend()
fig.show()


# Save you data somewhere.
