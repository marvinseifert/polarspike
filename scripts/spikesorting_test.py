import h5py
import hdf5plugin
import dask.array as da
import numpy as np
from h5py.h5fd import LOG_ALL
from sklearn.decomposition import FastICA
import einops
import matplotlib.pyplot as plt
import configparser
from pathlib import Path
from tqdm import tqdm
from polarspike.sc_curation import fast_obvious_clusters
import kilosort.io as io
import torch
from polarspike import Overview
import polars as pl
from circus import scripts


# %%
def get_spike_waveforms_means(
        spikes,
        binary_file,
        n_channels=252,
        whitening_mat=None,
        nt=60000,
        fs=40000,
        break_condition=100,
        **kwargs,
):
    """
    Extract spike waveforms from a binary file and store them in an HDF5 file.
    Parameters
    ----------
    spikes : array-like
        Array of spike times (in frames in the original recording).
    binary_file: str or Path
        Path to the binary file.
    n_channels: int
        Number of channels in the binary file.
    whitening_mat: np.ndarray or None
        Whitening matrix to apply to the data. If None, no whitening is applied.
    chan: array-like or None
        Channels to extract. If None, all channels are extracted.
    h5_path: str or Path or None
        Path to the HDF5 file to store the waveforms. This should be a local location, not network.
         If None, current working dir is used.
    nt: int
        Number of time points per batch in the binary file. Fewer time points means more batches and more I/O operations.
        More time points means fewer batches and more RAM usage.
    fs: int (I need to check why this is int)
        Sampling frequency of the recording.
    save_chunk_size: int
        Number of waveforms to save in each chunk. Larger chunks mean fewer I/O operations but more RAM usage.
    **kwargs:
        Additional arguments to pass to the BinaryFiltered reader.

    Returns
    -------

    """
    # Initialize the reader
    bin_reader = io.BinaryFiltered(
        binary_file,
        n_channels,
        whiten_mat=torch.from_numpy(whitening_mat).float()
        if whitening_mat is not None
        else None,
        do_CAR=True,
        device="cpu",
        NT=nt,
        fs=fs,
        **kwargs,
    )

    nt = bin_reader.NT
    snippet_len = bin_reader.nt
    n_spikes = spikes.size
    n_output_channels = n_channels

    # Pre-allocate the dataset
    # We use (Spikes, Channels, Time) for better write alignment
    # Chunks are critical for performance: (1000 spikes per chunk)
    batch_indices = np.unique(spikes // nt)

    print(f"Streaming {len(batch_indices)} batches from network to HDF5...")
    means = np.zeros(
        (n_output_channels, n_output_channels, bin_reader.nt), dtype=np.float32
    )
    channel_mean_counts = np.zeros(n_output_channels, dtype=np.uint64)
    try:
        for ibatch in tqdm(batch_indices):
            if np.all(channel_mean_counts >= break_condition):
                break
            t_start = ibatch * nt
            t_end = t_start + nt

            # 1. Find spikes in this time window
            idx_mask = (spikes >= t_start) & (spikes < t_end)
            current_spikes = spikes[idx_mask]
            current_store_indices = np.where(idx_mask)[0]

            if current_spikes.size == 0:
                continue

            # Network request for the full batch
            batch_data = bin_reader.padded_batch_to_torch(ibatch)
            # 2. Extract and write to HDF5
            # We convert to numpy once and slice locally
            batch_np = batch_data.cpu().numpy()

            # Calculate relative spike times within the batch
            rel_times = (current_spikes - t_start + bin_reader.nt).astype(int)
            # 3. Extract snippets for all valid spikes in this batch
            if rel_times.size > 0:
                # Create offsets for snippet extraction
                offsets = np.arange(
                    -bin_reader.nt0min, (bin_reader.nt - bin_reader.nt0min)
                )
                # Create on array with indices for all snippets for vectorized indexing
                time_indices = rel_times[:, None] + offsets  # for example (n_valid, 61)

                # Extract snippets: Transpose batch to (Time, Channels) to index time explicitly
                # Result shape: (n_valid, snippet_len, n_channels)
                snippets = batch_np[:, time_indices]
                batch_min_channels = np.argmin(
                    snippets[:, :, bin_reader.nt0min], axis=0
                )
                # save mean for each channel
                for ch in range(n_output_channels):
                    ch_mask = batch_min_channels == ch
                    ch_snippets = snippets[:, ch_mask, :]
                    if len(ch_snippets) > 0:
                        if ch_snippets.shape[1] > 0:
                            means[ch] = means[ch] + np.mean(ch_snippets, axis=1)
                            channel_mean_counts[ch] += ch_snippets.shape[1]

    finally:
        print("Closing binary file...")

    return means


# %%
def get_example_waveforms(
        spikes,
        binary_file,
        n_channels=252,
        whitening_mat=None,
        nt=60000,
        fs=40000,
        break_condition=1000,
        **kwargs,
):
    """
    Extract spike waveforms from a binary file and store them in an HDF5 file.
    Parameters
    ----------
    spikes : array-like
        Array of spike times (in frames in the original recording).
    binary_file: str or Path
        Path to the binary file.
    n_channels: int
        Number of channels in the binary file.
    whitening_mat: np.ndarray or None
        Whitening matrix to apply to the data. If None, no whitening is applied.
    chan: array-like or None
        Channels to extract. If None, all channels are extracted.
    h5_path: str or Path or None
        Path to the HDF5 file to store the waveforms. This should be a local location, not network.
         If None, current working dir is used.
    nt: int
        Number of time points per batch in the binary file. Fewer time points means more batches and more I/O operations.
        More time points means fewer batches and more RAM usage.
    fs: int (I need to check why this is int)
        Sampling frequency of the recording.
    save_chunk_size: int
        Number of waveforms to save in each chunk. Larger chunks mean fewer I/O operations but more RAM usage.
    **kwargs:
        Additional arguments to pass to the BinaryFiltered reader.

    Returns
    -------

    """
    # Initialize the reader
    bin_reader = io.BinaryFiltered(
        binary_file,
        n_channels,
        whiten_mat=torch.from_numpy(whitening_mat).float()
        if whitening_mat is not None
        else None,
        do_CAR=True,
        device="cpu",
        NT=nt,
        fs=fs,
        **kwargs,
    )

    nt = bin_reader.NT
    snippet_len = bin_reader.nt
    n_spikes = spikes.size
    n_output_channels = n_channels

    # Pre-allocate the dataset
    # We use (Spikes, Channels, Time) for better write alignment
    # Chunks are critical for performance: (1000 spikes per chunk)
    batch_indices = np.unique(spikes // nt)

    print(f"Streaming {len(batch_indices)} batches from network to HDF5...")
    example_spikes = np.zeros(
        (break_condition, n_output_channels, n_output_channels, bin_reader.nt),
        dtype=np.float32,
    )
    channel_counts = np.zeros(n_output_channels, dtype=np.uint64)
    try:
        for ibatch in tqdm(batch_indices):
            if np.all(channel_counts >= break_condition):
                break
            t_start = ibatch * nt
            t_end = t_start + nt

            # 1. Find spikes in this time window
            idx_mask = (spikes >= t_start) & (spikes < t_end)
            current_spikes = spikes[idx_mask]
            current_store_indices = np.where(idx_mask)[0]

            if current_spikes.size == 0:
                continue

            # Network request for the full batch
            batch_data = bin_reader.padded_batch_to_torch(ibatch)
            # 2. Extract and write to HDF5
            # We convert to numpy once and slice locally
            batch_np = batch_data.cpu().numpy()

            # Calculate relative spike times within the batch
            rel_times = (current_spikes - t_start + bin_reader.nt).astype(int)
            # 3. Extract snippets for all valid spikes in this batch
            if rel_times.size > 0:
                # Create offsets for snippet extraction
                offsets = np.arange(
                    -bin_reader.nt0min, (bin_reader.nt - bin_reader.nt0min)
                )
                # Create on array with indices for all snippets for vectorized indexing
                time_indices = rel_times[:, None] + offsets  # for example (n_valid, 61)

                # Extract snippets: Transpose batch to (Time, Channels) to index time explicitly
                # Result shape: (n_valid, snippet_len, n_channels)
                snippets = batch_np[:, time_indices]
                batch_min_channels = np.argmin(
                    snippets[:, :, bin_reader.nt0min], axis=0
                )
                # save mean for each channel
                for ch in range(n_output_channels):
                    if channel_counts[ch] >= break_condition:
                        continue
                    ch_mask = batch_min_channels == ch
                    ch_snippets = snippets[:, ch_mask, :]
                    if ch_snippets.shape[1] > 0:
                        n_to_add = min(
                            ch_snippets.shape[1], break_condition - channel_counts[ch]
                        )
                        example_spikes[
                            int(channel_counts[ch]): int(
                                channel_counts[ch] + n_to_add
                            ),
                            ch,
                            :,
                            :,
                        ] = ch_snippets[:, : int(n_to_add), :].transpose(1, 0, 2)
                        channel_counts[ch] += n_to_add

    finally:
        print("Closing binary file...")

    return example_spikes


# %%
def project_waveforms(
        spikes,
        models,
        n_components,
        binary_file,
        n_channels=252,
        whitening_mat=None,
        nt=60000,
        fs=40000,
        **kwargs,
):
    """
    Extract spike waveforms from a binary file and store them in an HDF5 file.
    Parameters
    ----------
    spikes : array-like
        Array of spike times (in frames in the original recording).
    binary_file: str or Path
        Path to the binary file.
    n_channels: int
        Number of channels in the binary file.
    whitening_mat: np.ndarray or None
        Whitening matrix to apply to the data. If None, no whitening is applied.
    chan: array-like or None
        Channels to extract. If None, all channels are extracted.
    h5_path: str or Path or None
        Path to the HDF5 file to store the waveforms. This should be a local location, not network.
         If None, current working dir is used.
    nt: int
        Number of time points per batch in the binary file. Fewer time points means more batches and more I/O operations.
        More time points means fewer batches and more RAM usage.
    fs: int (I need to check why this is int)
        Sampling frequency of the recording.
    save_chunk_size: int
        Number of waveforms to save in each chunk. Larger chunks mean fewer I/O operations but more RAM usage.
    **kwargs:
        Additional arguments to pass to the BinaryFiltered reader.

    Returns
    -------

    """
    # Initialize the reader
    bin_reader = io.BinaryFiltered(
        binary_file,
        n_channels,
        whiten_mat=torch.from_numpy(whitening_mat).float()
        if whitening_mat is not None
        else None,
        do_CAR=True,
        device="cpu",
        NT=nt,
        fs=fs,
        nt=120,
        nt0min=60,
        **kwargs,
    )

    spikes = np.sort(np.atleast_1d(spikes)).astype(np.int64)
    nt = bin_reader.NT
    snippet_len = bin_reader.nt
    n_spikes = spikes.size
    n_output_channels = n_channels

    # Pre-allocate the dataset
    # We use (Spikes, Channels, Time) for better write alignment
    # Chunks are critical for performance: (1000 spikes per chunk)
    batch_indices = np.unique(spikes // nt)

    projected_waveforms = np.zeros((len(spikes), n_components * 2), dtype=np.float32)
    max_channels = np.zeros(len(spikes), dtype=np.int32)

    try:
        spike_position = 0
        for ibatch in tqdm(batch_indices):
            t_start = ibatch * nt
            t_end = t_start + nt

            # 1. Find spikes in this time window
            idx_mask = (spikes >= t_start) & (spikes < t_end)
            current_spikes = spikes[idx_mask]
            current_store_indices = np.where(idx_mask)[0]

            if current_spikes.size == 0:
                continue

            # Network request for the full batch
            batch_data = bin_reader.padded_batch_to_torch(ibatch)
            # 2. Extract and write to HDF5
            # We convert to numpy once and slice locally
            batch_np = batch_data.cpu().numpy()

            # Calculate relative spike times within the batch
            rel_times = (current_spikes - t_start + bin_reader.nt).astype(int)
            # 3. Extract snippets for all valid spikes in this batch
            if rel_times.size > 0:
                # Create offsets for snippet extraction
                offsets = np.arange(
                    -bin_reader.nt0min, (bin_reader.nt - bin_reader.nt0min)
                )
                # Create on array with indices for all snippets for vectorized indexing
                time_indices = rel_times[:, None] + offsets  # for example (n_valid, 61)

                # Extract snippets: Transpose batch to (Time, Channels) to index time explicitly
                # Result shape: (n_valid, snippet_len, n_channels)
                snippets = batch_np[:, time_indices]
                batch_min_channels = np.argmin(
                    snippets[:, :, bin_reader.nt0min], axis=0
                )
                # for ch in range(n_output_channels):
                #     ch_mask = batch_min_channels == ch
                #     idx_masked = current_store_indices[ch_mask]
                #     ch_snippets = snippets[:, ch_mask, :]
                #     # for spike_idx in range(ch_snippets.shape[1]):
                #     #     projected_waveforms[idx_masked[spike_idx]] = (
                #     #         models[ch].components_
                #     #         @ ch_snippets[:, spike_idx, :].flatten()
                #     #     )
                #     #     max_channels[idx_masked[spike_idx]] = ch
                for ch in range(n_output_channels):
                    ch_mask = batch_min_channels == ch
                    if not np.any(ch_mask):
                        continue

                    # Get indices for this channel's spikes
                    target_indices = current_store_indices[ch_mask]

                    # Reshape snippets to (N_spikes, Total_Features)
                    # Ensure the flattening order matches how your PCA was trained!
                    # Usually: (N_spikes, Channels * Time)
                    ch_snippets = (
                        snippets[:, ch_mask, :]
                        .transpose(1, 0, 2)
                        .reshape(len(target_indices), -1)
                    )
                    ch_snippet = snippets[ch, ch_mask, :]

                    # Matrix-Matrix Multiplication (High Speed)
                    # (N_spikes, N_features) @ (N_features, N_components)
                    projected_waveforms[target_indices, :n_components] = (
                            ch_snippets @ models[ch].components_.T
                    )
                    # add projection of waveform of only the max channel as additional feature
                    projected_waveforms[target_indices, n_components:] = (
                            ch_snippet
                            @ einops.rearrange(
                        models[ch].components_, "n (c t) -> n c t", c=n_channels
                    )[:, ch, :].T
                    )
                    max_channels[target_indices] = ch

    finally:
        print("Closing binary file...")

    return projected_waveforms, max_channels


# %%


# %%
def get_key_params(path_to_params):
    """
    Reads a Spyking Circus .params file, prints key parameters,
    and returns them as a nested dictionary.
    """
    params_path = Path(path_to_params)
    results = {}

    if not params_path.exists():
        print(f"Error: The file '{params_path}' was not found.")
        return None

    config = configparser.ConfigParser()
    config.read(str(params_path))

    # Define the 'high-value' parameters
    targets = {
        "data": ["sampling_rate", "mapping", "nb_channels"],
        "detection": ["N_t", "radius"],
    }

    print(f"--- Summary for {params_path.name} ---")

    for section, keys in targets.items():
        results[section] = {}
        if config.has_section(section):
            print(f"\n[{section.upper()}]")
            for key in keys:
                value = config.get(section, key, fallback=None)
                results[section][key] = value

                display_val = value if value is not None else "Not Defined"
                print(f"  {key:<20} : {display_val}")
        else:
            print(f"\n[{section.upper()}] - Section not found.")

    return results


# %%
root = Path(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_18_11_2025/Phase_00"
)
combinations = (
        np.load(r"/mnt/data/electrical_imaging/channel_positions.npy")
        / 100  # Divide by 100 to get int positions (0, 1, 2...)
).astype(int)

params_file = root / "alldata.params"

params = get_key_params(params_file)
# Global parameters
electrode_spacing = 30.0  # in microns
combinations_um = combinations * electrode_spacing
channel_row_col = int(np.ceil(np.sqrt(combinations.shape[0])))
zero_position = 20
stimulus_id = 1
with h5py.File(root / "alldata/alldata.basis.hdf5", "r") as f:
    whitening_mat_inv = f["spatial"][:]
    whitening_mat_spatial = f["spatial"][:]
whitening_mat_inv = whitening_mat_inv.astype(np.float16)  # invert later to save memory
# %%
# Load the overview and get spikes from the recording
recording = Overview.Recording.load(root / "overview")
# recording.dataframes["subset"] = recording.spikes_df.query("stimulus_index == 1").sample(n=50)
spikes = recording.get_spikes_triggered(
    [{"stimulus_index": stimulus_id}],
    time="frames",
    pandas=False,
)

# spikes_subset = spikes.filter(
#     pl.int_range(pl.len()).shuffle().over("cell_index") < 5000
# )
# Get as numpy array
example_time = spikes["times"].to_numpy()
# We need to sort the times by frame in which they appear to optimize disk access. Also need to keep track of
# which time belongs to which cell after sorting:
sorted_indices = np.argsort(example_time)
inverse_sorted_indices = np.argsort(sorted_indices)
shuffled_cell_indices = spikes["cell_index"].to_numpy()[sorted_indices]
example_time = example_time[sorted_indices]

# %%
get_means = False
if get_means:
    example_waveforms = get_example_waveforms(
        example_time[shuffled_cell_indices == 287],
        r"/mnt/data/waveform_snippets/alldata.dat",
        n_channels=252,
        nt=6000,
        fs=int(recording.sampling_freq),
        whitening_mat=whitening_mat_inv,
        break_condition=10000,
    )

    # %%
    np.save("/mnt/data/waveform_snippets/example_waveforms_287.npy", example_waveforms)
else:
    example_waveforms = np.load(r"/mnt/data/waveform_snippets/example_waveforms.npy")

# %%
# plot random channel
square_dummy = np.empty((channel_row_col, channel_row_col))
square_dummy[:] = np.NaN

channel_to_plot = np.random.choice(252, 100, replace=False)
fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(20, 20))
for idx, ch in enumerate(channel_to_plot):
    data_to_plot = np.var(np.mean(example_waveforms[:, ch], axis=0), axis=1)
    square_dummy[combinations[:, 1], combinations[:, 0]] = data_to_plot
    # plot argmax as scatter
    ax[idx // 10, idx % 10].scatter(
        combinations[np.argmax(data_to_plot), 0],
        combinations[np.argmax(data_to_plot), 1],
        color="red",
        s=5,
        label="Max Variance",
    )
    ax[idx // 10, idx % 10].imshow(square_dummy, cmap="gray", aspect="auto")
    ax[idx // 10, idx % 10].set_title(f"Channel {ch} Mean Waveform")
fig.suptitle("Random Channel Mean Waveforms")
fig.show()

# %%
fig, ax = plt.subplots(figsize=(15, 7), ncols=2)
square_dummy[combinations[:, 1], combinations[:, 0]] = np.log(
    np.var(example_waveforms[202, 198, :, :], axis=1)
)
ax[0].imshow(square_dummy, cmap="gray", aspect="auto")
square_dummy[combinations[:, 1], combinations[:, 0]] = np.log(
    np.var(np.mean(example_waveforms[:, 198], axis=0), axis=1)
)
ax[1].imshow(square_dummy, cmap="gray", aspect="auto")
fig.show()
# %%
n_ch, n_time = example_waveforms.shape[1], example_waveforms.shape[2]
# %%
n_components = 5
reconstructions = np.zeros((n_ch, n_components, n_ch * n_time), dtype=np.float32)
models = []
for channel in tqdm(range(n_ch)):
    ica = FastICA(
        n_components=n_components, random_state=0, max_iter=100000, whiten=False
    )
    S_ = ica.fit_transform(
        einops.rearrange(example_waveforms[:, channel], "n c t -> n (c t)")
    )  # Reconstructed signals
    models.append(ica)

# %% PCA try
from sklearn.decomposition import KernelPCA, PCA, SparsePCA, FastICA

models = []
for channel in tqdm(range(n_ch)):
    pca = PCA(n_components=n_components, random_state=0)
    _ = pca.fit_transform(
        einops.rearrange(example_waveforms[:, channel], "n c t -> n (c t)")
    )  # Reconstructed signals
    models.append(pca)

# %%
channel = 206
from polarspike.r_pca import RobustPCA

random_indices = np.random.choice(example_waveforms.shape[0], 200, replace=False)
pca = RobustPCA(
    einops.rearrange(example_waveforms[random_indices, channel], "n c t -> n (c t)")
)
L, S = pca.fit(max_iter=1000, iter_print=20)

# %% plot S
fig, ax = plt.subplots()
square_dummy[combinations[:, 1], combinations[:, 0]] = (
    einops.rearrange(S, "n (c t) -> n c t", n=200, c=n_ch).mean(axis=0).var(axis=1)
)
ax.imshow(np.log(np.abs(square_dummy)), cmap="gray", aspect="auto")
fig.show()

# %% outlier removal using isolation forest
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=0)
outlier_labels = iso_forest.fit_predict(S)
S_clean = S[outlier_labels == 1]
# %%
S_pca = PCA(n_components=10, random_state=0, whiten=True)
transformed = S_pca.fit_transform(S_clean)
# reconstructed = S_pca.inverse_transform(transformed)
# plot first two components
fig, ax = plt.subplots()
ax.scatter(transformed[:, 0], transformed[:, 1], s=5)
fig.show()

# %%
# %%
S_pca = PCA(n_components=2, random_state=0, whiten=True)
S_pca = S_pca.fit(S_clean)
# print explained variance
print("Explained variance ratio:", S_pca.explained_variance_ratio_)
# %%
transformed = S_pca.transform(
    einops.rearrange(example_waveforms[:, channel], "n c t -> n (c t)")
)

# %%
from sktime.transformations.panel.rocket import Rocket

X = example_waveforms[:, channel, channel]
X = X[:, np.newaxis, :] if X.ndim == 2 else X
rocket = Rocket(num_kernels=2000, normalise=False, random_state=0)
transformed_rocket = rocket.fit_transform(X)

# %% plot first two components of rocket transformation
from umap import UMAP

reducer = UMAP(n_neighbors=50, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(transformed_rocket)
fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1], s=5)
fig.show()

# add transformed rocket features to transformed S_pca
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
transformed_combined = np.hstack(
    [scaler.fit_transform(transformed), scaler.fit_transform(embedding) * 3]
)

# %%
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].plot(L_channel_mean)
ax[0].set_title("Low-rank (L) of single channel")
ax[1].plot(S_channel_mean)
ax[1].set_title("Sparse (S) of single channel")
fig.show()
# %% project single channel from all waveforms onto L_channel and S_channel
L_channel_projections = np.tensordot(
    example_waveforms[:, channel, channel], np.mean(L_channel, axis=0), axes=1
)
S_channel_projections = np.tensordot(
    example_waveforms[:, channel, channel], np.mean(S_channel, axis=0), axes=1
)

# %%
fig, ax = plt.subplots()
ax.scatter(L_channel_projections, S_channel_projections, s=5)
fig.show()
# %% cluster using HDBSCAN
from sklearn.cluster import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=50, min_samples=1)
labels = clusterer.fit_predict(transformed_combined)
fig, ax = plt.subplots()
scatter = ax.scatter(transformed[:, 0], transformed[:, 1], s=5, c=labels, cmap="tab10")
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
fig.show()

# %%
mean_clusters = []
max_channel_waveforms = []
for label in np.unique(labels):
    mean_clusters.append(
        example_waveforms[labels == label, channel].mean(axis=0).var(axis=1)
    )
    max_channel_waveforms.append(
        example_waveforms[labels == label, channel][:, channel]
    )

# %% plot mean clusters
fig, ax = plt.subplots(
    ncols=np.unique(labels).shape[0],
    nrows=3,
    figsize=(20, 10),
    gridspec_kw={
        "height_ratios": [0.3, 0.8, 0.2],
    },
)
for idx, mean_cluster in enumerate(mean_clusters):
    square_dummy[combinations[:, 1], combinations[:, 0]] = mean_cluster
    ax[0, idx].imshow(np.log(square_dummy), cmap="viridis", aspect="auto")
    ax[1, idx].imshow(max_channel_waveforms[idx], cmap="inferno")
    # plot mean
    ax[2, idx].plot(max_channel_waveforms[idx].mean(axis=0))
fig.suptitle(f"Cluster on channel {channel}, first column is noise")
fig.show()
# %%
L_reshaped = einops.rearrange(L, "n (c t) -> n c t", n=500, c=n_ch).mean(axis=0)
S_reshaped = einops.rearrange(S, "n (c t) -> n c t", n=500, c=n_ch).mean(axis=0)
max_channel = np.argmax(L_reshaped.var(axis=1))

# %% plot time course of max channel
fig, ax = plt.subplots(ncols=2)
ax[0].plot(L_reshaped[max_channel], label="Low-rank (L)")

fig.show()

# %%
fig, ax = plt.subplots(ncols=2)
square_dummy[combinations[:, 1], combinations[:, 0]] = (
    einops.rearrange(L, "n (c t) -> n c t", n=500, c=n_ch).mean(axis=0).var(axis=1)
)
ax[0].imshow(square_dummy, cmap="gray", aspect="auto")
square_dummy[combinations[:, 1], combinations[:, 0]] = (
    einops.rearrange(S, "n (c t) -> n c t", n=500, c=n_ch).mean(axis=0).var(axis=1)
)
ax[1].imshow(square_dummy, cmap="gray", aspect="auto")
fig.show()

# %% project all example waveforms of channel 198 onto S
projections = np.tensordot(example_waveforms[:, 198], S_reshaped, axes=((1, 2), (0, 1)))
projections_noise = np.tensordot(
    example_waveforms[:, 198], L_reshaped, axes=((1, 2), (0, 1))
)

# %%
fig, ax = plt.subplots()
ax.scatter(projections[random_indices], transformed[:, 1], s=5)
fig.show()

# %%
fig, ax = plt.subplots()
ax.hist(projections, bins=50)
fig.show()

# %% split using
from sklearn.mixture import GaussianMixture

# 1. Reshape projections for sklearn (n_samples, n_features)
X = projections.reshape(-1, 1)

# 2. Fit a 2-component GMM (or 3 if you want to capture that 'tail' as noise)
gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(X)

# %%
mean_1 = example_waveforms[labels == 0, 198].mean(axis=0).var(axis=1)
mean_2 = example_waveforms[labels == 1, 198].mean(axis=0).var(axis=1)
square_dummy[combinations[:, 1], combinations[:, 0]] = mean_1
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(np.log(square_dummy), cmap="viridis", aspect="auto")
square_dummy[combinations[:, 1], combinations[:, 0]] = mean_2
ax[1].imshow(np.log(square_dummy), cmap="viridis", aspect="auto")
square_dummy[combinations[:, 1], combinations[:, 0]] = (
    example_waveforms[:, 198].mean(axis=0).var(axis=1)
)
ax[2].imshow(np.log(square_dummy), cmap="viridis", aspect="auto")
fig.show()
# %% plot second Ls and Ss
fig, ax = plt.subplots(ncols=2)
square_dummy[combinations[:, 1], combinations[:, 0]] = (
    einops.rearrange(L_second, "n (c t) -> n c t", n=100, c=n_ch)
    .mean(axis=0)
    .var(axis=1)
)
ax[0].imshow(square_dummy, cmap="gray", aspect="auto")
square_dummy[combinations[:, 1], combinations[:, 0]] = (
    einops.rearrange(S_second, "n (c t) -> n c t", n=100, c=n_ch)
    .mean(axis=0)
    .var(axis=1)
)
ax[1].imshow(square_dummy, cmap="gray", aspect="auto")
fig.show()
# %%
fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(15, 15))
channel_to_plot = np.random.choice(252, 25, replace=False)
for idx, ch in enumerate(channel_to_plot):
    square_dummy[combinations[:, 1], combinations[:, 0]] = einops.rearrange(
        models[ch].components_[0], "(c t) -> c t", c=n_ch
    ).var(axis=1)

    ax[idx // 5, idx % 5].imshow(square_dummy, cmap="gray", aspect="auto")
fig.show()

# %% plot all components from one channel
example_channel = 198
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(15, 15 // 5))
for i in range(5):
    square_dummy[combinations[:, 1], combinations[:, 0]] = einops.rearrange(
        models[example_channel].components_[i], "(c t) -> c t", c=n_ch
    ).var(axis=1)
    ax[0, i].imshow(square_dummy, cmap="gray", aspect="auto")
    ax[1, i].hist(models[example_channel].components_[i], bins=50)
fig.show()

# %% perform shapiro test of normality for each component and plot p-values
from scipy.stats import kurtosis

all_selected_components = []
channel_signal_to_noise_max = []
channel_signal_to_noise_min = []
for ch in range(n_ch):
    ks = np.zeros(n_components)
    for i in range(n_components):
        ks[i] = kurtosis(models[ch].components_[i])

    channel_signal_to_noise_max.append(np.max(ks) / 3)
    channel_signal_to_noise_min.append(np.min(ks) / 3)
    all_selected_components.append(ks > 3)
# %%
fig, ax = plt.subplots(ncols=2)
square_dummy[combinations[:, 1], combinations[:, 0]] = np.array(
    channel_signal_to_noise_max
)
ax[0].imshow(square_dummy, cmap="viridis", aspect="auto")
square_dummy[combinations[:, 1], combinations[:, 0]] = np.array(
    channel_signal_to_noise_min
)
ax[1].imshow(square_dummy, cmap="viridis", aspect="auto")
fig.show()

# %%


# %% plot histogram of p-values for all components
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(p_values.flatten(), bins=np.arange(0.005, 1.005, 0.005))
ax.set_title("Histogram of Shapiro-Wilk p-values for ICA Components")
ax.set_xlabel("p-value")
ax.set_ylabel("Frequency")
ax
fig.show()

# %%
# %% plot time courses of components for random channel
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
for idx, ch in enumerate(channel_to_plot):
    ch_max = np.argmax(
        einops.rearrange(models[ch].components_[0], "(c t) -> c t", c=n_ch).var(axis=1)
    )
    for comp in range(n_components):
        time_course = einops.rearrange(
            models[ch].components_[comp], "(c t) -> c t", c=n_ch
        )[ch_max]
        ax[idx // 5, idx % 5].plot(time_course, label=f"Comp {comp}")
    ax[idx // 5, idx % 5].set_title(f"Channel {ch} Component Time Courses")
fig.show()

# %%

# %%
projected_waveforms, max_channels = project_waveforms(
    example_time,
    models,
    n_components,
    str(r"/mnt/data/waveform_snippets/alldata.dat"),
    n_channels=252,
    nt=200000 * 10,
    fs=int(recording.sampling_freq),
    whitening_mat=whitening_mat_inv,
)

# %%
np.save(
    r"/mnt/data/waveform_snippets/projected_waveforms_long.npy", projected_waveforms
)
np.save(r"/mnt/data/waveform_snippets/max_channels_long.npy", max_channels)
# %%
projected_waveforms = np.load(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_18_11_2025/Phase_00/projected_waveforms.npy"
)
# %%
channel_to_plot = np.random.choice(252, 25, replace=False)
fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(15, 15))
for idx, ch in enumerate(channel_to_plot):
    # take subset
    ax[idx // 5, idx % 5].hexbin(
        projected_waveforms[max_channels == ch, 0],
        projected_waveforms[max_channels == ch, 1],
        cmap="viridis",
        gridsize=50,
        mincnt=1,
    )
    ax[idx // 5, idx % 5].set_title(f"Channel {ch} Projections")
fig.show()
# %%
from sklearn.cluster import HDBSCAN

all_electrods_labels = []
channel_indices = []
for ch in tqdm(range(252)):
    data = projected_waveforms[max_channels == ch]
    if data.shape[0] < 10:
        all_electrods_labels.append(np.array([]))
        continue
    clusterer = HDBSCAN(min_cluster_size=50, min_samples=10)
    labels = clusterer.fit_predict(data)
    all_electrods_labels.append(labels)

# %% Only cluster one channel for test

example_channel = 198
data = projected_waveforms[max_channels == example_channel]
clusterer = HDBSCAN(min_cluster_size=50, min_samples=10, n_jobs=-1)
labels = clusterer.fit_predict(data)
print(np.unique(labels, return_counts=True))

# %%
cluster_lengths = [len(np.unique(labels)) for labels in all_electrods_labels]

# %% get all spike times for all clusters
all_clusters_spike_times = []
for channel in range(252):
    channel_times = example_time[max_channels == channel]
    for cluster in np.unique(all_electrods_labels[channel]):
        cluster_times = channel_times[all_electrods_labels[channel] == cluster]
        all_clusters_spike_times.append(cluster_times)

# %%
spike_lengths = [array.shape for array in all_clusters_spike_times]

# %% create polars dataframe and then explode spikes
df = pl.DataFrame(
    {
        "cell_index": np.arange(len(all_clusters_spike_times)),
        "spike_times": all_clusters_spike_times,
    }
)
df = df.explode("spike_times")
df = df.with_columns(
    pl.col("spike_times")
    - recording.stimulus_df.query(f"stimulus_index=={stimulus_id}")["begin_fr"].values[
        0
    ]
)
df.write_parquet(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_18_11_2025/Phase_00/alternative_spike_times.parquet"
)
# %% plot clustering
example_channel = 198
data = projected_waveforms[max_channels == example_channel]
labels = all_electrods_labels[example_channel]
fig, ax = plt.subplots(figsize=(15, 15), ncols=5, nrows=5, sharex=True, sharey=True)
ax = ax.flatten()
component_pairs = [(i, j) for i in range(5) for j in range(5)]
for idx, (comp_x, comp_y) in enumerate(component_pairs):
    scatter = ax[idx].scatter(
        data[:, comp_x], data[:, comp_y], c=labels, s=1, cmap="tab10", alpha=0.5
    )
    ax[idx].set_xlabel(f"Comp {comp_x}")
    ax[idx].set_ylabel(f"Comp {comp_y}")
ax[0].set_xscale("symlog")
ax[0].set_yscale("symlog")
fig.suptitle(f"HDBSCAN Clustering on Channel {example_channel}")

fig.tight_layout()
fig.show()

# %%
fig, ax = plt.subplots(figsize=(15, 15), ncols=5, nrows=5)
ax = ax.flatten()
component_pairs = [(i, j) for i in range(5) for j in range(5)]
for idx, (comp_x, comp_y) in enumerate(component_pairs):
    scatter = ax[idx].hexbin(
        data[:, comp_x], data[:, comp_y], cmap="viridis", gridsize=50, mincnt=1
    )
    ax[idx].set_xlabel(f"Comp {comp_x}")
    ax[idx].set_ylabel(f"Comp {comp_y}")
fig.suptitle(f"HDBSCAN Clustering on Channel {example_channel}")
fig.tight_layout()
fig.show()
# %% plot 25 example channels
channel_to_plot = np.random.choice(252, 25, replace=False)
fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(15, 15), sharex=True, sharey=True)
for idx, ch in enumerate(channel_to_plot):
    data = projected_waveforms[max_channels == ch]
    data_subset_indices = np.random.choice(
        data.shape[0], min(500, data.shape[0]), replace=False
    )
    data_subset = data[data_subset_indices]
    labels = all_electrods_labels[ch][data_subset_indices]
    scatter = ax[idx // 5, idx % 5].scatter(
        data_subset[:, 0], data_subset[:, 1], c=labels, s=1, cmap="tab10", alpha=0.5
    )
    ax[idx // 5, idx % 5].set_title(f"Channel {ch} Clusters")
# use double log axis
ax[0, 0].set_xscale("symlog")
ax[0, 0].set_yscale("symlog")
fig.suptitle("HDBSCAN Clustering on Random Channels")
fig.show()

# %%


# %% get waveforms for channel
waveform_times = example_time[max_channels == example_channel]
cells_times = shuffled_cell_indices[max_channels == example_channel]
cluster1 = waveform_times[labels == -1]
np.unique(cells_times[labels == -1], return_counts=True)

# %%
bin_reader = io.BinaryFiltered(
    str(r"/mnt/data/waveform_snippets/alldata.dat"),
    252,
    # whiten_mat=torch.from_numpy(whitening_mat_inv).float(),
    do_CAR=True,
    device="cpu",
    NT=60000,
    fs=int(recording.sampling_freq),
)
# create indices -20 to +41 after spikes
all_spikes_index = np.array(np.split(waveform_times, waveform_times.shape[0]))
all_spikes_index = all_spikes_index - 20
all_spikes_index = all_spikes_index + np.arange(61)[None, :]
waveforms = bin_reader[all_spikes_index]
# %%
waveforms = waveforms.numpy()

# %%
# %% plot traces

all_channel_waveforms = waveforms[example_channel]
unique_labels = np.unique(labels)
fig, ax = plt.subplots(
    nrows=unique_labels.shape[0] + 1,
    ncols=1,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
)
colours = plt.cm.tab10(np.linspace(0, 1, unique_labels.shape[0]))
mean_maxes = []
for l_idx, label in enumerate(unique_labels):
    cluster_waveforms = all_channel_waveforms[:, labels == label].T
    mean_wf = np.mean(cluster_waveforms, axis=0)
    mean_maxes.append(np.max(np.abs(mean_wf)))
    # define random subset of waveforms to plot
    random_indices = np.random.choice(
        cluster_waveforms.shape[0],
        size=min(200, cluster_waveforms.shape[0]),
        replace=False,
    )
    cluster_waveforms = cluster_waveforms[random_indices]
    for wf in cluster_waveforms:
        ax[l_idx].plot(wf, alpha=0.1, color=colours[l_idx])

    ax[l_idx].plot(
        mean_wf,
        label=f"Cluster {label} Mean",
        linewidth=2,
        color=colours[l_idx],
    )
    ax[-1].plot(
        mean_wf,
        label=f"Cluster {label} Mean",
        linewidth=2,
        color=colours[l_idx],
    )
ax[0].set_ylim(-max(mean_maxes) * 1.5, max(mean_maxes) * 1.5)
fig.show()

# %% plot electrical image of specific cluster
square_dummy = np.empty((channel_row_col, channel_row_col))
square_dummy[:] = np.NaN
fig, ax = plt.subplots(
    ncols=unique_labels.shape[0], nrows=1, figsize=(unique_labels.shape[0] * 3, 3)
)
ax = np.atleast_1d(ax)
for l_idx, label in enumerate(unique_labels):
    cluster_waveforms = waveforms[:, :, labels == label].T
    footprint = np.var(np.mean(cluster_waveforms, axis=0), axis=0)
    square_dummy[combinations[:, 1], combinations[:, 0]] = footprint
    im = ax[l_idx].imshow(np.log(square_dummy), aspect="auto", cmap="viridis")
fig.show()

# %%
means = []
for l_idx, label in enumerate(unique_labels):
    cluster_waveforms = all_channel_waveforms[:, labels == label].T
    means.append(np.mean(cluster_waveforms, axis=0))
# %% project waveforms onto components

# projected_waveforms = np.zeros()
#     [
#     np.zeros((channel_count, n_components), dtype=np.float32)
#     for channel_count in channel_counts
# ]
# for channel in tqdm(range(n_ch)):
#     for store_idx, wave_idx in enumerate(np.where(min_channels == channel)[0]):
#         waveform = waveforms[wave_idx]
#         for component in range(n_components):
#             comp = reconstructions[channel, component]
#             comp_flat = einops.rearrange(comp, "c t -> (c t)")
#             waveform_flat = einops.rearrange(waveform, "c t -> (c t)")
#             projected_waveforms[channel][store_idx, component] = np.dot(
#                 waveform_flat, comp_flat
#             )

# %% plot 25 random channels' projections
channel_ids = np.random.choice(n_ch, size=25, replace=False)
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
ax = ax.flatten()
for plot_idx, channel in enumerate(channel_ids):
    ax[plot_idx].scatter(
        projected_waveforms[channel][:, 0],
        projected_waveforms[channel][:, 1],
        s=1,
        alpha=0.5,
    )
    ax[plot_idx].set_title(f"Channel {channel}")
fig.suptitle("Projections onto ICA Components for Random Channels")
fig.show()

# %% Try clustering on example channels
from sklearn.cluster import HDBSCAN

example_channel = 72
all_channel_waveforms = waveforms[np.where(min_channels == example_channel)[0]]
data = projected_waveforms[example_channel]
clustering = HDBSCAN(min_cluster_size=50)
labels = clustering.fit_predict(data)
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"Unique labels: {unique_labels}")
print(f"Counts: {counts}")

# %% Mean SHift
from sklearn.cluster import MeanShift

hdbscan_centers = np.array(
    [data[labels == l].mean(axis=0) for l in unique_labels if l != -1]
)
mean_shift = MeanShift(seeds=hdbscan_centers, cluster_all=False)
labels = mean_shift.fit_predict(data)
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"Unique labels: {unique_labels}")
print(f"Counts: {counts}")
# %%
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, s=1, cmap="tab20", alpha=0.5)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title(f"HDBSCAN Clustering on Channel {example_channel}")
fig.show()

# %% plot waveforms
fig, ax = plt.subplots(nrows=unique_labels.shape[0] + 1, ncols=1, figsize=(10, 10))
colours = plt.cm.tab20(np.linspace(0, 1, unique_labels.shape[0]))
for l_idx, label in enumerate(unique_labels):
    cluster_waveforms = all_channel_waveforms[labels == label]
    for wf in cluster_waveforms:
        ax[l_idx].plot(wf[example_channel], alpha=0.1, color=colours[l_idx])
    mean_wf = np.mean(cluster_waveforms, axis=0)
    ax[l_idx].plot(
        mean_wf[example_channel],
        label=f"Cluster {label} Mean",
        linewidth=2,
        color=colours[l_idx],
    )
    ax[-1].plot(
        mean_wf[example_channel],
        label=f"Cluster {label} Mean",
        linewidth=2,
        color=colours[l_idx],
    )

ax[0].set_title(f"Waveforms for Channel {example_channel}")
ax[0].legend()
fig.show()

# %% plot imshow
fig, ax = plt.subplots(ncols=unique_labels.shape[0], figsize=(20, 3))
for l_idx, label in enumerate(unique_labels):
    cluster_waveforms = all_channel_waveforms[labels == label]
    mean_wf = np.mean(cluster_waveforms, axis=0)
    square_dummy[combinations[:, 1], combinations[:, 0]] = np.var(mean_wf, axis=1)
    im = ax[l_idx].imshow(square_dummy, aspect="auto", cmap="viridis")
    ax[l_idx].set_title(f"Cluster {label} Mean Waveform")
    fig.colorbar(im, ax=ax[l_idx])
fig.show()

# %% project all waveforms onto all components
projected_waveforms = np.zeros((subset, n_ch, 2), dtype=np.float32)
for channel in tqdm(range(n_ch)):
    for component in range(2):
        comp = reconstructions[channel, component]
        comp_flat = einops.rearrange(comp, "c t -> (c t)")
        for spike_idx in range(subset):
            waveform_flat = einops.rearrange(waveforms[spike_idx], "c t -> (c t)")
            projected_waveforms[spike_idx, channel, component] = np.dot(
                waveform_flat, comp_flat
            )

# %%
# %%
projection_std = np.std(projected_waveforms)

fig, ax = plt.subplots(figsize=(15, 15))
ax.hist(projected_waveforms.flatten(), bins=500)
ax.vlines([-5 * projection_std, 5 * projection_std], ymin=0, ymax=10000, colors="red")
fig.show()

# %% kick out extreme projections
projected_waveforms = np.clip(
    projected_waveforms, -5 * projection_std, 5 * projection_std
)
projected_waveforms[projected_waveforms == -5 * projection_std] = 0
projected_waveforms[projected_waveforms == 5 * projection_std] = 0

# %%
fig, ax = plt.subplots(figsize=(15, 15))
ax.hist(projected_waveforms.flatten(), bins=500)
fig.show()
# %%
from sklearn.preprocessing import StandardScaler

waveform_features = einops.rearrange(projected_waveforms, "n c comp -> n (c comp)")
scaler = StandardScaler()
waveform_features_scaled = scaler.fit_transform(waveform_features)
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(waveform_features_scaled[:500, :], aspect="auto", cmap="viridis")
fig.show()

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
projected = pca.fit_transform(waveform_features_scaled)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(projected[:, 0], projected[:, 1], s=1, alpha=0.5)
fig.show()
# %% hdbscan for clustering
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.1)
labels = clustering.fit_predict(waveform_features_scaled)
np.unique(labels, return_counts=True)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(waveform_features_scaled[labels == 1][:500, :], aspect="auto", cmap="viridis")
fig.show()
