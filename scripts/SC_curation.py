"""
This script extracts waveforms of single cells based on spike sorted spike times. This version of the file works on
binary files in which frames are stored sequentially (all channels for frame 0, all channels for frame 1, etc.).
The waveforms are extracted in batches using a binary file handler from the kilosort package and pytorch.
The extracted waveforms are stored in an HDF5 file on disk to avoid memory issues. This way RAM limitations can be avoided
and electrical images can be calcualted on large datasets.
Be aware that the hdf5 file stored on disk is chunked based on frames not cell id. This means that accessing
extracting waveforms for a single cell is slow, while extracting waveforms for many cells in order is fast.
The script also calculates nearest neighbours based on template similarity and creates plots comparing the waveforms of
those neighbours. While not exhaustive, this can give a good indication of potential problematic mixed up units/cells.
@author: Marvin Seifert, 2026
"""

# Imports
from circus.shared.parser import CircusParser
from polarspike import Overview
import numpy as np
from kilosort import io
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import polars as pl
import einops
import torch
from tqdm import tqdm
import hdf5plugin
import dask.array as da
import subprocess
from scipy.ndimage import zoom
import warnings
import configparser
from scipy.signal import hilbert
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import IncrementalPCA
from tslearn.piecewise import SymbolicAggregateApproximation
from polarspike.sc_curation import get_spike_waveforms_parquet


# Functions
def get_spike_waveforms_hdf5(
        spikes,
        binary_file,
        n_channels=252,
        whitening_mat=None,
        chan=None,
        h5_path=None,
        nt=60000,
        fs=40000,
        save_chunk_size=1000,
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

    spikes = np.sort(np.atleast_1d(spikes)).astype(np.int64)
    nt = bin_reader.NT
    snippet_len = bin_reader.nt
    n_spikes = spikes.size
    n_output_channels = 1 if chan is not None else n_channels

    # 1. Create HDF5 file for output
    if h5_path is None:
        h5_path = "waveforms_cache.h5"
    try:
        f = h5py.File(h5_path, "w")
    except OSError as e:
        print(f"Error opening file {h5_path}: {e}")
        raise
    # Pre-allocate the dataset
    # We use (Spikes, Channels, Time) for better write alignment
    # Chunks are critical for performance: (1000 spikes per chunk)
    batch_indices = np.unique(spikes // nt)
    dset = f.create_dataset(
        "waveforms",
        shape=(n_spikes, n_output_channels, snippet_len),
        dtype="f4",  # int16
        chunks=(min(spikes.shape[0], save_chunk_size), n_output_channels, snippet_len),
        **hdf5plugin.Zstd(clevel=5),
    )
    # science: We load the batches that contain spikes, since we want to get frames in the window -20 to +41 after the spike
    # science: we need to calculate the batches accordingly

    print(f"Streaming {len(batch_indices)} batches from network to HDF5...")

    try:
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

            if chan is not None:
                batch_data = batch_data[chan, :]
                if batch_data.ndim == 1:
                    batch_data = batch_data.unsqueeze(0)

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

                # Transpose to (n_valid, n_channels, snippet_len) to match HDF5 dataset
                snippets = einops.rearrange(snippets, "c n t -> n c t")

                # Write all valid snippets in one operation
                dset[current_store_indices, :, :] = snippets

            # Periodically flush to disk to keep RAM clean
            if ibatch % 10 == 0:
                f.flush()

    finally:
        # Close the file
        f.close()
        print(f"Finished. Data saved to {h5_path}")

    return h5_path


def to_avi(out_path, imgs, fps, img_repeat=1):
    """Create a lossless AVI video from a list of images.

    Note: the images are _not_ flipped, so you may want to do this before if
    you are planning on using QDSpy to project onto a screen.
    """
    if out_path.suffix != ".avi":
        raise ValueError(
            "Output path must have .avi extension (extension is interpreted by ffmpeg)."
        )
    n, h, w, c = imgs.shape
    assert c == 3, "RGB images expected."
    # FFmpeg command
    # Ref: https://superuser.com/questions/347433/how-to-create-an-uncompressed-avi-from-a-series-of-1000s-of-png-images-using-ff/864520#864520
    # fmt: off
    command = [
        "ffmpeg",
        "-loglevel", "error",
        # Input is raw video with raw codec
        "-f", "rawvideo", "-vcodec", "rawvideo",
        # Input resolution
        "-s", f"{w}x{h}",
        # Input pixel format is 8-bit RGB (8x3 = 24 bits per pixel)
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        # Input comes from pipe
        "-i", "-",
        # Use lossless RGB H.264
        '-c:v', 'libx264rgb',
        # Maximum encoding speed
        '-preset', 'ultrafast',
        # Lossless quality
        '-qp', '0',
        "-an",  # No audio
        out_path,
    ]
    # fmt: on
    # Open subprocess with pipe
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Write frames to pipe
    for frame in imgs:
        for _ in range(img_repeat):
            try:
                proc.stdin.write(frame.tobytes())
            # catch broken pipe
            except BrokenPipeError:
                # get stderr output
                out, err = proc.communicate()
                print("FFmpeg error:", err.decode())
                break
    # Close pipe and wait for process to finish
    out, err = proc.communicate()
    # No need to wait, as proc.communicate() will wait for the process.
    # proc.stderr.close()
    # proc.stdin.close()
    # proc.wait()


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


def complex_pca(signals, n_components=None):
    """
    Complex PCA via SVD on the analytic signal.
    Args:
      signals       : array, shape (n_signals, n_samples), real-valued
      n_components  : int or None, number of components to keep
    Returns:
      U     : array, shape (n_signals, k), spatial maps (left singular vectors)
      S     : array, shape (k,), singular values
      Vh    : array, shape (k, n_samples), right singular vectors
      PCs   : array, shape (k, n_samples), S * Vh (complex principal time series)
      var_explained : array, shape (k,), fractional variance explained per component
    """
    # 1. Analytic signal
    A = hilbert(signals, axis=1)
    # 2. Center
    A = A - A.mean(axis=1, keepdims=True)
    # 3. SVD
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        Vh = Vh[:n_components, :]
    # 4. PCs and variance explained
    PCs = np.diag(S) @ Vh
    var_explained = (S ** 2) / np.sum(S ** 2)
    return U, S, Vh, PCs, var_explained


def complex_varimax(U, gamma=1.0, max_iter=100, tol=1e-6):
    """
    Performs Varimax rotation on a complex matrix U.
    """
    p, k = U.shape
    V = np.copy(U)
    d = 0

    for i in range(max_iter):
        d_old = d
        # Varimax criterion for complex numbers
        # This maximizes the variance of the squared magnitudes
        # to encourage 'sparsity' (some high, many low)
        V_sq = V * np.conj(V)
        mu = np.mean(V_sq, axis=0)

        # Calculate the rotation target
        target = V * (V_sq - gamma * mu)

        # Use SVD to find the closest unitary rotation matrix
        L, s, Mh = np.linalg.svd(np.conj(U.T) @ target)
        R = L @ Mh
        V = U @ R

        d = np.sum(s)
        if d_old != 0 and abs(d - d_old) < tol:
            break

    return V, R


# --- Example Usage ---
# display_key_params("alldata.params")

# %%
# Path parameters

# The root directory of the recording
root = Path(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_02_12_2025/Phase_01"
)

# The channel positions of the MEA array (used for plotting)
combinations = (
        np.load(r"/media/mawa/New Volume/electrical_imaging/channel_positions.npy")
        / 100  # Divide by 100 to get int positions (0, 1, 2...)
).astype(int)

params_file = root / "alldata.params"

params = get_key_params(params_file)
# Global parameters
electrode_spacing = 30.0  # in microns
combinations_um = combinations * electrode_spacing
channel_row_col = int(np.ceil(np.sqrt(combinations.shape[0])))
output_batch_size = 1000  # batch size of the output hdf5 file
nr_frames_per_waveform = 121  # number of time points to extract per waveform
batch_read_size = (
        60000 * 2
)  # number of time points to read from the binary file in each batch
stimulus_id = [4]  # This is the stimulus index to extract spikes from
max_spike_nr = 10000  # Maximum number of spikes to extract waveforms for (per cell). Increasing this will
# increase the time needed to extract waveforms but also improve the quality of the average waveform and electrical image.
print(f"Starting waveform extraction for recording at {root}.")
warnings.warn(
    "All existing analysis files and figures for this recording will be overwritten!"
)
# %%
# Load similarity matrix from Spyking-Circus output
with h5py.File(root / "alldata.templates.hdf5", "r") as f:
    similar = f["maxoverlap"][:]

# plot similarity matrix's upper triangle
similar[np.triu_indices_from(similar)] = 0
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(similar, cmap="viridis")
fig.colorbar(im, ax=ax)
fig.show()

# %%
# Load whitening matrix from the spikesorting output
with h5py.File(root / "alldata.basis.hdf5", "r") as f:
    whitening_mat_inv = f["spatial"][:]
    whitening_mat_spatial = f["spatial"][:]
whitening_mat_inv = whitening_mat_inv.astype(np.float16)  # invert later to save memory
# %% find the nearest neighbour for each cell
nearest_all = np.nanargmax(similar, axis=0)
# define nearest neighbour pairs
nn_pairs = np.hstack([np.arange(similar.shape[0])[:, None], nearest_all[:, None]])

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
spikes_subset = (
    spikes.with_columns(random_index=pl.lit(1).shuffle())
    .filter(pl.col("random_index").rank("ordinal").over("cell_index") <= max_spike_nr)
    .drop("random_index")
)
# Get as numpy array
example_time = spikes_subset["times"].to_numpy()
# We need to sort the times by frame in which they appear to optimize disk access. Also need to keep track of
# which time belongs to which cell after sorting:
sorted_indices = np.argsort(example_time)
inverse_sorted_indices = np.argsort(sorted_indices)
shuffled_cell_indices = spikes_subset["cell_index"].to_numpy()[sorted_indices]
example_time = example_time[sorted_indices]
# %%
# Get the waveforms from the file. This can take about one hour to multiple hours depending on the number of spikes
# per cell
waves = get_spike_waveforms_hdf5(
    example_time,
    root / "alldata.dat",
    whitening_mat=whitening_mat_inv,
    h5_path=r"/media/mawa/New Volume/waveform_snippets/waveforms_zebrafish.parquet",
    nt=batch_read_size,
    fs=int(recording.sampling_freq),
    save_chunk_size=output_batch_size,
    # Cast to int for compatibility, will cause problem if sampling freq is not integer. (On MCS systems it is integer)
)

# %%
# Here we lazy load the HDF5 file with dask to avoid loading everything into RAM. We need the lazy array to calculate
# averages per cell in the next step. Warning: Should you materialize the dask array, it will potentially crash your system
# due to out of memory issues.
h5_file = h5py.File(r"/media/mawa/New Volume/waveform_snippets/waveforms_new.h5", "r")
waveforms_dask = da.from_array(
    h5_file["waveforms"],
    chunks=(
        1000,
        h5_file["waveforms"].shape[1],
        h5_file["waveforms"].shape[2],
    ),
)
print(waveforms_dask.shape)

# %%
# Here, we calculate the mean waveform. This is done by loading chunks of waveforms and updating the mean per cell
# incrementally to avoid loading everything into RAM.
dset = h5_file["waveforms"]

# 1. Load the cell index mapping (Must match the sorted order of the HDF5 rows)
# Assuming 'shuffled_cell_indices' corresponds 1:1 to the rows in HDF5
cell_ids_per_spike = shuffled_cell_indices
n_cells = cell_ids_per_spike.max() + 1
n_spikes, n_ch, n_time = dset.shape
alphabet_size = 16
# 2. Initialize Accumulators in RAM
# (If 252ch * 61t is too large for all cells, process in batches of 100 cells)
mean_per_cell = np.zeros((n_cells, n_ch, n_time), dtype=np.float32)

# 3. Stream Processing (The "Inverted Loop")
sax_1d = SymbolicAggregateApproximation(n_segments=5, alphabet_size_avg=5)
print("Streaming data to compute means...")
for start_idx in tqdm(range(0, n_spikes, output_batch_size)):
    end_idx = min(start_idx + output_batch_size, n_spikes)

    # A. Read one big contiguous block from disk (FAST)
    waveforms_batch = dset[start_idx:end_idx]

    # B. Get corresponding cell IDs
    batch_cell_ids = cell_ids_per_spike[start_idx:end_idx]
    unique_ids = np.unique(batch_cell_ids)
    # C. vectorised aggregation is hard, simple loop is often fast enough for RAM operations
    # Alternatively, use np.add.at for vectorization
    for i, cell_id in enumerate(unique_ids):
        # Accumulate
        mean_per_cell[cell_id] += np.mean(
            waveforms_batch[batch_cell_ids == cell_id], axis=0
        )

# %%
fig, ax = plt.subplots()
ax.plot(mean_per_cell[100, :, :].T, color="gray", alpha=0.3)
fig.show()
# %%
projected_waveforms = [[] for _ in range(n_cells)]
residuals = [[] for _ in range(n_cells)]
for start_idx in tqdm(range(0, n_spikes, output_batch_size)):
    end_idx = min(start_idx + output_batch_size, n_spikes)

    # A. Read one big contiguous block from disk (FAST)
    waveforms_batch = dset[start_idx:end_idx]

    # B. Get corresponding cell IDs
    batch_cell_ids = cell_ids_per_spike[start_idx:end_idx]
    unique_ids = np.unique(batch_cell_ids)
    # C. vectorised aggregation is hard, simple loop is often fast enough for RAM operations
    # Alternatively, use np.add.at for vectorization
    for i, cell_id in enumerate(unique_ids):
        # Project each waveform to the mean waveform
        mean_waveform = mean_per_cell[cell_id]
        waveforms_temp = waveforms_batch[batch_cell_ids == cell_id]
        for j in range(waveforms_temp.shape[0]):
            projected_waveforms[cell_id].append(
                mean_waveform.flatten() @ waveforms_temp[j].flatten()
            )
            residuals[cell_id].append(
                np.linalg.norm(waveforms_temp[j].flatten() - mean_waveform.flatten())
            )

# %%
fig, ax = plt.subplots()
ax.hist(residuals[49], bins=50)
fig.show()

# %%
import diptest

dip_results = np.zeros((n_cells, 2))
for cell_id in range(n_cells):
    if len(projected_waveforms[cell_id]) > 100:
        dip, val = diptest.diptest(np.array(residuals[cell_id]))
        dip_results[cell_id, 0] = dip
        dip_results[cell_id, 1] = val
# %%
fig, ax = plt.subplots()
ax.hist(
    dip_results[:, 1], bins=50, weights=np.ones(len(dip_results)) / len(dip_results)
)
fig.show()
# %% split bimodal cells using gaussian mixture model
from sklearn.mixture import GaussianMixture

example_cell = 49
data = np.array(projected_waveforms[example_cell]).reshape(-1, 1)
gmm = GaussianMixture(n_components=2)
gmm.fit(data)
labels = gmm.predict(data)
fig, ax = plt.subplots()
ax.hist(data[labels == 0], bins=50, alpha=0.5, label="Cluster 0", color="k")
ax.hist(data[labels == 1], bins=50, alpha=0.5, label="Cluster 1", color="r")
fig.show()
# %% split the spikes based on this
all_spikes = []
for start_idx in tqdm(range(0, n_spikes, output_batch_size)):
    end_idx = min(start_idx + output_batch_size, n_spikes)

    # A. Read one big contiguous block from disk (FAST)
    waveforms_batch = dset[start_idx:end_idx]

    # B. Get corresponding cell IDs
    batch_cell_ids = cell_ids_per_spike[start_idx:end_idx]
    unique_ids = np.unique(batch_cell_ids)
    # C. vectorised aggregation is hard, simple loop is often fast enough for RAM operations
    # Alternatively, use np.add.at for vectorization

    waveforms_temp = waveforms_batch[batch_cell_ids == example_cell]
    all_spikes.append(waveforms_temp)
# %%
all_spikes = np.vstack(all_spikes)
label_0_spikes = all_spikes[labels == 0]
label_1_spikes = all_spikes[labels == 1]
mean_label_0 = np.mean(label_0_spikes, axis=0)
mean_label_1 = np.mean(label_1_spikes, axis=0)

# %%
# find best channel for each
min_first = np.min(mean_label_0, axis=1)
best_channel_first = np.argmin(min_first)
min_second = np.min(mean_label_1, axis=1)
best_channel_second = np.argmin(min_second)

fig, ax = plt.subplots(
    figsize=(30, 15), ncols=2, nrows=1, gridspec_kw={"width_ratios": [1, 2]}
)
ax[0].plot(
    mean_label_0[best_channel_first, :],
    label=f"Label 0",
    lw=3,
    c="k",
)
ax[0].plot(
    mean_label_1[best_channel_first, :],
    label=f"Label 1",
    lw=3,
    c="r",
)
ax[0].set_xlabel("Time (samples)")
ax[0].set_ylabel("Amplitude (a.u.)")
ax[0].legend()
n_ch, n_t = mean_label_0.shape
t = np.linspace(-0.4, 0.4, n_t)
amp_scale = -0.35 / (np.nanmax(np.abs(mean_label_0)) + 1e-12)
amp_scale_similar = -0.35 / (np.nanmax(np.abs(mean_label_1)) + 1e-12)

for i in range(n_ch):
    y, x = combinations[i]
    ax[1].plot(x + t, y + amp_scale * mean_label_0[i], color="k", lw=3)
    ax[1].plot(
        x + t,
        y + amp_scale_similar * mean_label_1[i],
        color="r",
        lw=3,
    )

ax[1].set_aspect("equal")
ax[1].set_xlim(combinations[:, 1].min() - 0.6, combinations[:, 1].max() + 0.6)
ax[1].set_ylim(combinations[:, 0].min() - 0.6, combinations[:, 0].max() + 0.6)
ax[1].invert_yaxis()
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
fig.show()

# %%
mean_example = mean_per_cell[example_cell, :, :]
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(mean_example.T)
# %% plot pc 1 and 2
square_dummy = np.empty((channel_row_col, channel_row_col))
square_dummy[:] = np.NaN
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
for i in range(10):
    square_dummy[combinations[:, 1], combinations[:, 0]] = pca.components_[i]
    ax[i // 5, i % 5].imshow(square_dummy, label=f"PC {i + 1}")
    ax[i // 5, i % 5].legend()
fig.show()
# print variance explained
print("Variance explained by PCs:", pca.explained_variance_ratio_)

# %% Do the same with complex pca
U, S, Vh, PCs, var_explained = complex_pca(mean_example, n_components=10)
# print variance explained
fig, ax = plt.subplots()
ax.scatter(np.arange(len(var_explained)), var_explained * 100)
ax.set_xlabel("Component")
ax.set_ylabel("Variance Explained (%)")
fig.show()
# %%


channel_phases = np.angle(U[:, 0])
magnitudes = np.abs(U[:, 0])
# check the phase of the most important pixel
most_important_phase = np.angle(U[np.argmax(magnitudes), 0])
# % normalize the phases so that 0 is the most important pixel
channel_phases = np.angle(np.exp(1j * (channel_phases - most_important_phase)))

# %%
fig, ax = plt.subplots()
square_dummy[combinations[:, 1], combinations[:, 0]] = channel_phases
im = ax.imshow(square_dummy, cmap="hsv", vmin=-np.pi, vmax=np.pi)
fig.colorbar(im, ax=ax, label="Phase (radians)")
fig.show()
# %%
fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
ax[0].plot(np.real(PCs).T, label="Real part")
ax[1].plot(np.imag(PCs).T, label="Imaginary part")
ax[0].legend()
ax[1].legend()
fig.show()

# %%
fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
ax[0].plot(np.degrees(np.angle(PCs[:2])).T, label="Real part")
ax[1].plot(np.real(PCs[:2]).T, label="Real part")
fig.show()
# %%
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
for i in range(10):
    square_dummy[combinations[:, 1], combinations[:, 0]] = np.abs(U[:, i])
    ax[i // 5, i % 5].imshow(square_dummy, label=f"PC {i + 1}")
    ax[i // 5, i % 5].legend()
fig.show()

# %%
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
for i in range(10):
    square_dummy[combinations[:, 1], combinations[:, 0]] = np.degrees(np.angle(U[:, i]))
    ax[i // 5, i % 5].imshow(
        square_dummy, label=f"PC {i + 1}", vmin=-180, vmax=180, cmap="hsv"
    )
    ax[i // 5, i % 5].legend()
fig.show()

# %%
U_rotated, R = complex_varimax(U, gamma=1.0, max_iter=100, tol=1e-6)

# %%
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
for i in range(10):
    square_dummy[combinations[:, 1], combinations[:, 0]] = np.abs(U_rotated[:, i])
    ax[i // 5, i % 5].imshow(square_dummy, label=f"Rotated PC {i + 1}")
    ax[i // 5, i % 5].legend()
fig.show()
# %%
PCs_rotated = np.conj(R.T) @ PCs
ref_phase_rad = np.angle(PCs_rotated[0, 20])
# 2. Align all PCs by this reference phase
# This preserves the relative phase between different components
PCs_aligned = PCs_rotated * np.exp(-1j * ref_phase_rad)
# 3. Calculate relative timing (phase in degrees)
# We use np.angle on the aligned PCs and wrap to [-180, 180]
phase_deg = np.degrees(np.angle(PCs_aligned))
relative_timing = (phase_deg.T - phase_deg[0, :][:, np.newaxis] + 180) % 360 - 180

# %%
anchor_phase = np.angle(PCs_rotated[0, 20])

# 2. Rotate all components by this single scalar value
# This aligns the whole system so PC0 has phase 0 at t=20
PCs_aligned = PCs_rotated * np.exp(-1j * anchor_phase)

# 3. Calculate the timing (phase) for all components
# Now PC0[20] will be 0, but other time points for PC0 will show progression
timing_deg = np.degrees(np.angle(PCs_aligned))

# 4. (Optional) Wrap to [-180, 180] for clean plotting
timing_deg = (timing_deg + 180) % 360 - 180
# %%
import colorcet as cc

fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
ax[0].plot(np.real(PCs_rotated).T, label="Real part")
ax[1].plot(np.imag(PCs_rotated).T, label="Imaginary part")
ax[0].legend()
ax[1].legend()
fig.show()

# %%
phase1 = np.degrees(np.angle(PCs_rotated[0, :]))
relative_timing = (
        np.degrees(np.angle(PCs_rotated)).T - phase1[:, np.newaxis] + 180 % 360 - 180
)
# %%
fig, ax = plt.subplots(nrows=2, figsize=(15, 7), sharex=True)
im = ax[0].imshow(
    timing_deg, label="Real part", vmin=-180, vmax=180, cmap=cc.cm.CET_C4s
)
# fig.colorbar(im, ax=ax[0], label="Phase (degrees)")
# flip y axis
# add 0 line
ax[1].axhline(0, color="k", linestyle="--", label="Reference time point")
ax[1].axvline(20, color="r", linestyle="--", label="Anchor time point")
ax[0].invert_yaxis()
ax[1].plot(timing_deg[:2, :].T, label="Real part")
fig.show()

# %%
# 1. Get the phase of every PC at your anchor point (t=20)
# We use the aligned PCs so PC0 is the 0-degree baseline
anchor_phases = np.angle(PCs_aligned[:, 20])

# 2. Find the sorting order (from smallest phase to largest)
# This orders them by their "timing" relative to your anchor
sort_idx = np.argsort(anchor_phases)

# 3. Reorder your data
PCs_sorted = PCs_aligned[sort_idx, :]
U_sorted = U_rotated[:, sort_idx]  # Don't forget to reorder the spatial maps!
var_explained_sorted = var_explained[sort_idx]  # If you want to keep track of variance

# %%
fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
ax[0].plot(np.real(PCs_sorted).T, label="Real part")
ax[1].plot(np.imag(PCs_sorted).T, label="Imaginary part")
ax[0].legend()
ax[1].legend()
fig.show()

# %%
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
for i in range(10):
    square_dummy[combinations[:, 1], combinations[:, 0]] = np.abs(U_sorted[:, i])
    ax[i // 5, i % 5].imshow(square_dummy, label=f"Rotated PC {i + 1}")
    ax[i // 5, i % 5].legend()
fig.show()

# %%
u_1 = U_sorted[:, 3:4]
pc_1 = PCs_sorted[3:4, :]

# 2. Reconstruct the complex analytic signal
complex_reconstruction = u_1 @ pc_1

# 3. Project back to real space
# This is the signal as it would appear in your original raw data
real_reconstruction = np.real(complex_reconstruction)

# %%
envelopes = np.abs(PCs_sorted)

# Check the correlation between envelopes
# If two PCs are parts of the same "jumping" process,
# their envelopes will be highly related (either correlated or perfectly complementary)
env_corr = np.corrcoef(envelopes)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(env_corr, cmap="viridis", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label="Envelope Correlation")
fig.show()

# %%
jumping_indices = [4]  # Based on your visual smear
u_jumping = U_sorted[:, jumping_indices]
pc_jumping = PCs_sorted[jumping_indices, :]

# This gives you a single complex matrix for the jumping signal
jumping_signal_complex = u_jumping @ pc_jumping
real_jumping_signal = np.real(jumping_signal_complex)
# %%
video_zoom = 50
fps = 10
video_array = np.zeros((real_jumping_signal.shape[1], channel_row_col, channel_row_col))
video_array[:, combinations[:, 1], combinations[:, 0]] = real_jumping_signal.T
video_array_rgb = np.repeat(video_array[..., None], 3, axis=-1)
# transform to uint8 by normalizing to 0-255
eps = 1e-9
log_scaled = np.log1p(video_array_rgb - video_array_rgb.min() + eps)
video_array_rgb = (
        (log_scaled - log_scaled.min()) / (log_scaled.max() - log_scaled.min() + eps) * 255
)
video_array_rgb = zoom(
    video_array_rgb, (1, video_zoom, video_zoom, 1), order=1, mode="constant"
)
# repeat frames if needed
video_array_rgb = video_array_rgb.astype(np.uint8, copy=False)
# ensure C‑contiguous (t, h, w, 3)
video_array_rgb = np.ascontiguousarray(video_array_rgb)

to_avi(
    Path(r"/media/mawa/New Volume/electrical_imaging/test_stationary.avi"),
    video_array_rgb,
    fps=fps,
    img_repeat=1,
)

# %%
from sklearn.decomposition import FastICA

ica = FastICA(n_components=3, random_state=0, max_iter=10000)
S_ = ica.fit_transform(mean_example.T)  # Reconstructed signals
A_ = ica.mixing_  # Get estimated mixing matrix
# %% plot ica components
fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
for i in range(3):
    square_dummy[combinations[:, 1], combinations[:, 0]] = A_[:, i]

    ax[i // 5, i % 5].imshow(square_dummy, label=f"ICA Component {i + 1}")
    ax[i // 5, i % 5].legend()
fig.show()

# %%
# find max channel for each component
maxes = []
for i in range(3):
    max_channel = np.nanargmax(A_[:, i])
    maxes.append(max_channel)

# %%
spikes_important_channels = np.zeros((all_spikes.shape[0], 3, all_spikes.shape[2]))
for i in range(3):
    spikes_important_channels[:, i, :] = all_spikes[
        :, maxes[i], :
    ]  # get spikes for important channel

# %%
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.viridis(np.linspace(0, 1, 200))
)
fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(15, 5), sharey="row")
for i in range(3):
    ax[0, i].imshow(spikes_important_channels[:, i, :], aspect="auto", cmap="viridis")
    random_subset = np.random.choice(
        spikes_important_channels.shape[0], size=200, replace=False
    )
    ax[1, i].plot(spikes_important_channels[random_subset, i, :].T, alpha=0.3)
    ax[1, i].plot(np.mean(spikes_important_channels, axis=(0, 1)), c="black", lw=3)
    # calculate residual after removing mean
    residuals = spikes_important_channels[random_subset, i, :] - np.mean(
        spikes_important_channels, axis=(0, 1)
    )
    ax[2, i].plot(residuals.T)
    pca = PCA(n_components=2)
    pca.fit(residuals)
    pc_scores = pca.transform(spikes_important_channels[:, i, :])
    ax[3, i].scatter(
        pc_scores[:, 0],
        pc_scores[:, 1],
        color=plt.cm.viridis(np.linspace(0, 1, pc_scores.shape[0])),
    )
    ax[3, i].set_title(f"Explained var: {pca.explained_variance_ratio_}")

fig.show()

# %%
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Assuming spikes_important_channels is already defined
# shape: (n_spikes, n_channels, n_timepoints)
n_spikes, n_channels, n_timepoints = spikes_important_channels.shape
subset_size = 200
random_subset = np.random.choice(n_spikes, size=subset_size, replace=False)

# Calculate global mean for residuals (as per your original code)
global_mean = np.mean(spikes_important_channels, axis=(0, 1))

# Initialize Subplots
fig = make_subplots(
    rows=4,
    cols=3,
    subplot_titles=[f"Col {i}" for i in range(3)],
    vertical_spacing=0.05,
    shared_yaxes=True,  # Mimics sharey="row"
    row_heights=[0.25, 0.25, 0.25, 0.25],
)

for i in range(3):
    # --- Row 1: Imshow (Heatmap) ---
    fig.add_trace(
        go.Heatmap(
            z=spikes_important_channels[:, i, :], colorscale="Viridis", showscale=False
        ),
        row=1,
        col=i + 1,
    )

    # --- Row 2: Line Plots ---
    # Plotting mean
    fig.add_trace(
        go.Scatter(
            y=global_mean,
            line=dict(color="black", width=3),
            name="Mean",
            showlegend=False,
        ),
        row=2,
        col=i + 1,
    )

    # Plot subset lines
    # We add them as individual traces to allow for selection, but use legendgroup for organization
    for idx in random_subset:
        fig.add_trace(
            go.Scatter(
                y=spikes_important_channels[idx, i, :],
                line=dict(color="rgba(100, 100, 100, 0.3)", width=1),
                hoverinfo="skip",
                showlegend=False,
                meta={"spike_idx": idx},  # Storing index for reference
            ),
            row=2,
            col=i + 1,
        )

    # --- Row 3: Residuals ---
    residuals = spikes_important_channels[random_subset, i, :] - global_mean
    for j, idx in enumerate(random_subset):
        fig.add_trace(
            go.Scatter(
                y=residuals[j, :],
                line=dict(width=1),
                showlegend=False,
                meta={"spike_idx": idx},
            ),
            row=3,
            col=i + 1,
        )

    # --- Row 4: PCA Scatter ---
    pca = PCA(n_components=2)
    # Using the subset residuals for PCA fit to match your logic
    pca.fit(residuals)
    pc_scores = pca.transform(spikes_important_channels[:, i, :])

    fig.add_trace(
        go.Scatter(
            x=pc_scores[:, 0],
            y=pc_scores[:, 1],
            mode="markers",
            marker=dict(
                color=np.arange(n_spikes), colorscale="Viridis", size=5, opacity=0.7
            ),
            name=f"PCA Col {i}",
            text=[f"Spike ID: {j}" for j in range(n_spikes)],
            showlegend=False,
        ),
        row=4,
        col=i + 1,
    )
    # Update row 4 title with Explained Var
    var_ratio = np.round(pca.explained_variance_ratio_.sum(), 3)
    fig.update_xaxes(title_text=f"Expl. Var: {var_ratio}", row=4, col=i + 1)

# Update layout for selection
fig.update_layout(
    height=1000,
    width=1200,
    template="simple_white",
    dragmode="lasso",  # Default tool for selection
    hovermode="closest",
)

fig.show(renderer="browser")
# %%
fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(15, 12), sharey="row")

for i in range(3):
    ax[0, i].imshow(spikes_important_channels[:, i, :], aspect="auto", cmap="viridis")

    random_subset = np.random.choice(
        spikes_important_channels.shape[0], size=100, replace=False
    )

    # 1. Plot lines and capture the color of each line
    lines = ax[1, i].plot(spikes_important_channels[random_subset, i, :].T, alpha=0.3)
    colors = [l.get_color() for l in lines]  # Extract colors used in the line plot

    ax[1, i].plot(np.mean(spikes_important_channels, axis=(0, 1)), c="black", lw=3)

    # Calculate residuals
    mean_signal = np.mean(spikes_important_channels, axis=(0, 1))
    residuals = spikes_important_channels[random_subset, i, :] - mean_signal

    # 2. Apply the same colors to the residual lines
    for j in range(len(random_subset)):
        ax[2, i].plot(residuals[j, :], color=colors[j], alpha=0.3)

    # PCA
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(residuals)

    # 3. Use the extracted 'colors' list for the scatter plot
    ax[3, i].scatter(pc_scores[:, 0], pc_scores[:, 1], c=colors, s=10)
    ax[3, i].set_title(f"Exp var: {pca.explained_variance_ratio_.sum():.2f}")
fig.show()
# %% Reconstruct the signals from ICA components
stationary_source = S_[:, 1:2]  # (Time, 1)
stationary_mixing = A_[:, 1:2]  # (Sensors, 1)

# Resulting shape: (Time, Sensors)
reconstructed_stationary = stationary_source @ stationary_mixing.T

# 3. Transpose back if you need it in the original (Sensors, Time) shape
final_output = reconstructed_stationary.T

# %% plot argmax of each component as scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(10):
    max_channel = np.argmax(np.abs(U_rotated[:, i]))
    y, x = combinations[max_channel]
    ax.scatter(x, y, label=f"Rotated PC {i + 1}")
ax.invert_yaxis()
ax.legend()
ax.set_ylim(0, channel_row_col)
ax.set_xlim(0, channel_row_col)
fig.show()
# %%
U, S, Vh, PCs, var_explained = complex_pca(
    einops.rearrange(all_spikes, "n c t -> n (c t)"), n_components=10
)

# %% plot pc 1 and 2
# %% plot p values histogram
fig, ax = plt.subplots()
ax.hist(dip_results[:, 1], bins=50)
fig.show()
# %%
np.where(dip_results[:, 1] < 0.05)
# %% Export the means as numpy array
# We need to create a square array to store the mean waveforms for each cell, since the MEA layout is not square (252 channels)
# with missing corners.
mean_per_cell_square = np.empty(
    (mean_per_cell.shape[0], channel_row_col, channel_row_col, mean_per_cell.shape[2]),
    dtype=mean_per_cell.dtype,
)
mean_per_cell_square[:] = np.NaN
mean_per_cell_square[:, combinations[:, 1], combinations[:, 0], :] = mean_per_cell
np.save(
    root / "mean_waveforms_per_cell.npy",
    mean_per_cell_square,
)
# %%
# Here, we create plots comparing the waveforms of nearest neighbour cells based on template similarity
# and save them to disk for each pair.
output_dir = root / "nn_waveforms"
output_dir.mkdir(exist_ok=True, parents=True)
for pair in tqdm(nn_pairs, desc="Plotting nearest neighbour waveforms to files"):
    try:
        # find best channel for each
        min_first = np.min(mean_per_cell[pair[0]], axis=1)
        best_channel_first = np.argmin(min_first)
        min_second = np.min(mean_per_cell[pair[1]], axis=1)
        best_channel_second = np.argmin(min_second)
        mean_first = mean_per_cell[pair[0], :, :]
        mean_second = mean_per_cell[pair[1], :, :]

        fig, ax = plt.subplots(
            figsize=(30, 15), ncols=2, nrows=1, gridspec_kw={"width_ratios": [1, 2]}
        )
        ax[0].plot(
            mean_per_cell[pair[0], best_channel_first, :],
            label=f"Cell {pair[0]}",
            lw=3,
            c="k",
        )
        ax[0].plot(
            mean_per_cell[pair[1], best_channel_second, :],
            label=f"Cell {pair[1]}",
            lw=3,
            c="r",
        )
        ax[0].set_xlabel("Time (samples)")
        ax[0].set_ylabel("Amplitude (a.u.)")
        ax[0].legend()
        n_ch, n_t = mean_first.shape
        t = np.linspace(-0.4, 0.4, n_t)
        amp_scale = -0.35 / (np.nanmax(np.abs(mean_first)) + 1e-12)
        amp_scale_similar = -0.35 / (np.nanmax(np.abs(mean_second)) + 1e-12)

        for i in range(n_ch):
            y, x = combinations[i]
            ax[1].plot(x + t, y + amp_scale * mean_first[i], color="k", lw=3)
            ax[1].plot(x + t, y + amp_scale_similar * mean_second[i], color="r", lw=3)

        ax[1].set_aspect("equal")
        ax[1].set_xlim(combinations[:, 1].min() - 0.6, combinations[:, 1].max() + 0.6)
        ax[1].set_ylim(combinations[:, 0].min() - 0.6, combinations[:, 0].max() + 0.6)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        fig.tight_layout()
        fig.savefig(output_dir / f"cell_{pair[0]}_vs_cell_{pair[1]}.png")
        plt.close(fig)
    except IndexError:
        pass

h5_file.close()

# %%
# Here we create electrical images for each cell and save them as AVI videos to disk
video_zoom = 20  # zoom factor for better visibility
fps = 25
output_dir = root / "electrical_images"
output_dir.mkdir(exist_ok=True, parents=True)
# Need to delete all .avi files in the output dir first to avoid confusion
for avi_file in output_dir.glob("*.avi"):
    avi_file.unlink()
for cell in tqdm(
        range(recording.nr_cells), desc="Saving electrical image videos to file"
):
    waves_subset = mean_per_cell[cell, :, :]
    video_array = np.zeros((waves_subset.shape[1], channel_row_col, channel_row_col))
    video_array[:, combinations[:, 1], combinations[:, 0]] = waves_subset.T
    video_array_rgb = np.repeat(video_array[..., None], 3, axis=-1)
    # transform to uint8 by normalizing to 0-255
    eps = 1e-9
    log_scaled = np.log1p(video_array_rgb - video_array_rgb.min() + eps)
    video_array_rgb = (
            (log_scaled - log_scaled.min())
            / (log_scaled.max() - log_scaled.min() + eps)
            * 255
    )
    video_array_rgb = zoom(
        video_array_rgb, (1, video_zoom, video_zoom, 1), order=1, mode="constant"
    )
    # repeat frames if needed
    video_array_rgb = video_array_rgb.astype(np.uint8, copy=False)
    # ensure C‑contiguous (t, h, w, 3)
    video_array_rgb = np.ascontiguousarray(video_array_rgb)

    to_avi(
        Path(output_dir / f"example_{cell}.avi"),
        video_array_rgb,
        fps=fps,
        img_repeat=1,
    )
