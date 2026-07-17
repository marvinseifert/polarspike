import configparser
import subprocess
from pathlib import Path
import h5py
import hdf5plugin
import numpy as np
import einops
from scipy.signal import hilbert
from tqdm import tqdm
from kilosort import io
import torch
import polars as pl
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


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


def get_spike_waveforms_parquet(
        spikes,
        binary_file,
        n_channels=252,
        whitening_mat=None,
        chan=None,
        parquet_path="waveforms.parquet",
        nt=60000,
        fs=40000,
        **kwargs,
):
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
    # ... [Keep your bin_reader initialization here] ...

    spikes = np.sort(np.atleast_1d(spikes)).astype(np.int64)
    nt = bin_reader.NT
    snippet_len = bin_reader.nt
    n_spikes = spikes.size
    n_output_channels = 1 if chan is not None else n_channels
    batch_indices = np.unique(spikes // nt)
    writer = None  # We'll initialize this when we have our first batch

    print(f"Streaming {len(batch_indices)} batches to Parquet...")

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

                # 2. Create a Polars DataFrame for this chunk
                batch_df = pl.DataFrame(
                    {
                        "current_store_indices": current_store_indices,  # Stores as Int64
                        "waveform": snippets,  # Stores as List(Float32)
                    }
                )

                # 3. Initialize ParquetWriter on the first batch
                if writer is None:
                    # Convert to Arrow Table to get schema
                    arrow_table = batch_df.to_arrow()
                    writer = pq.ParquetWriter(
                        parquet_path,
                        arrow_table.schema,
                        compression="zstd",
                        compression_level=5,
                    )

                # 4. Write this chunk to the file
                writer.write_table(batch_df.to_arrow())

    finally:
        if writer:
            writer.close()
        print(f"Finished. Data saved to {parquet_path}")

    return parquet_path


# %% Now we perform a gaussian mixture to test if there are more than two components
def fast_obvious_clusters(
        data: np.ndarray, max_components: int = 6, covariance_type="tied"
) -> int:
    best_icl = float("inf")
    best_n = 1

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=n, covariance_type=covariance_type, random_state=42
        ).fit(data)
        bic = gmm.bic(data)
        if np.any(
                gmm.weights_ < 0.1
        ):  # Ignore models where any cluster is < 10% of data
            continue

        if n == 1:
            icl = bic
        else:
            # predict_proba returns [n_samples, n_components]
            probs = gmm.predict_proba(data)
            # Standard ICL Entropy penalty
            # We use 2 * entropy to match the -2*log-likelihood scale of BIC
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            icl = bic + 2 * entropy

        if icl < best_icl:
            best_icl = icl
            best_n = n

    return best_n


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance.
    xy: position
    width/height: calculated from eigenvalues
    angle: rotation in degrees anti-clockwise
    """
    ax = ax or plt.gca()

    # 1. Handle Alpha: Pop it so it's not in kwargs when we call Ellipse
    base_alpha = kwargs.pop("alpha", 0.3)

    # 2. Standardize Covariance Shape
    if covariance.shape == (2, 2):
        cov = covariance
    elif covariance.shape == (2,):  # 'diag'
        cov = np.diag(covariance)
    else:  # 'spherical' or 'tied'
        cov = np.eye(2) * covariance if np.isscalar(covariance) else covariance

    # 3. Calculate Geometry (Eigenvalues and Eigenvectors)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # angle is the rotation of the first eigenvector
    angle_deg = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # width/height are the diameters (2 * radius) along principal axes
    # vals are variances, so sqrt(vals) is the standard deviation
    w, h = 2 * np.sqrt(vals)

    # 4. Draw 1, 2, and 3-sigma boundaries
    for nsig in range(1, 4):
        # Calculate a fading alpha for outer rings
        ring_alpha = base_alpha / nsig

        # Explicitly map to Ellipse arguments: xy, width, height, angle
        ell = Ellipse(
            xy=position,
            width=nsig * w,
            height=nsig * h,
            angle=angle_deg,
            alpha=ring_alpha,
            fill=False,
            **kwargs,  # Remaining styles like edgecolor, linewidth, etc.
        )
        ax.add_patch(ell)
