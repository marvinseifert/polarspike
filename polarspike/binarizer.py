"""
Binarizer module. This module contains functions to convert timestamps to binary signals. This process is basically binning
the timestamps into bins, so that each bin contains a 1 if a spike occurred in that bin, and a 0 otherwise.
This module makes heavy use of the Polars library to increase performance.
@ Marvin Seifert 2024
"""

import polars as pl
from functools import partial
import numpy as np
import scipy.signal as signal


def timestamps_to_binary_polars(
        df_timestamps: pl.DataFrame, sample_rate: int = None, max_window: int = None
) -> pl.DataFrame:
    """
    This function converts timestamps to binary signals. It takes a DataFrame with timestamps and converts them to binary
    signals. Conversion is done at a given sample rate and considering a window of a given size.
    Parameters
    ----------
    df_timestamps : pl.DataFrame
        DataFrame with timestamps
    sample_rate : int
        The sample rate at which the timestamps will be binarized
    max_window : int
        The maximum window size for the binary signal
    Returns
    -------
    pl.DataFrame
        DataFrame with the binary signals

    """
    if sample_rate is None:
        sample_rate = int(1 / df_timestamps["times_triggered"].sort().diff().min())
    if max_window is None:
        max_window = np.ceil(df_timestamps["times_triggered"].max()).astype(int)
    # Create a DataFrame with the timestamps
    # Calculate the indices for the timestamps based on the sampling rate
    df_timestamps = df_timestamps.with_columns(
        (df_timestamps["times_triggered"] / sample_rate).cast(pl.Int64).alias("index")
    )

    # Create a DataFrame with a sequence of zeros up to the maximum index
    # max_index = df_timestamps.select(pl.col("index").max()).item()

    df_binary_signal = pl.DataFrame(
        {"index": list(range(max_window + 1)), "value": [0] * (max_window + 1)},
        schema={"index": pl.Int64, "value": pl.UInt8},
    )

    # Overlay the ones onto the zeros based on the calculated indices
    df_result = df_binary_signal.join(
        df_timestamps, on="index", how="left"
    ).with_columns(
        pl.when(pl.col("times_triggered").is_null())
        .then(0)
        .otherwise(1)
        .cast(pl.UInt8)
        .alias("value")
    )
    if len(df_result) != max_window + 1:
        df_result = df_result.unique(subset="index")

    df_result = df_result.with_columns(
        pl.col("cell_index").fill_null(strategy="backward")
    ).with_columns(pl.col("repeat").fill_null(strategy="backward"))
    df_result = (
        df_result.with_columns(pl.col("cell_index").fill_null(strategy="forward"))
        .with_columns(pl.col("repeat").fill_null(strategy="forward"))
        .select("cell_index", "value", "repeat", "index")
    )

    # Fill dataframes with single spikes
    if df_result["cell_index"].unique().max() is None:
        df_result = df_result.with_columns(
            cell_index=pl.lit(df_timestamps["cell_index"])
        )
        df_result = df_result.with_columns(repeat=pl.lit(df_timestamps["repeat"]))

    # If cell didn't spike in a repeat, fill with zeros

    return df_result[:max_window]


def fill_missing_repeats(
        max_repeat: int, nr_bins: int, df_result: pl.DataFrame
) -> pl.DataFrame:
    """
    This function fills missing repeats with zeros. This is necessary to have a consistent DataFrame that is all
    zeros for missing repeats.

    Parameters
    ----------
    max_repeat : int
        The maximum number of repeats
    nr_bins : int
        The number of bins per repeat
    df_result : pl.DataFrame
        The DataFrame with the binary signals

    Returns
    -------
    pl.DataFrame
        The DataFrame with the missing repeats filled with zeros

    """
    if len(df_result["repeat"].unique()) == max_repeat:
        return df_result

    repeats = df_result["repeat"].unique().to_numpy()
    cell_index = df_result["cell_index"][0]
    expected_repeats = np.arange(max_repeat)
    missing_repeats = np.setdiff1d(expected_repeats, repeats)
    dfs = []
    for missing_repeat in missing_repeats:
        dfs.append(
            pl.DataFrame(
                data={
                    "cell_index": [cell_index] * nr_bins,
                    "value": [0] * nr_bins,
                    "repeat": [missing_repeat] * nr_bins,
                    "index": np.arange(nr_bins),
                },
                schema={
                    "cell_index": pl.Int64,
                    "value": pl.UInt8,
                    "repeat": pl.Int64,
                    "index": pl.Int64,
                },
            )
        )

    dfs.append(df_result)
    df_result = pl.concat(dfs)
    return df_result


def apply_on_group(
        sample_rate: int, max_window: int, group_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Apply the timestamps_to_binary_polars function on a group of a DataFrame.
    This is just a wrapper function for parallelization.

    Parameters
    ----------
    sample_rate : int
        The sample rate at which the timestamps will be binarized
    max_window : int
        The maximum window size for the binary signal
    group_df : pl.GroupBy
        The group of the DataFrame

    Returns
    -------
    pl.DataFrame
        The DataFrame with the binary signals

    """
    try:
        return timestamps_to_binary_polars(group_df, sample_rate, max_window)
    except Exception as e:  # This is broad, but we want to catch all exceptions
        print(group_df)


def timestamps_to_binary_multi(
        df: pl.DataFrame, bin_size: int, max_window: int, max_repeat: int
) -> pl.DataFrame:
    """
    Convert timestamps to binary signals for multiple cells.
    Warning: Cannot handle multiple recordings in one DataFrame.
    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame with the timestamps
    bin_size : int
        The bin size
    max_window : int
        The maximum window size of the binary signal
    max_repeat : int
        The maximum number of repeats of the stimulus
    """
    max_window = np.ceil(max_window / bin_size).astype(int)
    nr_cells = df["cell_index"].unique().len()
    # Reject spikes larger than max_window
    df = df.filter(pl.col("times_triggered") <= max_window)

    df = reject_single_spikes(
        df
    )  # This filters cells with single spikes or spikes only in one repeat
    if nr_cells > 1:  # Special case if we have only one cell.
        func = partial(apply_on_group, bin_size, max_window)
        results = df.group_by("cell_index", "repeat").map_groups(func)
        fill_func = partial(fill_missing_repeats, max_repeat, max_window)
        results = results.group_by("cell_index").map_groups(fill_func)
    else:  # One cell
        results = timestamps_to_binary_polars(df, bin_size, max_window)
        results = fill_missing_repeats(max_repeat, max_window, results)

    results = results.sort("cell_index", "repeat", "index")

    return results


def binary_as_array(repeats: int, nr_bins: int, df: pl.DataFrame) -> np.ndarray:
    """
    Extract the binary signal from a DataFrame and reshape it into an array.

    Parameters
    ----------
    repeats : int
        The number of repeats
    nr_bins : int
        The number of bins
    df : pl.DataFrame
        The DataFrame with the binary signals

    Returns
    -------
    np.ndarray
        The reshaped binary signal

    """
    return np.reshape(
        df["value"].to_numpy().astype(np.uint8),
        (repeats, nr_bins),
    )


def calc_qis(result_df: pl.DataFrame) -> np.ndarray:
    """
    Calculate the quality index for a DataFrame with binary signals.

    Parameters
    ----------
    result_df : pl.DataFrame
        The DataFrame with the binary signals

    Returns
    -------
    np.ndarray
        The quality indices
    """
    repeats = result_df["repeat"].max() + 1
    bins_per_repeat = np.ceil(
        (len(result_df.filter(pl.col("cell_index") == result_df["cell_index"][0])) - 1)
        / repeats
    ).astype(int)
    result_df = result_df.partition_by("cell_index")
    qis = np.zeros(len(result_df))
    for idx, cell in enumerate(result_df):
        qis[idx] = quality_index(repeats, bins_per_repeat, cell)
    return qis


def quality_index(repeats: int, nr_bins: int, df: pl.DataFrame) -> float:
    """
    Calculate the quality index for a single cell.

    Parameters
    ----------
    repeats : int
        The number of repeats
    nr_bins : int
        The number of bins
    df : pl.DataFrame
        The DataFrame with the binary signals


    """
    gauswins = np.tile(kernel_template(sampling_freq=1000)[::-1], (repeats, 1))
    array = binary_as_array(repeats, nr_bins, df)
    exs = signal.oaconvolve(array, gauswins, mode="valid", axes=1)
    return calc_tradqi(exs)


# array = np.reshape(result["value"].to_numpy(), (5, 2401), order="C")
#
# gauswins = np.tile(kernel_template(width=0.0125 / 100)[::-1], (5, 1))
#
# exs = signal.oaconvolve(array, gauswins, mode="valid", axes=1)
#


def calc_tradqi(kernel_fits: np.ndarray) -> float:
    """
    Calculates the Quality_index of the (Gaussian-kernel) fitted spike data. Cell-specific and stimulus-specific,
    but of nr-of-trials length

    Parameters
    ----------
    kernel_fits : np.ndarray
        The convolved binary signals

    Returns
    -------
    float
        The quality index
    """
    return np.var(np.mean(kernel_fits, 0)) / np.mean(np.var(kernel_fits, 0))


def kernel_template(
        width: float = 0.0100, sampling_freq: float = 17852.767845719834
) -> np.ndarray:
    """
    Create Gaussian kernel by providing the width (FWHM) value. According to wiki, this is approx. 2.355*std of dist.
    width=given in seconds, converted internally in sampling points.

    Parameters
    ----------
    width : float
        The width of the Gaussian kernel in seconds
    sampling_freq : float
        The sampling frequency

    Returns
    -------
    np.ndarray
        The Gaussian kernel
    """
    fwhm = int((sampling_freq) * width)  # in points

    # normalized time vector in ms
    k = int((sampling_freq) * 0.02)
    gtime = np.arange(-k, k)

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
    gauswin = gauswin / np.sum(gauswin)

    # initialize filtered signal vector
    return gauswin


def reject_single_spikes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reject cells with single spikes or spikes only in one repeat. Calculating a quality index for these cells is not
    possible.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame with the binary signals

    Returns
    -------
    pl.DataFrame
        The DataFrame with the rejected cells removed
    """
    unique_spikes = df.group_by("cell_index").count().filter(pl.col("count") == 1)
    unique_repeat = df.group_by("repeat").count().filter(pl.col("count") == 1)
    all_unique = np.concatenate(
        [unique_spikes["cell_index"].to_numpy(), unique_repeat["repeat"].to_numpy()]
    )
    unique_cells = np.unique(all_unique)

    return df.filter(pl.col("cell_index").is_in(unique_cells) == False)
