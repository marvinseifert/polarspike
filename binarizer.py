import polars as pl
from functools import partial
import numpy as np
import multiprocessing
import scipy.signal as signal


def timestamps_to_binary_polars(df_timestamps, sample_rate, max_window):
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

    return df_result


def fill_missing_repeats(max_repeat, nr_bins, df_result):
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
                    "repeat": pl.Int32,
                    "index": pl.Int64,
                },
            )
        )

    dfs.append(df_result)
    df_result = pl.concat(dfs)
    return df_result


def apply_on_group(sample_rate, max_window, group_df):
    try:
        return timestamps_to_binary_polars(group_df, sample_rate, max_window)
    except Exception as e:
        print(group_df)


def timestamps_to_binary_multi(df, bin_size, max_window, max_repeat):
    max_window = np.ceil(max_window / bin_size + bin_size).astype(int)

    # Reject spikes larger than max_window
    df = df.filter(pl.col("times_triggered") <= max_window)

    df = reject_single_spikes(
        df
    )  # This filters cells with single spikes or spikes only in one repeat
    func = partial(apply_on_group, bin_size, max_window)
    results = df.group_by("cell_index", "repeat").map_groups(func)

    fill_func = partial(fill_missing_repeats, max_repeat, max_window + 1)

    results = results.group_by("cell_index").map_groups(fill_func)
    results = results.sort("cell_index", "repeat", "index")

    return results


def binary_as_array(repeats, bins, df):
    return np.reshape(
        df["value"].to_numpy().astype(np.uint8),
        (repeats, bins),
    )


def calc_qis(result_df):
    repeats = result_df["repeat"].max() + 1
    bins_per_repeat = np.ceil(
        len(result_df.filter(pl.col("cell_index") == result_df["cell_index"][0]))
        / repeats
    ).astype(int)
    result_df = result_df.partition_by("cell_index")
    qis = np.zeros(len(result_df))
    for idx, cell in enumerate(result_df):
        qis[idx] = quality_index(repeats, bins_per_repeat, cell)
    return qis


def quality_index(repeats, bins, df):
    gauswins = np.tile(kernel_template(sampling_freq=1000)[::-1], (repeats, 1))
    array = binary_as_array(repeats, bins, df)
    exs = signal.oaconvolve(array, gauswins, mode="valid", axes=1)
    return calc_tradqi(exs)


# array = np.reshape(result["value"].to_numpy(), (5, 2401), order="C")
#
# gauswins = np.tile(kernel_template(width=0.0125 / 100)[::-1], (5, 1))
#
# exs = signal.oaconvolve(array, gauswins, mode="valid", axes=1)
#


def calc_tradqi(kernelfits):
    """
    Calculates the Quality_index of the (Gaussian-kernel) fitted spike data. Cell-specific and stimulus-specific,
    but of nr-of-trials length

    Returns
    The QI value for the cells-response to this stimulus
    """
    return np.var(np.mean(kernelfits, 0)) / np.mean(np.var(kernelfits, 0))


def kernel_template(width=0.0100, sampling_freq=17852.767845719834):
    """
    create Gaussian kernel by providing the width (FWHM) value. According to wiki, this is approx. 2.355*std of dist.
    width=given in seconds, converted internally in sampling points.

    Returns
    Gaussian kernel
    """
    fwhm = int((sampling_freq) * width)  # in points

    # normalized time vector in ms
    k = int((sampling_freq) * 0.02)
    gtime = np.arange(-k, k)

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime**2) / fwhm**2)
    gauswin = gauswin / np.sum(gauswin)

    # initialize filtered signal vector
    return gauswin


def reject_single_spikes(df):
    unique_spikes = df.group_by("cell_index").count().filter(pl.col("count") == 1)
    unique_repeat = df.group_by("repeat").count().filter(pl.col("count") == 1)
    all_unique = np.concatenate(
        [unique_spikes["cell_index"].to_numpy(), unique_repeat["repeat"].to_numpy()]
    )
    unique_cells = np.unique(all_unique)

    return df.filter(pl.col("cell_index").is_in(unique_cells) == False)
