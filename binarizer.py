import polars as pl
from functools import partial


def timestamps_to_binary_polars(df_timestamps, sample_rate, max_window):
    # Create a DataFrame with the timestamps
    # Calculate the indices for the timestamps based on the sampling rate
    df_timestamps = df_timestamps.with_columns(
        (df_timestamps["times_triggered"] / sample_rate).cast(pl.Int64).alias("index")
    )

    # Create a DataFrame with a sequence of zeros up to the maximum index
    max_index = df_timestamps.select(pl.col("index").max()).item()
    print(max_index)
    df_binary_signal = pl.DataFrame(
        {"index": list(range(max_window + 1)), "value": [0] * (max_window + 1)}
    )
    print(len(df_binary_signal))

    # Overlay the ones onto the zeros based on the calculated indices
    df_result = df_binary_signal.join(
        df_timestamps, on="index", how="left"
    ).with_columns(
        pl.when(pl.col("times_triggered").is_null()).then(0).otherwise(1).alias("value")
    )
    df_result = df_result.with_columns(
        pl.col("cell_index").fill_null(strategy="backward")
    ).with_columns(pl.col("repeat").fill_null(strategy="backward"))
    df_result = (
        df_result.with_columns(pl.col("cell_index").fill_null(strategy="forward"))
        .with_columns(pl.col("repeat").fill_null(strategy="forward"))
        .select("cell_index", "value", "repeat")
    )

    return df_result


def apply_on_group(sample_rate, max_window, group_df):
    return timestamps_to_binary_polars(group_df, sample_rate, max_window)


def timestamps_to_binary_multi(df, sample_rate, max_window):
    func = partial(apply_on_group, sample_rate, max_window)
    results = df.group_by("cell_index", "repeat").map_groups(func)
    results = results.sort(by="repeat", descending=False)
    return results
