import polars as pl
import numpy as np


def count_spikes(df, columns="trigger", name="spikes_sum"):
    """
    Count the number of spikes per unique value in a column

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    columns : str
        The column name of the DataFrame containing the trigger values.
    name : str
        The name of the new column containing the spike counts.

    Returns
    -------
    df : pandas.DataFrame or polars.DataFrame
        The DataFrame with the new column.
    """
    return_polars = True
    if type(df) is not pl.DataFrame:
        return_polars = False
        df = pl.from_pandas(df)
    if return_polars:
        return df.groupby(columns).agg(pl.count().alias(name)).sort(columns)
    else:
        return df.groupby(columns).agg(pl.count().alias(name)).sort(columns).to_pandas()


def collect_as_arrays(df, indices, column, name):
    """
    Collects the values of a column into a numpy array, stored in a new column.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    indices : str
        The column name of the DataFrame containing the indices by which to split the
        frame into arrays.
    column : str
        The column name of the DataFrame containing the values to collect.
    name : str
        The name of the new column containing the array.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the new column.

    """
    return_polars = True
    if type(df) is not pl.DataFrame:
        return_polars = False
        df = pl.from_pandas(df)

    df = df.group_by(indices).agg(pl.col(column).alias(name))
    df = df.sort(indices)
    if return_polars:
        return df
    else:
        return df.to_pandas()


def align_on_condition(df, condition, con_star_times, times="times_triggered"):
    """
    Aligns the spike times on a condition. That is, for each condition the new spike times
    are the old spike times minus the condition time. This is useful for aligning the spikes within a single stimulus.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        A DataFrame containing the spike trains
    condition : str
        The column name of the DataFrame containing the condition times.
    con_star_times : np.ndarray
        The condition times. Must have equal number of elements than unique values in the condition column.

    Returns
    -------
    df : pandas.DataFrame or polars.DataFrame
        The DataFrame with the new column.
    """
    return_polars = True
    if type(df) is not pl.DataFrame:
        return_polars = False
        df = pl.from_pandas(df)

    nr_uniques = df[condition].n_unique()
    uniques = df[condition].unique().sort()
    assert (
        nr_uniques == con_star_times.shape[0]
    ), "Number of unique values in condition column must match number of condition times"

    df = df.with_columns(aligned_times=pl.col(times))
    for idx, unique in enumerate(uniques):
        df = df.with_columns(
            pl.when(pl.col(condition) == unique)
            .then(pl.col(times) - con_star_times[idx])
            .otherwise(pl.col("aligned_times"))
            .alias("aligned_times")
        )
    if return_polars:
        return df
    else:
        return df.to_pandas()
