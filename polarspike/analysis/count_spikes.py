"""
This script provides functions to count the spikes elicited by a stimulus. Several methods are provided.
"""

import numpy as np
import polars as pl
import pandas as pd


def sum_spikes(
        spikes: pl.DataFrame or pd.DataFrame,
        mean_trigger: np.ndarray,
        window: float = 0.5,
        group_by: str | list = "cell_index",
) -> np.ndarray and np.ndarray:
    """
    Sum the spikes in a given window for each trigger time.

    Parameters
    ----------
    spikes : pl.DataFrame or pd.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    mean_trigger : np.ndarray
        The mean trigger times for the stimulus, return of the stimulus_spikes.mean_trigger_times function.
    window : float
        The window size in seconds over which the response will be summed. The float represents the right edge of the
        window, relative to the stimulus start.
    group_by : str
        The column to group by. If "cell_index" the spikes will be summed for each cell.

    Returns
    -------
    spikes_summed : np.ndarray
        The summed spikes for each trigger time.
    cell_index : np.ndarray
        The cell index for each row in the spikes_summed

    """
    if type(group_by) == str:
        group_by = [group_by]
    n_indices = len(group_by)
    trig = 0
    spikes_summed = []
    for trig_idx, trigger in enumerate(mean_trigger):
        spikes_summed.append(
            (
                spikes.filter(
                    (pl.col("times_triggered") > trig)
                    & (pl.col("times_triggered") < (trig + window))
                )
                .group_by(group_by)
                .agg(pl.count("times_triggered").alias(f"spikes_{trig_idx}"))
            )
        )
        trig += trigger

    all_trigger_df = pl.concat(spikes_summed, how="align").to_numpy()
    spikes_summed = all_trigger_df[:, n_indices:].astype(float)
    spikes_summed[np.isnan(spikes_summed)] = 0
    cell_index = all_trigger_df[:, :n_indices]

    return spikes_summed, cell_index
