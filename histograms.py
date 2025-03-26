"""
This module contains functions for calculating peri-stimulus time histograms (PSTHs) from spike times. This is done using
polars DataFrames.
"""
import numpy as np
import pandas as pd
import polars as pl


def psth(
    df: pl.DataFrame | pd.DataFrame,
    bin_size: float = 0.05,
    start: float = 0,
    end: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the peri-stimulus time histogram (PSTH) for a given cell and stimulus.
    Warning: This function drops cells without spikes. For example, if a cell has no spikes in the given time window,
    it will not be included in the output DataFrame.

    Parameters
    ----------
    df : polars.DataFrame or pd.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    bin_size : float, optional
        The size of the bins in seconds. The default is 0.05.
    start : float, optional
        The start time of the PSTH in seconds. The default is 0.
    end : float, optional
        The end time of the PSTH in seconds. The default is None which means the end time is the maximum time in the
        DataFrame.

    Returns
    -------
    psth : np.ndarray
        The PSTH for the given cell and stimulus.
    bins : np.ndarray
        The bin edges of the PSTH.

    """
    try:
        df = pl.from_pandas(df)
    except TypeError:
        pass
    if end is not None:
        bins = np.arange(start, end, bin_size)
    else:
        bins = np.arange(start, df["times_triggered"].max(), bin_size)
    # Calculate the number of spikes in each bin
    psth, bin_edges = np.histogram(df["times_triggered"], bins=bins)

    return psth, bins


def psth_by_index(
    df: pl.DataFrame | pd.DataFrame,
    bin_size: str = 0.05,
    index: list[str] = "cell_index",
    to_bin: str = "times_triggered",
    return_idx: bool = False,
    window_end: float = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the peri-stimulus time histogram (PSTH) averaged over multiple indices.
    Warning: This function drops entries without any spikes. For example, if a cell did not spike during a stimulus
    presentation, the entry will be missing in the DataFrame.

    Parameters
    ----------
    df : polars.DataFrame or pd.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    bin_size : float, optional
        The size of the bins in seconds. The default is 0.05.
    index : list[str], optional
        The columns to group the DataFrame by. The default is ["cell_index"]. If only one column is given, the return
        index will be reshaped to a 2D array.
    to_bin : str, optional
        The column containing the spike times. The default is "times_triggered".
    return_idx : bool, optional
        Whether to return the index. The default is False. This is important, as entries without spikes will be
        missing in the DataFrame. return_idx offers a quick way to check which entries are missing.
    window_end : float, optional
        The end of the window in seconds. The default is None which means the end time is the maximum time in the
        DataFrame.

    Returns
    -------
    histograms : np.ndarray
        The PSTH for the given cell and stimulus.
    bins : np.ndarray
        The bin edges of the PSTH.
    return_index : np.ndarray
        The index of the DataFrame. Only returned if return_idx is True.
    """
    try:
        df = pl.from_pandas(df)
    except TypeError:
        pass
    if window_end is None:
        window_end = df.max()[to_bin][0]
    cell_spikes = df.group_by(index).agg(to_bin)[index + [to_bin]].to_numpy()

    bins = np.arange(0, window_end, bin_size)
    histograms = np.zeros((cell_spikes[:, 1].shape[0], bins.shape[0] - 1))
    return_index = []
    for idx, spiketrain in enumerate(cell_spikes[:, -1]):
        histograms[idx], _ = np.histogram(spiketrain, bins=bins)
        return_index.append(cell_spikes[idx, : len(index)])
    if not return_idx:
        return histograms, bins
    else:
        if len(index) == 1:
            return_index = np.array(return_index).reshape(-1, 1)
        else:
            return_index = np.array(return_index)
        return (
            histograms,
            bins,
            return_index,
        )
