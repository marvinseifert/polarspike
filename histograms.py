import numpy as np
import polars as pl


def psth(df, bin_size=0.05, start=0, end=None):
    """
    Calculate the peri-stimulus time histogram (PSTH) for a given cell and stimulus.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.

    Returns
    -------
    psth : np.ndarray
        The PSTH for the given cell and stimulus.

    """
    if end is not None:
        bins = np.arange(start, end, bin_size)
    else:
        bins = np.arange(start, df["times_triggered"].max(), bin_size)
    # Calculate the number of spikes in each bin
    psth, bin_edges = np.histogram(df["times_triggered"], bins=bins)

    return psth, bins


def psth_by_index(
        df,
        bin_size=0.05,
        index="cell_index",
        to_bin="times_triggered",
        return_idx=False,
        window_end=None,
):
    try:
        df = pl.from_pandas(df)
    except TypeError:
        pass
    if window_end is None:
        window_end = df.max()[to_bin][0]
    cell_spikes = df.group_by(index).agg(to_bin)[[index, to_bin]].to_numpy()

    bins = np.arange(0, window_end, bin_size)
    histograms = np.zeros((cell_spikes[:, 1].shape[0], bins.shape[0] - 1))
    for idx, spiketrain in enumerate(cell_spikes[:, 1]):
        histograms[idx], _ = np.histogram(spiketrain, bins=bins)
    if not return_idx:
        return histograms, bins
    else:
        return histograms, bins, cell_spikes[:, 0]
