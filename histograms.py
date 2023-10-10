import numpy as np
import polars as pl

def psth(df, bin_size=0.05):
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
    # Calculate the number of spikes in each bin
    psth, _ = np.histogram(df["times_triggered"], bins=100)

    # Calculate the bin edges
    bin_edges = np.linspace(0, 1, 101)

    return psth, bin_edges


def psth_by_cell(df, bin_size=0.05, return_idx=False):
    try:
        df = pl.from_pandas(df)
    except TypeError:
        pass

    max_spike = df.max()["times_triggered"][0]
    cell_spikes = df.group_by("cell_index").agg("times_triggered")[["cell_index", "times_triggered"]].to_numpy()
    bins = np.arange(0, max_spike, bin_size)
    histograms = np.zeros((cell_spikes[:,1].shape[0], bins.shape[0]-1))
    for idx, spiketrain in enumerate(cell_spikes[:,1]):
        histograms[idx], _ = np.histogram(spiketrain, bins=bins)
    if not return_idx:
        return histograms, bins
    else:
        return histograms, bins, cell_spikes[:,0]






