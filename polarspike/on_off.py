import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlate_on_offs(bined_spikes, bins, trigger, interval=2):
    """
    This function correlates the trigger with a cell's response psth to determine whether the cell is ON, OFF or ONOFF.
    Parameters
    ----------
    bined_spikes : np.array
        Array of binned spike trains.
    bins : np.array
        np.array of bin edges.
    trigger : np.array
        np.array of trigger times.
    interval : int, optional
        The interval of the stimulus. That is the time in seconds between ON and OFF stimulus. The default is 2.
    Returns
    -------
    on_offs : np.array
        Array containing the on_off index (ONOFF = 0, ON = 1, OFF = -1) for each cell.


    """
    # First, bin the trigger:
    trigger_bins = np.histogram(trigger, bins=bins)[0]
    # Second, correlate the trigger with the bined_spikes:
    interval_bins = int(interval / np.diff(bins)[0])
    print(interval_bins)
    on_offs = np.ones(bined_spikes.shape[0])
    for idx, spike_train in enumerate(bined_spikes):
        corr = np.correlate(trigger_bins, spike_train, mode="full")
        # We only need the second half of the correlation:
        half_corr = corr[int(len(corr) / 2) :]
        on = np.max(half_corr[: int(interval_bins / 2)])
        off = np.max(
            half_corr[int(interval_bins - 1) : interval_bins + interval_bins // 2]
        )
        if off != 0 and on != 0:
            on_offs[idx] = (off - on) / (off + on)
        else:
            on_offs[idx] = np.NaN

    return on_offs


def histogram(on_offs):
    """
    This function plots a histogram of the on_offs array.
    Parameters
    ----------
    on_offs : np.array
        Array containing the on_off index (ONOFF = 0, ON = 1, OFF = -1) for each cell.
    Returns
    -------
    f, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis on which the histogram is plotted.
    """
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    ax = sns.histplot(
        x=on_offs,
        edgecolor=".3",
        color="grey",
        linewidth=0.5,
        ax=ax,
        bins=np.arange(-1.1, 1.1, 0.2),
    )
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["OFF", "ONOFF", "ON"])
    ax.set_ylabel("Nr of cells")
    return f, ax
