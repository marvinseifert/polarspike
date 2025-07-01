import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def expo_chirp(start_time, end_time, start_freq, end_freq):
    """

    Generate a frequency chirp that exponentially increases from start_freq to end_freq.



    Parameters:

        start_time (float): The time at which the chirp starts.

        end_time (float): The time at which the chirp ends.

        start_freq (float): The starting frequency of the chirp in Hz.

        end_freq (float): The ending frequency of the chirp in Hz.



    Returns:

        x (ndarray): The time values from start_time to end_time.

        freq (ndarray): The corresponding frequency values of the chirp.

    """

    # Number of points per second

    points_per_second = 10

    # Generate x values

    x = np.linspace(
        start_time, end_time, int((end_time - start_time) * points_per_second)
    )

    # Calculate b value for exponential

    b = end_freq / start_freq

    # Generate the frequency values, adjusting the exponential to start at x=0 for the chirp part

    freq = start_freq * b ** ((x - start_time) / (end_time - start_time))

    return x, freq


def plot(df, start_time, end_time, start_freq, end_freq):
    """
    Plots the spike count histogram and the frequency chirp in a searborn figure and calculcates a kernel density estimate.

    Parameters:
    df : DataFrame
        The DataFrame containing the spike count data.
    start_time : float
        The time at which the chirp starts.
    end_time : float
        The time at which the chirp ends.
    start_freq : float
        The starting frequency of the chirp in Hz.
    end_freq : float
        The ending frequency of the chirp in Hz.

    Returns:
    fig : Figure
        The figure containing the plot (matplotlib).
    axs : Axes
        The axes containing the plot(matplotlib)   .
    """
    plt.ioff()
    fig, axs = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw=dict(height_ratios=[1, 2])
    )

    axs[0] = sns.histplot(
        data=df,
        x="times_triggered",
        kde=True,
        bins=789,
        color="black",
        edgecolor="none",
        element="step",
        ax=axs[0],
    )

    axs[1] = sns.lineplot(
        x=expo_chirp(start_time, end_time, start_freq, end_freq)[0],
        y=expo_chirp(start_time, end_time, start_freq, end_freq)[1],
        ax=axs[1],
        color="black",
    )
    for ax in axs:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines["left"].set_position(("outward", 10))

    axs[0].set_ylabel("Spike Count")
    axs[0].set_xlabel("Time in s")

    axs[1].set_ylabel("Frequency")
    axs[1].set_xlabel("Time in s")
    axs[1].set_box_aspect(19 / 20)
    axs[1].set_box_aspect(1 / 20)
    fig.tight_layout(pad=0.1)
    return fig, axs


def plot_interactive(df, start_time, end_time, start_freq, end_freq):
    """
    Plots the spike count histogram and the frequency chirp in a searborn figure and calculcates a kernel density estimate.
    This function directly shows the plot, which will be displayed interactively in Jupyter if %matplotlib widget magic is used.

    Parameters:
    df : DataFrame
        The DataFrame containing the spike count data.
    start_time : float
        The time at which the chirp starts.
    end_time : float
        The time at which the chirp ends.
    start_freq : float
        The starting frequency of the chirp in Hz.
    end_freq : float
        The ending frequency of the chirp in Hz.

    Returns:
        Shows the plot.

    """

    fig, ax = plot(df, start_time, end_time, start_freq, end_freq)
    return fig.show()
