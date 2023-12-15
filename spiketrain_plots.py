import plotly.graph_objects as go
import polars as pl
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds
import numpy as np
import pandas as pd
from polarspike import histograms
from polarspike import spiketrains


def whole_stimulus_plotly(df, stacked=False):
    y_key = "repeat"
    if type(df) is pl.DataFrame:
        df = df.to_pandas()

    if stacked:
        df, unique_indices = map_index(df, "cell_index", "repeat")
        y_key = "index_linear"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df["times_triggered"],
            y=df[y_key],
            name=None,
            marker=dict(color=df["cell_index"], size=10, opacity=1),
            marker_symbol="line-ns",
            marker_line_width=1,
        )
    )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Cell(s), repeat(s)")
    fig.update_layout(template="simple_white")
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=np.arange(1, df[y_key].max() + 1, 2),
            ticktext=np.repeat(df["cell_index"].unique(), df["repeat"].max() + 1)[1::2],
        )
    )
    return fig


def whole_stimulus(
    df,
    how="times_triggered",
    index="cell_index",
    stacked=True,
    height=10,
    width=10,
    cmap="Greys",
    bin_size=0.05,
    norm="linear",
    y_key="repeat",
):
    # Store some information about the data
    unique_indices = np.unique(df[index])
    nr_repeats = df[y_key].max() + 1
    nr_unique = unique_indices.shape[0]  # In case of a single cell or a single color
    max_time = np.max(df[how])

    if height is None:
        if nr_unique > 3:
            height = int(5 * nr_unique)
        else:
            height = 10
    plot_width = int(max_time / bin_size)

    # Create a figure and axis to draw
    fig, axs = plt.subplots(
        3,
        2,
        figsize=(width, height),
        width_ratios=[1, 0.01],
        height_ratios=[0.1, 1, 0.1],
        sharex="col",
    )

    # Map 'cell_index' to 'cell_index_linear' using the mapping
    df = df.copy()
    if stacked:
        # To space the cells, we need to map the combined repeats and cells indices to a linear range.
        df, unique_indices = map_index(df, index, y_key)
        y_key = "index_linear"
    #     psth, bins = histograms.psth_by_index(
    #         df, bin_size=bin_size, index=index, window_end=max_time
    #     )
    # else:
    psth, bins = histograms.psth(df, bin_size=bin_size, start=0, end=max_time)
    psth = psth / unique_indices.shape[0] * bin_size

    # Plot the PSTH

    axs[0, 0].plot(bins[1:], psth, color="black", alpha=0.5)

    # Switch data format to categorical
    df["index"] = df[index].astype("category")
    df["index"] = df["index"].cat.as_ordered()

    if stacked:
        plot_height = int(nr_unique * nr_repeats)
    else:
        plot_height = nr_repeats

    if type(cmap) is str:
        draw_artist(
            df,
            fig,
            axs,
            how,
            y_key,
            cmap,
            norm,
            plot_height,
            plot_width,
            bin_size,
        )

    else:
        if len(cmap) != len(unique_indices):
            cmap = cmap * len(unique_indices)

        for index_id, c in zip(unique_indices, cmap):
            df_temp = df.query(f"index == {index_id}")
            draw_artist(
                df_temp,
                fig,
                axs,
                how,
                y_key,
                c,
                norm,
                plot_height,
                plot_width,
                bin_size,
            )

    if stacked:
        axs[1, 0].yaxis.set_ticks(np.arange(1, nr_unique * nr_repeats + 1, 2))
        axs[1, 0].set_yticklabels(np.repeat(unique_indices.to_numpy(), nr_repeats)[::2])
    axs[0, 0].set_ylabel("Spikes / s")
    axs[1, 0].set_ylabel(f"{index} and repeat(s)")
    axs[2, 0].set_xlabel("Time in s")
    axs[2, 1].axis("off")
    axs[0, 1].axis("off")
    axs[0, 0].spines[["right", "top"]].set_visible(False)
    stim_ax = axs[1, 0].get_position()
    cbar_ax = axs[1, 1].get_position()
    cbar_max_height = 0.1
    if stim_ax.height < cbar_max_height:
        cbar_max_height = stim_ax.height

    axs[1, 1].set_position(
        [stim_ax.x0 + stim_ax.width + 0.01, stim_ax.y0, cbar_ax.width, cbar_max_height]
    )
    stim_trace_pos = axs[2, 0].get_position()
    axs[2, 0].set_position(
        [stim_ax.x0, stim_ax.y0 - 0.1, stim_ax.width, stim_trace_pos.height]
    )
    axs[2, 0].spines["top"].set_visible(False)
    axs[2, 0].spines["right"].set_visible(False)
    axs[2, 0].spines["bottom"].set_visible(False)
    axs[2, 0].spines["left"].set_visible(False)
    axs[2, 0].tick_params(axis="y", which="both", length=0)
    axs[2, 0].set_yticks([])
    axs[1, 0].set_xlim(0, np.max(df[how]))
    return fig, axs


def draw_artist(
    df, fig, axs, how, y_key, cmap, norm, plot_height, plot_width, bin_size
):
    """Draws the artist on the figure and returns the figure and axis.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    fig, axs : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis on which to draw the artist.
    how : str
        The column name of the DataFrame containing the spike times, valid are ('times_triggered', 'times_stimulus')
    y_key : str
        The column name of the DataFrame containing the cell indices.
    cmap : str or list
        The colormap to use for the plot.
    norm : str
        The normalization to use for the plot. Valid are ("linear", "eq_hist" "cbrt").
    plot_height : int
        The height of the plot in pixels used for drawing the heatmap.
    width : int
        The width of the plot in pixels used for drawing the heatmap.
    bin_size : float
        The bin size in seconds used for drawing the heatmap.
    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis on which the artist was drawn.
    """

    # Add the datashader artist to the figure
    artist = dsshow(
        df,
        ds.Point(how, y_key),
        plot_height=plot_height,
        cmap=cmap,
        plot_width=plot_width,
        vmin=0,
        norm=norm,
        ax=axs[1, 0],
        aspect="auto",
    )
    cbar = fig.colorbar(artist, cax=axs[1, 1], shrink=0.1)

    cbar.set_label(f"Nr of spikes, binsize={bin_size} s")


def map_index(df, index="cell_index", y_key="repeat"):
    """Maps the cell indices to a linear range.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the cell indices mapped to a linear range.
    unique_cells : pandas.Series
        A Series containing the unique cell indices.

    """

    unique_index = pd.Series(df[index].unique())
    mapping = pd.Series(index=unique_index, data=range(len(unique_index)))
    df["index_linear"] = df[index].map(mapping)
    # Assign a new unique index for each cell and repeat:
    combined_indices = np.array(list(zip(df["index_linear"], df[y_key])))

    _, inverse_indices = np.unique(combined_indices, axis=0, return_inverse=True)
    df["index_linear"] = inverse_indices + 1

    return df, unique_index


# def plot_vertical(df, )
