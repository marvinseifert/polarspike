import plotly.graph_objects as go
import polars as pl
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib import cm
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from bokeh.models import LinearColorMapper
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable


def whole_stimulus_plotly(df):
    if type(df) is pl.DataFrame:
        df = df.to_pandas()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df["times_triggered"],
            y=df["repeat"],
            name=None,
            marker=dict(color=df["cell_index"], size=10, opacity=1),
            marker_symbol="line-ns",
            marker_line_width=1,
        )
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Repeats")
    fig.update_layout(template="simple_white")
    return fig


def whole_stimulus(
    df,
    how="times_triggered",
    spaced=True,
    height=None,
    width=None,
    cmap="Greys",
    bin_size=0.05,
    norm="linear",
):
    # Store some information about the data
    y_key = "repeat"
    unique_cells = np.unique(df["cell_index"])
    nr_repeats = df["repeat"].max() + 1
    nr_cells = unique_cells.shape[0]  # In case of a single cell or a single color

    if height is None:
        if nr_cells > 3:
            height = int(5 * nr_cells)
        else:
            height = 10
    if width is None:
        width = int(np.max(df[how]) / bin_size)

    # Create a figure and axis to draw
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(20, height),
        width_ratios=[1, 0.01],
        height_ratios=[1, 0.1],
        sharex="col",
    )
    axs[1, 1].axis("off")
    # Map 'cell_index' to 'cell_index_linear' using the mapping
    df = df.copy()
    if spaced:
        # To space the cells, we need to map the combined repeats and cells indices to a linear range.
        df, unique_cells = map_cells(df)
        y_key = "cell_index_linear"

    # Switch data format to categorical
    df["cell_index"] = df["cell_index"].astype("category")
    df["cell_index"] = df["cell_index"].cat.as_ordered()

    if spaced:
        plot_height = int(nr_cells * nr_repeats)
    else:
        plot_height = nr_repeats

    if type(cmap) is str:
        draw_artist(df, fig, axs, how, y_key, cmap, norm, plot_height, width, bin_size)

    else:
        if len(cmap) != len(unique_cells):
            cmap = cmap * len(unique_cells)
        for cell, c in zip(unique_cells, cmap):
            df_temp = df.query(f"cell_index == {cell}")
            draw_artist(
                df_temp,
                fig,
                axs,
                how,
                y_key,
                c,
                norm,
                plot_height,
                width,
                bin_size,
            )

    if spaced:
        axs[0, 0].yaxis.set_ticks(np.arange(1, nr_cells * nr_repeats + 1, 2))
        axs[0, 0].set_yticklabels(np.repeat(unique_cells.to_numpy(), nr_repeats)[::2])

    stim_ax = axs[0, 0].get_position()
    cbar_ax = axs[0, 1].get_position()
    axs[0, 1].set_position(
        [stim_ax.x0 + stim_ax.width + 0.01, stim_ax.y0, cbar_ax.width, stim_ax.height]
    )
    stim_trace_pos = axs[1, 0].get_position()
    axs[1, 0].set_position(
        [stim_ax.x0, stim_ax.y0 - 0.1, stim_trace_pos.width, stim_trace_pos.height]
    )
    axs[1, 0].spines["top"].set_visible(False)
    axs[1, 0].spines["right"].set_visible(False)
    axs[1, 0].spines["bottom"].set_visible(False)
    axs[1, 0].spines["left"].set_visible(False)
    axs[1, 0].tick_params(axis="y", which="both", length=0)

    return fig, axs


def draw_artist(df, fig, axs, how, y_key, cmap, norm, plot_height, width, bin_size):
    """Draws the artist on the figure and returns the figure and axis.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
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
        plot_width=width,
        vmin=0,
        norm=norm,
        ax=axs[0, 0],
    )
    cbar = fig.colorbar(artist, cax=axs[0, 1], shrink=0.1)

    cbar.set_label(f"Nr of spikes, binsize={bin_size} s")
    axs[0, 0].set_ylabel("Cell(s) - Repeat(s)")
    axs[1, 0].set_xlabel("Time in seconds")


def map_cells(df):
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
    unique_cells = pd.Series(df["cell_index"].unique()).sort_values()
    mapping = pd.Series(index=unique_cells, data=range(len(unique_cells)))
    df["cell_index_linear"] = df["cell_index"].map(mapping)
    # Assign a new unique index for each cell and repeat:
    combined_indices = np.array(list(zip(df["cell_index_linear"], df["repeat"])))

    _, inverse_indices = np.unique(combined_indices, axis=0, return_inverse=True)
    df["cell_index_linear"] = inverse_indices + 1

    return df, unique_cells
