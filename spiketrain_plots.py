import plotly.graph_objects as go
import polars as pl
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds
import numpy as np
import pandas as pd
from polarspike import histograms
from polarspike import spiketrains
from plotly.subplots import make_subplots
from polarspike import histograms
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from polarspike import backbone
from bokeh.io import curdoc
from bokeh.themes import built_in_themes


def whole_stimulus_plotly(df, stacked=False):
    y_key = "repeat"
    if type(df) is pl.DataFrame:
        df = df.to_pandas()

    if stacked:
        df, unique_indices = map_index(df, ["cell_index", "repeat"])
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
    indices=None,
    stacked=True,
    height=10,
    width=10,
    cmap="Greys",
    bin_size=0.05,
    norm="linear",
    line_colour="white",
):
    if indices is None:
        indices = ["cell_index", "repeat"]
    # Store some information about the data
    unique_indices = np.unique(df[indices[0]])
    nr_repeats = df[indices[1]].max() + 1
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
        df, unique_indices = map_index(df, indices)
        y_key = "index_linear"
    #     psth, bins = histograms.psth_by_index(
    #         df, bin_size=bin_size, index=index, window_end=max_time
    #     )
    # else:
    psth, bins = histograms.psth(df, bin_size=bin_size, start=0, end=max_time)
    psth = psth / df[indices[0]].unique().shape[0] * (1 / bin_size)

    # Plot the PSTH

    axs[0, 0].plot(bins[1:], psth, color=line_colour, alpha=0.5)

    # Switch data format to categorical
    df["index"] = df[indices[0]].astype("category")
    df["index"] = df["index"].cat.as_ordered()

    if stacked:
        plot_height = len(df["index_linear"].unique())
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
        axs[1, 0].yaxis.set_ticks(np.arange(1, len(df["index_linear"].unique()) + 1, 2))
        axs[1, 0].set_yticklabels(unique_indices.to_numpy()[::2])
    axs[0, 0].set_ylabel("Spikes / s/ nr_cells")
    seperator = ", "
    axs[1, 0].set_ylabel(seperator.join(indices))
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
    fig.subplots_adjust(left=0.2)
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


def spikes_and_trace(
    df, stacked=False, indices=None, width=1400, height=500, bin_size=0.05, theme="dark"
):
    assert (
        len(df) < 10000
    ), "The number of spikes is too high for this plot, use spiketrain_plots.whole_stimulus instead"
    y_key = "repeat"
    if indices is None:
        indices = ["cell_index", "repeat"]
    if type(df) is pl.DataFrame:
        df = df.to_pandas()

    if stacked:
        df, unique_indices = map_index(df, indices)
        y_key = "index_linear"
    try:
        current_theme = curdoc().theme
        theme_name = next(k for k, v in built_in_themes.items() if v is current_theme)
        if theme_name == "dark_minimal":
            line_color = "white"
        else:
            line_color = "black"
    except StopIteration:
        if theme == "dark":
            line_color = "white"
        else:
            line_color = "black"
    print(line_color)

    psth, bins = histograms.psth(
        df, bin_size=bin_size, start=0, end=df["times_triggered"].max()
    )
    psth = psth / df[indices[0]].unique().shape[0] * (1 / bin_size)

    # First subplot
    s1 = figure(width=width, height=int(0.2 * height), title=None, sizing_mode="fixed")
    s1.line(bins[1:], psth, line_width=2, color=line_color)
    s1.xaxis.major_label_text_font_size = "0pt"

    # Second subplot
    s2 = figure(
        width=width,
        height=int(0.8 * height),
        title=None,
        x_range=s1.x_range,
        sizing_mode="fixed",
    )
    source = ColumnDataSource(df)
    s2.dash(
        "times_triggered",
        y_key,
        size=10,
        source=source,
        color=line_color,
        alpha=1,
        angle=1.5708,
    )
    s2.xgrid.grid_line_color = None
    s2.ygrid.grid_line_color = None

    s2.xaxis.axis_label = "Time (s)"
    if stacked:
        seperator = ", "
        s2.yaxis.axis_label = seperator.join(indices)
    else:
        s2.yaxis.axis_label = "Repeat(s)"
    s1.yaxis.axis_label = "Spikes / s / nr_cells"

    if stacked:
        s2.yaxis.ticker = np.arange(1, df[y_key].max() + 1, 2)
        s2.yaxis.major_label_overrides = {
            i: str(label)
            for i, label in backbone.enumerate2(
                unique_indices.to_numpy()[1::2],
                start=1,
                step=2,
            )  #
        }

    # Combine plots vertically
    grid = gridplot(
        [[s1], [s2]], width=width, sizing_mode="scale_width", merge_tools=True
    )

    return grid


def map_index(df, index_columns):
    """
    Maps the indices specified by index_columns to a linear range.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing data with indices to be linearly mapped.
    index_columns : list
        A list of column names to be used for creating a multi-dimensional index.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the indices mapped to a linear range.
    unique_indices : pandas.Series
        A Series containing the unique indices created from the specified columns.
    """

    # Create a multi-dimensional index based on the specified columns
    df = df.sort_values(index_columns)
    multi_index = df[index_columns].apply(tuple, axis=1)

    # Get unique indices and create a mapping
    unique_indices = pd.Series(multi_index.unique())
    mapping = pd.Series(index=unique_indices, data=range(len(unique_indices)))

    # Map the multi-dimensional index to a linear index
    df["index_linear"] = multi_index.map(mapping) + 1

    return df, unique_indices
