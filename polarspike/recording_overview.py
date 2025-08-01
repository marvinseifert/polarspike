"""
This module provides functions to plot spike counts, interspike intervals, spike trains and spike amplitudes from a parquet file.
This enables the user to visualize the data of a single recording fast and easy to get a quick overview.
@ Marvin Seifert 2024
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from bokeh.plotting import figure, show
from pathlib import Path
import pandas as pd
import datashader as ds
import holoviews as hv
from holoviews.operation.datashader import datashade
from bokeh.models import Span
from bokeh.models import BoxAnnotation, Label
from bokeh.models import FixedTicker


def spike_counts_from_file(
        file_parquet: str | Path, cmap: str = "Oranges"
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the number of spikes for each cell in a recording using DataShader.

    Parameters
    ----------
    file_parquet : str | Path
        The path to the parquet file containing the spike times.
    cmap : str, optional
        The colormap to use for the plot. The default is "Oranges".

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.

    """
    df = pl.scan_parquet(file_parquet)
    df = df.select(pl.col("cell_index"), pl.col("times"))
    return plot_spike_counts_bokeh(df)


def isi_from_file(
        file_parquet: str | Path,
        freq: float = 1,
        x: str = "times",
        cmap: str = "viridis",
        cutoff: float = 0.001,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the interspike intervals for a recording using DataShader.

    Parameters
    ----------
    file_parquet : str | Path
        The path to the parquet file containing the spike times.
    freq : float, optional
        The frequency of the recording in Hz. The default is 1 (which is essentially plotting on a frame base).
    x : str, optional
        The x-axis to plot the data on. The default is "times".
    cmap : str, optional
        The colormap to use for the plot. The default is "viridis".
    cutoff : float, optional
        The cutoff value in seconds. The default is 0.001.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.


    """
    cutoff = (1 / freq) * cutoff
    df = pl.scan_parquet(file_parquet)
    df = df.select(pl.col("cell_index"), pl.col("times"))
    df = df.with_columns(pl.col("times").truediv(freq).alias("times"))
    return plot_isi_bokeh(df, x, cmap=cmap, cutoff=cutoff)


def spiketrains_from_file(
        file_parquet: str | Path,
        freq: float = 1,
        cmap: str = "Greys",
        height: int = None,
        width: int = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the spike trains per cell in a recording using DataShader.

    Parameters
    ----------
    file_parquet : str | Path
        The path to the parquet file containing the spike times.
    freq : float, optional
        The frequency of the recording in Hz. The default is 1 (which is essentially plotting on a frame base).
    cmap : str, optional
        The colormap to use for the plot. The default is "Greys".
    height : int, optional
        The height of the plot. The default is None, which defaults to nr_cells.
    width : int, optional
        The width of the plot. The default is None, which defaults to nr_bins.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    """
    df = pl.scan_parquet(file_parquet)
    df = df.select(pl.col("cell_index"), pl.col("times"))
    df = df.with_columns(pl.col("times").truediv(freq).alias("times"))
    return plot_spiketrains_bokeh(df, cmap=cmap, height=height, width=width)


def spike_amplitudes_from_file(
        file_parquet: str | Path, freq: float = 1, cmap: str = "viridis"
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the spike amplitudes for a recording using DataShader. This plots the amplitude of the waveforms of the individual
    cells.

    Parameters
    ----------
    file_parquet : str | Path
        The path to the parquet file containing the spike times.
    freq : float, optional
        The frequency of the recording in Hz. The default is 1 (which is essentially plotting on a frame base).
    cmap : str, optional
        The colormap to use for the plot. The default is "viridis".

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.

    """
    df = pl.scan_parquet(file_parquet)
    df = df.with_columns(pl.col("times").truediv(freq).alias("times"))
    waves = df.columns[2:]
    df_melt = df.melt(
        id_vars=["cell_index", "times"],
        value_vars=waves,
        variable_name="waveform",
        value_name="voltage",
    )
    df_group = df_melt.group_by(["cell_index", "times"])
    df_min = df_group.agg(pl.col("voltage").min())
    df_results = df_min.collect(streaming=True)
    return plot_spike_amplitudes(
        df_results.sort(["cell_index", "times"], descending=False), cmap
    )


def plot_spike_counts(df: pl.DataFrame, cmap: str) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the number of spikes for each cell in a recording using DataShader.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing the spike times.
    cmap : str
        The colormap to use for the plot.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    """
    results_df = df.group_by("cell_index").agg(
        pl.col("times").count().alias("nr_spikes")
    )
    results_df = results_df.collect(streaming=True).to_pandas()

    results_df["log_nr_spikes"] = np.log10(results_df["nr_spikes"])
    max_spike_count = results_df["log_nr_spikes"].max().astype(int)
    fig, ax = plt.subplots(figsize=(15, 10))

    artist = dsshow(
        results_df,
        ds.Point("cell_index", "log_nr_spikes"),
        norm="linear",
        cmap=cmap,
        ax=ax,
    )
    ax.set_aspect("auto")

    ax.set_yticks(
        np.log10([10 ** i for i in range(max_spike_count)])
    )  # Replace with the desired ticks in the original scale
    ax.set_yticklabels([10 ** i for i in range(max_spike_count)])

    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Number of Spikes")
    ax.set_title("Recording Overview")
    return fig, ax


def plot_spike_counts_bokeh(df: pl.DataFrame) -> figure:
    """
    Plot the number of spikes for each cell as a simple Bokeh line plot.
    """
    # Aggregate spike counts
    results_df = df.group_by("cell_index").agg(
        pl.col("times").count().alias("nr_spikes")
    ).collect(streaming=True).to_pandas()

    # Log-transform
    results_df["log_nr_spikes"] = np.log10(results_df["nr_spikes"])

    results_df = results_df.sort_values(by="cell_index")
    # Prepare figure
    p = figure(
        title="Recording Overview",
        x_axis_label="Cell Index",
        y_axis_label="Number of Spikes",
        width=1500,
        height=800
    )

    # Line plot
    p.line(
        x=results_df["cell_index"],
        y=results_df["log_nr_spikes"],
        line_width=2,
        color="navy"
    )

    # Set log-like y-ticks
    max_log = int(results_df["log_nr_spikes"].max()) + 1
    ticks = [np.log10(10 ** i) for i in range(max_log)]
    labels = {tick: str(10 ** i) for i, tick in enumerate(ticks)}
    p.yaxis.ticker = FixedTicker(ticks=ticks)
    p.yaxis.major_label_overrides = labels

    return p


def plot_isi(
        df: pl.DataFrame, x: str, cmap: str = "viridis", cutoff: float = 18.0
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the interspike intervals for a recording using DataShader.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing the spike times.
    x : str
        x variable to plot the data on in y.
    cmap : str, optional
        The colormap to use for the plot. The default is "viridis".
    cutoff : float, optional
        The cutoff value in seconds. The default is 18.0.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.

    """
    df = df.sort(["cell_index", "times"], descending=False)
    df = df.with_columns(pl.col("times").diff(null_behavior="ignore").alias("isi"))
    df = df.with_columns(pl.when(pl.col("isi") < 0).then(0).alias("diff"))
    filtered_df = df.filter(pl.col("isi") > 0).collect(streaming=True)

    plot_df = filtered_df.to_pandas()
    plot_df["log_isi"] = np.log10(plot_df["isi"])
    max_isi = plot_df["log_isi"].max().astype(int)
    max_x = plot_df[x].max().astype(int)

    fig, ax = plt.subplots(figsize=(15, 10))

    artist = dsshow(plot_df, ds.Point(x, "log_isi"), norm="eq_hist", cmap=cmap, ax=ax)
    ax.set_aspect("auto")

    ax.set_yticks(
        np.log10([10 ** i for i in range(max_isi)])
    )  # Replace with the desired ticks in the original scale
    ax.set_yticklabels([10 ** i for i in range(max_isi)])

    ax.hlines(
        np.log10(cutoff),
        0,
        max_x,
        color="red",
        linestyle="--",
        linewidth=2,
        label="1ms cutoff",
    )

    norm = mcolors.LogNorm(vmin=plot_df["isi"].min(), vmax=plot_df["isi"].max())

    ax.legend()
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(
        []
    )  # This line is needed to create the ScalarMappable object but the array is not used

    # Create a colorbar with the ScalarMappable object
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("nr spikes")

    ax.set_xlabel(x)
    ax.set_ylabel("isi")
    ax.set_title("Recording Overview")
    return fig, ax


def plot_isi_bokeh(df: pl.DataFrame, x: str, cmap: str = "viridis", cutoff: float = 18.0):
    """
    Plot the interspike intervals using Bokeh and HoloViews (with DataShader).
    """
    # Preprocessing
    df = df.sort(["cell_index", "times"], descending=False)
    df = df.with_columns(pl.col("times").diff(null_behavior="ignore").alias("isi"))
    df = df.with_columns(pl.when(pl.col("isi") < 0).then(0).otherwise(pl.col("isi")).alias("isi"))
    filtered_df = df.filter(pl.col("isi") > 0).collect(streaming=True)

    plot_df = filtered_df.to_pandas()
    plot_df["log_isi"] = np.log10(plot_df["isi"])

    # Prepare datashader canvas and image
    points = hv.Points(plot_df, kdims=[x, 'log_isi'])
    shaded = datashade(points, cmap=cmap, aggregator=ds.count(), normalization='eq_hist')

    # Bokeh figure
    p = hv.render(shaded, backend='bokeh')
    p.title.text = "Recording Overview"
    p.xaxis.axis_label = x
    p.yaxis.axis_label = "isi (s)"
    p.width = 1500  # Set desired width
    p.height = 800  # Set desired height

    # Y ticks back to linear scale
    max_isi_log = int(plot_df["log_isi"].max())
    ticks = [np.log10(10 ** i) for i in range(max_isi_log + 1)]
    labels = {np.log10(10 ** i): str(10 ** i) for i in range(max_isi_log + 1)}
    p.yaxis.ticker = ticks
    p.yaxis.major_label_overrides = labels

    # Horizontal cutoff line
    cutoff_line = Span(location=np.log10(cutoff), dimension='width',
                       line_color='red', line_dash='dashed', line_width=2)
    p.add_layout(cutoff_line)

    return p


def plot_spiketrains(
        df: pl.DataFrame, cmap: str, height: int = None, width: int = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the spike trains per cell in a recording using DataShader.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing the spike times.
    cmap : str
        The colormap to use for the plot.
    height : int, optional
        The height of the plot. The default is None, which defaults to nr_cells.
    width : int, optional
        The width of the plot. The default is None, which defaults to nr_bins.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    """
    if height is None:
        height = int(df.select(pl.max("cell_index")).collect().item())
    if width is None:
        width = int(df.select(pl.max("times")).collect().item() / 10)
    plot_df = df.collect(streaming=True).to_pandas()
    fig, ax = plt.subplots(figsize=(20, 10))

    artist = dsshow(
        plot_df,
        ds.Point("times", "cell_index"),
        plot_height=height,
        plot_width=width,
        norm="eq_hist",
        cmap=cmap,
        ax=ax,
    )
    ax.set_aspect("auto")

    norm = mcolors.Normalize(vmin=plot_df["times"].min(), vmax=plot_df["times"].max())

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("nr spikes")

    ax.set_xlabel("time in seconds")
    ax.set_ylabel("cell_index")
    ax.set_title("Recording Overview")

    return fig, ax


def plot_spiketrains_bokeh(
        df: pl.DataFrame, cmap: str = "viridis", height: int = None, width: int = None
):
    """
    Plot the spike trains per cell in a recording using Bokeh and HoloViews (with Datashader).
    """
    # Determine plot dimensions
    if height is None:
        height = int(df.select(pl.max("cell_index")).collect().item())
    if width is None:
        width = int(df.select(pl.max("times")).collect().item() / 10)

    # Convert to pandas
    plot_df = df.collect(streaming=True).to_pandas()

    # Build HoloViews Points and apply Datashader
    points = hv.Points(plot_df, kdims=["times", "cell_index"])
    shaded = datashade(points, cmap=cmap, aggregator=ds.count(), normalization='eq_hist', width=width, height=height)

    # Render to Bokeh
    p = hv.render(shaded, backend='bokeh')
    p.width = 1500
    p.height = 800
    p.title.text = "Recording Overview"
    p.xaxis.axis_label = "time in seconds"
    p.yaxis.axis_label = "cell_index"

    return p


def plot_spike_amplitudes(df: pl.DataFrame, cmap: str) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the spike amplitudes for a recording using DataShader. This plots the amplitude of the waveforms of the individual
    cells.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing the spike times.
    cmap : str
        The colormap to use for the plot.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    """
    plot_df = df.to_pandas()
    fig, ax = plt.subplots(figsize=(20, 10))

    artist = dsshow(
        plot_df,
        ds.Point("times", "voltage"),
        plot_height=120,
        norm="eq_hist",
        cmap=cmap,
        ax=ax,
    )
    ax.set_aspect("auto")

    norm = mcolors.Normalize(vmin=plot_df["times"].min(), vmax=plot_df["times"].max())

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)

    ax.set_xlabel("time")
    ax.set_ylabel("voltage")
    ax.set_title("Recording Overview")
    ax.set_ylim(-100, 20)
    return fig, ax


def empty_spike_counts_figure() -> tuple[plt.Figure, plt.Axes]:
    """
    Create an empty figure for the spike counts plot.
    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Number of Spikes")
    ax.set_title("Recording Overview")
    return fig, ax


def add_stimulus_df(fig: plt.Figure, df: pd.DataFrame) -> plt.Figure:
    """
    Add stimulus information to a spike count plot as vertical shaded areas.

    Parameters
    ----------
    fig : plt.Figure
        The figure containing the plot.
    df : plt.DataFrame
        The DataFrame containing the stimulus information.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot with the stimulus information.
    """
    colours = ["green", "orange"]
    nr_stimuli = len(df)
    if nr_stimuli > len(colours):
        colours = colours * (nr_stimuli // len(colours) + 1)
    for i, row in df.iterrows():
        fig.axes[0].axvspan(
            row["begin_fr"] / row["sampling_freq"],
            row["end_fr"] / row["sampling_freq"],
            facecolor=colours[i],
            linestyle="dashed",
            alpha=0.1,
            zorder=0,
        )
        midpoint = (row["begin_fr"] + row["end_fr"]) / 2
        fig.axes[0].text(
            midpoint / row["sampling_freq"],
            5,
            row["stimulus_name"],
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=10,
            zorder=1,
            color="red",
            backgroundcolor="white",
        )

    return fig


def plot_isi_new(df: pl.DataFrame, x: str, cmap: str = "viridis", cutoff: float = 18.0):
    """
    Under construction
    """
    df = df.sort(["cell_index", "times"], descending=False)
    df = df.with_columns(pl.col("times").diff(null_behavior="ignore").alias("isi"))
    df = df.with_columns(pl.when(pl.col("isi") < 0).then(0).alias("diff"))
    filtered_df = df.filter(pl.col("isi") > 0).collect(streaming=True)

    plot_df = filtered_df.to_pandas()
    plot_df["log_isi"] = np.log10(plot_df["isi"])
    max_isi = plot_df["log_isi"].max().astype(int)
    max_x = plot_df[x].max().astype(int)

    fig, ax = plt.subplots(figsize=(15, 10))

    artist = dsshow(
        plot_df,
        ds.Point(x, "log_isi"),
        norm="eq_hist",
        cmap=cmap,
        ax=ax,
        plot_height=50,
        plot_width=480,
    )
    ax.set_aspect("auto")
    # define y ticks in log scale from 0.001 to max_isi
    ax.set_yticks(np.log10([0.0001, 0.001, 0.01, 0.1, 1, 10, 100]))
    ax.set_yticklabels([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

    norm = mcolors.LogNorm(vmin=plot_df["isi"].min(), vmax=plot_df["isi"].max())

    ax.legend()
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(
        []
    )  # This line is needed to create the ScalarMappable object but the array is not used

    # Create a colorbar with the ScalarMappable object
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("nr spikes")

    ax.set_xlabel(x)
    ax.set_ylabel("isi")
    ax.set_title("Recording Overview")
    return fig, ax


def add_stimulus_df_bokeh(p, df: pd.DataFrame):
    """
    Add stimulus information to a Bokeh spike count plot as vertical shaded areas.

    Parameters
    ----------
    p : bokeh.plotting.figure.Figure
        The Bokeh figure containing the plot.
    df : pd.DataFrame
        The DataFrame containing the stimulus information.

    Returns
    -------
    p : bokeh.plotting.figure.Figure
        The figure containing the plot with stimulus overlays.
    """
    colours = ["green", "orange"]
    nr_stimuli = len(df)
    if nr_stimuli > len(colours):
        colours = colours * ((nr_stimuli // len(colours)) + 1)

    for i, row in df.iterrows():
        x_start = row["begin_fr"] / row["sampling_freq"]
        x_end = row["end_fr"] / row["sampling_freq"]

        # Add shaded area
        box = BoxAnnotation(
            left=x_start,
            right=x_end,
            fill_color=colours[i],
            fill_alpha=0.1,
            line_dash="dashed",
            level="underlay"
        )
        p.add_layout(box)

        # Add label at midpoint
        midpoint = (x_start + x_end) / 2
        if row["stimulus_name"] != row["stimulus_name"]:  # if stimulus_name is NaN
            row["stimulus_name"] = "Unknown"
        label = Label(
            x=midpoint,
            y=5,
            text=row["stimulus_name"],
            text_align="center",
            text_baseline="bottom",
            angle=120,
            text_color="red",
            background_fill_color="white",
            text_font_size="10pt"
        )
        p.add_layout(label)
