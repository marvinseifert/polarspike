import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from bokeh.plotting import figure, show


def spike_counts_from_file(file_parquet, cmap="Oranges"):
    df = pl.scan_parquet(file_parquet)
    df = df.select(pl.col("cell_index"), pl.col("times"))
    return plot_spike_counts(df, cmap=cmap)


def isi_from_file(file_parquet, freq=1, x="times", cmap="viridis", cutoff=0.001):
    cutoff = (1 / freq) * cutoff
    df = pl.scan_parquet(file_parquet)
    df = df.select(pl.col("cell_index"), pl.col("times"))
    df = df.with_columns(pl.col("times").truediv(freq).alias("times"))
    return plot_isi(df, x, cmap=cmap, cutoff=cutoff)


def spiketrains_from_file(file_parquet, freq=1, cmap="Greys", height=None, width=None):
    df = pl.scan_parquet(file_parquet)
    df = df.select(pl.col("cell_index"), pl.col("times"))
    df = df.with_columns(pl.col("times").truediv(freq).alias("times"))
    return plot_spiketrains(df, cmap=cmap, height=height, width=width)


def spike_amplitudes_from_file(file_parquet, freq=1, cmap="viridis"):
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


def plot_spike_counts(df, cmap):
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
        np.log10([10**i for i in range(max_spike_count)])
    )  # Replace with the desired ticks in the original scale
    ax.set_yticklabels([10**i for i in range(max_spike_count)])

    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Number of Spikes")
    ax.set_title("Recording Overview")
    return fig, ax


def plot_isi(df, x, cmap="viridis", cutoff=18.0):
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
        np.log10([10**i for i in range(max_isi)])
    )  # Replace with the desired ticks in the original scale
    ax.set_yticklabels([10**i for i in range(max_isi)])

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


def plot_spiketrains(df, cmap, height=None, width=None):
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

    ax.set_xlabel("time in seconds")
    ax.set_ylabel("cell_index")
    ax.set_title("Recording Overview")

    return fig, ax


def plot_spike_amplitudes(df, cmap):
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


def empty_spike_counts_figure():
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Number of Spikes")
    ax.set_title("Recording Overview")
    return fig, ax
