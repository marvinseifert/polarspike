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
    df, how="times_triggered", spaced=True, height=None, width=None, cmap="Dark2"
):
    if height is None:
        height = int(np.max(df["cell_index"]))
    if width is None:
        width = int(np.max(df[how]) / 0.05)

    unique_cells = pd.Series(df["cell_index"].unique()).sort_values()
    mapping = pd.Series(index=unique_cells, data=range(len(unique_cells)))
    max_repeat = df["repeat"].max().item()

    # Map 'cell_index' to 'cell_index_linear' using the mapping
    df = df.copy()
    y_key = "repeat"
    if spaced:
        df["cell_index_linear"] = df["cell_index"].map(mapping)
        df["cell_index_linear"] = (
            df["cell_index_linear"] * (max_repeat + 1) + df["repeat"]
        )
        y_key = "cell_index_linear"

    nr_cells = np.unique(df["cell_index"]).shape[0]
    if nr_cells < 10:
        height = 10

    df["cell_index"] = df["cell_index"].astype("category")
    df["cell_index"] = df["cell_index"].cat.as_ordered()
    fig, ax = plt.subplots(figsize=(20, 1 * height))
    colors = cm.get_cmap(cmap, nr_cells)
    cmap = [(255, 255, 255)] + [
        colors(c, bytes=True)[:-1][::-1] for c in range(nr_cells)
    ]

    artist = dsshow(
        df,
        ds.Point(how, y_key),
        ds.count_cat("cell_index"),
        plot_height=height,
        color_key=cmap,
        plot_width=width,
        norm="eq_hist",
        ax=ax,
    )
    ax.set_aspect("auto")

    # norm = mcolors.Normalize(vmin=df[how].min(), vmax=df[how].max())

    # cbar = plt.colorbar(artist, ax=ax)

    ax.set_xlabel("Time")
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_ylabel("Repeats and Cell(s)")
    ax.set_title("Recording Overview")

    return fig, ax
