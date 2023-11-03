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
        df, how="times_triggered", spaced=True, height=None, width=None, cmap="Greys"
):
    if height is None:
        height = int(5 * np.unique(df["cell_index"]).shape[0])
    if width is None:
        width = int(np.max(df[how]) / 0.05)

    unique_cells = pd.Series(df["cell_index"].unique()).sort_values()
    mapping = pd.Series(index=unique_cells, data=range(len(unique_cells)))

    # Map 'cell_index' to 'cell_index_linear' using the mapping
    df = df.copy()
    y_key = "repeat"
    if spaced:
        df["cell_index_linear"] = df["cell_index"].map(mapping)
        # Assign a new unique index for each cell and repeat:
        combined_indices = np.array(list(zip(df["cell_index_linear"], df["repeat"])))

        _, inverse_indices = np.unique(combined_indices, axis=0, return_inverse=True)
        df["cell_index_linear"] = inverse_indices + 1
        print(np.unique(df["cell_index_linear"]))

        y_key = "cell_index_linear"

    nr_cells = np.unique(df["cell_index"]).shape[0]
    nr_repeats = df["repeat"].max() + 1
    if nr_cells < 3:
        height = 10

    df["cell_index"] = df["cell_index"].astype("category")
    df["cell_index"] = df["cell_index"].cat.as_ordered()
    fig, ax = plt.subplots(figsize=(20, 1 * height))

    if spaced:
        plot_height = int(nr_cells * nr_repeats)
    else:
        plot_height = nr_repeats

    if type(cmap) == str:
        artist = dsshow(
            df,
            ds.Point(how, y_key),
            plot_height=plot_height,
            # color_key=cmap,
            cmap=cmap,
            plot_width=width,
            height_scale=5,
            norm="eq_hist",
            ax=ax,
        )

    else:
        if len(cmap) != len(unique_cells):
            cmap = cmap * len(unique_cells)
        for cell, c in zip(unique_cells, cmap):
            artist = dsshow(
                df.query(f"cell_index == {cell}"),
                ds.Point(how, y_key),
                plot_height=plot_height,
                # color_key=cmap,
                cmap=c,
                plot_width=width,
                norm="eq_hist",
                ax=ax,
            )
    # ax.set_aspect("auto")

    # norm = mcolors.Normalize(vmin=df[how].min(), vmax=df[how].max())

    # cbar = plt.colorbar(artist, ax=ax)

    ax.set_xlabel("Time")
    if spaced:
        ax.yaxis.set_ticks(np.arange(1, nr_cells * nr_repeats + 1, 2))
        ax.set_yticklabels(np.repeat(unique_cells.to_numpy(), nr_repeats)[::2])
    # ax.hlines(
    # ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_ylabel("Cell(s) - Repeat(s)")
    ax.set_title("Recording Overview")

    return fig, ax
