"""
This module contains functions for plotting spike train data. Several different means of plotting are available, including
interactive plots using Plotly and Bokeh, as well as static plots using Matplotlib.

"""
import plotly.graph_objects as go
import polars as pl
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds
import numpy as np
import pandas as pd
from polarspike import histograms
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from polarspike import backbone
import matplotlib.cm as cm
import warnings
from scipy.stats import gaussian_kde


def whole_stimulus_plotly(df, stacked=False):
    warnings.warn("Deprecated, use spikes_and_trace instead", DeprecationWarning)

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
    df: pl.DataFrame | pd.DataFrame,
    how: str = "times_triggered",
    indices: list[str] = None,
    height: int = 10,
    width: int = 10,
    cmap: str | list[str] = "Greys",
    bin_size: float = 0.05,
    norm: str = "linear",
    single_psth: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Parameters:
    -----------
    df : DataFrame
        The input DataFrame containing the spike train data.
    how : str, optional
        The column name in `df` representing the time of each spike. Defaults to "times_triggered".
    indices : List[str], optional
        The column names in `df` to use as indices for partitioning the data. Defaults to ["cell_index", "repeat"].
    height : int, optional
        The height of the resulting plot. Defaults to 10.
    width : int, optional
        The width of the resulting plot. Defaults to 10.
    cmap : str or List[str], optional
        The colormap(s) to use for plotting. Defaults to "Greys".
    bin_size : float, optional
        The bin size for calculating the PSTH. (line and histogram) Defaults to 0.05.
    norm : str, optional
        The normalization method for the PSTH histogram. Defaults to "linear". Available options: see datashader documentation.
    single_psth : bool, optional
        Whether to calculate a single PSTH for all indices or separate PSTHs for each index. Defaults to True.
        If False, the PSTH will be calculated individually for the first index in `indices`.

    Returns:
    --------
    Tuple[Figure, Axes]
        A tuple containing the figure and axis objects of the resulting plot.
    """

    # Standard values:
    df, cmap, indices = _preprocess_input(df, indices, cmap)

    # Store some information about the data

    unique_indices = np.unique(df[indices[0]])
    max_time = np.max(df[how])
    # X range
    x_range = (0, max_time)

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
    # To space the cells, we need to map the combined repeats and cells indices to a linear range.
    df, repeated_indices, rows_per_index = map_index(df.copy(), indices)
    y_key = "index_linear"
    plot_height = len(repeated_indices)

    # Extend the colour map according to first index
    cmap = cmap * np.unique(np.stack(repeated_indices)[:, 0]).shape[0]

    # Get psths per category
    psth_list, bins_list = _calculate_psth(df, bin_size, single_psth, indices, max_time)

    # Plot the PSTH
    _plot_psth(psth_list, bins_list, axs, cmap)

    y_ranges = [
        (i, j)
        for i, j in zip(
            np.arange(0, plot_height, rows_per_index),
            np.arange(rows_per_index - 1, plot_height, rows_per_index),
        )
    ]
    df["index"] = df[indices[0]].astype("category")
    df["index"] = df["index"].cat.as_ordered()

    for index_id, c, y_r in zip(unique_indices, cmap, y_ranges):
        if type(index_id) is str:
            df_temp = df.query(f"index == '{index_id}'")
        else:
            df_temp = df.query(f"index == {index_id}")

        _draw_artist(
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
            (x_range, y_r),
        )

    fig, ax = _whole_stimulus_beautified(fig, axs, repeated_indices, indices, how, df)

    return fig, axs


def _calculate_psth(
    df: pl.DataFrame | pl.DataFrame,
    bin_size: float,
    single_psth: bool,
    indices: list[str],
    max_time: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Calculate the PSTH (Peri-Stimulus Time Histogram) for the given DataFrame.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame containing the spike train data.
    bin_size : float
        The bin size for calculating the PSTH.
    single_psth : bool
        Whether to calculate a single PSTH for all indices or separate PSTHs for each index.
    indices : List[str]
        The column names in `df` to use as indices for partitioning the data.
    max_time : float
        The maximum time value for the PSTH calculation.

    Returns:
    --------
    Tuple[List[np.ndarray], List[np.ndarray]]
        A tuple containing lists of the calculated PSTHs and corresponding bins.
    """
    if not single_psth:
        df_split = pl.from_pandas(df.loc[:, df.columns != "index"]).partition_by(
            indices[0]
        )  # This is a quick fix. The bigger problem is that the datatypes in the dataframe are not consistent.
        # If the dtype is categorical, the conversion to polars will fail here if the index column isnt excluded.
        psth_list = []
        bins_list = []
        for df_cat in df_split:
            psth, bins = histograms.psth(
                df_cat, bin_size=bin_size, start=0, end=max_time
            )
            psth = psth / df_cat[indices[0]].unique().shape[0] * (1 / bin_size)
            psth_list.append(psth)
            bins_list.append(bins)
        return psth_list, bins_list
    else:
        psth, bins = histograms.psth(df, bin_size=bin_size, start=0, end=max_time)
        psth = psth / df[indices[0]].unique().shape[0] * (1 / bin_size)
        return [psth], [bins]


def _plot_psth(
    psth_list: list[np.ndarray],
    bins_list: list[np.ndarray],
    axs: plt.Axes,
    cmap: str | list[str],
) -> None:
    """
    Plot the PSTH (Peri-Stimulus Time Histogram) using the given PSTHs and bins.

    Parameters:
    -----------
    psth_list : List[np.ndarray]
        A list of arrays representing the PSTHs.
    bins_list : List[np.ndarray]
        A list of arrays representing the bins for the PSTHs.
    axs : Any
        The axis object for plotting.
    cmap : str or List[str]
        The colormap(s) to use for plotting.

    Returns:
    --------
    None
    """
    for idx, (psth, bins) in enumerate(zip(psth_list, bins_list)):
        c = cm.get_cmap(cmap[idx])
        axs[0, 0].plot(bins[1:], psth, color=c(0.5), alpha=0.5)


def _whole_stimulus_beautified(
    fig: plt.Figure,
    axs: plt.Axes,
    repeated_indices: pd.DataFrame,
    indices: list[str],
    how: str,
    df: pl.DataFrame | pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Adjust the appearance of the whole stimulus plot.

    Parameters:
    -----------
    fig : plt.Figure
        The figure object of the plot.
    axs : plt.Axes
        The axis object of the plot.
    repeated_indices : pd.DataFrame
        The repeated indices.
    indices : List[str]
        The indices.
    how : str
        The 'how' parameter.
    df : pd.DataFrame
        The DataFrame.

    Returns:
    --------
    Tuple[Figure, Any]
        A tuple containing the adjusted figure and axis objects.
    """
    new_labels = np.arange(0, len(repeated_indices), 2)
    axs[1, 0].yaxis.set_ticks(new_labels)
    axs[1, 0].set_yticklabels(repeated_indices.to_numpy()[new_labels])
    axs[0, 0].set_ylabel(f"Spikes / s\n / {indices[0]}")
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
    axs[1, 0].set_ylim(0, len(repeated_indices))
    fig.subplots_adjust(left=0.2)
    return fig, axs


def _draw_artist(
    df: pd.DataFrame,
    fig: plt.Figure,
    axs: plt.Axes,
    how: str,
    y_key: str,
    cmap: str,
    norm: str,
    plot_height: int,
    plot_width: int,
    bin_size: float,
    ranges: tuple[tuple[float, float], tuple[float, float]],
) -> None:
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
        x_range=ranges[0],
        y_range=ranges[1],
    )
    cbar = fig.colorbar(artist, cax=axs[1, 1], shrink=0.1)

    cbar.set_label(f"Nr of spikes, binsize={bin_size} s")


def _preprocess_input(
    df: pl.DataFrame | pd.DataFrame, indices: list[str], cmap: str | list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Preprocess the input DataFrame and colormap.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame containing the spike train data.
    indices : List[str]
        The column names in `df` to use as indices for partitioning the data.
    cmap : str or List[str]
        The colormap(s) to use for plotting.

    Returns:
    --------
    Tuple[DataFrame, List[str]]
        A tuple containing the preprocessed DataFrame and colormap.
    """
    if type(df) is pl.DataFrame:
        df = df.to_pandas()

    if type(cmap) is str:
        cmap = [cmap]

    if indices is None:
        indices = ["cell_index", "repeat"]

    return df, cmap, indices


def bokeh_psth_plot(
    df: pl.DataFrame | pd.DataFrame,
    single_trace: bool,
    indices: list[str],
    line_colours: list[str],
    width: int,
    height: int,
    bin_size: float,
) -> figure:
    """
    Generate a plot of the PSTH trace for the bokeh plotting library.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame containing the spike train data.
    single_trace : bool
        Whether to plot a single PSTH.
    indices : List[str]
        The column names in `df` to use as indices for partitioning the data.
    line_colours : List[str]
        The colors of the lines in the plot, must match the logic of single_trace.
    width : int
        The width of the plot.
    height : int
        The height of the plot.
    bin_size : float
        The bin size for calculating the PSTH.

    Returns:
    --------
    bokeh.plotting.figure
        The generated plot as a figure object.
    """
    psth_list, bins_list = _calculate_psth(
        df, bin_size, single_trace, indices, df["times_triggered"].max()
    )
    bins_list = bins_list[0][:-1]
    bins_list = [bins_list] * len(psth_list)
    s1 = figure(width=width, height=int(0.2 * height), title=None, sizing_mode="fixed")

    s1.multi_line(bins_list, psth_list, line_width=2, color=line_colours, alpha=0.5)
    s1.xaxis.major_label_text_font_size = "0pt"
    return s1


def bokeh_kde_plot(
    df,
    single_trace,
    indices,
    line_colours,
    width,
    height,
    nr_samples,
    max_time,
    **kwargs,
):
    """
    Under construction
    """
    categories = df["index"].unique()
    s1 = figure(width=width, height=int(0.2 * height), title=None, sizing_mode="fixed")
    kdes = []
    for category in categories:
        kde_array = df.loc[df["index"] == category, "times_triggered"].values
        kde = gaussian_kde(kde_array)
        x = np.linspace(-1, max_time, nr_samples)
        kde_list = np.zeros(100)
        for i, x_i in enumerate(x):
            kde_list[i] = kde(x_i)[0]
        kdes.append(kde_list / np.max(kde_list))
    s1.multi_line([x] * len(kdes), kdes, line_width=2, color=line_colours, alpha=0.5)
    s1.y_range.start = 0
    s1.y_range.end = np.max(kdes) + 0.5
    return s1


def bokeh_raster_plot(
    df: pd.DataFrame,
    y_key: str,
    line_colours: list[str],
    width: int,
    height: int,
    plot_height: int,
    category_values: list[str],
    x_range: tuple[float, float],
    **kwargs: dict,
) -> figure:
    """
    Generate a raster plot for a bokeh figure.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame containing the spike train data.
    y_key : str
        The column name of the DataFrame containing the cell indices.
    line_colours : List[str]
        The colors of the lines in the plot.
    width : int
        The width of the plot.
    height : int
        The height of the plot.
    plot_height : int
        The height of the plot.
    category_values : List[str]
        The category values to use for the plot.
    x_range : Tuple[float, float]
        The range of the x-axis.

    Returns:
    --------
    bokeh.plotting.figure
        The generated plot as a figure object.


    """

    s2 = figure(
        width=width,
        height=int(0.8 * height),
        title=None,
        x_range=x_range,
        y_range=(-0.5, plot_height),
        sizing_mode="fixed",
    )
    if len(category_values) != len(line_colours):
        line_colours = line_colours * len(category_values)

    for index_id, c in zip(category_values, line_colours):
        source = df.loc[df["index"].astype(int) == int(index_id)]

        s2.dash(
            "times_triggered",
            y_key,
            size=10,
            source=source,
            color=c,
            alpha=1,
            angle=1.5708,
        )
    return s2


def spikes_and_trace(
    df: pl.DataFrame | pd.DataFrame,
    indices: list[str],
    width: int = 1400,
    height: int = 500,
    bin_size: float = 0.05,
    line_colour: str | list[str] = "black",
    single_psth: bool = True,
) -> gridplot:
    """
    Generate a plot of spikes and psth trace.

    Parameters:
        df : DataFrame
            The input DataFrame containing spike data.
        indices : list, optional
            The column names to use as indices. Defaults to None.
        width : int, optional
            The width of the plot. Defaults to 1400.
        height : int, optional
            The height of the plot. Defaults to 500.
        bin_size : float, optional
            The bin size for calculating the PSTH. Defaults to 0.05.
        line_colour : str, optional
            The color of the lines in the plot. Defaults to None.
        single_psth : bool, optional
            Whether to plot a single PSTH. Defaults to False.

    Returns:
        grid : GridPlot
            The generated plot as a grid of subplots.

    Raises:
        AssertionError
            If the number of spikes is too high (over 10000) for this plot.

    """

    grid = _bokeh_plotting(
        df,
        indices,
        width,
        height,
        single_trace=single_psth,
        line_colour=line_colour,
        bin_size=bin_size,
        s1_func=bokeh_psth_plot,
        s2_func=bokeh_raster_plot,
    )

    return grid


def first_spikes_plot(
    spikes: pd.DataFrame or pl.DataFrame,
    category_name: str,
    category_values: list,
    category_colours: list,
    nr_stim_repeats: int = None,
    secondary_index: str = "cell_index",
    max_time: float = None,
    nr_samples: int = 100,
    width: int = 1400,
    height: int = 500,
    bin_size: float = 0.05,
    single_trace: bool = False,
):
    if nr_stim_repeats is None:
        nr_stim_repeats = (spikes["repeat"].max() + 1) // len(category_values)
    if type(spikes) is pd.DataFrame:
        spikes = pl.from_pandas(spikes)

    nr_categories = len(category_values)
    repeat_max = nr_stim_repeats * nr_categories
    if max_time is None:
        max_time = spikes["times_triggered"].max()
    # spikes = spikes.with_columns(**{category_name: pl.lit("")})
    spikes = spikes.with_columns([pl.lit("").alias(category_name)])
    for category in range(nr_categories):
        spikes = spikes.with_columns(
            [
                (
                    pl.when(
                        pl.col("repeat").is_in(
                            np.arange(category, repeat_max, nr_categories)
                        )
                    )
                    .then(pl.lit(category_values[category], dtype=str))
                    .otherwise(pl.col(category_name))
                ).alias(category_name)
            ]
        )

    spikes = spikes.to_pandas().sort_values(by=category_name)
    indices = [category_name, secondary_index]
    grid = _bokeh_plotting(
        spikes,
        indices,
        width,
        height,
        single_trace=single_trace,
        line_colour=category_colours,
        bin_size=bin_size,
        s1_func=bokeh_kde_plot,
        s2_func=bokeh_raster_plot,
        max_time=max_time,
        nr_samples=nr_samples,
    )
    grid.children[0][0].yaxis.axis_label = f"kernel density \n [norm]"
    return grid


def _bokeh_plotting(
    df: pd.DataFrame,
    indices: list[str],
    width: int = 1400,
    height: int = 500,
    s1_func: callable = bokeh_psth_plot,
    s2_func: callable = bokeh_raster_plot,
    single_trace: bool = False,
    line_colour: str | list[str] = "black",
    **kwargs,
) -> gridplot:
    """
    Generate a plot of spikes and psth trace.

    Parameters:
        df : DataFrame
            The input DataFrame containing spike data.
        indices : list, optional
            The column names to use as indices. Defaults to None.
        width : int, optional
            The width of the plot. Defaults to 1400.
        height : int, optional
            The height of the plot. Defaults to 500.
        bin_size : float, optional
            The bin size for calculating the PSTH. Defaults to 0.05.
        line_colour : str, optional
            The color of the lines in the plot. Defaults to None.
        single_psth : bool, optional
            Whether to plot a single PSTH. Defaults to False.

    Returns:
        grid : GridPlot
            The generated plot as a grid of subplots.

    Raises:
        AssertionError
            If the number of spikes is too high (over 10000) for this plot.

    """

    df, line_colours, indices = _preprocess_input(df, indices, line_colour)
    if len(line_colours) == 1 or df[indices[0]].nunique() == 1:
        single_trace = True
    if len(indices) > 1:
        df, repeated_indices, rows_per_index = map_index(df, indices)
        y_key = "index_linear"
    else:
        y_key = indices[0]
        repeated_indices = pd.Index(np.unique(df[y_key]))
    plot_height = len(repeated_indices)

    # Map colours to the indices
    category_values = df[indices[0]].unique().astype(str).tolist()
    if not single_trace:
        line_colours = line_colours * len(category_values)
        line_colours = line_colours[: len(category_values)]

    df["index"] = df[indices[0]].astype("category")
    df["index"] = df["index"].cat.as_ordered()
    # First subplot
    s1 = s1_func(df, single_trace, indices, line_colours, width, height, **kwargs)

    # Second subplot
    s2 = s2_func(
        df,
        y_key,
        line_colours,
        width,
        height,
        plot_height,
        category_values,
        s1.x_range,
        **kwargs,
    )

    grid = _bokeh_beautified(s1, s2, width, indices, repeated_indices)

    return grid


def _bokeh_beautified(
    s1: figure,
    s2: figure,
    width: int,
    indices: list[str],
    repeated_indices: pd.MultiIndex,
) -> gridplot:
    """
    Beautify the Bokeh plot and arrange the subplots.

    Parameters
    ----------
    s1, s2 : Figure
        The figure objects for the subplots.
    width : int
        The width of the plot.
    indices : list
        The column names to use as indices.
    repeated_indices : Series
        The repeated indices.


    Returns
    -------
    grid : GridPlot
        The generated plot as a grid of subplots.

    """
    s2.xgrid.grid_line_color = None
    s2.ygrid.grid_line_color = None

    s2.xaxis.axis_label = "Time (s)"

    seperator = ", "
    s2.yaxis.axis_label = seperator.join(indices)

    s1.yaxis.axis_label = f"Spikes / s \n / {indices[0]}"
    s1.yaxis.axis_label_text_font_size = "9pt"

    if len(repeated_indices) >= 10:
        s2.yaxis.ticker = np.arange(0, len(repeated_indices), 2)
        s2.yaxis.major_label_overrides = {
            i: str(label)
            for i, label in backbone.enumerate2(
                repeated_indices.to_numpy()[1::2],
                start=0,
                step=2,
            )  #
        }
    else:
        s2.yaxis.ticker = np.arange(0, len(repeated_indices), 1)
        s2.yaxis.major_label_overrides = {
            i: str(label)
            for i, label in backbone.enumerate2(
                repeated_indices.to_numpy(),
                start=0,
                step=1,
            )  #
        }

    # Make sure the last label is included if len(repeated_indices) is odd
    if len(repeated_indices) % 2 != 0:
        s2.yaxis.major_label_overrides[len(repeated_indices)] = str(
            repeated_indices.to_numpy()[-1]
        )

    # Combine plots vertically
    grid = gridplot(
        [[s1], [s2]], width=width, sizing_mode="scale_width", merge_tools=True
    )
    return grid


def map_index(
    df: pd.DataFrame, index_columns: list[str]
) -> tuple[pd.DataFrame, pd.MultiIndex, int]:
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

    # Create an ideal index as expected from the index_columns
    # get unique entries per column
    unique_indices = df[index_columns].apply(pd.unique)
    rows_per_index = len(unique_indices[index_columns[-1]])

    ideal_index = pd.MultiIndex.from_product(np.asarray(unique_indices).T)
    # Get unique indices and create a mapping
    # unique_indices = pd.Series(multi_index.unique())
    mapping = pd.Series(index=ideal_index, data=range(len(ideal_index)))

    dummy_df = pd.DataFrame(
        index=ideal_index, data=ideal_index.map(mapping), columns=["index_linear"]
    )
    # Extract the actual index_linear
    index_linear = dummy_df.loc[multi_index].values.flatten()

    # Map the multi-dimensional index to a linear index
    df["index_linear"] = index_linear

    return df, ideal_index, rows_per_index
