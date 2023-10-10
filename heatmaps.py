import numpy as np
import plotly.graph_objs as go
from xarray import DataArray
import datashader as ds
import datashader.transfer_functions as tf
import plotly.express as px
from warnings import warn







def heatmap(spikes_df, width=600, height=600):
    """
    Deprecated function. Use heatmap_figure instead, or create individual traces using psth_heatmap or psth_mean.
    Creates a heatmap figure of the PSTHs of all cells in the recording and the according trace
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the psths.
    width : int
        Width of the figure
    height : int
        Height of the figure
    Returns
    -------
    heatmap_trace : go.Scatter
        Trace of the average PSTH
    heatmap_raster : go.Heatmap
        Heatmap of the raster of the PSTHs
    heatmap_trace_std_u :
        Upper bound of the standard deviation
    heatmap_trace_std_l :
        Lower bound of the standard deviation
    """

    warn('This function will be deprecated, '
         'Use heatmap_figure instead, or create individual traces using psth_heatmap or psth_mean.', DeprecationWarning,
         stacklevel=2)
    data = create_heatmap_data(spikes_df)
    nr_cells = data["histogram_arr_heatmap"].shape[0]
    # Transfer data into DataArray
    histogram_arr_heatmap = DataArray(data=data["histogram_arr_heatmap"], dims=("cell_id", "Time"),
                                      coords={"cell_id": np.arange(0, data["histogram_arr_heatmap"].shape[0], 1),
                                              "Time": np.linspace(0, np.max(data["bins"]),
                                                                  data["histogram_arr_heatmap"].shape[1])})

    # Transfer data into datashader
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    img = tf.shade(cvs.raster(histogram_arr_heatmap, agg="mean"), how="log")
    # Create traces, mean trace first
    heatmap_trace = go.Scatter(
        x=data["bins"],
        y=data["histogram_mean"],
        mode="lines",
        name="Average PSTH",
        line=dict(color="#000000"),
    )
    # Create upper and lower bound of the standard deviation
    heatmap_trace_std_u = go.Scatter(
        name="Upper Bound",
        x=data["bins"],
        y=data["std_upper"],
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        fill="tonexty",
        showlegend=False,
    )

    heatmap_trace_std_l = go.Scatter(
        name="lower Bound",
        x=data["bins"],
        y=data["std_lower"],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(68, 68, 68, 0.6)",
        fill="tonexty",
        showlegend=False,
    )
    # Create the heatmap
    x = np.linspace(0, np.max(data["bins"]), width)
    y = np.linspace(0, nr_cells, height)

    heatmap_raster = go.Heatmap(x=x, y=y, z=img, colorscale=[
        [0, "rgb(0, 0, 0)"],
        [0.2, "rgb(50, 50, 50)"],
        [0.4, "rgb(100, 100, 100)"],
        [0.6, "rgb(150, 150, 150)"],
        [0.8, "rgb(200, 200, 200)"],
        [1.0, "rgb(250, 250, 250)"],
    ], showscale=False)

    return heatmap_trace, heatmap_raster, heatmap_trace_std_u, heatmap_trace_std_l


def psth_heatmap(spikes_df, width=600, height=600, agg="mean", how="log"):
    """
    Creates a heatmap figure of the PSTHs of all cells in the recording and the according trace
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the psths.
    width : int
        Width of the figure in pixels
    height: int
        Height of the figure in pixels
    **kwargs:
        Keyword arguments for datashader

    Returns
    -------
    heatmap_raster : go.Heatmap
        Heatmap of the raster of the PSTHs
    """
    # Get the data from the dataframe
    data = create_heatmap_data(spikes_df)
    nr_cells = data["histogram_arr_heatmap"].shape[0]
    # Transfer data into DataArray
    histogram_arr_heatmap = DataArray(data=data["histogram_arr_heatmap"], dims=("cell_id", "Time"),
                                      coords={"cell_id": np.arange(0, data["histogram_arr_heatmap"].shape[0], 1),
                                              "Time": np.linspace(0, np.max(data["bins"]),
                                                                  data["histogram_arr_heatmap"].shape[1])})

    # Transfer data into datashader. Make sure pixel dont effect the plotting
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    img = tf.shade(cvs.raster(histogram_arr_heatmap, agg=agg), how=how)
    # Create the heatmap
    x = np.linspace(0, np.max(data["bins"]), width)
    y = np.linspace(0, nr_cells, height)
    heatmap_raster = go.Heatmap(x=x, y=y, z=img, colorscale=[
        [0, "rgb(0, 0, 0)"],
        [0.2, "rgb(50, 50, 50)"],
        [0.4, "rgb(100, 100, 100)"],
        [0.6, "rgb(150, 150, 150)"],
        [0.8, "rgb(200, 200, 200)"],
        [1.0, "rgb(250, 250, 250)"],
    ], showscale=False)
    return heatmap_raster


def psth_mean(spikes_df):
    """
    Creates a trace of the mean PSTH of all cells in the recording.
    Parameters
    ----------
    spikes_df: pd.DataFrame
        Dataframe containing the psths.

    Returns
    -------
    heatmap_trace : go.Scatter
        Trace of the mean PSTH
    heatmap_trace_std_u : go.Scatter
        Trace of the upper bound of the standard deviation
    heatmap_trace_std_l : go.Scatter
        Trace of the lower bound of the standard deviation
    """
    # Get the data from the dataframe
    data = create_heatmap_data(spikes_df)
    # Create traces, mean trace first
    heatmap_trace = go.Scatter(
        x=data["bins"],
        y=data["histogram_mean"],
        mode="lines",
        name="Average PSTH",
        line=dict(color="#000000"),
    )
    # Create upper and lower bound of the standard deviation
    heatmap_trace_std_u = go.Scatter(
        name="Upper Bound",
        x=data["bins"],
        y=data["std_upper"],
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        fill="tonexty",
        showlegend=False,
    )

    heatmap_trace_std_l = go.Scatter(
        name="lower Bound",
        x=data["bins"],
        y=data["std_lower"],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(68, 68, 68, 0.6)",
        fill="tonexty",
        showlegend=False,
    )
     
    return heatmap_trace, heatmap_trace_std_u, heatmap_trace_std_l



def get_max_bins(histogram_column, bins_column):
    """
    Checks for the maximal number of bins in any recording.
    Parameters
    ----------
    histogram_column: pd.Series
        Column containing the PSTHs
    bins_column: pd.Series
        Column containing the bins

    Returns
    -------
    min_bins: int
        The maximal number of bins in any recording
    bins: np.array
        The bins

    """
    # Variables for the loop
    tries = 0
    min_bins = 0
    bins = np.array([0])
    # Loop over all PSTHs
    for array, bins_array in zip(histogram_column, bins_column):
        # Check if the PSTH is a numpy array and not None
        if type(array).__module__ == np.__name__:
            # First iteration, set the number of bins and bins.
            if tries == 0:
                min_bins = array.shape[0]
                bins = bins_array[1:].astype(float)
                tries = tries + 1
            # Next iterations, check if the number of bins is smaller than the current number of bins
            elif array.shape[0] < min_bins:
                min_bins = array.shape[0]
                bins = bins_array[1:].astype(float)
    return min_bins, bins


def create_heatmap_data(spikes_df, data_column="PSTH", bin_column="PSTH_x"):
    """
    Collects the data for the heatmap figure.
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the psths.
    data_column: str
        Name of the column containing the PSTHs
    bin_column: str
        Name of the column containing the bins

    Returns
    -------
    data: dict
        Dictionary containing the data for the heatmap figure.
        Keys:
            histogram_arr: np.array
                Array containing the PSTHs of all cells
            histogram_arr_heatmap: np.array
                Array containing the PSTHs of all cells for the heatmap
            histogram_mean: np.array
                Array containing the mean PSTH of all cells
            std_upper: np.array
                Array containing the upper bound of the standard deviation of the PSTHs
            std_lower: np.array
                Array containing the lower bound of the standard deviation of the PSTHs
            bins: np.array
                Array containing the bins

    """
    # Extract the PSTH column from the dataframe
    histogram_column = spikes_df.loc[:, data_column]
    histograms = histogram_column.values
    bins_column = spikes_df.loc[:, bin_column]
    # Check the maximal number of bins in any recording:
    min_bins, bins = get_max_bins(histogram_column, bins_column)
    # Create an array with the same number of bins for all recordings
    histogram_arr = np.zeros((len(spikes_df), min_bins))
    histogram_arr_heatmap = np.zeros((len(spikes_df), min_bins))
    nr_cells = np.shape(histograms)[0]

    for cell in range(nr_cells):
        # Don't run if the cell has no spikes
        if np.max(histograms[cell]) > 0:
            # Normalize the histogram to zscores
            histogram_arr[cell, :] = histograms[cell][:min_bins] - np.nanmean(
                histograms[cell][:min_bins]
            ) / np.nanstd(histograms[cell][:min_bins])
            # Normalize the histogram to the maximum value
            histogram_arr_heatmap[cell, :] = histograms[cell][:min_bins] / np.max(
                histograms[cell][:min_bins]
            )

        else:
            # Fill the array with zeros if the cell has no spikes
            histogram_arr[cell, :] = 0
            histogram_arr_heatmap[cell, :] = 0

    # Calculate the mean and standard deviation of the PSTHs
    histogram_mean = np.nanmean(histogram_arr, axis=0)
    histogram_std = np.nanstd(histogram_arr, axis=0)
    std_upper = np.add(histogram_mean, histogram_std)
    std_lower = np.subtract(histogram_mean, histogram_std)
    std_lower[std_lower < 0] = np.min(histogram_mean)

    data = {"bins": bins,
            "histogram_mean": histogram_mean,
            "histogram_std": histogram_std,
            "histogram_arr_heatmap": histogram_arr_heatmap,
            "std_upper": std_upper,
            "std_lower": std_lower,
            }
    return data


def get_kernel_dimensions(spikes_df, kernel_column="Kernel_norm"):
    """
    Get the dimensions of the kernel array for the plotting.
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the kernels.
    kernel_column : str
        Name of the column containing the kernels

    Returns
    -------
    nr_bins : int
        Number of bins in the kernel
    nr_colours : int
        Number of colours in the kernel

    """
    nr_bins = 0
    nr_colours = 0
    # Loop over all cells
    for cell in range(len(spikes_df)):
        try:
            # Get the number of bins and colours
            nr_bins = spikes_df.iloc[cell][kernel_column].shape[0]
            nr_colours = spikes_df.iloc[cell][kernel_column].shape[1]
            break
        except AttributeError:
            # If the cell has no spikes, continue to the next cell
            print(spikes_df.iloc[cell][kernel_column])
            continue
    return nr_bins, nr_colours


def get_kernel_data(spikes_df, kernel_column="Kernel_norm"):
    """
    Collects the data for the kernel figure.
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the kernels.
    kernel_column : str
        Name of the column containing the kernels.

    Returns
    -------
    data: dict
        Dictionary containing the data for the kernel figure.
        Keys:
            mean_kernels: np.array
                Array containing the mean kernel of all cells
            std_kernels: np.array
                Array containing the standard deviation of the kernels
            std_upper: np.array
                Array containing the upper bound of the standard deviation of the kernels
            std_lower: np.array
                Array containing the lower bound of the standard deviation of the kernels
            bins: np.array
                Array containing the bins
    """
    # Get all the cell indices
    cell_indices = spikes_df.index.get_level_values(
        spikes_df.index.names.index("Cell index")
    ).to_numpy()

    nr_bins, nr_colours = get_kernel_dimensions(spikes_df, kernel_column)

    nr_cells = np.shape(cell_indices)[0]
    # Create an array with the same number of bins for all recordings
    kernels = np.zeros((nr_cells, nr_bins, nr_colours), dtype=float)
    all_kernel = spikes_df["Kernel_norm"].to_numpy()

    for cell in range(nr_cells):
        kernels[cell, :, :] = all_kernel[cell]

    kernels_flat = kernels.reshape((nr_cells, nr_bins * nr_colours), order="F")

    mean_kernel = np.mean(kernels, axis=0)
    std_kernel = np.std(kernels, axis=0)
    std_upper = np.add(mean_kernel, std_kernel)
    std_lower = np.subtract(mean_kernel, std_kernel)
    data = {"kernels": kernels, "mean_kernel": mean_kernel, "std_upper": std_upper, "std_lower": std_lower,
            "nr_bins": nr_bins, "nr_colours": nr_colours, "kernels_flat": kernels_flat}
    return data


def plot_kernel_trace(spikes_df, kernel_column="Kernel_norm", colour_name="FF_Noise"):
    """
    Plot the kernel trace.
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the kernels in kernel_column.
    kernel_column : str
        Name of the column containing the kernels.
    colour_name : str
        Name of the colourmap to use.
    Returns
    -------
    traces : list
        List containing the traces for the kernel figure.
    """
    data = get_kernel_data(spikes_df, kernel_column)
    color_temp = ["#fe7c7c", "#8afe7c", "#7cfcfe", "#7c86fe"]
    names_temp = ["630nm", "505nm", "480nm", "420nm"]

    traces = [
        go.Scatter(
            x=np.arange(0, data["nr_bins"], 1),
            y=data["mean_kernel"][:, i],
            line=dict(color=color_temp[i], width=2.5),
            name=names_temp[i],
        )
        for i in range(data["nr_colours"])
    ]
    return traces


def plot_kernel_heatmap(spikes_df, kernel_column="Kernel_norm", width=300, height=400, agg="mean", how="log",
                        colour_name="FF_Noise"):
    """
    Plot the kernel heatmap.
    Parameters
    ----------
    spikes_df : pd.DataFrame
        Dataframe containing the kernels in kernel_column.
    kernel_column : str
        Name of the column containing the kernels.
    colour_name : str
        Name of the colour for the heatmap

    Returns
    -------

    """
    data = get_kernel_data(spikes_df, kernel_column)
    nr_cells = data["kernels"].shape[0]
    # Transfer data into DataArray
    histogram_arr_heatmap = DataArray(data=data["kernels_flat"], dims=("cell_id", "Time"),
                                      coords={"cell_id": np.arange(0, data["kernels_flat"].shape[0], 1),
                                              "Time": np.linspace(0, data["nr_bins"],
                                                                  data["kernels_flat"].shape[1])})

    # Transfer data into datashader. Make sure pixel dont effect the plotting
    heatmap_raster = plot_heatmap_from_xarray(histogram_arr_heatmap, width, height, agg, how)
    return heatmap_raster


def plot_heatmap_from_xarray(histogram_arr_heatmap, width=300, height=400, agg="mean", how="log"):
    """
    Plot the heatmap from the xarray.
    Parameters
    ----------
    histogram_arr_heatmap : xarray.DataArray
        DataArray containing the data for the heatmap.

    width : int
        Width of the heatmap canvas
    height : int
        Height of the heatmap canvas
    agg : str
        Aggregation method for the heatmap.
    how: str
        How to the colour map is implemented. Log or linear.

    Returns
    -------
    heatmap_raster : go.Heatmap
        Heatmap object for a plotly figure.

    """
    nr_cells = histogram_arr_heatmap.shape[0]
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    img = tf.shade(cvs.raster(histogram_arr_heatmap, agg=agg), how=how)
    # Create the heatmap
    x = histogram_arr_heatmap.coords["Time"].values
    y = np.linspace(0, nr_cells, height)
    heatmap_raster = go.Heatmap(x=x, y=y, z=img, colorscale=[
        [0, "rgb(0, 0, 0)"],
        [0.2, "rgb(50, 50, 50)"],
        [0.4, "rgb(100, 100, 100)"],
        [0.6, "rgb(150, 150, 150)"],
        [0.8, "rgb(200, 200, 200)"],
        [1.0, "rgb(250, 250, 250)"],
    ], showscale=False)
    return heatmap_raster
