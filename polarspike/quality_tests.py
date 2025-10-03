"""
This module contains functions to calculate quality indices for spiketrains.
Currently, only the quality index according to Baden et al. 2016 is implemented.
@ Marvin Seifert 2024
"""

import polars.exceptions
from polarspike import binarizer
import polars as pl
import pandas as pd


def spiketrain_qi(
        spikes: pl.DataFrame,
        max_window: int = 1,
        bin_size: float = 0.001,
        max_repeat: int = 24,
) -> pd.DataFrame:
    """
    Calculate the quality index of a spiketrain or multiple spiketrains from multiple cells/ recordings.
    Warning! This function does not split the spiketrains by stimulus. Provide a single stimulus or multiple stimuli.
    In the latter case, the quality index will be calculated for all stimuli combined.
    The quality index is calculated according to Baden et al. 2016 after gaussian convolution of the binary spike train.
    The quality index is calculated for each cell in the spiketrain.

    Parameters
    ----------
    spikes : polars.DataFrame or pandas.DataFrame
        The spiketrain(s) to calculate the quality index for.
    max_window : int, optional
        The maximum window size in seconds. The default is 1.
    bin_size : float, optional
        The bin size in seconds. The default is 0.001.
    max_repeat : int, optional
        The maximum number of repeats of the stimulus. The default is 24.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the quality index for each cell in the spiketrain(s).

    """
    # In case a pandas DataFrame is provided, convert it to a polars DataFrame
    if isinstance(spikes, pd.DataFrame):
        spikes = pl.from_pandas(spikes)

    # Check if the DataFrame contains a column "recording" and split the DataFrame by recording
    if any("recording" in column for column in spikes.columns):
        if len(spikes["recording"].unique()) > 1:
            spikes_by_recording = spikes.partition_by("recording")
        else:
            spikes_by_recording = [spikes]
    else:  # Create a single item list.
        spikes_by_recording = [spikes]

    rec_result = []
    # Loop over the recordings and calculate the quality index for each cell
    for recording in spikes_by_recording:
        # First convert the timestamps to binary signals
        binary_df = binarizer.timestamps_to_binary_multi(
            recording, max_window=max_window, bin_size=bin_size, max_repeat=max_repeat
        )
        cell_indices = binary_df["cell_index"].unique().to_numpy()
        # Next, calculate the quality index
        qis = binarizer.calc_qis(binary_df)
        # Create a new DataFrame with the quality indices
        result_df = pd.DataFrame(
            data=qis,
            columns=["qi"],
            index=pd.Index(data=cell_indices, name="cell_index"),
        )
        try:
            result_df["recording"] = recording["recording"][0]
            result_df = result_df.set_index("recording", append=True)
        except pl.exceptions.ColumnNotFoundError:
            pass
        rec_result.append(result_df)
    return pd.concat(rec_result)
