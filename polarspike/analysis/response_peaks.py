# This script provides functions to calculate peaks in the responses of neurons to stimuli.
import pandas as pd
from polarspike import (
    histograms,
)
import numpy as np
import polars as pl
from scipy.signal import find_peaks as find_peaks_scipy


def find_peaks(
        spikes_df: pl.DataFrame or pd.DataFrame,
        mean_trigger: np.ndarray,
        bin_size: float = 0.05,
) -> np.ndarray and np.ndarray:
    """
    Find the peaks in the response of a cell to a stimulus. Then align the peaks with the trigger signal.
    Returns the height of the peaks.

    Parameters
    ----------
    spikes_df : pl.DataFrame or pd.DataFrame
        A DataFrame containing the times of spikes relative to the stimulus onset.
    mean_trigger : np.ndarray
        The mean trigger times for the stimulus, return of the stimulus_spikes.mean_trigger_times function.
    bin_size : float
        The bin size for the PSTH.

    Returns
    -------
    all_heights : np.ndarray
        The height of the peaks.
    peak_locations : np.ndarray
        The location of the peaks.

    """
    trace, psth_bins = histograms.psth(
        spikes_df, bin_size=bin_size, end=np.sum(mean_trigger)
    )
    peaks, peak_properties = find_peaks_scipy(
        trace, height=np.std(trace), distance=(np.mean(mean_trigger) / bin_size) - 5
    )
    peak_locations = psth_bins[peaks]

    possible_peak_loc = np.zeros((len(mean_trigger)))
    for idx, trigger in enumerate(mean_trigger):
        possible_peak_loc[idx] = np.round(np.sum(mean_trigger[:idx]))

    # Matching
    abs_diff = np.abs(peak_locations[:, np.newaxis] - possible_peak_loc)
    nearest_index = np.argmin(abs_diff, axis=1)

    all_heights = np.zeros((len(mean_trigger)))
    all_heights[nearest_index] = peak_properties["peak_heights"]

    return all_heights, peak_locations
