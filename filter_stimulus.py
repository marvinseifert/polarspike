"""
This module contains functions to filter stimulus information from a stimulus DataFrame.
@ Marvin Seifert 2024
"""

import pandas as pd
import numpy as np


def get_stimulus_index(df: pd.DataFrame, stimulus: int) -> pd.DataFrame:
    """
    Filters the DataFrame df for the given stimulus index.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the stimulus information.
    stimulus : int
        The stimulus index to filter for.
    Returns
    -------
    DataFrame
        DataFrame containing the filtered stimulus index.
    """
    return df.query(f"stimulus_index == {stimulus}")


def get_stimulus_name(df: pd.DataFrame, stimulus: str) -> pd.DataFrame:
    """
    Filters the DataFrame df for the given stimulus name.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the stimulus information.
    stimulus : str
        The stimulus name to filter for.
    Returns
    -------
    DataFrame
        DataFrame containing the filtered stimulus name.
    """
    return df.query(f"stimulus_name == '{stimulus}'")


def get_stimulus_info(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the stimulus information from the DataFrame df as numpy arrays.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the stimulus information.
    Returns
    -------
    begin_end : numpy array
        Array containing the begin and end frame of the stimuli.
    trigger : numpy array
        Array containing the trigger frame of the stimuli.
    trigger_end : numpy array
        Array containing the trigger end frame of the stimuli.
    stim_logic : numpy array
        Array containing the stimulus repeat logic.

    """
    begin_end = df[["begin_fr", "end_fr"]].to_numpy()
    trigger = df[["trigger_fr_relative"]].to_numpy()[0][0]
    trigger_end = df[["trigger_fr_relative"]].to_numpy()[0][0][1:]
    # trigger_end = df[["trigger_ends"]].to_numpy()[0][0]
    stim_logic = df[["stimulus_repeat_logic"]].to_numpy()[0][0]
    return begin_end, trigger, trigger_end, stim_logic


def stim_names_to_indices(stimuli: list[str], stimulus_df: pd.DataFrame) -> list:
    """
    Converts stimulus names to stimulus indices according to the stimulus_df.

    Parameters
    ----------
    stimuli : list
        List of stimuli to convert.
    stimulus_df : DataFrame
        DataFrame containing the stimulus information.


    """

    stimuli_int = [
        [index]
        for stimulus in stimuli
        for index in (
            get_stimulus_name(stimulus_df, stimulus[0])["stimulus_index"].tolist()
            if isinstance(stimulus[0], str)
            else stimulus
        )
    ]
    return stimuli_int
