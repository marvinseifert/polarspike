import pandas as pd
import polars as pl


def find_stimuli(df, stimuli):
    """
    Filters the DataFrame df for the given stimuli. Allows for a range of ways to denote the stimuli.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the stimulus information.
    stimuli : int, list, str, slice, dict
        The stimuli to filter for. Can be an integer, a list of integers, a string, a slice, or a dictionary.

    """
    if isinstance(stimuli, int):
        return get_stimulus_index(df, stimuli)
    elif isinstance(stimuli, list):
        if isinstance(stimuli[0], int):
            df_filtered = [get_stimulus_index(df, stimuli[0])]
            for stimulus in stimuli[1:]:
                df_filtered.append(get_stimulus_index(df, stimulus))
            return pd.concat(df_filtered)
        else:
            df_filtered = [get_stimulus_name(df, stimuli[0])]
            for stimulus in stimuli[1:]:
                df_filtered.append(get_stimulus_name(df, stimulus))
            return pd.concat(df_filtered)
    elif isinstance(stimuli, str):
        return df.query(f"stimulus_name == '{stimuli}'")
    elif isinstance(stimuli, slice):
        return df.query(
            f"stimulus_index >= {stimuli.start} & stimulus_index <= {stimuli.stop}"
        )
    elif isinstance(stimuli, dict):
        for key in stimuli.keys():
            df_filtered = df.query(f"stimulus_name == '{key}'")
            return df_filtered.query(
                f"stimulus_index >= {stimuli[key][0]} & stimulus_index <= {stimuli[key][1]}"
            )


def get_stimulus_index(df, stimulus):
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


def get_stimulus_name(df, stimulus):
    """
    Filters the DataFrame df for the given stimulus index.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the stimulus information.
    stimulus : str
        The stimulus index to filter for.
    Returns
    -------
    DataFrame
        DataFrame containing the filtered stimulus index.
    """
    return df.query(f"stimulus_name == '{stimulus}'")


def get_stimulus_info(df):
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


def stim_names_to_indices(stimuli, stimulus_df):
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
