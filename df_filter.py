import pandas as pd
import polars as pl


def find_stimuli(df, stimuli):
    if type(stimuli) == int:
        return get_stimulus_index(df, stimuli)
    elif type(stimuli) == list:
        if type(stimuli[0]) == int:
            df_filtered = [get_stimulus_index(df, stimuli[0])]
            for stimulus in stimuli[1:]:
                df_filtered.append(get_stimulus_index(df, stimulus))
            return pd.concat(df_filtered)
        else:
            df_filtered = [get_stimulus_name(df, stimuli[0])]
            for stimulus in stimuli[1:]:
                df_filtered.append(get_stimulus_name(df, stimulus))
            return pd.concat(df_filtered)
    elif type(stimuli) == str:
        return df.query(f"stimulus_name == '{stimuli}'")
    elif type(stimuli) == slice:
        return df.query(f"stimulus_index >= {stimuli.start} & stimulus_index <= {stimuli.stop}")
    elif type(stimuli) == dict:
        for key in stimuli.keys():
            df_filtered = df.query(f"stimulus_name == '{key}'")
            return df_filtered.query(f"stimulus_index >= {stimuli[key][0]} & stimulus_index <= {stimuli[key][1]}")


def get_stimulus_index(df, stimulus):
    return df.query(f"stimulus_index == {stimulus}")


def get_stimulus_name(df, stimulus):
    return df.query(f"stimulus_name == '{stimulus}'")


def get_stimulus_info(df):
    begin_end = df[["begin_fr", "end_fr"]].to_numpy()
    trigger = df[["trigger_fr_relative"]].to_numpy()[0][0]
    trigger_end = df[["trigger_ends"]].to_numpy()[0][0]
    stim_logic = df[["stimulus_repeat_logic"]].to_numpy()[0][0]
    return begin_end, trigger, trigger_end, stim_logic


def stim_names_to_indices(stimuli, df):
    # if any(isinstance(i, list) for i in stimuli):
    #     stimuli_temp = []
    #     for stimulus in stimuli:
    #         stimuli_temp = stimuli_temp + stimulus
    #     stimuli = stimuli_temp

    stimuli_int = [index
                   for stimulus in stimuli
                   for index in (get_stimulus_name(df, stimulus)["stimulus_index"].tolist()
                                 if isinstance(stimulus, str)
                                 else [stimulus])]
    return stimuli_int
