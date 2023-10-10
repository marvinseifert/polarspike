
def stimulus_column_name_corrections(df):
    if "stimulus_name" in df.columns:
        print("stimulus_df: no column names to correct")
        return df
    df.reset_index(inplace=True)
    if "Stimulus_index" in df.columns:
        df.rename(columns={"Stimulus_index": "stimulus_index"}, inplace=True)
        print("Stimulus_index corrected")
    if "Stimulus_name" in df.columns:
        df.rename(columns={"Stimulus_name": "stimulus_name"}, inplace=True)
        print("Stimulus_name corrected")
    if "Begin_Fr" in df.columns:
        df.rename(columns={"Begin_Fr": "begin_fr"}, inplace=True)
        print("Begin_Fr corrected")
    if "End_Fr" in df.columns:
        df.rename(columns={"End_Fr": "end_fr"}, inplace=True)
        print("End_Fr corrected")
    if "Trigger_Fr_relative" in df.columns:
        df.rename(
            columns={"Trigger_Fr_relative": "trigger_fr_relative"}, inplace=True
        )
        print("Trigger_Fr_relative corrected")
    if "Trigger_int" in df.columns:
        df.rename(columns={"Trigger_int": "trigger_int"}, inplace=True)
        print("Trigger_int corrected")
    if "Stimulus_repeat_logic" in df.columns:
        df.rename(
            columns={"Stimulus_repeat_logic": "stimulus_repeat_logic"}, inplace=True
        )
        print("Stimulus_repeat_logic corrected")
    if "Stimulus_repeat_sublogic" in df.columns:
        df.rename(
            columns={"Stimulus_repeat_sublogic": "stimulus_repeat_sublogic"}, inplace=True
        )
        print("Stimulus_repeat_sublogic corrected")

    return df
    # End of column name corrections

def spikes_df_column_name_corrections(df):
    # Correct column names. First in spikes_df
    if "stimulus_name" in df.columns:
        print("spikes_df: No column names to correct")
        return df
    df.reset_index(inplace=True)
    if "Stimulus name" in df.columns:
        print("Stimulus name corrected")
        df.rename(columns={"Stimulus name": "stimulus_name"}, inplace=True)
    if "Stimulus ID" in df.columns:
        df.rename(columns={"Stimulus ID": "stimulus_index"}, inplace=True)
        print("Stimulus ID corrected")
    if "Recording" in df.columns:
        df.rename(columns={"Recording": "recording"}, inplace=True)
        print("Recording corrected")
    if "Cell index" in df.columns:
        df.rename(columns={"Cell index": "cell_index"}, inplace=True)
        print("Cell index corrected")
    if "Centres x" in df.columns:
        df.rename(columns={"Centres x": "centres_x"}, inplace=True)
        print("Centres x corrected")
    if "Centres y" in df.columns:
        df.rename(columns={"Centres y": "centres_y"}, inplace=True)
        print("Centres y corrected")
    if "Nr of Spikes" in df.columns:
        df.rename(columns={"Nr of Spikes": "nr_of_spikes"}, inplace=True)
        print("Nr of Spikes corrected")
    if "Area" in df.columns:
        df.rename(columns={"Area": "area"}, inplace=True)
        print("Area corrected")
    if "Spikes" in df.columns:
        df.rename(columns={"Spikes": "spikes"}, inplace=True)
        print("Spikes corrected")
    if "Idx" in df.columns:
        df.rename(columns={"Idx": "idx"}, inplace=True)
        print("Idx corrected")

    return df
