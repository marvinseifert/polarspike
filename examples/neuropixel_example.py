# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import Overview
from pathlib import Path
import pandas as pd

working_folder = Path(r"C:\Users\Florencia\OneDrive - University of Sussex\Polarspikes\Neuropixels recordings\FG002_2022-10-25_1_g0_imec0\sorting")
stimulus_df = pd.read_pickle(working_folder/"stimulus_df")
stimulus_df["trigger_fr_relative"] = stimulus_df["trigger_fr_relative"] - stimulus_df["begin_fr"]





spikes_df = pd.read_pickle(working_folder/"spikes_df")
spikes_df["recording"] = "FG002_2022_10_25"


#%%


recording = Overview.Recording(working_folder/"spike_times.parquet", working_folder/"test.bin", dataframes=
                               {"stimulus_df":stimulus_df, "spikes_df": spikes_df}, sampling_freq=30000)



#%%
recording.dataframes["gratings"] = recording.stimulus_df.copy()
recording.dataframes["gratings"]["stimulus_repeat_logic"] = 1

#%%
spikes = recording.get_spikes_triggered([[0]], [[0,1]])

#%%
spikes_g = spikes = recording.get_spikes_triggered([[0]], [[0,1]], stimulus_df="gratings", cell_df="spikes_df")