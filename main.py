from pathlib import Path
import recording_overview
import matplotlib.pyplot as plt
from IPython import display
import Overview
import spiketrain_plots
import colour_template
import stimulus_spikes
import binarizer
import numpy as np
import stimulus_trace
import stimulus_dfs

plt.style.use("dark_background")
import Extractors

if __name__ == "__main__":
    stimulus_folder = Path(r"D:\Florencia_data\StimulusInfo_Experiment_1")
    starts = np.load(stimulus_folder / "stimuli_start.npy")
    ends = np.load(stimulus_folder / "stimuli_end.npy")
    df_creator = stimulus_dfs.Stimulus_df_schroeder(recording_name="FG002_2022_10_25", sampling_freq=30000)
    df_creator.add_stimulus("Gratings", starts=starts, ends=ends, stimulus_repeat_logic=24, stimulus_repeat_sublogic=1)
