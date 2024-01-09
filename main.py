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
import stimulus_dfs

plt.style.use("dark_background")
import Extractors

if __name__ == "__main__":
    recordings = Overview.Recording_s.load("D:\\combined_analysis\\all_recordings")
    recordings.dataframes["test_df"] = recordings.dataframes["fff_df"].loc[
        (recordings.dataframes["fff_df"]["cell_index"] == 5)
        | (recordings.dataframes["fff_df"]["cell_index"] == 6)
    ]

    spikes = recordings.get_spikes_df(cell_df="test_df", stimulus_df="fff_stim_normal")

    fig, ax = spiketrain_plots.whole_stimulus(
        spikes, index=["recording", "cell_index", "repeat"], width=14
    )
