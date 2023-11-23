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
    recording = Overview.Recording.load(r"D:\Zebrafish_14_11_23\ks_sorted\overview")
    recording.dataframes["fff_analysis"] = recording.spikes_df.query(
        "stimulus_name=='FFF'"
    )
    recording.dataframes["fff_analysis"] = recording.dataframes["fff_analysis"].loc[
        recording.dataframes["fff_analysis"]["nr_of_spikes"] > 20
    ]
    spikes_df = recording.get_spikes_df("fff_analysis")
