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
    recording = Overview.Recording.load(r"D:\Zebrafish_14_11_23\ks_sorted\overview")
    recording.dataframes["fff_analysis"] = recording.spikes_df.query("stimulus_name=='FFF'").copy()
    recording.get_spikes_df("fff_analysis")
