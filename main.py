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

plt.style.use("dark_background")
import Extractors

if __name__ == "__main__":
    recording = Overview.Recording.load(r"D:\zebrafish_26_10_23\ks_sorted\overview")
    recording.dataframes["fff_analysis"] = recording.extract_df_subset(
        [1, 2, 3, 4, 5], stimulus_name=["FFF"]
    )
