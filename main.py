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
    recordings = Overview.Recording_s.load_from_single(
        r"D:\combined_analysis",
        "test_analysis",
        r"D:\zebrafish_14_11_23\ks_sorted\overview",
    )
    spikes = recordings.get_spikes_triggered(
        [["zebrafish_14_11_23"]], [[[0], [5]]], [[[270, 271, 272, 273]]]
    )
    fig = spiketrain_plots.spikes_and_trace(spikes, stacked=True)
