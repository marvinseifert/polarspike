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
    recordings.add_from_saved(r"D:\zebrafish_26_10_23\ks_sorted\overview")
    recordings.dataframes["fff_all"] = recordings.dataframes["spikes_df"].query(
        "stimulus_name=='FFF'"
    )
    recordings.get_spikes_df(cell_df="fff_all")
