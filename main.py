from pathlib import Path
import recording_overview
import matplotlib.pyplot as plt
from IPython import display
import Overview
import spiketrain_plots

plt.style.use("dark_background")
import Extractors

if __name__ == "__main__":
    recording = Overview.Recording.load(r"D:\zebrafish_26_10_23\ks_sorted\overview")
    spikes_df = recording.get_spikes_triggered([["all"]], [["FFF"]])
    fig, ax = spiketrain_plots.whole_stimulus(
        spikes_df, stacked=True, height=10, index="stimulus_index"
    )
