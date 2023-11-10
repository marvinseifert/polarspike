from pathlib import Path
import recording_overview
import matplotlib.pyplot as plt
from IPython import display
import Overview
import spiketrain_plots
import colour_template
import stimulus_spikes

plt.style.use("dark_background")
import Extractors

if __name__ == "__main__":
    CT = colour_template.Colour_template()
    CT.pick_stimulus("FFF_6")
    recording = Overview.Recording.load(r"D:\zebrafish_26_10_23\ks_sorted\overview")
    flash_durations = stimulus_spikes.mean_trigger_times(recording.stimulus_df, [0, 5])
    spikes_df = recording.get_spikes_triggered([["all"]], [["FFF"]])
    fig, ax = spiketrain_plots.whole_stimulus(
        spikes_df, stacked=True, height=10, index="stimulus_index"
    )
    fig = CT.add_stimulus_to_plot(fig, flash_durations, names=False)
