import Overview
import stimulus_spikes
import binarizer
import numpy as np
import spiketrain_plots
import colour_template


recordings = Overview.Recording_s(r"D:\combined_analysis", "chicken")
recordings.add_recording(
    Overview.Recording.load("D:\Chicken_19_08_21\Phase_00\overview")
)
recordings.add_recording(
    Overview.Recording.load("D:\Chicken_19_08_21\Phase_01\overview")
)
spikes_df = recordings.get_spikes_triggered([["all"]], [[10], [100]], [[0]])
fig, ax = single_cell_plots.whole_stimulus(
    spikes_df,
    stacked=True,
    indices="cell_index",
    height=10,
    norm="linear",
    bin_size=0.05,
)
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
