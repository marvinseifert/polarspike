import Overview
import stimulus_spikes
import binarizer
import numpy as np
import single_cell_plots
import colour_template

CT = colour_template.Colour_template()
CT.pick_stimulus("Contrast_Step")

overview_df = Overview.Recording.load(r"D:\zebrafish_26_10_23\ks_sorted\overview")

spikes_df = overview_df.get_spikes_triggered([["all"]], [[6]])
fig, ax = single_cell_plots.whole_stimulus(spikes_df, cmap=["Greys", "Reds"], spaced=False, height=10)
fig = colour_template.add_stimulus_to_matplotlib(fig, CT.colours, [2] * 20)
fig.show()
