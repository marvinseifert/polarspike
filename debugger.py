import Overview
import stimulus_spikes
import binarizer
import numpy as np
import single_cell_plots
import colour_template

CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6")

overview_df = Overview.Recording.load(r"D:\zebrafish_26_10_23\ks_sorted\overview")

spikes_df = overview_df.get_spikes_triggered([[10, 12]], [[5]])
fig, ax = single_cell_plots.whole_stimulus(
    spikes_df, spaced=True, height=10, cmap="Greys"
)
fig = colour_template.add_stimulus_to_matplotlib(fig, CT.colours, [2] * 12)
fig.show()
