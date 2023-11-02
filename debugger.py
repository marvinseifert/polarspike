import Overview
import stimulus_spikes
import binarizer
import numpy as np

overview_df = Overview.Recording.load(r"D:\zebrafish_26_10_23\ks_sorted\overview")

spikes_df = overview_df.get_spikes_triggered([["all"], [12]], [[0], [5]])

print(spikes_df)
