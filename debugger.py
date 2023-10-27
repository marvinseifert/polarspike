import Overview
import stimulus_spikes
import binarizer
import numpy as np

overview_df = Overview.Recording.load("D:\Chicken_03_08_21\Phase_01\overview")
recordings = Overview.Recording_s(r"D:\combined_analysis\chicken_test", "Chicken_re")
recordings.add_recording(
    Overview.Recording.load("D:\Chicken_03_08_21\Phase_01\overview")
)
spikes_df = recordings.get_spikes_triggered(["all"], [[101]], [["FFF"]], pandas=False)
result = binarizer.timestamps_to_binary_multi(
    spikes_df,
    0.001,
    np.sum(stimulus_spikes.mean_trigger_times(overview_df.stimulus_df, 0)),
)
