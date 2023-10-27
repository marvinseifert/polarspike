import Overview
import stimulus_spikes
import binarizer
import numpy as np

if __name__ == "__main__":
    overview_df = Overview.Recording.load("D:\zebrafish_20_10_2023\ks_sorted\overview")
    recordings = Overview.Recording_s(
        r"D:\combined_analysis\chicken_test", "Chicken_re"
    )
    recordings.add_recording(
        Overview.Recording.load("D:\zebrafish_20_10_2023\ks_sorted\overview")
    )
    spikes_df = recordings.get_spikes_triggered(
        ["all"], [np.arange(100).tolist()], [["FFF"]], pandas=False
    )
    result = binarizer.timestamps_to_binary_multi(
        spikes_df,
        0.001,
        np.sum(stimulus_spikes.mean_trigger_times(overview_df.stimulus_df, 0)),
        recordings.stimulus_df.loc[0]["nr_repeats"],
    )
    result_list = binarizer.calc_qis(result)
