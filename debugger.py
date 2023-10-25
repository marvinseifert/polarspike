import Overview

recordings = Overview.Recording_s(r"D:\combined_analysis", "test_analysis")
overview_df = Overview.Recording.load(r"D:\zebrafish_19_10_2023\ks_sorted\overview")
recordings.add_recording(overview_df)
second_rec = Overview.Recording.load(r"D:\zebrafish_20_10_2023\ks_sorted\overview")
recordings.add_recording(second_rec)
df_test = recordings.get_spikes_triggered(["all"], [[10, 11, 12]], [["FFF"], ["FFF"]])
