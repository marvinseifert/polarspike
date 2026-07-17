from polarspike import Overview
from polarspike import histograms

recordings = Overview.Recording_s(r"/media/mawa/fast_data/oil_analysis", "oil_no_oil")
# %%
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_18_07_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_11_09_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_05_09_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_30_08_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_09_09_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_17_07_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_04_09_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_19_07_2024/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_13_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_14_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_09_05_2024/Phase_00/overview")

# %%
spikes = recordings.get_spikes_triggered(
    [{"stimulus_name": "fff"}], carry=["qi"], pandas=False
)
# %%
psth, indices = histograms.psth_by_index(spikes, index=["recording", "cell_index"])
