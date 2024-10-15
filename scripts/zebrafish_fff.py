from polarspike import Overview, quality_tests
from pathlib import Path

# %%
root = Path(r"A:\Marvin\fff_clustering_zf")

# %%
recordings = Overview.Recording_s.load_from_single(
    root, "records", Path(r"A:\Marvin\zebrafish_05_07_2024\Phase_00\overview")
)
recordings.add_from_saved(Path(r"A:\Marvin\zebrafish_08_07_2024\Phase_00\overview"))
recordings.add_from_saved(Path(r"B:\Marvin\Zebrafish_04_22_2024\Phase_00\overview"))
recordings.add_from_saved(Path(r"B:\Marvin\Zebrafish_04_23_2024\Phase_01\overview"))
recordings.add_from_saved(Path(r"B:\Marvin\Zebrafish_19_04_2024\Phase_01\overview"))
recordings.add_from_saved(Path(r"B:\Marvin\Zebrafish_20_11_23\ks_sorted\overview"))
recordings.add_from_saved(Path(r"B:\Marvin\Zebrafish_21_11_23\ks_sorted\overview"))
recordings.add_from_saved(Path(r"B:\Marvin\zebrafish_25_10_23\ks_sorted\overview"))
recordings.add_from_saved(Path(r"D:\zebrafish_26_10_23\ks_sorted\overview"))
recordings.add_from_saved(Path(r"D:\zebrafish_25_10_23\ks_sorted\overview"))
recordings.save(root / "records")
# %%
recordings.dataframes["fff"] = recordings.dataframes["spikes_df"].query(
    "stimulus_name == 'fff'"
)
recordings.save(root / "records")
recordings.synchronize_dataframes()

# %%
spikes = recordings.get_spikes_df("fff")
# %%
qis = quality_tests.spiketrain_qi(spikes, max_window=24, max_repeat=10)
# %%
recordings.dataframes["fff"] = recordings.dataframes["fff"].set_index(
    ["cell_index", "recording"]
)
recordings.dataframes["fff"].update(qis)
recordings.dataframes["fff"] = recordings.dataframes["fff"].reset_index()
# %%
recordings.dataframes["fff_filtered"] = recordings.dataframes["fff"].query("qi>0.3")
# %%
recordings.save(root / "records")
