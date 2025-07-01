from polarspike import (
    Overview,
    quality_tests,
)
from pathlib import Path
import polars as pl
import pickle

# %%


recordings = Overview.Recording_s.load_from_single(
    r"D:\fff_clustering",
    "chicken_fff",
    "" r"A:\Marvin\chicken_19_07_2024\Phase_00\overview",
)
recordings.add_from_saved(r"A:\Marvin\chicken_30_08_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_05_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_11_09_2024\Phase_00\overview")

# %%
recordings.dataframes["fff"] = recordings.spikes_df.query("stimulus_name == 'fff'")
spikes = recordings.get_spikes_df("fff")
qis = quality_tests.spiketrain_qi(spikes, max_window=24, max_repeat=10)
recordings.dataframes["fff"] = recordings.dataframes["fff"].set_index(
    ["cell_index", "recording"]
)
recordings.dataframes["fff"].update(qis)
recordings.dataframes["fff"] = recordings.dataframes["fff"].reset_index()

# %%
recordings.dataframes["fff_filtered"] = recordings.dataframes["fff"].query("qi>0.3")

# %%
recordings.save(r"D:\cluster_all\records")
# %%
noise_paths = []
for rec in recordings.recordings.keys():
    noise_paths.append(
        Path(recordings.recordings[rec].load_path).parent / "noise_analysis"
    )
# %%
rf_dicts = []
rf_dfs = []
cut_sizes = []
recording_names = list(recordings.recordings.keys())
for path_idx, path in enumerate(noise_paths):
    rf_dict = pickle.load(open(path / "settings.pkl", "rb"))
    cut_sizes.append(rf_dict["noise"]["cut_size"])
    rf_dicts.append(rf_dict)
    df = pl.scan_parquet(path / "noise_df")
    df = df.with_columns(recording=pl.lit(recording_names[path_idx]))
    rf_dfs.append(df)

# %%
rf_dfs[2].dtypes
