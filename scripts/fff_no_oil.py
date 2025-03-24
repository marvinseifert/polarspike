from polarspike import Overview
from pathlib import Path

# %%
root = Path("A:/Marvin/fff_clustering")
recordings = Overview.Recording_s.load(root / "records")

# %%
recordings.dataframes["fff_top"] = recordings.dataframes["spikes_df"].query(
    "stimulus_name == 'fff_top'"
)
recordings.dataframes["fff_top"] = recordings.dataframes["fff_top"].query("qi>0.2")
recordings.save_save()
