from polarspike import Overview, spiketrain_plots, colour_template
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# %%
cluster_df = pd.read_pickle(r"D:\zebrafish_combined\rf_clusters.pkl")

# %%
paths = cluster_df["recording"].unique()

# %% Load the recordings
recordings_list = []
cluster_df_corrected = cluster_df.copy()
for path in paths:
    recording_path = Path(path).parent / "overview"
    recording = Overview.Recording.load(recording_path)
    cluster_df_corrected = cluster_df_corrected.replace(path, recording.name)
    recordings_list.append(recording)


# %%
recordings = Overview.Recording_s("D:\zebrafish_combined", "rf_clusters")
for rec in recordings_list:
    recordings.add_recording(rec)

# %%
recordings = Overview.Recording_s.load(r"D:\zebrafish_combined\rf_combined")
recordings.dataframes["rf_df_chirp"] = recordings.dataframes["rf_df"].query(
    "stimulus_name=='chirp' & qi>0.4"
)
# %%
spikes = recordings.get_spikes_df("rf_df_chirp", pandas=False, carry=["cluster"])

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
fig, ax = spiketrain_plots.whole_stimulus(
    spikes, indices=["cluster", "repeat"], cmap=["Greys", "Reds"], norm="eq_hist"
)
# fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()
