from polarspike import Overview
import polars as pl

# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering_zf\records")

# %%
stimulus_df = pl.from_pandas(recordings.stimulus_df)
spikes_df = pl.from_pandas(recordings.spikes_df)
# %%
old_stimulus = "csteps420"
new_stimulus = "csteps413"

stimulus_df = stimulus_df.with_columns(
    pl.col("stimulus_name").replace(old_stimulus, new_stimulus)
)
spikes_df = spikes_df.with_columns(
    pl.col("stimulus_name").replace(old_stimulus, new_stimulus)
)
# %%
recordings.dataframes["stimulus_df"] = stimulus_df.to_pandas()
recordings.dataframes["spikes_df"] = spikes_df.to_pandas()
# %%
recordings.synchronize_dataframes()
# %%
recordings.save_save()
