from polarspike import Overview, spiketrain_plots, colour_template
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# %%
CT_fff = colour_template.Colour_template()
CT_fff.pick_stimulus("FFF_6_MC")
CT_contrast = colour_template.Colour_template()
CT_contrast.pick_stimulus("Contrast_Step")

# %% Filter good cells
contrast_df = pl.scan_parquet(
    r"D:\chicken_analysis\changepoint_df_csteps_test.parquet"
).collect()
posterior_columns = [f"first_spike_posterior_{i}" for i in range(0, 20, 2)]
good_cells = pl.DataFrame(contrast_df.select(posterior_columns).mean_horizontal())
good_cells = pl.concat(
    [good_cells, contrast_df.select("cell_index"), contrast_df.select("recording")],
    how="horizontal",
)
good_cells = good_cells.sort("mean", descending=True).drop_nulls()
good_cells = good_cells.filter(pl.col("mean") > 0.4)

# %%
idx = np.random.randint(0, len(good_cells))
cell_df_fff = (
    pl.scan_parquet(r"D:\chicken_analysis\changepoint_df_fff_test.parquet")
    .filter(
        (pl.col("recording") == good_cells[idx].select("recording").item())
        & (pl.col("cell_index") == good_cells[idx].select("cell_index").item())
    )
    .collect()
)
cell_df_contrast = (
    pl.scan_parquet(r"D:\chicken_analysis\changepoint_df_csteps.parquet")
    .filter(
        (pl.col("recording") == good_cells[idx].select("recording").item())
        & (pl.col("cell_index") == good_cells[idx].select("cell_index").item())
    )
    .collect()
)
# %% Raw spike data
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
cell = good_cells[idx].select("cell_index").item()
recording = good_cells[idx].select("recording").item()
recordings.dataframes["single_cell_csteps"] = recordings.spikes_df.query(
    f"recording==@recording&cell_index==@cell&stimulus_name=='csteps'"
)
recordings.dataframes["single_cell_fff"] = recordings.spikes_df.query(
    f"recording==@recording&cell_index==@cell&stimulus_name=='fff'"
)
recordings.dataframes["single_cell_chirp"] = recordings.spikes_df.query(
    f"recording==@recording&cell_index==@cell&stimulus_name=='chirp'"
)
recordings.dataframes["single_cell_noise"] = recordings.spikes_df.query(
    f"recording==@recording&cell_index==@cell&stimulus_name=='20px_20Hz_shuffle_w'"
)

# %%
spikes_csteps = recordings.get_spikes_df("single_cell_csteps", pandas=False)
spikes_fff = recordings.get_spikes_df("single_cell_fff", pandas=False)
# spikes_noise = recordings.get_spikes_df("single_cell_noise", pandas=False)
spikes_chirp = recordings.get_spikes_df("single_cell_chirp", pandas=False)
# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_chirp,
    indices=["stimulus_index", "cell_index", "repeat"],
    width=20,
    height=3,
    bin_size=0.05,
)
# fig = CT_contrast.add_stimulus_to_plot(fig, [2] * 20)
fig.show()

# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_csteps,
    indices=["stimulus_index", "cell_index"],
    width=20,
    height=3,
    bin_size=0.01,
)
fig = CT_contrast.add_stimulus_to_plot(fig, [2] * 20)
fig.show()

# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_fff,
    indices=["stimulus_index", "cell_index"],
    width=20,
    height=3,
)
fig = CT_fff.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %%
spikes = spikes_chirp["times_relative"].to_numpy()
spikes_isi = np.diff(spikes)
isi_hist, bins = np.histogram(
    spikes_isi, bins=np.arange(0, 0.05, 0.00001), density=True
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(bins[:-1], isi_hist, width=np.diff(bins), align="edge")
ax.set_xlabel("ISI (s)")
ax.set_ylabel("Density")
fig.show()
# %%
spikes_diff = spikes_csteps.group_by(["trigger"]).agg(
    pl.col("times_triggered").diff(null_behavior="drop").alias("spikes_diff")
)
fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(10, 20), sharex=True, sharey=True)
for rep in range(10):
    isi_hist, bins = np.histogram(
        spikes_diff.filter(pl.col("trigger") == rep)["spikes_diff"].item().to_numpy(),
        bins=np.arange(0, 0.15, 0.001),
    )
    ax[rep].bar(bins[:-1], isi_hist, width=np.diff(bins), align="edge")
fig.show()
# %%
from scipy.signal import detrend, correlate

fig, ax = plt.subplots(figsize=(10, 5))
for rep in range(4):
    spikes_chirp_arr = spikes_csteps.filter(pl.col("repeat") == rep)[
        "times_triggered"
    ].to_numpy()
    # cumsum bining
    bin_size = 0.01
    bins = np.arange(0, 40, bin_size)
    psth, _ = np.histogram(spikes_chirp_arr, bins=bins)
    cum_psth = np.cumsum(psth)
    cum_psth = cum_psth / np.max(cum_psth)
    cum_detrend = detrend(cum_psth)
    # %%  plot
    ax.plot(bins[:-1], cum_psth, label=f"Repeat {rep}")
    ax.plot(bins[:-1], cum_detrend, label=f"Repeat {rep} detrended")
ax.legend()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cumulative spike count")
fig.show()
# %%
correlation = correlate(cum_psth, cum_psth, mode="same")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(correlation)
ax.set_xlabel("Lag")
ax.set_ylabel("Correlation")
fig.show()
