from sympy.printing.pretty.pretty_symbology import line_width

from polarspike import (
    Overview,
    spiketrain_plots,
    colour_template,
    histograms,
)
import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import polars as pl
from matplotlib.patches import Rectangle
from polarspike import quality_tests
from polarspike.Overview import Recording

# %% -------------------------------------------------------------------------

CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_8_MC")
# %%

recordings = Overview.Recording_s(
    r"/mnt/nas_b/Marvin/combined_analysis",
    "fff_analysis",
)
recordings.add_from_saved(r"/mnt/nas_b/Marvin/chicken_11_02_2026/Phase_01/overview")
recordings.add_from_saved(r"/mnt/nas_b/Marvin/chicken_12_02_2026/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_b/Marvin/chicken_13_11_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_b/Marvin/chicken_17_11_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_b/Marvin/chicken_18_11_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_01_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_05_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_07_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_13_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_14_05_2025/Phase_00/overview")
recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_09_05_2024/Phase_00/overview")

# %%
spikes_df = recordings.get_spikes_triggered(
    [{"stimulus_name": ["fff", "scf(0.0ND)"]}],
    pandas=False,
    carry=["stimulus_name", "qi"],
)
# %%
unique_cell_rec = spikes_df["recording", "stimulus_index"].unique()

# %%
unique_for_selection = (
    unique_cell_rec.sample(fraction=1.0, shuffle=True)
    .group_by(["recording"])
    .agg(pl.first("stimulus_index"))
)
# %%
spikes_filtered = spikes_df.join(
    unique_for_selection,
    on=["recording", "stimulus_index"],
    how="semi",
)
# %%
dataframes_split = spikes_filtered.partition_by(["recording", "stimulus_index"])
qualities = []
for df in dataframes_split:
    temp_df = quality_tests.spiketrain_qi(df, max_window=8 * 4, max_repeat=5)
    temp_df["stimulus_index"] = df["stimulus_index"][0]
    qualities.append(temp_df)

# %%

new_spikes = spikes_filtered.update(
    pl.concat([pl.from_pandas(df.reset_index()) for df in qualities]),
    on=["recording", "cell_index", "stimulus_index"],
)
# %%
spikes_df = new_spikes.filter(pl.col("qi") >= 0.3)
psth, bins, cell_index = histograms.psth_by_index(
    spikes_df,
    index=["recording", "cell_index", "repeat"],
    return_idx=True,
    window_end=2 * 16 + 0.05,
)

# %%
left_df = pl.DataFrame(
    {
        "psths": psth.tolist(),
        "recording": cell_index[:, 0],
        "cell_index": cell_index[:, 1].astype(int),
        "repeat": cell_index[:, 2].astype(int),
    }
)

# %%
right_df = spikes_df.group_by(["recording", "cell_index"]).agg(pl.col("qi").mean())

# %%
combined_df = left_df.join(right_df, on=["recording", "cell_index"])

# %%
psths = np.vstack(combined_df["psths"].to_numpy())
qi = combined_df["qi"].to_numpy()

# %%
np.save(r"/home/mawa/Downloads/green_step_chicken.npy", psths)
np.save(r"/home/mawa/Downloads/quality_chicken.npy", qi)
# %%
np.savetxt(
    r"/home/mawa/Downloads/green_step_chicken.csv",
    psths.astype(int),
    delimiter=",",
)
np.savetxt(r"/home/mawa/Downloads/quality_chicken.csv", qi, delimiter=",")
# %%
step_positions = np.arange(0, 32, 2)
time_per_step = 2
mean_psth = np.mean(psth, axis=0)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(bins[:-1], mean_psth, c="black", linewidth=2)
ax.set_ylabel("Spike rate (avg)")
ax.set_xlabel("Time (s)")
for step in range(16):
    ax.add_patch(
        Rectangle(
            (step_positions[step], 0),
            time_per_step,
            np.max(mean_psth),
            color=CT.colours[step],
            alpha=0.2,
        )
    )
ax.hlines(np.median(mean_psth), 0, 32, color="red", linestyles="--")
fig.show()
fig.savefig(r"/home/mawa/Documents/labmeeting_may27/average_response_median.png")
# %%
unique_pairs = left_df.select(["recording", "cell_index"]).unique()
all_repeats = pl.DataFrame({"repeat": pl.Series(range(5), dtype=pl.UInt8)})

# Cross join to get all expected combinations
expected = unique_pairs.join(all_repeats, how="cross")

# Left join back to the actual data, filling missing with zeros
zero_array = [0] * 640

summed_df = (
    expected.join(left_df, on=["recording", "cell_index", "repeat"], how="left")
    .sort(["recording", "cell_index", "repeat"])
    .with_columns(
        pl.col("psths").fill_null(pl.lit(zero_array, dtype=pl.Array(pl.Float64, 640)))
    )
)
# %%
arrays = [
    np.vstack(df["psths"].to_numpy())
    for df in summed_df.partition_by(["recording", "cell_index"], maintain_order=True)
]
# %%
rec_cells = summed_df.group_by(["recording", "cell_index"], maintain_order=True).agg()
rec_cells.write_parquet(r"/media/mawa/fast_data/swa_embedding_all_fff/cell_indices")
# %%
arrays = np.array(arrays)
# %%
np.save(r"/media/mawa/fast_data/swa_embedding_all_fff/psths.npy", arrays)
# %%


fig, ax = spiketrain_plots.whole_stimulus(
    spikes_df,
    indices=["stimulus_index", "cell_index"],
    cmap=["Greys", "Reds"],
    single_psth=False,
)
fig = CT.add_stimulus_to_plot(fig, [2] * 16)

fig.show()

# %%

fig, ax = spiketrain_plots.whole_stimulus(
    spikes_top,
    indices=["stimulus_index", "repeat"],
    cmap=["Greys"],
    single_psth=False,
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)

fig.show()

# %%
psth, bins, cell_index = histograms.psth_by_index(
    spikes, index=["cell_index"], return_idx=True, window_end=24
)

psth_top, bins, cell_index_top = histograms.psth_by_index(
    spikes_top, index=["cell_index"], return_idx=True, window_end=24
)

# %%
all_cells = np.zeros((recording.nr_cells, (len(bins) - 1) * 2))
all_cells[cell_index.astype(int).flatten(), : len(bins) - 1] = psth
all_cells[cell_index_top.astype(int).flatten(), len(bins) - 1:] = psth_top

# %% scaling
scaler = StandardScaler()
all_cells_scaled = scaler.fit_transform(all_cells)

# %%
clustering = AffinityPropagation(random_state=5, max_iter=1000).fit(all_cells)
labels = clustering.labels_
print(np.max(labels) + 1)

# %%
clustering = AgglomerativeClustering(n_clusters=20, linkage="complete")
labels = clustering.fit_predict(all_cells_scaled)
# %% Plot cluster means
fig, axs = plt.subplots(nrows=21, ncols=1, figsize=(10, 10), sharex=True)
for cluster in range(20):
    mean = np.mean(all_cells[labels == cluster], axis=0)
    axs[cluster].plot(bins[:-1], mean[: len(bins) - 1], label=f"Cluster {cluster}")
    axs[cluster].plot(bins[:-1], mean[len(bins) - 1:], label=f"Cluster {cluster}")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()
