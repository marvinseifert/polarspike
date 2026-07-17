from polarspike import Overview
from polarspike import spiketrain_plots
from polarspike import quality_tests
import pandas as pd
import numpy as np
from polarspike import histograms
import polars as pl
import matplotlib.pyplot as plt
from polarspike import colour_template

# %%
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
# %%
fff_df = pd.read_pickle(r"/mnt/nas_a/Marvin/fff_clustering/fff_df.pkl")
fff_df.index.get_level_values(0).unique()
# %%
test = np.vstack(fff_df["psth"].to_numpy())
# %%
recordings = Overview.Recording_s(r"/media/mawa/fast_data/oil_analysis", "oil_no_oil")
# %%
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_18_07_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_11_09_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_05_09_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_30_08_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_09_09_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_17_07_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_04_09_2024/Phase_00/overview")
# recordings.add_from_saved(r"/mnt/nas_a/Marvin/chicken_19_07_2024/Phase_00/overview")
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
spikes = recordings.get_spikes_triggered(
    [{"stimulus_name": ["fff", "scf(0.0ND)"]}], carry=["qi"], pandas=False
)
# %%
unique_cell_rec = spikes["recording", "stimulus_index"].unique()

# %%
unique_for_selection = (
    unique_cell_rec.sample(fraction=1.0, shuffle=True)
    .group_by(["recording"])
    .agg(pl.first("stimulus_index"))
)
# %%
spikes_filtered = spikes.join(
    unique_for_selection,
    on=["recording", "stimulus_index"],
    how="semi",
)
# %%
psths, indices = histograms.psth_by_index(
    spikes_filtered,
    index=["recording", "cell_index", "repeat"],
    window_end=8 * 4 + 0.05,
)
# %%
dataframes_split = spikes.partition_by(["recording", "stimulus_index"])
qualities = []
for df in dataframes_split:
    qualities.append(quality_tests.spiketrain_qi(df, max_window=6 * 4, max_repeat=10))
# %%
spikes = spikes.filter(pl.col("qi") > 0.3)

# %%

# %%
unique_cell_rec = spikes["recording", "stimulus_index"].unique()
# %%
dataframes_split = spikes.partition_by(["recording", "stimulus_index"])

# %%
fig, ax = spiketrain_plots.whole_stimulus(
    dataframes_split[0], indices=["stimulus_index", "cell_index"]
)
fig.show()

# %%
qualities = []
for df in dataframes_split:
    qualities.append(quality_tests.spiketrain_qi(df, max_window=6 * 4, max_repeat=10))

# %%
fig, ax = spiketrain_plots.whole_stimulus(
    spikes, indices=["recording", "stimulus_index"]
)
fig.show()

# %%
cell_indices = pl.read_parquet(
    r"/media/mawa/fast_data/swa_embedding_all_fff/cell_indices"
)
labels = np.load(r"/media/mawa/fast_data/swa_embedding_all_fff/labels.npy")
cell_indices = cell_indices.with_columns(labels=labels)

# %%
fff_top_spikes = recordings.get_spikes_triggered(
    [{"stimulus_name": "fff_top"}], pandas=False
)

combined_df = fff_top_spikes.join(cell_indices, on=["recording", "cell_index"])
unique_labels = (
    combined_df.group_by(["recording", "cell_index"])
    .agg(pl.col("labels").first().alias("labels"))["labels"]
    .value_counts()
)

# %%
psths = np.load(r"/media/mawa/fast_data/swa_embedding_all_fff/psths.npy")
psths = np.mean(psths, axis=1)
#
time_bins = np.arange(0, 4 * 6, 0.05)
# %% plot individual clusters
mean_top = np.zeros((len(unique_labels) + 1, 480))
mean_bottom = np.zeros((len(unique_labels) + 1, 640))
fig, ax = plt.subplots(nrows=len(unique_labels), figsize=(10, 20), sharex=True)
for idx, label in enumerate(unique_labels.sort("labels")["labels"].to_numpy()):
    if label == -1:
        continue
    #
    psth_top = histograms.psth(
        combined_df.filter(pl.col("labels") == label), end=4 * 6 + 0.05
    )[0]
    psth_top = (psth_top - np.min(psth_top)) / (np.max(psth_top) - np.min(psth_top))
    ax[idx - 1].plot(time_bins, psth_top, color="red", label=f"C = {label}, top")
    mean_top[label] = psth_top  # psth = np.mean(
    #     psths[labels == label],
    #     axis=0,
    # )[40 * 2: -40 * 2]
    # psth = (psth - np.min(psth)) / (np.max(psth) - np.min(psth))
    # mean_bottom.append(psth)
    # ax[idx - 1].plot(time_bins, psth, color="black", label=f"C = {label}, bottom")
    # ax[idx - 1].spines[["right", "top"]].set_visible(False)
    # ax[idx - 1].legend(loc=1)
    psth = np.mean(
        psths[labels == label],
        axis=0,
    )
    psth = (psth - np.min(psth)) / (np.max(psth) - np.min(psth))
    mean_bottom[label] = psth
# fig = ct.add_stimulus_to_plot(fig, [2] * 16)
# fig.show()
# %%
np.save(
    r"/home/mawa/PycharmProjects/linear_nonlinear_model/top_psth.npy",
    np.vstack(mean_top),
)
np.save(
    r"/home/mawa/PycharmProjects/linear_nonlinear_model/bottom_psth.npy",
    np.vstack(mean_bottom),
)
