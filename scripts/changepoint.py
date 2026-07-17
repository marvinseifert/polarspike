from polarspike import Overview
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ruptures as rpt
from polarspike import histograms
from polarspike.Overview import Recording
from polarspike import colour_template

# %%
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
# %%
colours = ct.colours[::2]
colours = np.repeat(colours, 80, axis=0)
# %%
recording = Overview.Recording.load(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_11_02_2026/Phase_01/overview"
)
# recording = Overview.Recording.load(
#     r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_05_11_2025/Phase_00/overview"
# )
recording_1 = Overview.Recording.load(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_11_02_2026/Phase_01/overview"
)
# %%
spikes = recording.get_spikes_df(pandas=False, carry=["stimulus_name", "qi"])
# %%
df = pl.DataFrame(
    {
        "times_relative": spikes["times_relative"],
        "stimulus_name": spikes["stimulus_name"],
        "cell_index": spikes["cell_index"],
        "repeat": spikes["repeat"],
        "qi": spikes["qi"],
    }
).lazy()

# %%
df = df.sort(["stimulus_name", "cell_index", "times_relative"], descending=False)
df = df.with_columns(pl.col("times_relative").diff(null_behavior="ignore").alias("isi"))

df = df.filter((pl.col("isi") > 0) & (pl.col("isi") > 0.005))
df_isi = (
    df.group_by(["stimulus_name", "cell_index"])
    .agg(pl.min("isi").alias("isi_mean"))
    .collect()
)
# %%
psths, bins, cells = histograms.psth_by_index(
    spikes,
    index=["stimulus_name", "cell_index"],
    bin_size=0.05,
    window_end=2 * 16,
    return_idx=True,
)

# %%
max_psth_colour = np.argmax(psths, axis=1)
# %% Check how many bins are zero per cell

zero_bins_per_cell = np.sum(psths == 0, axis=1) / psths.shape[1]
zero_bins_df = pl.DataFrame(
    {
        "cell_index": cells[:, 1].astype(int),
        "zero_bins_proportion": zero_bins_per_cell,
        "max_psth_colour": max_psth_colour,
        "stimulus_name": cells[:, 0],
    }
)
combined_df = zero_bins_df.join(df_isi, on=["stimulus_name", "cell_index"])

# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(
    combined_df["isi_mean"], bins=np.arange(0, 1, 0.01), color="grey", edgecolor="black"
)
fig.show()
# %%
fig, ax = plt.subplots(
    figsize=(30, 5),
    ncols=np.unique(combined_df["stimulus_name"]).shape[0],
    sharex=True,
    sharey=True,
)
for i, stimulus in enumerate(np.unique(combined_df["stimulus_name"])):
    ax[i].scatter(
        combined_df.filter(pl.col("stimulus_name") == stimulus)["isi_mean"],
        combined_df.filter(pl.col("stimulus_name") == stimulus)["zero_bins_proportion"],
        c=colours[
            combined_df.filter(pl.col("stimulus_name") == stimulus)[
                "max_psth_colour"
            ].to_numpy()
        ],
        edgecolor="black",
        alpha=0.5,
    )
    # if i<7:
    #     ax[i].scatter(
    #         combined_df.filter(pl.col("stimulus_name") == "scf_2nd")["isi_mean"],
    #         combined_df.filter(pl.col("stimulus_name") == "scf_2nd")[
    #             "zero_bins_proportion"
    #         ],
    #         c="grey",
    #         edgecolor="black",
    #         alpha=0.1,
    #     )
ax[0].set_xlabel("95th percentile of ISI (s)")
ax[0].set_ylabel("Proportion of zero bins in PSTH")
# set axes to log
ax[0].set_xscale("log")
fig.show()
# %%
spikes_fff = recording.get_spikes_triggered(
    [{"stimulus_name": "fff"}], pandas=False, carry=["stimulus_name", "qi"]
)

psth_fff, bins, cells = histograms.psth_by_index(
    spikes_fff,
    index=["stimulus_index", "cell_index"],
    bin_size=0.1,
    window_end=2 * 16,
    return_idx=True,
)
# %%
cell = np.where(cells[:, 1] == 261)[0][0]
# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(bins[:-1], psth_fff[cell, :], c="black")
fig.show()

# %%
df_fff = pl.DataFrame(
    {
        "times_relative": spikes_fff["times_relative"],
        "times_triggered": spikes_fff["times_triggered"],
        "stimulus_name": spikes_fff["stimulus_name"],
        "cell_index": spikes_fff["cell_index"],
        "repeat": spikes_fff["repeat"],
        "qi": spikes_fff["qi"],
    }
).lazy()

# %%
df_fff = df_fff.sort(
    ["stimulus_name", "cell_index", "times_relative"], descending=False
)
df_fff = df_fff.with_columns(
    pl.col("times_relative").diff(null_behavior="ignore").alias("isi")
)

df_fff = df_fff.filter((pl.col("isi") > 0) & (pl.col("isi") > 0.005)).collect()

# %%
fig, ax = plt.subplots(
    figsize=(12, 7), nrows=4, gridspec_kw={"height_ratios": [2, 10, 10, 1]}, sharex=True
)
ax[0].plot(bins[:-1], psth_fff[cell, :], c="black")
ax[1].plot(
    df_fff.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
    df_fff.filter(pl.col("cell_index") == cells[cell, 1])["isi"],
    # c=df_fff.filter(pl.col("cell_index") == cells[cell, 1])["repeat"].to_numpy(),
    # cmap="Set1",
    # edgecolor="black",
    alpha=0.5,
    marker="_",
)
ax[2].scatter(
    df_fff.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
    df_fff.filter(pl.col("cell_index") == cells[cell, 1])["repeat"],
    # c=df_fff.filter(pl.col("cell_index") == cells[cell, 1])["repeat"].to_numpy(),
    # cmap="Set1",
    # edgecolor="black",
    marker="|",
)
# set y axis to log
ax[1].set_yscale("log")
fig = ct.add_stimulus_to_plot(fig, [2] * 16, names=False)
fig.show()

# %%
fig, ax = plt.subplots(
    figsize=(20, 15),
    nrows=6,
    gridspec_kw={"height_ratios": [10] * 5 + [1]},
    sharex=True,
)
for i in range(5):
    fff_sub = df_fff.filter(pl.col("repeat") == i)

    ax[i].scatter(
        fff_sub.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
        fff_sub.filter(pl.col("cell_index") == cells[cell, 1])["isi"],
        c=fff_sub.filter(pl.col("cell_index") == cells[cell, 1])["repeat"].to_numpy(),
        cmap="Set1",
        edgecolor="black",
        alpha=0.5,
        marker="_",
    )
    ax[i].set_yscale("log")
    ax[i].set_ylim(
        df_fff.filter(pl.col("cell_index") == cells[cell, 1])["isi"].min(),
        df_fff.filter(pl.col("cell_index") == cells[cell, 1])["isi"].max(),
    )

fig = ct.add_stimulus_to_plot(fig, [2] * 16, names=False)
fig.show()
# %%

fig, ax = plt.subplots(figsize=(20, 7))
for i in range(5):
    fff_sub = df_fff.filter(pl.col("repeat") == i)
    ax.plot(
        fff_sub.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
        fff_sub.filter(pl.col("cell_index") == cells[cell, 1])["isi"],
        c="black",
        marker=".",
    )
ax.set_yscale("log")
fig.show()

# %%
spikes_noise = recording.get_spikes_triggered(
    [{"stimulus_index": 2}],
    pandas=False,
    carry=["stimulus_name"],
)

spikes_noise = spikes_noise.filter(pl.col("times_triggered") < 1000)

psth_noise, bins, cells = histograms.psth_by_index(
    spikes_noise,
    index=["stimulus_index", "cell_index"],
    bin_size=0.05,
    window_end=1000,
    return_idx=True,
)
# %%
cell = 2
# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(bins[:-1], psth_noise[cell, :], c="black")
fig.show()

# %%
df_noise = pl.DataFrame(
    {
        "times_relative": spikes_noise["times_relative"],
        "times_triggered": spikes_noise["times_triggered"],
        "stimulus_name": spikes_noise["stimulus_name"],
        "cell_index": spikes_noise["cell_index"],
        "repeat": spikes_noise["repeat"],
    }
).lazy()

# %%
df_noise = df_noise.sort(
    ["stimulus_name", "cell_index", "times_relative"], descending=False
)
df_noise = df_noise.with_columns(
    pl.col("times_relative").diff(null_behavior="ignore").alias("isi")
)

df_noise = df_noise.filter((pl.col("isi") > 0) & (pl.col("isi") > 0.005)).collect()

# %%
fig, ax = plt.subplots(
    figsize=(12, 7), nrows=4, gridspec_kw={"height_ratios": [2, 10, 10, 1]}, sharex=True
)
ax[0].plot(bins[:-1], psth_noise[cell, :], c="black")
ax[1].plot(
    df_noise.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
    df_noise.filter(pl.col("cell_index") == cells[cell, 1])["isi"],
    # c=df_fff.filter(pl.col("cell_index") == cells[cell, 1])["repeat"].to_numpy(),
    # cmap="Set1",
    # edgecolor="black",
    alpha=0.5,
    marker="_",
)
ax[2].scatter(
    df_noise.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
    df_noise.filter(pl.col("cell_index") == cells[cell, 1])["repeat"],
    # c=df_fff.filter(pl.col("cell_index") == cells[cell, 1])["repeat"].to_numpy(),
    # cmap="Set1",
    # edgecolor="black",
    marker="|",
)
# set y axis to log
ax[1].set_yscale("log")
fig.show()
# %%
fig, ax = plt.subplots(figsize=(20, 7))
ax.hist(
    np.log10(df_noise.filter(pl.col("cell_index") == cells[cell, 1])["isi"]),
    color="grey",
    edgecolor="black",
)
fig.show()
# %%
noise_sub = df_noise.filter(pl.col("repeat") == 0)
# %%
fig, ax = plt.subplots(
    figsize=(12, 7), nrows=2, gridspec_kw={"height_ratios": [10, 1]}, sharex=True
)
ax[0].scatter(
    noise_sub.filter(pl.col("cell_index") == cells[cell, 1])["times_triggered"],
    noise_sub.filter(pl.col("cell_index") == cells[cell, 1])["isi"],
    c=noise_sub.filter(pl.col("cell_index") == cells[cell, 1])["repeat"].to_numpy(),
    cmap="Set1",
    edgecolor="black",
    alpha=0.5,
    marker="_",
)
# set y axis to log
ax[0].set_yscale("log")
fig.show()
# %%
algo = rpt.Dynp(model="l1", min_size=5).fit(psths[cell, :])
result = algo.predict(16 * 2)
# %%

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(bins[:-1], psths[cell, :], c="black")
ax.vlines(
    bins[result], ymin=0, ymax=np.max(psths[cell, :]), color="red", linestyle="--"
)
fig.show()
# %%
parquet_df = pl.scan_parquet(recording_1.parquet_path)
cell_df = parquet_df.filter(pl.col("cell_index") == cells[0, 1]).collect()
# %%
start_indices = np.random.randint(0, high=len(cell_df) - 10, size=500)
intervals = []
for i in range(500):
    intervals.append(
        np.mean(
            np.diff(cell_df["times"].slice(start_indices[i], start_indices[i] + 10))
        )
        / recording.sampling_freq
    )

# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(
    intervals,
    bins=np.arange(0, np.max(intervals), 0.05),
    color="grey",
    edgecolor="black",
)
ax.set_xlabel("Inter-spike interval (s)")
ax.set_ylabel("Count")
fig.show()
# %%
spike_indices = np.arange(len(spikes) - 10)
spike_intervals = []
for i in range(len(spikes.filter(pl.col("cell_index") == cells[0, 1])) - 10):
    spike_intervals.append(
        np.mean(
            np.diff(
                spikes.filter(pl.col("cell_index") == cells[0, 1])
                .sort("times_relative")["times_relative"]
                .slice(spike_indices[i], spike_indices[i] + 10)
            )
        )
    )
# %%
spike_intervals = np.array(spike_intervals)
# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(
    intervals,
    bins=np.arange(0, np.max(intervals), 0.05),
    color="grey",
    edgecolor="black",
    alpha=0.5,
    density=True,
)
ax.hist(
    spike_intervals,
    bins=np.arange(0, np.max(spike_intervals), 0.05),
    color="blue",
    edgecolor="black",
    alpha=0.5,
    density=True,
)
ax.set_xlabel("Inter-spike interval (s)")
ax.set_ylabel("Count")
fig.show()
