import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from polarspike import colour_template

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")

# %% import feature df

feature_df = pl.from_pandas(
    pd.read_pickle(r"A:\Marvin\fff_clustering\feature_df").reset_index()
)
contrast_df = pl.scan_parquet(
    r"D:\chicken_analysis\changepoint_df_csteps_test.parquet"
).collect()
fff_df = pl.scan_parquet(
    r"D:\chicken_analysis\changepoint_df_fff_test.parquet"
).collect()
# %%
dfs = []
for step in range(12):
    dfs.append(
        fff_df.group_by(["recording", "cell_index"]).agg(
            pl.col(f"first_spike_{step}").median().alias(f"first_spike_{step}"),
            pl.col(f"first_spike_posterior_{step}")
            .median()
            .alias(f"first_spike_posterior_{step}"),
        )
    )
fff_average = pl.concat(dfs, how="align")

# %%
dfs = []
for step in range(20):
    dfs.append(
        contrast_df.group_by(["recording", "cell_index"]).agg(
            pl.col(f"first_spike_{step}").median().alias(f"first_spike_{step}"),
            pl.col(f"first_spike_posterior_{step}")
            .median()
            .alias(f"first_spike_posterior_{step}"),
        )
    )
contrast_average = pl.concat(dfs, how="align")
# %%
contrast_and_fff = fff_average.join(
    contrast_average, on=["recording", "cell_index"], how="full"
)

# %% calculate median response time
medians = []
for step in range(12):
    median_temp = fff_average.select(f"first_spike_{step}").to_numpy().flatten()
    weights = fff_average.select(f"first_spike_posterior_{step}").to_numpy().flatten()
    nan_medians = np.isnan(median_temp)
    median_temp = median_temp[~nan_medians]
    weights = weights[~nan_medians]
    medians.append(np.average(median_temp, weights=weights))

# %%
medians_contrast = []
for step in range(20):
    median_temp = contrast_average.select(f"first_spike_{step}").to_numpy().flatten()
    weights = (
        contrast_average.select(f"first_spike_posterior_{step}").to_numpy().flatten()
    )
    nan_medians = np.isnan(median_temp)
    median_temp = median_temp[~nan_medians]
    weights = weights[~nan_medians]
    medians_contrast.append(np.average(median_temp, weights=weights))
# %%
wavelengths = ["610", "560", "535", "460", "413", "365"]
rate_columns = [f"{wavelengths[i]}_max" for i in range(6)]
rate_columns.extend([f"{wavelengths[i]}_off_max" for i in range(6)])

# %%
plot_df = pl.concat(
    [fff_average, feature_df.select(["recording", "cell_index"] + rate_columns)],
    how="align",
)

# %%
plot_df_contrast = pl.concat(
    [contrast_average, feature_df.select(["recording", "cell_index"] + rate_columns)],
    how="align",
)
# %%
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    ax[0, step].scatter(
        plot_df.select(rate_columns[step]).to_numpy().flatten(),
        plot_df.select(f"first_spike_{trigger}").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=plot_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    # plot dotted h line for median
    ax[0, step].axhline(medians[trigger], linestyle="--", color="black")
for step, trigger in zip(range(6), range(1, 12, 2)):
    ax[1, step].scatter(
        plot_df.select(rate_columns[step + 6]).to_numpy().flatten(),
        plot_df.select(f"first_spike_{trigger}").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=plot_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    ax[1, step].axhline(medians[trigger], linestyle="--", color="black")


for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("First spike time (s)", fontsize=15)
ax[1, 0].set_xlabel("Max firing rate (Hz)", fontsize=15)

# set y axis to log
# ax[0, 0].set_yscale("log")
ax[0, 0].set_xscale("log")
ax[0, 0].set_ylim([0.01, 0.1])
fig.show()
# %%
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    temp_df = pl.concat(
        [
            plot_df.select(
                [
                    "recording",
                    "cell_index",
                    rate_columns[step],
                    f"first_spike_posterior_{trigger}",
                ]
            ),
            plot_df_contrast.select(["recording", "cell_index", "first_spike_0"]),
        ],
        how="align",
    )
    temp_df = temp_df.drop_nulls()

    ax[0, step].scatter(
        temp_df.select(rate_columns[step]).to_numpy().flatten(),
        temp_df.select(f"first_spike_0").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=temp_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    # plot dotted h line for median
    ax[0, step].axhline(medians_contrast[trigger], linestyle="--", color="black")
for step, trigger in zip(range(6), range(1, 12, 2)):
    temp_df = pl.concat(
        [
            plot_df.select(
                [
                    "recording",
                    "cell_index",
                    rate_columns[step + 6],
                    f"first_spike_posterior_{trigger}",
                ]
            ),
            plot_df_contrast.select(["recording", "cell_index", "first_spike_1"]),
        ],
        how="align",
    )
    temp_df = temp_df.drop_nulls()
    ax[1, step].scatter(
        temp_df.select(rate_columns[step + 6]).to_numpy().flatten(),
        temp_df.select(f"first_spike_1").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=temp_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    ax[1, step].axhline(medians_contrast[trigger], linestyle="--", color="black")


for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("First spike time (s)", fontsize=15)
ax[1, 0].set_xlabel("Max firing rate (Hz)", fontsize=15)

# set y axis to log
# ax[0, 0].set_yscale("log")
ax[0, 0].set_xscale("log")
ax[0, 0].set_ylim([0.01, 0.1])
fig.show()
# %% plot all colour latencies against white
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    ax[0, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_0_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
    # plot dotted h line for median
for step, trigger in zip(range(6), range(1, 12, 2)):
    ax[1, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_1_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("Contrast first spike", fontsize=15)
ax[1, 0].set_xlabel("FFF first spike", fontsize=15)
ax[0, 0].set_ylim([0.01, 0.1])
ax[0, 0].set_xlim([0.01, 0.1])
fig.show()

# %% plot all colour latencies against white
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    ax[0, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_0_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
    # plot diagonal
    ax[0, step].plot([0.01, 0.1], [0.01, 0.1], linestyle="--", color="black", alpha=0.5)
    ax[0, step].scatter(
        medians[trigger],
        medians_contrast[0],
        color="red",
        s=100,
        marker="x",
    )
for step, trigger in zip(range(6), range(1, 12, 2)):
    ax[1, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_1_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
    ax[1, step].plot([0.01, 0.1], [0.01, 0.1], linestyle="--", color="black", alpha=0.5)
    ax[1, step].scatter(
        medians[trigger],
        medians_contrast[1],
        color="red",
        s=100,
        marker="x",
    )
for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("Contrast first spike", fontsize=15)
ax[1, 0].set_xlabel("FFF first spike", fontsize=15)
ax[0, 0].set_ylim([0.01, 0.1])
ax[0, 0].set_xlim([0.01, 0.1])
fig.show()

# %% Same for std
#
#
#
# %%
dfs = []
for step in range(12):
    dfs.append(
        fff_df.group_by(["recording", "cell_index"]).agg(
            pl.col(f"first_spike_{step}").std().alias(f"first_spike_{step}"),
            pl.col(f"first_spike_posterior_{step}")
            .median()
            .alias(f"first_spike_posterior_{step}"),
        )
    )
fff_average = pl.concat(dfs, how="align")

# %%
dfs = []
for step in range(20):
    dfs.append(
        contrast_df.group_by(["recording", "cell_index"]).agg(
            pl.col(f"first_spike_{step}").std().alias(f"first_spike_{step}"),
            pl.col(f"first_spike_posterior_{step}")
            .median()
            .alias(f"first_spike_posterior_{step}"),
        )
    )
contrast_average = pl.concat(dfs, how="align")
# %%
contrast_and_fff = fff_average.join(
    contrast_average, on=["recording", "cell_index"], how="full"
)

# %% calculate median response time
medians = []
for step in range(12):
    median_temp = fff_average.select(f"first_spike_{step}").to_numpy().flatten()
    weights = fff_average.select(f"first_spike_posterior_{step}").to_numpy().flatten()
    nan_medians = np.isnan(median_temp)
    median_temp = median_temp[~nan_medians]
    weights = weights[~nan_medians]
    medians.append(np.average(median_temp, weights=weights))

# %%
medians_contrast = []
for step in range(20):
    median_temp = contrast_average.select(f"first_spike_{step}").to_numpy().flatten()
    weights = (
        contrast_average.select(f"first_spike_posterior_{step}").to_numpy().flatten()
    )
    nan_medians = np.isnan(median_temp)
    median_temp = median_temp[~nan_medians]
    weights = weights[~nan_medians]
    medians_contrast.append(np.average(median_temp, weights=weights))
# %%
wavelengths = ["610", "560", "535", "460", "413", "365"]
rate_columns = [f"{wavelengths[i]}_max" for i in range(6)]
rate_columns.extend([f"{wavelengths[i]}_off_max" for i in range(6)])

# %%
plot_df = pl.concat(
    [fff_average, feature_df.select(["recording", "cell_index"] + rate_columns)],
    how="align",
)

# %%
plot_df_contrast = pl.concat(
    [contrast_average, feature_df.select(["recording", "cell_index"] + rate_columns)],
    how="align",
)
# %%
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    ax[0, step].scatter(
        plot_df.select(rate_columns[step]).to_numpy().flatten(),
        plot_df.select(f"first_spike_{trigger}").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=plot_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    # plot dotted h line for median
    ax[0, step].axhline(medians[trigger], linestyle="--", color="black")
for step, trigger in zip(range(6), range(1, 12, 2)):
    ax[1, step].scatter(
        plot_df.select(rate_columns[step + 6]).to_numpy().flatten(),
        plot_df.select(f"first_spike_{trigger}").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=plot_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    ax[1, step].axhline(medians[trigger], linestyle="--", color="black")


for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("Std first spike (s)", fontsize=15)
ax[1, 0].set_xlabel("Max firing rate (Hz)", fontsize=15)

# set y axis to log
# ax[0, 0].set_yscale("log")
ax[0, 0].set_xscale("log")
ax[0, 0].set_ylim([0, 0.1])
fig.show()
# %%
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    temp_df = pl.concat(
        [
            plot_df.select(
                [
                    "recording",
                    "cell_index",
                    rate_columns[step],
                    f"first_spike_posterior_{trigger}",
                ]
            ),
            plot_df_contrast.select(["recording", "cell_index", "first_spike_0"]),
        ],
        how="align",
    )
    temp_df = temp_df.drop_nulls()

    ax[0, step].scatter(
        temp_df.select(rate_columns[step]).to_numpy().flatten(),
        temp_df.select(f"first_spike_0").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=temp_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    # plot dotted h line for median
    ax[0, step].axhline(medians_contrast[trigger], linestyle="--", color="black")
for step, trigger in zip(range(6), range(1, 12, 2)):
    temp_df = pl.concat(
        [
            plot_df.select(
                [
                    "recording",
                    "cell_index",
                    rate_columns[step + 6],
                    f"first_spike_posterior_{trigger}",
                ]
            ),
            plot_df_contrast.select(["recording", "cell_index", "first_spike_1"]),
        ],
        how="align",
    )
    temp_df = temp_df.drop_nulls()
    ax[1, step].scatter(
        temp_df.select(rate_columns[step + 6]).to_numpy().flatten(),
        temp_df.select(f"first_spike_1").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=temp_df.select(f"first_spike_posterior_{trigger}").to_numpy().flatten() * 10,
    )
    ax[1, step].axhline(medians_contrast[trigger], linestyle="--", color="black")


for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("Std first spike (s)", fontsize=15)
ax[1, 0].set_xlabel("Max firing rate (Hz)", fontsize=15)

# set y axis to log
# ax[0, 0].set_yscale("log")
ax[0, 0].set_xscale("log")
ax[0, 0].set_ylim([0, 0.1])
fig.show()
# %% plot all colour latencies against white
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    ax[0, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_0_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
    # plot dotted h line for median
for step, trigger in zip(range(6), range(1, 12, 2)):
    ax[1, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_1_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("Contrast first spike", fontsize=15)
ax[1, 0].set_xlabel("FFF first spike", fontsize=15)
ax[0, 0].set_ylim([0, 0.1])
ax[0, 0].set_xlim([0, 0.1])
fig.show()

# %% plot all colour latencies against white
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(20, 10), sharey=True, sharex=True)
for step, trigger in zip(range(6), range(0, 12, 2)):
    ax[0, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_0_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
    # plot diagonal
    ax[0, step].plot([0.01, 0.1], [0.01, 0.1], linestyle="--", color="black", alpha=0.5)
    ax[0, step].scatter(
        medians[trigger],
        medians_contrast[0],
        color="red",
        s=100,
        marker="x",
    )
for step, trigger in zip(range(6), range(1, 12, 2)):
    ax[1, step].scatter(
        contrast_and_fff.select(f"first_spike_{trigger}").to_numpy().flatten(),
        contrast_and_fff.select(f"first_spike_1_right").to_numpy().flatten(),
        alpha=0.5,
        color="black",
        s=contrast_and_fff.select(f"first_spike_posterior_{trigger}")
        .to_numpy()
        .flatten()
        # * contrast_and_fff.select(f"first_spike_posterior_{trigger}_right")
        # .to_numpy()
        # .flatten()
        * 10,
    )
    ax[1, step].plot([0.01, 0.1], [0.01, 0.1], linestyle="--", color="black", alpha=0.5)
    ax[1, step].scatter(
        medians[trigger],
        medians_contrast[1],
        color="red",
        s=100,
        marker="x",
    )
for a_sub in ax:
    for idx, a in enumerate(a_sub):
        # remove top and right spines
        a.spines.right.set_visible(False)
        a.spines.top.set_visible(False)
        # change background color to light gray
        a.set_facecolor(CT.colours[::2][idx])
ax[1, 0].set_ylabel("Contrast std", fontsize=15)
ax[1, 0].set_xlabel("FFF std", fontsize=15)
ax[0, 0].set_ylim([0, 0.1])
ax[0, 0].set_xlim([0, 0.1])
fig.show()
