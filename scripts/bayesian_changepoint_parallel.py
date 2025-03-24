import pandas as pd
from polarspike import Overview, stimulus_spikes, bayesian, colour_template

import numpy as np
import polars as pl
import itertools
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from functools import reduce
from sklearn.linear_model import RANSACRegressor

# %%
window_neg = -0.2
window_neg_abs = np.abs(window_neg)
window_pos = 0.2
shuffle_window = 5
bin_width = 0.001
bins = np.arange(window_neg, window_pos + bin_width, bin_width)
bin_freq = 1 / bin_width
prior_std = 0.01

# %% global variables
# Check global parallel variables below

results_df = pl.scan_parquet(r"D:\chicken_analysis\all_results_fff_test.parquet")  # (
# r"D:\chicken_analysis\all_results_fff.parquet"
# )  # (r"D:\chicken_analysis\all_results_csteps.parquet")
single_cells_spikes = pl.scan_parquet(r"D:\chicken_analysis\spikes_fffs.parquet")  # (
#   r"D:\chicken_analysis\spikes_fffs.parquet"
# )  # (r"D:\chicken_analysis\spikes_csteps.parquet")

recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [1])
cum_triggers = np.hstack([np.array([0]), np.cumsum(mean_trigger_times)])
unique_indices = results_df.select("cell_index", "recording").unique().collect()
recordings = unique_indices["recording"].to_list()
cell_indices = unique_indices["cell_index"].to_list()
recs_n_cells = list(zip(recordings, cell_indices))
nr_repeats = single_cells_spikes.select("repeat").max().collect().item() + 1

all_trigger_times = []
trigger_indices = []
for rep in range(nr_repeats):
    all_trigger_times.append(cum_triggers[:-1] + rep * cum_triggers[-1])
    trigger_indices.append(np.arange(len(cum_triggers) - 1))
trigger_indices.append(np.array([mean_trigger_times.shape[0]]))
all_trigger_times.append(
    np.array([all_trigger_times[-1][-1] + all_trigger_times[0][1]])
)
trigger_indices = np.concatenate(trigger_indices)
all_trigger_times = np.concatenate(all_trigger_times)
# Here we add columns (called f"{trigger_idx}_fs") to the dataframe that save the spike times relative to the trigger
# and the window size indicated above.


spikes_fs = single_cells_spikes.clone().lazy()
# Get the unique trigger indices
unique_trigger_indices = np.unique(trigger_indices)

# Loop over each unique trigger index
for trig in unique_trigger_indices:
    # Get all trigger times for this trigger index
    triggers_for_index = [
        t for idx, t in zip(trigger_indices, all_trigger_times) if idx == trig
    ]

    # Build a piecewise expression that checks each trigger's window
    expr = None
    for t in triggers_for_index:
        condition = (pl.col("times_relative") > t + window_neg) & (
            pl.col("times_relative") < t + window_pos
        )
        # If the condition is met, subtract the corresponding trigger time t
        if expr is None:
            expr = pl.when(condition).then(pl.col("times_relative") - t)
        else:
            expr = expr.when(condition).then(pl.col("times_relative") - t)

    # Set a fallback value (e.g., None) if no condition is met
    expr = expr.otherwise(None)

    # Update (or add) the column for this trigger index.
    # Using .over(...) if you need to apply it per group
    spikes_fs = spikes_fs.with_columns(
        expr.alias(f"{trig}_fs").over(["recording", "cell_index", "repeat"])
    )

for trigger_idx, trigger in enumerate(cum_triggers):
    results_df = results_df.with_columns(
        pl.col(f"slopes_{trigger_idx}").list.eval(
            pl.element() / nr_repeats
        )  # slope is calculated for given bin width and over all repeats
    )

spikes_fs = spikes_fs.collect()

# Calculate the median response start time for each cell.
response_start_columns = [
    f"response_start_{trigger}" for trigger in range(len(cum_triggers))
]

median_response = results_df.select(
    ["recording", "cell_index"] + response_start_columns
)
median_response = median_response.with_columns(
    median_starts=pl.concat_list(response_start_columns).list.median()
)
median_response = median_response.with_columns(
    median_starts=pl.col("median_starts") * bin_width + window_neg
)
# median_response.collect()["median_starts"].std()
median_start = (
    median_response.select("median_starts").median().collect().item() + window_neg_abs
)
# Find the break points for each cell. The break is defined as the break point which is closest to the median response
# start time.

updated_results = pl.concat([results_df, median_response], how="align").lazy()
dfs = []

for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(f"breaks_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            (
                pl.col(f"breaks_{trigger_idx}").abs()
                - pl.col(
                    "median_starts"
                )  # This finds the closest break point to the median response start time
            )
            .arg_min()
            .alias(f"break_idx_{trigger_idx}"),
        )
    )
# Filter out the cells that do not have a break point
break_positions_df = pl.concat(dfs, how="align")
break_positions_df = break_positions_df.filter(
    ~pl.all_horizontal(
        pl.col(
            [
                f"break_idx_{trigger_idx}"
                for trigger_idx, trigger in enumerate(cum_triggers)
            ]
        ).is_null()
    )
)
updated_results = pl.concat(
    [updated_results, break_positions_df], how="align"
).collect()


# Find the firing rates (slopes) before and after the break point for each cell.
# Unfortunately, some slopes might be negative, which is impossible, as spike rate cannot be negative. It also cannot be
# zero, as we are working with the log of the spike rate. We will set these values to the smallest float value.
updated_results = updated_results.lazy()
dfs = []
# for trigger_idx, trigger in enumerate(cum_triggers):
#     temp_df = (
#         updated_results.explode(f"slopes_{trigger_idx}")
#         .group_by(["recording", "cell_index"])
#         .agg(
#             pl.col(f"slopes_{trigger_idx}")
#             .gather(pl.col(f"break_idx_{trigger_idx}").gather(0) - 1)
#             .alias(f"slope_before{trigger_idx}")
#         )
#         .explode(f"slope_before{trigger_idx}")
#     )
#     dfs.append(
#         temp_df.with_columns(
#             pl.when(pl.col(f"slope_before{trigger_idx}") <= 0)
#             .then(1.0)
#             .otherwise(pl.col(f"slope_before{trigger_idx}"))
#             .alias(f"slope_before{trigger_idx}")
#         )
#     )
#
#     temp_df = (
#         updated_results.filter(
#             pl.col(f"slopes_{trigger_idx}").list.len()
#             > pl.col(f"break_idx_{trigger_idx}")
#         )
#         .explode(f"slopes_{trigger_idx}")
#         .group_by(["recording", "cell_index"])
#         .agg(
#             pl.col(f"slopes_{trigger_idx}")
#             .gather(pl.col(f"break_idx_{trigger_idx}").gather(0))
#             .alias(f"slope_after{trigger_idx}")
#         )
#         .explode(f"slope_after{trigger_idx}")
#     )
#     dfs.append(
#         temp_df.with_columns(
#             pl.when(pl.col(f"slope_after{trigger_idx}") <= 0)
#             .then(float(np.finfo(np.float64).tiny))
#             .otherwise(pl.col(f"slope_after{trigger_idx}"))
#             .alias(f"slope_after{trigger_idx}")
#         )
#     )

for trigger_idx, trigger in enumerate(cum_triggers):
    temp_df = (
        updated_results.explode(f"slopes_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"slopes_{trigger_idx}")
            .slice(0, pl.col(f"break_idx_{trigger_idx}").get(0))
            .mean()
            .alias(f"slope_before{trigger_idx}")
        )
    )
    dfs.append(
        temp_df.with_columns(
            pl.when(pl.col(f"slope_before{trigger_idx}") <= 0)
            .then(1.0)
            .otherwise(pl.col(f"slope_before{trigger_idx}"))
            .alias(f"slope_before{trigger_idx}")
        )
    )

    temp_df = (
        updated_results.filter(
            pl.col(f"slopes_{trigger_idx}").list.len()
            > pl.col(f"break_idx_{trigger_idx}")
        )
        .explode(f"slopes_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"slopes_{trigger_idx}")
            .slice(pl.col(f"break_idx_{trigger_idx}").get(0), None)
            .mean()
            .alias(f"slope_after{trigger_idx}")
        )
    )
    dfs.append(
        temp_df.with_columns(
            pl.when(pl.col(f"slope_after{trigger_idx}") <= 0)
            .then(float(np.finfo(np.float64).tiny))
            .otherwise(pl.col(f"slope_after{trigger_idx}"))
            .alias(f"slope_after{trigger_idx}")
        )
    )


slopes_df = pl.concat(dfs, how="align")


updated_results = pl.concat([updated_results, slopes_df], how="align").collect()


# # We save the trigger times for each cell in a new dataframe. This is useful for the parallel processing later.
df_add = spikes_fs.select(["recording", "cell_index", "repeat"]).unique().lazy()
nr_unique = df_add.select(pl.len()).collect().item()
fs_columns = []
for trigger_idx, trigger in enumerate(cum_triggers):
    df_add = df_add.with_columns(
        pl.Series(
            f"{trigger_idx}_fs",
            [[window_neg]] * nr_unique,
        )
    )
    fs_columns.append(f"{trigger_idx}_fs")

df_add = df_add.explode(fs_columns).select(
    ["recording", "cell_index", "repeat"] + fs_columns
)

df_add = df_add.collect()


spikes_fs_only = spikes_fs.select(["recording", "cell_index", "repeat"] + fs_columns)
spikes_fs_only = pl.concat([spikes_fs_only, df_add]).lazy()

dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        spikes_fs_only.group_by(["recording", "cell_index", "repeat"]).agg(
            pl.col(f"{trigger_idx}_fs").drop_nulls().sort() + window_neg_abs
        )
    )


spikes_fs_list = pl.concat(dfs, how="align").collect()

updated_results = pl.concat([updated_results, spikes_fs_list], how="align")

updated_results = updated_results.lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = updated_results.with_columns(
        pl.col(f"{trigger_idx}_fs")
        .list.diff(null_behavior="drop")
        .alias(f"{trigger_idx}_diff")
    )
updated_results = updated_results.collect()

updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_diff"))
        .group_by(["recording", "cell_index", "repeat"])
        .agg(pl.col(f"{trigger_idx}_diff").cum_sum().alias(f"{trigger_idx}_cum_sum"))
    )
updated_results = pl.concat([updated_results, pl.concat(dfs, how="align")], how="align")
updated_results = updated_results.collect()


updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_cum_sum"))
        .group_by(["recording", "cell_index", "repeat"])
        .agg(
            (
                (
                    pl.int_range(1, pl.len(), dtype=pl.UInt32)
                    * pl.col(f"slope_before{trigger_idx}").slice(0, pl.len() - 1).log()
                )
                - (
                    pl.col(f"slope_before{trigger_idx}").slice(0, pl.len() - 1)
                    * pl.col(f"{trigger_idx}_cum_sum").slice(0, pl.len() - 1)
                )
            ).alias(f"L1_log_{trigger_idx}")
        )
    )
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_cum_sum"))
        .group_by(["recording", "cell_index", "repeat"])
        .agg(
            (
                (
                    (pl.len() - pl.int_range(1, pl.len(), dtype=pl.Int32))
                    * pl.col(f"slope_after{trigger_idx}").slice(0, pl.len() - 1).log()
                )
                - (
                    pl.col(f"slope_after{trigger_idx}").slice(0, pl.len() - 1)
                    * (
                        (
                            pl.repeat(
                                pl.col(f"{trigger_idx}_cum_sum").slice(-1), pl.len() - 1
                            )
                            - pl.col(f"{trigger_idx}_cum_sum").slice(0, pl.len() - 1)
                        )
                    )
                )
            ).alias(f"L2_log_{trigger_idx}")
        )
    )
updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align")], how="align"
).collect()

updated_results = updated_results.lazy()

for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = updated_results.with_columns(
        pl.col(f"L1_log_{trigger_idx}")
        .list.eval(pl.element() - pl.all().max())
        .alias(f"L1_log_{trigger_idx}")
    )
    updated_results = updated_results.with_columns(
        pl.col(f"L2_log_{trigger_idx}")
        .list.eval(pl.element() - pl.all().max())
        .alias(f"L2_log_{trigger_idx}")
    )
updated_results = updated_results.collect()


updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(f"L1_log_{trigger_idx}", f"L2_log_{trigger_idx}")
        .with_columns(
            (pl.col(f"L1_log_{trigger_idx}") + pl.col(f"L2_log_{trigger_idx}")).alias(
                f"log_combined{trigger_idx}"
            )
        )
        .group_by(["recording", "cell_index", "repeat"])
        .agg(
            pl.col(f"log_combined{trigger_idx}"),
        )
    )

updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align").collect()], how="align"
).collect()

updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    df_temp = (
        updated_results.group_by(["recording", "cell_index", "repeat"])
        .agg(
            pl.col(f"log_combined{trigger_idx}")
            .list.eval(pl.element() - pl.all().max())
            .list.eval(pl.element().exp())
            .alias(f"max_log_{trigger_idx}")
        )
        .explode(f"max_log_{trigger_idx}")
    )
    dfs.append(df_temp.with_columns(pl.col(f"max_log_{trigger_idx}").list.drop_nulls()))
updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align")], how="align"
).collect()


updated_results = updated_results.lazy()
dfs = []
cum_columns = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_cum_sum"))
        .group_by(["recording", "cell_index", "repeat"])
        .agg(
            pl.col(f"{trigger_idx}_cum_sum")
            .slice(0, pl.len() - 1)
            .alias(f"{trigger_idx}_cum_sum")
        )
    )
    cum_columns.append(f"{trigger_idx}_cum_sum")


updated_results = updated_results.drop(cum_columns).collect()
updated_results = pl.concat([updated_results, pl.concat(dfs, how="align")], how="align")


updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    df_temp_prior = (
        updated_results.explode(pl.col(f"{trigger_idx}_fs"))
        .group_by(["recording", "cell_index", "repeat"])
        .agg(
            (
                (
                    (
                        (
                            pl.col(f"{trigger_idx}_fs").slice(1, pl.len() - 2)
                            - median_start
                        )
                        / prior_std
                    )
                    ** 2
                )
                * -0.5
            )
            .exp()
            .alias(f"prior_{trigger_idx}")
        )
    )
    df_temp_prior = pl.concat(
        [
            df_temp_prior,
            updated_results.select(
                ["recording", "cell_index", "repeat", f"max_log_{trigger_idx}"]
            ),
        ],
        how="align",
    )
    dfs.append(
        df_temp_prior.explode(f"prior_{trigger_idx}", f"max_log_{trigger_idx}")
        .group_by(["recording", "cell_index", "repeat"])
        .agg(
            (pl.col(f"prior_{trigger_idx}") * pl.col(f"max_log_{trigger_idx}")).alias(
                f"posterior_{trigger_idx}"
            )
        )
        # .with_columns(
        #     pl.col(f"posterior_{trigger_idx}").list.eval(pl.element() / pl.all().sum())
        # )
    )
    dfs.append(
        df_temp_prior.select(
            ["recording", "cell_index", "repeat", pl.col(f"prior_{trigger_idx}")]
        )
    )

updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align")], how="align"
).collect()


# Check if there is any spike before the stimulus begin.
# In that case, we will only consider L2_log to determine the first spike.
updated_results = updated_results.lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = updated_results.with_columns(
        pl.when(
            (
                pl.col(f"{trigger_idx}_fs").list.get(1, null_on_oob=True)
                < window_neg_abs
            ).fill_null(False)
        )
        .then(True)
        .otherwise(False)
        .alias(
            f"spike_before_0_{trigger_idx}"
        )  # This new column will be True if there is a spike before the stimulus begin
    )
updated_results = updated_results.collect()


# Now we can get the predicted first spike and its probability.
updated_results = updated_results.lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = updated_results.with_columns(
        pl.when(pl.col(f"spike_before_0_{trigger_idx}"))
        .then(
            pl.col(f"posterior_{trigger_idx}")
            .list.arg_max()
            .alias(f"first_spike_idx_{trigger_idx}")
        )
        .otherwise(
            pl.col(f"L2_log_{trigger_idx}")
            .list.arg_max()
            .alias(f"first_spike_idx_{trigger_idx}")
        )
    )
    updated_results = updated_results.with_columns(
        (
            pl.col(f"{trigger_idx}_fs").list.get(
                pl.col(f"first_spike_idx_{trigger_idx}") + 1
            )
            - window_neg_abs  # Subtract the window_neg_abs to get the time centered at 0
        ).alias(f"first_spike_{trigger_idx}")
    )
    updated_results = updated_results.with_columns(
        pl.col(f"posterior_{trigger_idx}")
        .list.get(pl.col(f"first_spike_idx_{trigger_idx}"))
        .alias(f"first_spike_posterior_{trigger_idx}")
        # pl.when(pl.col(f"spike_before_0_{trigger_idx}"))
        # .then(
        #     pl.col(f"posterior_{trigger_idx}")
        #     .list.get(pl.col(f"first_spike_idx_{trigger_idx}"))
        #     .alias(f"first_spike_posterior_{trigger_idx}")
        # )
        # .otherwise(
        #     pl.col(f"L2_log_{trigger_idx}")
        #     .list.get(pl.col(f"first_spike_idx_{trigger_idx}"))
        #     .alias(f"first_spike_posterior_{trigger_idx}")
        # )
    )
updated_results = updated_results.collect()
# %%

updated_results.write_parquet(r"D:\chicken_analysis\changepoint_df_fff_test.parquet")

# %%
updated_results = updated_results.lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = updated_results.with_columns(
        nr_spikes=pl.col(f"{0}_fs").list.len()
    )
updated_results = updated_results.collect()
# %%
# df_test = updated_results.sample(1)
df_test = updated_results.filter(
    (pl.col("recording") == "chicken_05_09_2024_p0") & (pl.col("cell_index") == 8)
)

df_test = df_test.filter(pl.col("repeat") == 0)
trigger = 11
df_test = df_test.with_columns(
    pl.col(f"{trigger}_fs").list.eval(pl.element() + window_neg)
)


fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(15, 20))
ax[0].plot(
    df_test.select(f"{trigger}_fs").item()[1:-1],
    df_test.select(f"posterior_{trigger}").item(),
)
ax[0].scatter(
    df_test.select(f"{trigger}_fs").item(),
    np.ones(df_test.select(f"{trigger}_fs").item().len()),
)
ax[0].vlines(
    df_test.select(f"{trigger}_fs")
    .item()[1:-1]
    .to_numpy()[np.argmax(df_test.select(f"posterior_{trigger}").item().to_numpy())],
    0,
    1,
)
ax[1].plot(
    df_test.select(f"{trigger}_fs").item()[1:-1],
    df_test.select(f"L1_log_{trigger}").item(),
    label="L1",
)
ax[2].plot(
    df_test.select(f"{trigger}_fs").item()[1:-1],
    df_test.select(f"L2_log_{trigger}").item(),
    label="L2",
)
ax[3].plot(
    df_test.select(f"{trigger}_fs").item()[1:-1],
    df_test.select(f"log_combined{trigger}").item(),
    label="combined",
)

ax[4].plot(
    df_test.select(f"{trigger}_fs").item()[1:-1],
    df_test.select(f"prior_{trigger}").item(),
    label="prior",
)  #
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[4].set_xlim([window_neg, window_pos])
fig.show()
# %% plot over repeat
df_test = updated_results.filter(
    (pl.col("recording") == "chicken_05_09_2024_p0") & (pl.col("cell_index") == 244)
).collect()

# (pl.col("recording") == "chicken_19_07_2024_p0")
# & (pl.col("cell_index") == 335)
# ((pl.col("first_spike_6") < 0.05) & (pl.col("first_spike_6") > 0.07)))
# df_test = updated_results.filter(
#     (pl.col("recording") == df_test["recording"].item())
#     & (pl.col("cell_index") == df_test["cell_index"].item())
# )
df_test = df_test.with_columns(
    pl.col(f"{trigger}_fs").list.eval(pl.element() + window_neg)
)
cells_median = df_test.select(f"first_spike_{trigger}").median().item()
fig, ax = plt.subplots(nrows=10, sharex=True, figsize=(15, 20))
for rep in range(10):
    df_sub = df_test.filter(pl.col("repeat") == rep)
    if df_sub.select(f"{trigger}_fs").item().len() <= 2:
        continue
    ax[rep].plot(
        df_sub.select(f"{trigger}_fs").item()[1:-1],
        df_sub.select(f"posterior_{trigger}").item(),
        color="black",
    )
    ax[rep].scatter(
        df_sub.select(f"{trigger}_fs").item(),
        np.ones(df_sub.select(f"{trigger}_fs").item().len()) + 0.2,
        marker="|",
        color="black",
    )
    # ax[rep].vlines(
    #     df_sub.select(f"{trigger}_fs")
    #     .item()[1:-1]
    #     .to_numpy()[np.argmax(df_sub.select(f"posterior_{trigger}").item().to_numpy())],
    #     0,
    #     1,
    #     color="green",
    # )
    ax[rep].vlines(
        median_start - window_neg_abs,
        0,
        1,
        color="red",
        linestyle="--",
        linewidth=0.5,
    )
    ax[rep].set_title(f"Repeat {rep}")
    # remove top and right spines
    ax[rep].spines["top"].set_visible(False)
    ax[rep].spines["right"].set_visible(False)

ax[0].set_xlim([window_neg, window_pos])
ax[-1].set_xlabel("Time (s)")
ax[-1].set_ylabel("Posterior and spikes")
fig.show()
# %%
import plotly.express as px

updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.group_by(["recording", "cell_index"]).agg(
            pl.col(f"first_spike_{trigger_idx}")
            .median()
            .alias(f"av_first_spike_{trigger_idx}"),
            pl.col(f"first_spike_posterior_{trigger_idx}")
            .mean()
            .alias(f"av_posterior_{trigger_idx}"),
        )
    )

hist_df = pl.concat(dfs, how="align").collect()

# %%
# bin the spikes

# %%
fig, ax = plt.subplots(nrows=2, ncols=11, figsize=(30, 10), sharey=True)
col = 0
cs = ["grey", "black"]
times = np.arange(0, 0.4, 0.001)
prior = np.exp(-0.5 * ((times - median_start) / prior_std) ** 2)
times = times - window_neg_abs
nr_bins = 80
bins = np.linspace(window_neg, window_pos, nr_bins)
all_binned = np.zeros((len(cum_triggers), nr_bins - 1))

for trigger_idx, trigger in enumerate(cum_triggers):
    if np.mod(trigger_idx, 2) == 0:
        row = 0
    else:
        row = 1
    # spikes = updated_results.select(f"first_spike_{trigger_idx}").to_numpy().flatten()
    spikes = hist_df[f"av_first_spike_{trigger_idx}"].to_numpy()
    spikes = spikes[~np.isnan(spikes)]
    weights = hist_df[f"av_posterior_{trigger_idx}"].to_numpy()
    weights = weights[~np.isnan(weights)]

    values, bins = np.histogram(
        spikes,
        bins=bins,
        weights=weights,
        density=True,
    )
    all_binned[trigger_idx, :] = values
    ax[row, col].plot(bins[:-1], values, c=cs[row])
    ax[row, col].set_title(f"Trigger {trigger_idx}")
    ax[row, col].plot(times, prior * 100, c="red", linewidth=0.2)

    ax[row, col].vlines(
        0,  # np.nanmedian(hist_df[f"av_first_spike_{trigger_idx}"].to_numpy()),
        0,
        100,
        color=cs[row],
    )

    if np.mod(trigger_idx, 2) != 0:
        col = col + 1
fig.show()


# %%
peaks = bins[np.argmax(all_binned, axis=1)]
heights = np.max(all_binned, axis=1)
# %% scatter plot trigger vs first spike
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
real_trigger = np.arange(0, 20, 2)

first_spike = []
trigger_indices = []
weights = []
for trigger_idx, trigger in enumerate(cum_triggers[:-1][1::2]):
    first_spike.extend(hist_df[f"av_first_spike_{trigger_idx}"].to_numpy())
    weights.extend(hist_df[f"av_posterior_{trigger_idx}"].to_numpy())
    trigger_indices.extend(
        np.ones(hist_df[f"av_first_spike_{trigger_idx}"].to_numpy().shape)
        * real_trigger[trigger_idx]
    )

    ax.scatter(
        np.ones(hist_df[f"av_first_spike_{trigger_idx}"].to_numpy().shape)
        * trigger_idx,
        hist_df[f"av_first_spike_{trigger_idx}"].to_numpy(),
        c="black",
        alpha=0.5,
    )
fig.show()

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")

first_spike_on = []
first_spike_off = []
colours = []
for trigger_idx, trigger in enumerate(cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0:
        first_spike_on.extend(hist_df[f"av_first_spike_{trigger_idx}"].to_numpy())
        colours.extend(
            [CT.colours[trigger_idx]]
            * hist_df[f"av_first_spike_{trigger_idx}"].shape[0]
        )

    else:
        first_spike_off.extend(hist_df[f"av_first_spike_{trigger_idx}"].to_numpy())


# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax.scatter(first_spike_on, first_spike_off, c=colours, alpha=0.5)
ax.set_ylim([0, 0.2])
ax.set_xlim([0, 0.2])
ax.scatter(np.nanmean(first_spike_on), np.nanmean(first_spike_off), c="red")
ax.set_xlabel("First spike on")
ax.set_ylabel("First spike off")
fig.show()
# %% ransac regression

trigger_indices = np.asarray(trigger_indices, dtype=int)
first_spike = np.asarray(first_spike)
nan_idx = np.isnan(first_spike)
trigger_indices = trigger_indices[~nan_idx]
weights = np.asarray(weights)
weights = weights[~nan_idx]
first_spike = first_spike[~nan_idx]
minus_index = first_spike < 0
trigger_indices = trigger_indices[~minus_index]
first_spike = first_spike[~minus_index]
weights = weights[~minus_index]


model = RANSACRegressor()
model.fit(trigger_indices[:, np.newaxis], first_spike, weights)
# plot the line
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
for unique_trigger in np.unique(trigger_indices):
    ax.boxplot(
        first_spike[trigger_indices == unique_trigger], positions=[unique_trigger]
    )
ax.plot(trigger_indices, model.predict(trigger_indices[:, np.newaxis]), color="red")
fig.show()

# %%
# get slope
slope = model.estimator_.coef_[0]

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
for unique_trigger in np.unique(trigger_indices):
    ax.bar(
        unique_trigger,
        np.mean(first_spike[trigger_indices == unique_trigger]),
        yerr=np.std(first_spike[trigger_indices == unique_trigger]),
    )
fig.show()
# %%
median_starts = np.zeros((10, 2))
row = 0
for trigger_idx, trigger in enumerate(cum_triggers):
    if np.mod(trigger_idx, 2) == 0:
        idx = 0
    else:
        idx = 1
    median_starts[row, idx] = (
        updated_results.filter(
            (pl.col(f"first_spike_{trigger_idx}") > 0)
            & (pl.col(f"first_spike_{trigger_idx}") < window_pos)
        )
        .select(f"first_spike_{trigger_idx}")
        .median()
    ).item()
    if np.mod(trigger_idx, 2) != 0:
        row += 1
# %%
fastest_cell = updated_results.filter(
    (pl.col(f"first_spike_{1}") > 0) & (pl.col(f"first_spike_{1}") < window_pos)
)
# %%
prior = np.exp(
    -0.5 * ((np.linspace(window_neg, window_pos, 100) - median_start) / prior_std) ** 2
)
fig, ax = plt.subplots()
ax.plot(np.linspace(window_neg, window_pos, 100), prior)
fig.show()
