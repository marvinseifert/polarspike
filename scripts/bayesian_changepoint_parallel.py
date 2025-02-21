import pandas as pd
from polarspike import (
    Overview,
    stimulus_spikes,
    bayesian,
)

import numpy as np
import polars as pl
import itertools
import matplotlib.pyplot as plt

# %%
window_neg = -0.2
window_pos = 0.2
shuffle_window = 5
bin_width = 0.001
bins = np.arange(window_neg, window_pos + bin_width, bin_width)
prior_std = 0.01


# %% functions
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def get_slopes_and_breaks(
    spikes_fs: pl.DataFrame,
    single_cell_results: pl.DataFrame,
    cum_triggers: np.ndarray,
    median_response_start: float,
    window_neg: float,
    window_pos: float,
):
    nr_triggers = len(cum_triggers)
    slopes_before = np.zeros(nr_triggers)
    slopes_after = np.zeros(nr_triggers)
    breaks = np.zeros(nr_triggers)
    for trigger_idx, trigger in enumerate(cum_triggers):
        spikes_fs = spikes_fs.with_columns(
            pl.when(
                (pl.col("times_triggered") > trigger + window_neg)
                & (pl.col("times_triggered") < trigger + window_pos)
            )
            .then(pl.col("times_triggered"))
            .alias(f"{trigger_idx}_fs")
        )
        breaks[trigger_idx] = np.where(
            single_cell_results[f"breaks_{trigger_idx}"]
            .list.to_array(1)
            .to_numpy()
            .flatten()
            > median_response_start
        )[0][0]

        slopes_before[trigger_idx] = (
            single_cell_results[f"breaks_{trigger_idx}"]
            .list[breaks[trigger_idx] - 1]
            .item()
        )

        slopes_after[trigger_idx] = (
            single_cell_results[f"breaks_{trigger_idx}"]
            .list[breaks[trigger_idx]]
            .item()
        )
    return spikes_fs, slopes_before, slopes_after, breaks


def changepoint_detection(
    example_times, slope_before, slope_after, cum_trigger, median_response_start
):
    if not np.all(example_times > cum_trigger):
        if slope_before > 0:
            candidate_times, posterior = bayesian.changepoint_posterior(
                example_times,
                slope_before * 1000,
                slope_after * 1000,
                tau_guess=cum_trigger + median_response_start,
                prior_std=0.01,
            )
            changepoint_time = candidate_times[np.argmax(posterior)]
        else:
            (
                changepoint_time,
                posterior,
            ) = bayesian.detect_changepoint_single_rate(example_times, slope_after)
            candidate_times = example_times
        credibility_borders = bayesian.credible_interval(candidate_times, posterior)
    else:
        candidate_times = np.array([example_times[0]])
        changepoint_time, _ = bayesian.detect_changepoint_single_rate(
            example_times, slope_after
        )
        posterior = np.array([1])
        credibility_borders = np.array([example_times[0], example_times[0]])
    return changepoint_time, candidate_times, posterior, credibility_borders


def parallel_changepoint(
    recs_n_cells: np.ndarray, scanned_dfs: dict, cum_triggers: np.ndarray
):
    #
    # First scan the dataframes

    nr_of_repeats = (
        scanned_dfs["single_cells_spikes"]
        .select("repeat")
        .unique()
        .max()
        .collect()
        .item()
    )
    nr_triggers = len(cum_triggers)
    dfs = []
    for entry in recs_n_cells:
        recording, cell_index = entry
        # Get the spikes for the cell
        spikes_fs = (
            scanned_dfs["single_cells_spikes"]
            .filter(
                (pl.col("recording") == recording)
                & (pl.col("cell_index") == cell_index)
            )
            .collect()
        )
        # Get the rate results for the cell
        single_cell_results = (
            scanned_dfs["results_df"]
            .filter(
                (pl.col("recording") == recording)
                & (pl.col("cell_index") == cell_index)
            )
            .collect()
        )

        median_response_start = get_median_response_start(
            single_cell_results, cum_triggers, bins
        )
        spikes_fs, slopes_before, slopes_after, breaks = get_slopes_and_breaks(
            spikes_fs,
            single_cell_results,
            cum_triggers,
            median_response_start,
            window_neg,
            window_pos,
        )
        dfs_repeats = []
        for repeat, trigger_idx in itertools.product(
            range(nr_of_repeats), range(nr_triggers)
        ):
            # candidate_times, posterior = changepoint_posterior(
            example_times = (
                spikes_fs.filter([pl.col("repeat") == repeat])
                .sort(f"{trigger_idx}_fs")[f"{trigger_idx}_fs"]
                .to_numpy()
            )
            if (
                len(example_times) < 3
            ):  # Exclude cells with less than 3 spikes as no meaningful analysis can be done
                dfs_repeats.append(
                    pl.DataFrame(
                        {
                            "recording": [recording],
                            "cell_index": [cell_index],
                            "repeat": [repeat],
                            "trigger": [trigger_idx],
                            "change_point": [pl.Null],
                            "candidate_times": [[pl.Null]],
                            "posterior": [[pl.Null]],
                            "credibility_borders": [[pl.Null]],
                        }
                    )
                )
                continue
            (
                change_point,
                candidate_times,
                posterior,
                credibility_borders,
            ) = changepoint_detection(
                example_times,
                slopes_before[trigger_idx],
                slopes_after[trigger_idx],
                cum_triggers[trigger_idx],
                median_response_start,
            )
            dfs_repeats.append(
                pl.DataFrame(
                    {
                        "recording": [recording],
                        "cell_index": [cell_index],
                        "repeat": [repeat],
                        "trigger": [trigger_idx],
                        "change_point": [change_point],
                        "candidate_times": [candidate_times.tolist()],
                        "posterior": [posterior.tolist()],
                        "credibility_borders": [credibility_borders.tolist()],
                    }
                )
            )
        dfs.append(pl.concat(dfs_repeats, how="align"))
    return pl.concat(dfs)


# %% global variables
# Check global parallel variables below

results_df = pl.scan_parquet(r"D:\chicken_analysis\all_results_csteps.parquet")
single_cells_spikes = pl.scan_parquet(r"D:\chicken_analysis\spikes_csteps.parquet")
binned_spikes = pl.scan_parquet(r"D:\chicken_analysis\binned_spikes.parquet")


# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [12])
cum_triggers = np.hstack([np.array([0]), np.cumsum(mean_trigger_times)])
unique_indices = results_df.select("cell_index", "recording").unique().collect()
recordings = unique_indices["recording"].to_list()
cell_indices = unique_indices["cell_index"].to_list()
recs_n_cells = list(zip(recordings, cell_indices))
# %%
# for trigger_idx, trigger in enumerate(cum_triggers):
#     spikes_fs = spikes_fs.with_columns(
#         pl.when(
#             (pl.col("times_triggered") > trigger + window_neg)
#             & (pl.col("times_triggered") < trigger + window_pos)
#         )
#         .then(pl.col("times_triggered"))
#         .alias(f"{trigger_idx}_fs")
#     )


# %%
spikes_fs = single_cells_spikes.clone().lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    spikes_fs = spikes_fs.with_columns(
        pl.when(
            (pl.col("times_triggered") > trigger + window_neg)
            & (pl.col("times_triggered") < trigger + window_pos)
        )
        .then(pl.col("times_triggered"))
        .alias(f"{trigger_idx}_fs")
        .over(["recording", "cell_index"])
    )

spikes_fs = spikes_fs.collect()
# %%
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

# %%
updated_results = pl.concat([results_df, median_response], how="align").lazy()
dfs = []

for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(f"breaks_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            (
                pl.col(f"breaks_{trigger_idx}").abs() - pl.col("median_starts")
            ).arg_min()  # (pl.col(f"breaks_{trigger_idx}") > pl.col("median_starts"))
            # .arg_true()
            # .first()
            .alias(f"break_idx_{trigger_idx}"),
        )
    )

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
updated_results = pl.concat([updated_results, break_positions_df], how="align")
# %%
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(f"slopes_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"slopes_{trigger_idx}")
            .gather(pl.col(f"break_idx_{trigger_idx}").gather(0) - 1)
            .alias(f"slope_before{trigger_idx}")
        )
        .explode(f"slope_before{trigger_idx}")
    )
    dfs.append(
        updated_results.filter(
            pl.col(f"slopes_{trigger_idx}").list.len()
            > pl.col(f"break_idx_{trigger_idx}")
        )
        .explode(f"slopes_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"slopes_{trigger_idx}")
            .gather(pl.col(f"break_idx_{trigger_idx}").gather(0))
            .alias(f"slope_after{trigger_idx}")
        )
        .explode(f"slope_after{trigger_idx}")
    )
slopes_df = pl.concat(dfs, how="align")
# %%
updated_results = pl.concat([updated_results, slopes_df], how="align")

# %%
nr_cells = len(unique_indices)
df_add = unique_indices.clone().lazy()
fs_columns = []
for trigger_idx, trigger in enumerate(cum_triggers):
    df_add = df_add.with_columns(
        pl.Series(
            f"{trigger_idx}_fs",
            [[window_neg + trigger, window_pos + trigger]] * nr_cells,
        )
    )
    fs_columns.append(f"{trigger_idx}_fs")

df_add = df_add.explode(fs_columns).select(["recording", "cell_index"] + fs_columns)

df_add = df_add.collect()

# %%
spikes_fs_only = spikes_fs.select(["recording", "cell_index"] + fs_columns)
spikes_fs_only = pl.concat([spikes_fs_only, df_add])
# %%
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        spikes_fs_only.group_by(["recording", "cell_index"]).agg(
            pl.col(f"{trigger_idx}_fs").drop_nulls().sort() - trigger
        )
    )


spikes_fs_list = pl.concat(dfs, how="align")
# %%
updated_results = pl.concat([updated_results, spikes_fs_list], how="align")
# %%
updated_results = updated_results.lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = updated_results.with_columns(
        pl.col(f"{trigger_idx}_fs")
        .list.diff(null_behavior="drop")
        .alias(f"{trigger_idx}_diff")
    )
updated_results = updated_results.collect()
# %%
updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_diff"))
        .group_by(["recording", "cell_index"])
        .agg(pl.col(f"{trigger_idx}_diff").cum_sum().alias(f"{trigger_idx}_cum_sum"))
    )
updated_results = pl.concat([updated_results, pl.concat(dfs, how="align")], how="align")
updated_results = updated_results.collect()

# %%
updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_fs"))
        .group_by(["recording", "cell_index"])
        .agg(
            (
                pl.int_range(0, pl.len(), dtype=pl.UInt32)
                * pl.col(f"slope_before{trigger_idx}")
                - pl.col(f"slope_before{trigger_idx}")
                * pl.col(f"{trigger_idx}_fs").slice(
                    0, pl.col(f"{trigger_idx}_fs").len() - 1
                )
            ).alias(f"L1_log_{trigger_idx}")
        )
    )
    dfs.append(
        updated_results.explode(pl.col(f"{trigger_idx}_fs"))
        .group_by(["recording", "cell_index"])
        .agg(
            (
                (pl.len() - pl.int_range(0, pl.len(), dtype=pl.UInt32))
                * pl.col(f"slope_after{trigger_idx}")
                - pl.col(f"slope_after{trigger_idx}")
                * (
                    pl.col(f"{trigger_idx}_cum_sum").list.slice(-1).list.get(0)
                    - pl.col(f"{trigger_idx}_fs").slice(
                        0, pl.col(f"{trigger_idx}_fs").len() - 1
                    )
                )
            ).alias(f"L2_log_{trigger_idx}")
        )
    )
updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align")], how="align"
).collect()
# %%
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
        .group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"log_combined{trigger_idx}"),
        )
    )

updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align").collect()], how="align"
).collect()
# %%
updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"log_combined{trigger_idx}")
            .list.eval(pl.element() - pl.all().max())
            .list.eval(pl.element().exp())
            .alias(f"max_log_{trigger_idx}")
        )
        .explode(f"max_log_{trigger_idx}")
    )
updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align")], how="align"
).collect()
# %% Calculate prior
updated_results = updated_results.lazy()
dfs = []
for trigger_idx, trigger in enumerate(cum_triggers):
    dfs.append(
        updated_results.select(
            [
                "recording",
                "cell_index",
                f"{trigger_idx}_fs",
                f"max_log_{trigger_idx}",
                f"median_starts",
            ]
        )
        .explode(pl.col(f"{trigger_idx}_fs"), pl.col(f"max_log_{trigger_idx}"))
        .group_by(["recording", "cell_index"])
        .agg(
            (
                (
                    -0.5
                    * (
                        (pl.col(f"{trigger_idx}_fs") - pl.col("median_starts"))
                        / prior_std
                    )
                    ** 2
                ).exp()
                * pl.col(f"max_log_{trigger_idx}")
            ).alias(f"posterior_{trigger_idx}")
        )
    )
# %%
updated_results = pl.concat(
    [updated_results, pl.concat(dfs, how="align")], how="align"
).collect()
updated_results.write_parquet(r"D:\chicken_analysis\changepoint_df.parquet")
# %%
df_test = updated_results.sample(1)
fig, ax = plt.subplots()
ax.plot(df_test.select("0_fs").item(), df_test.select("posterior_0").item())
ax.scatter(df_test.select("0_fs").item(), np.ones(df_test.select("0_fs").item().len()))
fig.show()
