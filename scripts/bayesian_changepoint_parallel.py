from polarspike import (
    Overview,
    stimulus_spikes,
    bayesian,
)

import numpy as np
import polars as pl
import itertools

# %%
window_neg = -0.2
window_pos = 0.2
shuffle_window = 5
bin_width = 0.001
bins = np.arange(window_neg, window_pos + bin_width, bin_width)


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
unique_indices = (
    pl.scan_parquet(paths["results_df"])
    .select("cell_index", "recording")
    .unique()
    .collect()
)
recordings = unique_indices["recording"].to_list()
cell_indices = unique_indices["cell_index"].to_list()
recs_n_cells = list(zip(recordings, cell_indices))
# %%
for trigger_idx, trigger in enumerate(cum_triggers):
    spikes_fs = spikes_fs.with_columns(
        pl.when(
            (pl.col("times_triggered") > trigger + window_neg)
            & (pl.col("times_triggered") < trigger + window_pos)
        )
        .then(pl.col("times_triggered"))
        .alias(f"{trigger_idx}_fs")
    )


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
    median_starts=pl.col("median_starts") * bin_width
).collect()

# %%
updated_results = pl.concat([results_df, median_response], how="align").lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    updated_results = (
        updated_results.explode(f"breaks_{trigger_idx}")
        .group_by(["recording", "cell_index"])
        .agg(
            pl.col(f"breaks_{trigger_idx}"),
            pl.col(f"breaks_{trigger_idx}").filter(
                pl.col(f"breaks_{trigger_idx}")
                > pl.col("median_starts").min().alias(f"break_{trigger_idx}")
            ),
        )
    )

updated_results = updated_results.collect()
# %%
# breaks[trigger_idx] = np.where(
#             single_cell_results[f"breaks_{trigger_idx}"]
#             .list.to_array(1)
#             .to_numpy()
#             .flatten()
#             > median_response_start
#         )[0][0]
# %%
# spikes_fs = spikes_fs.with_columns(
#             pl.when(
#                 (pl.col("times_triggered") > trigger + window_neg)
#                 & (pl.col("times_triggered") < trigger + window_pos)
#             )
#             .then(pl.col("times_triggered"))
#             .alias(f"{trigger_idx}_fs")
#         )
#         breaks[trigger_idx] = np.where(
#             single_cell_results[f"breaks_{trigger_idx}"]
#             .list.to_array(1)
#             .to_numpy()
#             .flatten()
#             > median_response_start
#         )[0][0]
#
#         slopes_before[trigger_idx] = (
#             single_cell_results[f"breaks_{trigger_idx}"]
#             .list[breaks[trigger_idx] - 1]
#             .item()
#         )
#
#         slopes_after[trigger_idx] = (
#             single_cell_results[f"breaks_{trigger_idx}"]
#             .list[breaks[trigger_idx]]
#             .item()
#         )
