# Imports
from polarspike import (
    Overview,
    stimulus_dfs,
    stimulus_spikes,
    colour_template,
    spiketrains,
    spiketrain_plots,
)
import numpy as np
import polars as pl
import pwlf
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt


# %% Functions
def piece_wise_linear(
    x: np.ndarray, y: np.ndarray, segment_range: range = None, t: int = 0
):
    best_score = np.inf
    best_n_segments = 0
    if segment_range is None:
        segment_range = range(2, 10)

    # Loop over the range of possible segments
    for n_segments in segment_range:
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fitfast(n_segments)
        ssr_score = my_pwlf.ssr
        # The score is based on the lenghts of the segments, longer segments are rewarded, shorter segments are penalized
        # the relationship is logarithmic. This value is summed over all segments and multiplied by the sum of the squared residuals
        # This avoids overfitting with too many small segments
        score = np.sum(np.abs(np.log(np.diff(breaks))) * ssr_score)
        if score < best_score:
            best_score = score
            best_n_segments = n_segments

    # For the output we need to calculate break positions and slopes for each segment
    breaks = my_pwlf.fitfast(best_n_segments).astype(np.float64)
    breaks_pos = np.array([np.argmin(np.abs(x[:-1] - b)) for b in breaks]).astype(
        np.uint32
    )

    slopes = my_pwlf.calc_slopes().astype(np.float64)
    ssr_score = my_pwlf.ssr
    # the response start is the position of the first break > 0
    response_start = breaks_pos[np.argmax(slopes)]
    # build a polars.DataFrame with the results

    results_df = pl.DataFrame(
        {
            f"n_best_segments_{t}": [best_n_segments],
            f"breaks_{t}": [breaks.tolist()],
            f"breaks_pos_{t}": [breaks_pos.tolist()],
            f"slopes_{t}": [slopes.tolist()],
            f"response_start_{t}": [response_start],
            f"ssr_score_{t}": [ssr_score],
        }
    )

    return results_df


def piece_wise_parallel(data: list, bins: np.ndarray, triggers: list):
    nr_cells = len(data)
    all_results = np.empty(nr_cells, dtype=object)
    for cell in range(nr_cells):
        dfs = []
        for trigger in triggers:
            if data[cell].select(pl.col(f"{trigger}_fs_cumsum").sum()).item != 0:
                df_temp = piece_wise_linear(
                    bins,
                    data[cell]
                    .select(pl.col(f"{trigger}_fs_cumsum").sort_by("bins"))
                    .to_numpy()
                    .flatten(),
                    t=trigger,
                )
                df_temp = df_temp.with_columns(
                    recording=data[cell]["recording"].unique()
                )
                df_temp = df_temp.with_columns(
                    cell_index=data[cell]["cell_index"].unique()
                )
                dfs.append(df_temp)

        all_results[cell] = pl.concat(dfs, how="align")
    return pl.concat(all_results)


def generate_poisson_spike_train(segments):
    """
    Generate a spike train given segments with different constant firing rates.

    Parameters:
        segments (list of tuples): Each tuple should be of the form (start_time, end_time, rate)
                                   where rate is in spikes per second.

    Returns:
        np.ndarray: An array of spike times.
    """
    spike_times = []

    for start, end, rate in segments:
        t = start  # Start of the current segment
        while t < end:
            # Draw the next interspike interval from an exponential distribution
            dt = np.random.exponential(1.0 / rate)
            t += dt  # Next spike time
            if t < end:
                spike_times.append(t)
    return np.array(spike_times)


if __name__ == "__main__":
    # %%

    CT = colour_template.Colour_template()
    CT.pick_stimulus("FFF_6_MC")
    # # %% Load recording and create new triggers
    recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

    recordings.dataframes["fff_stim"] = recordings.stimulus_df.query(
        f"stimulus_name == 'fff_top'"
    )
    # recordings.dataframes["fff_stim"] = recordings.dataframes["csteps_stim"].query(
    #     "stimulus_index< 2"
    # )
    # recordings.dataframes["fff_filtered"] = recordings.dataframes["fff_filtered"].query(
    #     "stimulus_index<3"
    # )
    recordings.dataframes["fff_filtered"] = recordings.spikes_df.query(
        f"stimulus_name == 'fff_top'&qi >0.3"
    )
    old_triggers = np.stack(
        recordings.dataframes["fff_stim"]["trigger_fr_relative"].values, axis=0
    )
    new_triggers, new_intervals = stimulus_dfs.split_triggers(old_triggers, 1)

    new_triggers_stacked = np.empty(new_triggers.shape[0], dtype=object)
    new_intervals_stacked = np.empty_like(new_triggers_stacked)
    for idx, array in enumerate(new_triggers):
        new_triggers_stacked[idx] = array.flatten()
        new_intervals_stacked[idx] = new_intervals[idx, :].flatten()

    recordings.dataframes["fff_stim"]["trigger_fr_relative"] = new_triggers_stacked
    recordings.dataframes["fff_stim"]["trigger_int"] = new_triggers_stacked
    recordings.dataframes["fff_stim"]["stimulus_repeat_logic"] = 20
    recordings.dataframes["fff_stim"]["stimulus_repeat_sublogic"] = 1

    # %%
    # recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
    #
    # recordings.dataframes["csteps_filtered"] = recordings.spikes_df.query(
    #     "stimulus_name == 'csteps'&qi >0.3"
    # )
    #
    # recordings.dataframes["csteps_stim"] = recordings.stimulus_df.query(
    #     "stimulus_name == 'csteps'"
    # )
    # # %%
    # old_triggers = np.stack(
    #     recordings.dataframes["csteps_stim"]["trigger_fr_relative"].values, axis=0
    # )
    # new_triggers, new_intervals = stimulus_dfs.split_triggers(old_triggers, 1)
    #
    # new_triggers_stacked = np.empty(new_triggers.shape[0], dtype=object)
    # new_intervals_stacked = np.empty_like(new_triggers_stacked)
    # for idx, array in enumerate(new_triggers):
    #     new_triggers_stacked[idx] = array.flatten()
    #     new_intervals_stacked[idx] = new_intervals[idx, :].flatten()
    #
    # recordings.dataframes["csteps_stim"]["trigger_fr_relative"] = new_triggers_stacked
    # recordings.dataframes["csteps_stim"]["trigger_int"] = new_triggers_stacked
    # recordings.dataframes["csteps_stim"]["stimulus_repeat_logic"] = 20
    # recordings.dataframes["csteps_stim"]["stimulus_repeat_sublogic"] = 1

    # %% get triggered spikes from a subset of cells
    # recordings.dataframes["sub_df"] = recordings.dataframes["fff_filtered"]
    spikes = recordings.get_spikes_df(
        "fff_filtered", stimulus_df="fff_stim", pandas=False
    )
    # %% Define the analysis parameters
    window_neg = -0.2
    window_pos = 0.2
    shuffle_window = 5
    bin_width = 0.001
    max_trigger = spikes["trigger"].max()
    mean_trigger_times = stimulus_spikes.mean_trigger_times(
        recordings.stimulus_df, [49]
    )
    cum_triggers = np.hstack([np.array([0]), np.cumsum(mean_trigger_times)])
    # %% Get potential first spikes
    first_spike_potential = []
    all_cumsums = []
    all_steps = []
    indices = []
    bins = np.arange(window_neg, window_pos + bin_width, bin_width)
    # Partition the dataframe based on cell_index

    # %%
    spikes_fs = spikes.clone().lazy()
    for trigger_idx, trigger in enumerate(cum_triggers):
        spikes_fs = spikes_fs.with_columns(
            pl.when(
                (pl.col("times_triggered") > trigger + window_neg)
                & (pl.col("times_triggered") < trigger + window_pos)
            )
            .then(pl.col("times_triggered") - trigger)
            .alias(f"{trigger_idx}_fs")
        )

    dfs = []
    for trigger_idx, trigger in enumerate(cum_triggers):
        dfs.append(
            spikes_fs.group_by(["recording", "cell_index"]).agg(
                pl.col(f"{trigger_idx}_fs")
                .hist(bins=bins[:-1])
                .alias(f"{trigger_idx}_fs_binned")
            )
        )

    binned_spikes = pl.concat(dfs, how="align")
    binned_spikes = binned_spikes.with_columns(bins=bins.tolist())
    columns = [
        f"{trigger_idx}_fs_binned" for trigger_idx, _ in enumerate(cum_triggers)
    ] + ["bins"]
    binned_spikes = binned_spikes.explode(columns)
    # %%

    for trigger_idx, trigger in enumerate(cum_triggers):
        binned_spikes = binned_spikes.with_columns(
            pl.cum_sum(f"{trigger_idx}_fs_binned")
            .alias(f"{trigger_idx}_fs_cumsum")
            .over(["recording", "cell_index"])
        )
        # binned_spikes = binned_spikes.with_columns(
        #     pl.col(f"{trigger_idx}_fs_cumsum")
        #     / pl.sum_horizontal(pl.sum(columns[:-1]))
        #     .alias(f"{trigger_idx}_fs_cumsum")
        #     .over(["recording", "cell_index"])
        # )

    binned_spikes = binned_spikes.fill_nan(0)
    binned_spikes = binned_spikes.collect()

    # %%
    df_list = binned_spikes.partition_by(["recording", "cell_index"])

    # %% parallel results
    nr_cpus = 20  # np.min([mp.cpu_count(), len(df_list)])
    n = len(df_list) // nr_cpus
    par_func = partial(
        piece_wise_parallel,
        bins=bins,
        triggers=[trigger_idx for trigger_idx, _ in enumerate(cum_triggers)],
    )

    def chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    chunks = list(chunk_list(df_list, n))
    pool = mp.Pool(nr_cpus)
    all_results = pool.map(par_func, chunks)
    pool.close()
    pool.join()

    # np.save(r"D:\chicken_analysis\test.npy", all_results)
    # # %%
    all_results_df = pl.concat(all_results)

    # %%
    all_results_df.write_parquet(rf"D:\chicken_analysis\all_results_fff_top.parquet")
    binned_spikes.write_parquet(rf"D:\chicken_analysis\binned_spikes_fff_top.parquet")
#
#
# # %%
# def generate_line_data(starts: list, slopes: list, lengths: np.ndarray):
#     x_list = [0]
#     y_list = [0]
#     for start_idx in range(len(starts) - 1):
#         x = np.linspace(starts[start_idx], starts[start_idx] + lengths[start_idx], 10)
#         y = slopes[start_idx] * (x - starts[start_idx]) + y_list[-1]
#         x_list.extend(x)
#         y_list.extend(y)
#         # print(x, y)
#     return np.asarray(x_list[1:]).T, np.asarray(y_list[1:]).T
#
#
# def plot_cell(result_df: pl.DataFrame, cum_bin_df: pl.DataFrame, triggers: list = None):
#     if triggers is None:
#         triggers = [0]
#     unique_entries = result_df.unique(["recording", "cell_index"])[
#         "recording", "cell_index"
#     ]
#     fig, ax = plt.subplots(
#         nrows=len(result_df),
#         ncols=len(triggers),
#         figsize=(3 * len(triggers), 3 * len(unique_entries)),
#         sharex=True,
#         sharey=True,
#     )
#     ax = np.atleast_2d(ax)
#
#     for e_idx, entry in enumerate(unique_entries.to_numpy()):
#         for trigger_idx, trigger in enumerate(triggers):
#             cum_counts = (
#                 cum_bin_df.filter(
#                     (pl.col("recording") == entry[0])
#                     & (pl.col("cell_index") == entry[1])
#                 )[f"{trigger}_fs_cumsum"]
#                 .to_numpy()
#                 .flatten()
#             )
#             ax[e_idx, trigger_idx].plot(bins, cum_counts, c="black")
#             starts = result_df.filter(
#                 (pl.col("recording") == entry[0]) & (pl.col("cell_index") == entry[1])
#             )[f"breaks_{trigger}"].item()
#             slopes = result_df.filter(
#                 (pl.col("recording") == entry[0]) & (pl.col("cell_index") == entry[1])
#             )[f"slopes_{trigger}"].item()
#             response_start = result_df.filter(
#                 (pl.col("recording") == entry[0]) & (pl.col("cell_index") == entry[1])
#             )[f"response_start_{trigger}"].item()
#
#             if starts is None or len(starts) == 0:
#                 continue
#             lengths = np.diff(starts)
#
#             x, y = generate_line_data(starts, slopes, lengths)
#
#             ax[e_idx, trigger_idx].plot(x, y, c="red")
#             ax[e_idx, trigger_idx].set_title(f"Recording {entry[0]}, Cell {entry[1]}")
#             (
#                 ax[e_idx, trigger_idx].scatter(
#                     bins[response_start], cum_counts[response_start], c="blue"
#                 )
#             )
#             ax[e_idx, trigger_idx].annotate(
#                 f"Response start {bins[response_start]:.3f}",
#                 (0, 0.4),
#             )
#
#     ax[0, 0].set_ylabel("Firing rate")
#     ax[-1, 0].set_xlabel("Time (s)")
#     ax[0, 0].set_xlim([window_neg, window_pos])
#     ax[0, 0].set_ylim([0, 0.5])
#
#     return fig, ax
#
#
# # %%
# sample_df = all_results_df.sample(10)
# rec_cell = sample_df.unique(["recording", "cell_index"])["recording", "cell_index"]
# fig, ax = plot_cell(
#     sample_df,
#     binned_spikes.filter(
#         (pl.col("recording").is_in(rec_cell["recording"]))
#         & (pl.col("cell_index").is_in(rec_cell["cell_index"]))
#     ),
#     triggers=np.arange(1, 20, 2).tolist(),
# )
# fig.show()
# # %%
# fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(15, 10))
#
# for trigger_idx, trigger in enumerate(np.arange(0, 12, 2).tolist()):
#     values = bins[all_results_df[f"response_start_{int(trigger)}"].to_numpy().flatten()]
#     values = values[values > 0]
#     ax[trigger_idx].hist(
#         values,
#         bins=np.arange(window_neg, window_pos, 0.001),
#         label=f"Trigger {trigger}",
#         color=CT.colours[0::2][trigger_idx],
#         density=True,
#     )
#     ax[trigger_idx].axvline(
#         np.median(values),
#         color="black",
#     )
#     ax[trigger_idx].axvline(
#         np.quantile(values, 0.25),
#         color="black",
#         linestyle="--",
#     )
#     ax[trigger_idx].axvline(
#         np.quantile(values, 0.75),
#         color="black",
#         linestyle="--",
#     )
#
#
# ax[-1].set_xlabel("Response start (s)")
# ax[-1].set_ylabel("Count")
# fig.show()
#
#
# # %% plot spiketrain
# cell_spikes = spikes.filter(
#     (pl.col("recording") == "chicken_11_09_2024_p0") & (pl.col("cell_index") == 358)
# )
# fig, ax = spiketrain_plots.whole_stimulus(cell_spikes, indices=["cell_index", "repeat"])
# fig = CT.add_stimulus_to_plot(fig, [2] * 12)
# fig.show()
#
#
# # %% bayesian inference for actual first spike detection
# fig, ax = plot_cell(
#     sample_df[-1],
#     binned_spikes.filter(
#         (pl.col("recording").is_in(sample_df[-1]["recording"]))
#         & (pl.col("cell_index").is_in(sample_df[-1]["cell_index"]))
#     ),
#     triggers=np.arange(1, 20, 2).tolist(),
# )
# fig.show()
# # %%
# single_cell = binned_spikes.filter(
#     (pl.col("recording").is_in(sample_df[-1]["recording"]))
#     & (pl.col("cell_index").is_in(sample_df[-1]["cell_index"]))
# )
# recordings.dataframes["csteps_single"] = recordings.dataframes["csteps_filtered"].query(
#     f"recording == '{single_cell['recording'][0]}' & cell_index == {single_cell['cell_index'][0]}"
# )
# single_cell_spikes = recordings.get_spikes_df(
#     "csteps_single", stimulus_df="csteps_stim", pandas=False
# )
