# Imports
from polarspike import (
    Overview,
    stimulus_dfs,
    stimulus_spikes,
    colour_template,
    spiketrains,
)
import numpy as np
import polars as pl
import pwlf
import multiprocessing as mp


# %% Functions
def piece_wise_linear(x, y, segment_range):
    best_score = np.inf
    best_n_segments = None
    if segment_range is None:
        segment_range = range(2, 10)

    # Loop over the range of possible segments
    for n_segments in range(segment_range):
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
    breaks = my_pwlf.fitfast(best_n_segments)
    breaks_pos = np.array([np.argmin(np.abs(x[:-1] - b)) for b in breaks])
    slopes = my_pwlf.calc_slopes()
    # the response start is the position of the first break > 0
    response_start = breaks_pos[breaks > 0][0]
    return np.array(
        [best_n_segments, breaks, breaks_pos, slopes, response_start], dtype=object
    )


def piece_wise_parallel(data, bins):
    nr_cells = data.shape[0]
    all_results = np.empty(nr_cells, dtype=object)
    for cell in range(nr_cells):
        nr_trigger = data[cell].shape[0] - 2
        for trigger in range(nr_trigger):
            all_results[cell] = piece_wise_linear(bins, data[cell][trigger + 2])
    return all_results


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


# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %% Load recording and create new triggers
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

recordings.dataframes["fff_stim"] = recordings.stimulus_df.query(
    "stimulus_name == 'fff'"
)
recordings.dataframes["fff_stim"] = recordings.dataframes["fff_stim"].query(
    "recording!='chicken_04_09_2024_p0' & stimulus_index != 18"
)
recordings.dataframes["fff_filtered"] = recordings.dataframes["fff_filtered"].query(
    "recording!='chicken_04_09_2024_p0' & stimulus_index != 18"
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
recordings.dataframes["fff_stim"]["stimulus_repeat_logic"] = 12
recordings.dataframes["fff_stim"]["stimulus_repeat_sublogic"] = 1
# %% get triggered spikes from a subset of cells
recordings.dataframes["sub_df"] = recordings.dataframes["fff_filtered"].sample(10)
spikes = recordings.get_spikes_df("sub_df", stimulus_df="fff_stim", pandas=False)
# %% Define the analysis parameters
window_neg = -0.2
window_pos = 0.2
shuffle_window = 5
bin_width = 0.001
max_trigger = spikes["trigger"].max()
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [1])
cum_triggers = np.cumsum(mean_trigger_times)
# %% Get potential first spikes
first_spike_potential = []
all_cumsums = []
all_steps = []
indices = []
bins = np.arange(window_neg, window_pos + bin_width, bin_width)
# Partition the dataframe based on cell_index


# %%
spikes_array_df = spiketrains.collect_as_arrays(
    spikes, ["recording", "cell_index"], "times_triggered", "spikes"
).lazy()
for trigger_idx, trigger in enumerate(cum_triggers):
    spikes_array_df = spikes_array_df.with_columns(
        pl.col("spikes")
        .list.gather(
            pl.col("spikes")
            .list.eval(
                (pl.element() > trigger + window_neg)
                & (pl.element() < trigger + window_pos)
            )
            .list.eval(pl.element().arg_true())
        )
        .list.eval(pl.element().sub(trigger))
        .alias(f"{trigger_idx}_fs")
    )


# %%
columns = [f"{trigger_idx}_fs" for trigger_idx, _ in enumerate(cum_triggers)]


def bin_func(x):
    bined_times = np.histogram(x, bins=bins)[0]
    cum_times = np.cumsum(bined_times) / bined_times.sum()
    return cum_times.tolist()


spikes_array_df = spikes_array_df.with_columns(
    [
        pl.col(col)
        .list.eval(
            pl.element().map_elements(bin_func, return_dtype=pl.List(pl.Float64))
        )
        .alias(f"{col}_binned")
        for col in columns
    ]
)
spikes_array_df = spikes_array_df.collect()

# %%
selector = ["recording", "cell_index"] + [f"{col}_binned" for col in columns]
row_split = spikes_array_df[selector].rows()

# %% parallel results

nr_cpus = mp.cpu_count()
row_split = np.array_split(row_split, nr_cpus)
pool = mp.Pool(nr_cpus)
all_results = pool.map(piece_wise_parallel, row_split)
pool.close()
pool.join()
all_results = np.concatenate(all_results)
# %%
