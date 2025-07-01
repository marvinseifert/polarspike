from polarspike import (
    Overview,
    stimulus_dfs,
    stimulus_spikes,
    colour_template,
    bayesian,
)
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# %% global variables
window_neg = -0.2
window_pos = 0.2
shuffle_window = 5
bin_width = 0.001
bins = np.arange(window_neg, window_pos + bin_width, bin_width)

# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

recordings.dataframes["csteps_filtered"] = recordings.spikes_df.query(
    "stimulus_name == 'csteps'&qi >0.3"
)

recordings.dataframes["csteps_stim"] = recordings.stimulus_df.query(
    "stimulus_name == 'csteps'"
)

old_triggers = np.stack(
    recordings.dataframes["csteps_stim"]["trigger_fr_relative"].values, axis=0
)
new_triggers, new_intervals = stimulus_dfs.split_triggers(old_triggers, 1)

new_triggers_stacked = np.empty(new_triggers.shape[0], dtype=object)
new_intervals_stacked = np.empty_like(new_triggers_stacked)
for idx, array in enumerate(new_triggers):
    new_triggers_stacked[idx] = array.flatten()
    new_intervals_stacked[idx] = new_intervals[idx, :].flatten()

recordings.dataframes["csteps_stim"]["trigger_fr_relative"] = new_triggers_stacked
recordings.dataframes["csteps_stim"]["trigger_int"] = new_triggers_stacked
recordings.dataframes["csteps_stim"]["stimulus_repeat_logic"] = 20
recordings.dataframes["csteps_stim"]["stimulus_repeat_sublogic"] = 1

# %%
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [12])
cum_triggers = np.hstack([np.array([0]), np.cumsum(mean_trigger_times)])


# %%
def generate_line_data(starts: list, slopes: list, lengths: np.ndarray):
    x_list = [0]
    y_list = [0]
    for start_idx in range(len(starts) - 1):
        x = np.linspace(starts[start_idx], starts[start_idx] + lengths[start_idx], 10)
        y = slopes[start_idx] * (x - starts[start_idx]) + y_list[-1]
        x_list.extend(x)
        y_list.extend(y)
        # print(x, y)
    return np.asarray(x_list[1:]).T, np.asarray(y_list[1:]).T


def plot_cell(result_df: pl.DataFrame, cum_bin_df: pl.DataFrame, triggers: list = None):
    if triggers is None:
        triggers = [0]
    unique_entries = result_df.unique(["recording", "cell_index"])[
        "recording", "cell_index"
    ]
    fig, ax = plt.subplots(
        nrows=len(result_df),
        ncols=len(triggers),
        figsize=(3 * len(triggers), 3 * len(unique_entries)),
        sharex=True,
        sharey=True,
    )
    ax = np.atleast_2d(ax)

    for e_idx, entry in enumerate(unique_entries.to_numpy()):
        for trigger_idx, trigger in enumerate(triggers):
            cum_counts = (
                cum_bin_df.filter(
                    (pl.col("recording") == entry[0])
                    & (pl.col("cell_index") == entry[1])
                )[f"{trigger}_fs_cumsum"]
                .to_numpy()
                .flatten()
            )
            ax[e_idx, trigger_idx].plot(bins, cum_counts, c="black")
            starts = result_df.filter(
                (pl.col("recording") == entry[0]) & (pl.col("cell_index") == entry[1])
            )[f"breaks_{trigger}"].item()
            slopes = result_df.filter(
                (pl.col("recording") == entry[0]) & (pl.col("cell_index") == entry[1])
            )[f"slopes_{trigger}"].item()
            response_start = result_df.filter(
                (pl.col("recording") == entry[0]) & (pl.col("cell_index") == entry[1])
            )[f"response_start_{trigger}"].item()

            if starts is None or len(starts) == 0:
                continue
            lengths = np.diff(starts)

            x, y = generate_line_data(starts, slopes, lengths)

            ax[e_idx, trigger_idx].plot(x, y, c="red")
            ax[e_idx, trigger_idx].set_title(f"Recording {entry[0]}, Cell {entry[1]}")
            (
                ax[e_idx, trigger_idx].scatter(
                    bins[response_start], cum_counts[response_start], c="blue"
                )
            )
            ax[e_idx, trigger_idx].annotate(
                f"Response start {bins[response_start]:.3f}",
                (0, 0.4),
            )

    ax[0, 0].set_ylabel("Firing rate")
    ax[-1, 0].set_xlabel("Time (s)")
    ax[0, 0].set_xlim([window_neg, window_pos])
    ax[0, 0].set_ylim([0, 0.2])

    return fig, ax


# %%
all_results_df = pl.read_parquet(r"D:\chicken_analysis\all_results_csteps.parquet")
binned_spikes = pl.read_parquet(r"D:\chicken_analysis\binned_spikes_csteps.parquet")
# %%
break_columns = [f"response_start_{trigger}" for trigger in range(20)]
score_columns = [f"ssr_score_{trigger}" for trigger in range(20)]
fastest_cells = (
    all_results_df.group_by("recording", "cell_index")
    .agg(pl.mean_horizontal(break_columns).alias("response_start"))
    .explode("response_start")
)
fastest_cells = fastest_cells.with_columns(
    (pl.col("response_start") * 0.001).alias("response_start")
)
consistent_cells = (
    all_results_df.group_by("recording", "cell_index")
    .agg(var=pl.concat_list(break_columns).list.std())
    .explode("var")
)
linear_cells = (
    all_results_df.group_by("recording", "cell_index")
    .agg(score_mean=pl.concat_list(score_columns).list.mean())
    .explode("score_mean")
)
# %% combine dfs
all_results_stats = pl.concat(
    [fastest_cells, consistent_cells, linear_cells], how="align"
)
# %%
fig, ax = plt.subplots()
ax.hist(fastest_cells["response_start"].to_numpy(), bins=100)
fig.show()

# %%
sample_df = all_results_df.sample(10)
rec_cell = sample_df.unique(["recording", "cell_index"])["recording", "cell_index"]
fig, ax = plot_cell(
    sample_df,
    binned_spikes.filter(
        (pl.col("recording").is_in(rec_cell["recording"]))
        & (pl.col("cell_index").is_in(rec_cell["cell_index"]))
    ),
    triggers=np.arange(1, 20, 2).tolist(),
)
fig.show()
# %%
cell_idx = [367]
recording = ["chicken_18_07_2024_p0"]
single_cell = binned_spikes.filter(
    (pl.col("recording").is_in(recording)) & (pl.col("cell_index").is_in(cell_idx))
)
recordings.dataframes["csteps_single"] = recordings.dataframes["csteps_filtered"].query(
    f"recording == '{single_cell['recording'][0]}' & cell_index == {single_cell['cell_index'][0]}"
)
single_cell_spikes = recordings.get_spikes_df(
    "csteps_single", stimulus_df="csteps_stim", pandas=False
)
max_trigger = single_cell_spikes["trigger"].max()

fig, ax = plot_cell(
    all_results_df.filter(
        (pl.col("recording").is_in(recording)) & (pl.col("cell_index").is_in(cell_idx))
    ),
    binned_spikes.filter(
        (pl.col("recording").is_in(recording)) & (pl.col("cell_index").is_in(cell_idx))
    ),
    triggers=np.arange(0, 20, 2).tolist(),
)
fig.show()

# %% median start point
single_cell_results = all_results_df.filter(
    (pl.col("recording") == recording[0]) & (pl.col("cell_index") == cell_idx[0])
)
response_starts = []
response_starts.extend(
    bins[
        single_cell_results.filter(pl.col(f"response_start_{trigger}").is_not_null())[
            f"response_start_{trigger}"
        ]
    ]
    for trigger in range(20)
)

median_response_start = np.median(response_starts, axis=0)
# %%
# find breaks closest to median response start

breaks = []
breaks.extend(
    np.where(
        single_cell_results[f"breaks_{trigger}"].list.to_array(1).to_numpy().flatten()
        > median_response_start
    )[0][0]
    for trigger in range(20)
)
# %%
slopes_before = np.zeros(20)
slopes_after = np.zeros(20)
for trigger in range(20):
    slopes_before[trigger] = (
        single_cell_results[f"breaks_{trigger}"].list[breaks[trigger] - 1].item()
    )

    slopes_after[trigger] = (
        single_cell_results[f"breaks_{trigger}"].list[breaks[trigger]].item()
    )
# need to fill with smallest possible value

slopes_before[slopes_before < 0] = np.mean(slopes_before[slopes_before > 0])
slopes_after[slopes_after < 0] = np.mean(slopes_after[slopes_after > 0])

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
ax[0].bar(np.arange(0, 20, 1), slopes_before, color="black")
ax[0].set_title("Slopes before")
ax[1].bar(np.arange(0, 20, 1), slopes_after, color="black")
ax[1].set_title("Slopes after")
ax[0].set_ylim([-0.5, 0.5])
ax[1].set_ylim([-0.5, 0.5])
fig.show()

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("Contrast_Step")
# %%
spikes_fs = single_cell_spikes.clone()
for trigger_idx, trigger in enumerate(cum_triggers):
    spikes_fs = spikes_fs.with_columns(
        pl.when(
            (pl.col("times_triggered") > trigger + window_neg)
            & (pl.col("times_triggered") < trigger + window_pos)
        )
        .then(pl.col("times_triggered"))
        .alias(f"{trigger_idx}_fs")
    )

rep = 0
spikes_fs = spikes_fs.filter(pl.col("repeat") == rep)
# %%
fig_spikes, spikes_axes = plt.subplots(nrows=10, ncols=2, figsize=(15, 20), sharey=True)
fig_cred, cred_axes = plt.subplots(nrows=10, ncols=2, figsize=(15, 20), sharey=True)
row = 0
first_spikes_times = np.zeros((20, 2))
on_off_times = np.zeros((10, 2))
for test_trigger in range(20):
    col = 0 if test_trigger % 2 == 0 else 1
    # candidate_times, posterior = changepoint_posterior(
    example_times = (
        spikes_fs.filter(pl.col(f"{test_trigger}_fs").is_not_nan())
        .sort(f"{test_trigger}_fs")[f"{test_trigger}_fs"]
        .to_numpy()
    )
    if len(example_times) < 3:
        continue
    #
    spikes_axes[row, col].scatter(
        example_times,
        np.ones_like(example_times),
        c=CT.colours[test_trigger],
        marker="|",
    )
    spikes_axes[row, col].set_facecolor((112 / 255, 128 / 255, 144 / 255))
    spikes_axes[row, col].set_xlim(
        [
            cum_triggers[test_trigger] + window_neg,
            cum_triggers[test_trigger] + window_pos,
        ]
    )

    # %%
    if not np.all(example_times > [cum_triggers[test_trigger]]):
        if slopes_before[test_trigger] > 0:
            candidate_times, posterior = bayesian.changepoint_posterior(
                example_times,
                slopes_before[test_trigger] * 1000,
                slopes_after[test_trigger] * 1000,
                tau_guess=cum_triggers[test_trigger] + median_response_start,
                prior_std=0.01,
            )
            credibility_borders = bayesian.credible_interval(candidate_times, posterior)
        else:
            changepoint_time, posterior = bayesian.detect_changepoint_single_rate(
                example_times, slopes_after[test_trigger]
            )
            candidate_times = example_times
    else:
        candidate_times = np.array([example_times[0]])
        changepoint_time, _ = bayesian.detect_changepoint_single_rate(
            example_times, slopes_after[test_trigger]
        )
        posterior = np.array([1])

    # %% plot posterior

    cred_axes[row, col].plot(candidate_times, posterior, c="black")
    cred_axes[row, col].scatter(
        candidate_times[np.argmax(posterior)], np.max(posterior), c="red"
    )
    cred_axes[row, col].annotate(
        f"change_point = {candidate_times[np.argmax(posterior)]:.3f}",
        (candidate_times[np.argmax(posterior)], np.max(posterior)),
    )
    cred_axes[row, col].axvline(
        cum_triggers[test_trigger] + response_starts[test_trigger][0], c="blue"
    )
    if slopes_before[test_trigger] > 0 and not np.all(
        example_times > [cum_triggers[test_trigger]]
    ):
        cred_axes[row, col].axvline(credibility_borders[0], c="green")
        cred_axes[row, col].axvline(credibility_borders[1], c="green")
    cred_axes[row, col].set_xlim(
        [
            cum_triggers[test_trigger] + window_neg,
            cum_triggers[test_trigger] + window_pos,
        ]
    )
    on_off_times[row, col] = (
        candidate_times[np.argmax(posterior)] - cum_triggers[test_trigger]
    )
    if col == 1:
        row += 1

fig_spikes.savefig(rf"D:\chicken_analysis\first_spike_repeat{rep}.png")
fig_cred.savefig(rf"D:\chicken_analysis\changepoint_example_repeat{rep}.png")

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
ax[0].boxplot(on_off_times[:, 0], notch=True)
ax[1].boxplot(on_off_times[:, 1], notch=True)
ax[0].set_title("On time")
ax[1].set_title("Off time")
ax[0].set_ylim([0, 0.1])
fig.show()

# %% plotting
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
ax[0].scatter(np.arange(0, 10), on_off_times[:, 0])
# plot the regression line
model = make_pipeline(PolynomialFeatures((1, 2)), RANSACRegressor(random_state=0))
model.fit(np.arange(0, 10).reshape(-1, 1), on_off_times[:, 0])
ax[0].plot(
    np.arange(0, 10), model.predict(np.arange(0, 10).reshape(-1, 1)), color="red"
)
ax[1].scatter(np.arange(0, 10), on_off_times[:, 1])
model = make_pipeline(PolynomialFeatures((1, 2)), RANSACRegressor(random_state=0))
model.fit(np.arange(0, 10).reshape(-1, 1), on_off_times[:, 1])
ax[1].plot(
    np.arange(0, 10), model.predict(np.arange(0, 10).reshape(-1, 1)), color="red"
)
ax[0].set_ylim([0, 0.1])
fig.show()


# %%
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
test_trigger = 0
breaks = (
    single_cell_results.select(f"breaks_{test_trigger}")[f"breaks_{test_trigger}"]
    .list.to_array(1)
    .to_numpy()
    .flatten()
)
rates = (
    single_cell_results.select(f"slopes_{test_trigger}")[f"slopes_{test_trigger}"]
    .list.to_array(1)
    .to_numpy()
    .flatten()
    * 1000
)
rates[rates < 0] = 0
segments = [(breaks[i], breaks[i + 1], rates[i]) for i in range(len(breaks) - 1)]

# %%
spikes_fs_relative = single_cell_spikes.clone()
spikes_fs_relative = spikes_fs_relative.filter(pl.col("repeat") == 2)
for trigger_idx, trigger in enumerate(cum_triggers):
    spikes_fs_relative = spikes_fs_relative.with_columns(
        pl.when(
            (pl.col("times_triggered") > trigger + window_neg)
            & (pl.col("times_triggered") < trigger + window_pos)
        )
        .then(pl.col("times_triggered") - trigger)
        .alias(f"{trigger_idx}_fs")
    )
# %%
spike_times = generate_poisson_spike_train(segments)
psth, bins = np.histogram(spike_times, bins=bins)
psth_cell, bin = np.histogram(
    spikes_fs_relative.select(pl.col(f"{test_trigger}_fs"))[
        f"{test_trigger}_fs"
    ].to_numpy(),
    bins=bins,
)
fig, ax = plt.subplots(nrows=2)
ax[0].plot(bins[:-1], psth, color="red")
ax[1].plot(bins[:-1], psth_cell, color="black")
fig.show()

# %% plot isi difference
isi_real = np.diff(spikes_fs_relative[f"{test_trigger}_fs"].to_numpy())

# %%
isis = []
spiketimes_generated = []
for _ in range(1000):
    spiketimes_generated.append(generate_poisson_spike_train(segments))
    isis.append(np.diff(spiketimes_generated[-1]))

# %%
bins_isi = np.arange(window_neg, window_pos, 0.01)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
ax[0].hist(isi_real, bins=bins_isi, color="black", density=True)
ax[1].hist(isis[0], bins=bins_isi, color="red", density=True)
fig.show()
# %%
isis_int = np.interp(spike_times, spiketimes_generated[0][:-1], isis[0])
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10), sharey=True, sharex=True)
ax[0].plot(
    spikes_fs_relative.select(pl.col(f"{test_trigger}_fs"))[
        f"{test_trigger}_fs"
    ].to_numpy()[:-1],
    isi_real,
    color="black",
)
ax[1].plot(spike_times, isis_int, color="red")
ax[0].set_xlim([window_neg, window_pos])
fig.show()
