import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from polarspike import (
    Overview,
    stimulus_spikes,
    bayesian,
    colour_template,
    plotly_templates,
    histograms,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from sklearn.metrics import r2_score

import matplotlib.animation as animation
import seaborn as sns

# %%
window_neg = -0.2
window_neg_abs = np.abs(window_neg)
window_pos = 0.2
shuffle_window = 5
bin_width = 0.001
bins = np.arange(window_neg, window_pos + bin_width, bin_width)
bin_freq = 1 / bin_width
prior_std = 0.01


# %%
def first_spike_stats(
    lazy_df: pl.DataFrame,
    cum_triggers: np.ndarray,
    op_first=pl.median,
    op_posterior=pl.mean,
) -> pl.DataFrame:
    """
    Aggregates the dataframe using the provided operations for first_spike and first_spike_posterior.

    Parameters:
      lazy_df (pl.DataFrame): The input Polars dataframe (lazy).
      cum_triggers (list): List of triggers used to determine the column indices.
      op_first (function): The Polars function to aggregate the 'first_spike' columns.
      op_posterior (function): The Polars function to aggregate the 'first_spike_posterior' columns.

    Returns:
      pl.Dataframe: A list of lazy query results for each trigger.
    """
    # Convert to lazy mode if not already lazy.

    results = []
    for trigger_idx, _ in enumerate(cum_triggers):
        # Build the aggregation expressions using the provided functions.
        exprs = [
            op_first(f"first_spike_{trigger_idx}").alias(
                f"av_first_spike_{trigger_idx}"
            ),
            op_posterior(f"first_spike_posterior_{trigger_idx}").alias(
                f"av_posterior_{trigger_idx}"
            ),
        ]
        # Group and aggregate the lazy dataframe.
        results.append(lazy_df.group_by(["recording", "cell_index"]).agg(exprs))
    return pl.concat(results, how="align").collect()


def calc_prior(times, prior_mean, prior_std):
    return np.exp(-0.5 * ((times - prior_mean) / prior_std) ** 2)


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
            if rate <= 0:
                rate = 1
            dt = np.random.exponential(1.0 / rate)
            t += dt  # Next spike time
            if t < end:
                spike_times.append(t)
    return np.array(spike_times)


def generate_line_data(starts, slopes, lengths):
    x_list = [0]
    y_list = [0]
    for start_idx in range(len(starts)):
        x = np.linspace(starts[start_idx], starts[start_idx] + lengths[start_idx], 10)
        y = slopes[start_idx] * (x - starts[start_idx]) + y_list[-1]
        x_list.extend(x)
        y_list.extend(y)

    return np.asarray(x_list[:-1]).T, np.asarray(y_list[:-1]).T


# %%
contrast_df = pl.scan_parquet(r"D:\chicken_analysis\changepoint_df_csteps_test.parquet")
fff_df = pl.scan_parquet(r"D:\chicken_analysis\changepoint_df_fff.parquet")
fff_repeats = len(fff_df.select("repeat").unique().collect())
contrast_repeats = len(contrast_df.select("repeat").unique().collect())
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
# %% Contrast triggers
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [12])
contrast_cum_triggers = np.hstack([np.array([0]), np.cumsum(mean_trigger_times)])
# %% FFF triggers
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [1])
fff_cum_triggers = np.hstack([np.array([0]), np.cumsum(mean_trigger_times)])
# %% Get first spike statistics for contrast
contrast_hist = first_spike_stats(contrast_df, contrast_cum_triggers)
fff_hist = first_spike_stats(fff_df, fff_cum_triggers)
# %%
CT_contrast = colour_template.Colour_template()
CT_contrast.pick_stimulus("Contrast_Step")
CT_fff = colour_template.Colour_template()
CT_fff.pick_stimulus("FFF_6_MC")
# %%
fig = go.Figure()
for trigger_idx, _ in enumerate(contrast_cum_triggers):
    fig.add_trace(
        go.Scatter(
            x=contrast_hist[trigger_idx]["av_first_spike_0"],
            y=contrast_hist[trigger_idx]["av_posterior_0"],
            mode="markers",
            name=f"Trigger {trigger_idx}",
        )
    )
# %%
contrast_median_prior = contrast_df.select("median_starts").median().collect().item()
contrast_std_prior = (
    contrast_df.select(
        (
            pl.col("median_starts").quantile(0.75)
            - pl.col("median_starts").quantile(0.25)
        )
        / 1.349
    )
    .collect()
    .item()
)

fff_median_prior = fff_df.select("median_starts").median().collect().item()
fff_std_prior = (
    fff_df.select(
        (
            pl.col("median_starts").quantile(0.75)
            - pl.col("median_starts").quantile(0.25)
        )
        / 1.349
    )
    .collect()
    .item()
)


# %% plot the priors
times = np.arange(-0.2, 0.2, 0.001)
prior_contrast = calc_prior(times, contrast_median_prior, contrast_std_prior)
prior_fff = calc_prior(times, fff_median_prior, fff_std_prior)
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(times, prior_contrast, label="Contrast", color="black")
ax.plot(times, prior_fff, label="FFF", color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Prior")
fig.show()
# %%
nr_bins = 80
bins = np.linspace(window_neg, window_pos, nr_bins)
contrast_binned = np.zeros((len(contrast_cum_triggers), nr_bins - 1))
fff_binned = np.zeros((len(fff_cum_triggers), nr_bins - 1))
first_spikes_contrast = np.ma.zeros(
    (len(contrast_hist.select("recording")), len(contrast_cum_triggers))
)
weights_contrast = np.ma.zeros(
    (len(contrast_hist.select("recording")), len(contrast_cum_triggers))
)
first_spikes_fff = np.ma.zeros(
    (len(fff_hist.select("recording")), len(fff_cum_triggers))
)
peaks_contrast = np.zeros(contrast_cum_triggers.shape[0])
peaks_fff = np.zeros(fff_cum_triggers.shape[0])
std_contrast = np.zeros(contrast_cum_triggers.shape[0])
std_fff = np.zeros(fff_cum_triggers.shape[0])

weights_fff = np.ma.zeros((len(fff_hist.select("recording")), len(fff_cum_triggers)))
for trigger_idx, _ in enumerate(contrast_cum_triggers):
    contrast_spikes = contrast_hist[f"av_first_spike_{trigger_idx}"].to_numpy()
    first_spikes_contrast[:, trigger_idx] = contrast_spikes
    contrast_posterior = contrast_hist[f"av_posterior_{trigger_idx}"].to_numpy()
    weights_contrast[:, trigger_idx] = contrast_posterior
    contrast_binned[trigger_idx], _ = np.histogram(
        contrast_spikes, bins=bins, density=True, weights=contrast_posterior
    )
    peaks_contrast[trigger_idx] = bins[np.argmax(contrast_binned[trigger_idx])]

for trigger_idx, _ in enumerate(fff_cum_triggers):
    fff_spikes = fff_hist[f"av_first_spike_{trigger_idx}"].to_numpy()
    first_spikes_fff[:, trigger_idx] = fff_spikes
    fff_posterior = fff_hist[f"av_posterior_{trigger_idx}"].to_numpy()
    weights_fff[:, trigger_idx] = fff_posterior
    fff_binned[trigger_idx], _ = np.histogram(
        fff_spikes, bins=bins, density=True, weights=fff_posterior
    )
    peaks_fff[trigger_idx] = bins[np.argmax(fff_binned[trigger_idx])]


# %% set the mask where the first spike is nan
first_spikes_contrast = np.ma.masked_invalid(first_spikes_contrast)
weights_contrast = np.ma.masked_invalid(weights_contrast)
first_spikes_fff = np.ma.masked_invalid(first_spikes_fff)
weights_fff = np.ma.masked_invalid(weights_fff)
# %%
fig = go.Figure()
for trigger_idx, _ in enumerate(contrast_cum_triggers[:-1]):
    fig.add_trace(
        go.Scatter(
            x=bins[:-1],
            y=contrast_binned[trigger_idx],
            mode="lines",
            name=f"Contrast Trigger {trigger_idx}",
            line=dict(color=CT_contrast.colours[trigger_idx], width=2),
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=[
    #             np.average(
    #                 first_spikes_contrast[:, trigger_idx],
    #                 weights=weights_contrast[:, trigger_idx],
    #             )
    #         ],
    #         y=[1 * np.max(contrast_binned[trigger_idx])],
    #         mode="markers+lines",
    #         name=f"Contrast Trigger {trigger_idx} Median",
    #         marker=dict(color=CT_contrast.colours[trigger_idx]),
    #     )
    # )
fig.add_trace(
    go.Scatter(
        x=times,
        y=prior_contrast * 100,
        mode="lines",
        name="Contrast Prior",
        line=dict(color="black", dash="dash"),
    )
)
for trigger_idx, _ in enumerate(fff_cum_triggers[:-1]):
    fig.add_trace(
        go.Scatter(
            x=bins[:-1],
            y=fff_binned[trigger_idx],
            mode="lines",
            name=f"FFF Trigger {trigger_idx}",
            line=dict(color=CT_fff.colours[trigger_idx], width=2),
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=[
    #             np.average(
    #                 first_spikes_fff[:, trigger_idx],
    #                 weights=weights_fff[:, trigger_idx],
    #             )
    #         ],
    #         y=[1 * np.max(fff_binned[trigger_idx])],
    #         mode="markers+lines",
    #         name=f"FFF Trigger {trigger_idx} Median",
    #         marker=dict(color=CT_fff.colours[trigger_idx]),
    #     )
    # )
fig.add_trace(
    go.Scatter(
        x=times,
        y=prior_fff * 100,
        mode="lines",
        name="FFF Prior",
        line=dict(color="red", dash="dash"),
    )
)
# add ylabels
fig.update_yaxes(title_text="Cell count in %")
fig.update_xaxes(title_text="Time (s)")
fig.show(renderer="browser")
# %% get inter repeat variability
combined_contrast = contrast_hist.join(
    first_spike_stats(contrast_df, contrast_cum_triggers, op_first=pl.std),
    how="right",
    left_on=["recording", "cell_index"],
    right_on=["recording", "cell_index"],
)
combined_fff = fff_hist.join(
    first_spike_stats(fff_df, fff_cum_triggers, op_first=pl.std),
    how="right",
    left_on=["recording", "cell_index"],
    right_on=["recording", "cell_index"],
)
# %% seaborn ecfd plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True, sharex=True)
c = 0
for trigger_idx, _ in enumerate(contrast_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0:
        row = 0
        if trigger_idx != 0:
            c += 1
    else:
        row = 1
    sns.ecdfplot(
        data=combined_contrast.to_pandas(),
        x=f"av_first_spike_{trigger_idx}_right",
        weights=f"av_posterior_{trigger_idx}",
        ax=ax[row],
        palette="viridis",
        color=CT_contrast.colours[::2][c],
        label=CT_contrast.names[::2][c],
    )
# change background color
for a in ax:
    a.set_facecolor("lightblue")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_xlim([0, 0.025])
    a.set_ylim([0, 1])
ax[0].set_ylabel("Cumulative density")
ax[1].set_xlabel(None)
ax[0].set_xlabel("Inter repeat variability")


fig.show()
# %%
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True, sharex=True)
c = 0
for trigger_idx, _ in enumerate(fff_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0:
        row = 0
        if trigger_idx != 0:
            c += 1
    else:
        row = 1

    sns.ecdfplot(
        data=combined_fff.to_pandas(),
        x=f"av_first_spike_{trigger_idx}_right",
        weights=f"av_posterior_{trigger_idx}",
        ax=ax[row],
        palette="viridis",
        color=CT_fff.colours[::2][c],
        label=CT_fff.names[::2][c],
    )
# change background color
for a in ax:
    a.set_facecolor("lightblue")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_xlim([window_neg, window_pos])
    a.set_ylim([0, 1])
ax[0].set_ylabel("Cumulative density")
ax[1].set_xlabel(None)
ax[0].set_xlabel("Inter repeat variability")
ax[0].set_xlim([0, window_pos])
ax[1].legend()

ax[0].set_title("ON")
ax[1].set_title("OFF")
fig.show()

# %%
fig = make_subplots(
    rows=2,
    cols=int(np.ceil(contrast_cum_triggers[:-1].shape[0] / 2)),
    shared_yaxes=True,
    shared_xaxes=True,
)
col = 1
for trigger_idx, _ in enumerate(contrast_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0:
        row = 1
    else:
        row = 2
    fig.add_trace(
        go.Scatter(
            x=combined_contrast[f"av_first_spike_{trigger_idx}"],
            y=combined_contrast[f"av_first_spike_{trigger_idx}_right"],
            mode="markers",
            marker=dict(
                size=10,
                color=combined_contrast[f"av_posterior_{trigger_idx}"],
                coloraxis="coloraxis",  # Use coloraxis for a single colorbar
            ),
            text=combined_contrast.to_pandas().apply(
                lambda row: f"Recording: {row['recording']}, Cell Index: {row['cell_index']}, posterior: {row[f'av_posterior_{trigger_idx}']}",
                axis=1,
            ),
            hoverinfo="text",
        ),
        row=row,
        col=col,
    )
    # add title
    fig.update_xaxes(
        title_text=CT_contrast.names[trigger_idx],
        title_font=dict(color="black"),
        row=row,
        col=col,
    )
    if np.mod(trigger_idx, 2) == 1:
        col += 1
# add ylabel to first row
fig.update_yaxes(title_text="Inter repeat variability", row=1, col=1)
# add one colorbar
fig.update_layout(
    coloraxis=dict(
        cmin=0,
        cmax=1,
        colorscale="Viridis",
        colorbar=dict(title="Posterior"),
    )
)
fig.update_yaxes(type="log")
fig.update_layout(height=1300, width=2000)
# hide legend
fig.update_layout(showlegend=False)
fig.show(renderer="browser")
# %%


fig = make_subplots(
    rows=2,
    cols=int(np.ceil(fff_cum_triggers[:-1].shape[0] / 2)),
    shared_yaxes=True,
    shared_xaxes=True,
)
col = 1
for trigger_idx, _ in enumerate(fff_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0:
        row = 1
    else:
        row = 2
    fig.add_trace(
        go.Scatter(
            x=combined_fff[f"av_first_spike_{trigger_idx}"],
            y=combined_fff[f"av_first_spike_{trigger_idx}_right"],
            mode="markers",
            marker=dict(
                size=10,
                color=combined_fff[f"av_posterior_{trigger_idx}"],
                coloraxis="coloraxis",  # Use coloraxis for a single colorbar
            ),
            text=combined_fff.to_pandas().apply(
                lambda row: f"Recording: {row['recording']}, Cell Index: {row['cell_index']}, posterior: {row[f'av_posterior_{trigger_idx}']}",
                axis=1,
            ),
            hoverinfo="text",
        ),
        row=row,
        col=col,
    )
    # add title
    fig.update_xaxes(
        title_text=CT_fff.names[trigger_idx],
        title_font=dict(color=CT_fff.colours[trigger_idx]),
        row=row,
        col=col,
    )
    if np.mod(trigger_idx, 2) == 1:
        col += 1
# add ylabel to first row
fig.update_yaxes(title_text="Inter repeat variability", row=1, col=1)
# add one colorbar
fig.update_layout(
    coloraxis=dict(
        cmin=0,
        cmax=1,
        colorscale="Viridis",
        colorbar=dict(title="Posterior"),
    )
)
fig.update_yaxes(type="log")
fig.update_layout(height=1300, width=2000)
# hide legend
fig.update_layout(showlegend=False)
fig.show(renderer="browser")


# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharey=True)
ax[0, 0].bar(
    np.arange(contrast_cum_triggers[:-1].shape[0] // 2),
    peaks_contrast[:-1][::2],
    color=CT_contrast.colours[::2],
)

ax[1, 0].bar(
    np.arange(contrast_cum_triggers[:-1].shape[0] // 2),
    peaks_contrast[:-1][1::2],
    color=CT_contrast.colours[::2],
)
ax[0, 1].bar(
    np.arange(fff_cum_triggers[:-1].shape[0] // 2),
    peaks_fff[:-1][::2],
    color=CT_fff.colours[::2],
)
ax[1, 1].bar(
    np.arange(fff_cum_triggers[:-1].shape[0] // 2),
    peaks_fff[:-1][1::2],
    color=CT_fff.colours[::2],
)
# add subplot titles
ax[0, 0].set_title("ON")
ax[1, 0].set_title("OFF")
ax[0, 1].set_title("ON")
ax[1, 1].set_title("OFF")
# remove splines
for a_sub in ax:
    for a in a_sub:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_facecolor("lightblue")
# set plot area color to light blue
# set y ticks, use two decimal places
ax[0, 0].set_yticks(
    bins[
        np.logical_and(
            bins > 0, bins < np.max(np.concatenate([peaks_contrast, peaks_fff]))
        )
    ].round(3)
)
# ylabel
ax[1, 0].set_ylabel("First spike time (s)")
fig.show()
# %%
trigger = 2
# random_df = fff_df.collect().sample(1)
df_test = fff_df.filter(
    (pl.col("recording") == "chicken_30_08_2024_p0") & (pl.col("cell_index") == 187)
).collect()
# df_test = subset_df.sample(1)
# df_test = fff_df.filter(
#     (pl.col("recording") == df_test["recording"].item())
#     & (pl.col("cell_index") == df_test["cell_index"].item())
# ).collect()
#     (pl.col("recording") == random_df["recording"].item())
#     & (pl.col("cell_index") == random_df["cell_index"].item())
# ).collect()
#     (pl.col("recording") == "chicken_19_07_2024_p0") & (pl.col("cell_index") == 335)
# ).collect()
#     (pl.col("recording") == random_df["recording"].item())
#     & (pl.col("cell_index") == random_df["cell_index"].item())
# ).collect()
# repeats = len(df_test)
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

    ax[rep].set_title(f"Repeat {rep}")
    # remove top and right spines
    ax[rep].spines["top"].set_visible(False)
    ax[rep].spines["right"].set_visible(False)
# plot vertical line at 0
for a in ax:
    a.axvline(0, color="black", linestyle="--")
ax[0].set_xlim([window_neg, window_pos])
ax[-1].set_xlabel("Time (s)")
ax[-1].set_ylabel("Posterior and spikes")
fig.show()
# %% animation
# Create a figure with 10 subplots arranged in 5 rows x 2 columns.
fig, axes = plt.subplots(10, 1, figsize=(15, 20))
axes = axes.flatten()  # easier to iterate over all axes

# Set a common x-axis range for all subplots.
for ax in axes:
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, 1)

frame_times = np.linspace(-0.2, 0.2, 100)


def update(frame):
    # For each neuron (subplot), clear the axis and plot a new spike.
    for rep, ax in enumerate(axes):
        ax.cla()  # clear previous content
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0, 1)
        # Generate one random spike time between -0.2 and 0.2.
        spike_time = df_test.explode(f"{trigger}_fs").filter(
            (pl.col("repeat") == rep) & (pl.col(f"{trigger}_fs") < frame_times[frame])
        )
        if len(spike_time) > 0:
            spike_time = spike_time.select(f"{trigger}_fs").to_numpy()
        else:
            continue

        y_pos = np.ones_like(spike_time)
        ax.scatter(spike_time, y_pos, marker="|", color="black", s=100)

    # remove splines
    # set y_range to 0 - 2
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_ylim([0, 2])
    # remove xticks and yticks from all but the last subplot
    for a in axes[:-1]:
        a.set_xticks([])
        a.set_yticks([])
    # increase margins
    fig.subplots_adjust(hspace=0.5)
    axes[-1].set_xlabel("Time (s)")

    # No artist list is returned because we're redrawing the axes.
    return axes


# Create the animation; here, 100 frames with a 200ms interval between frames.
ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=False)

fig.tight_layout()
ani.save(r"D:\slide_app_presentation\spike_times.gif", writer="pillow", fps=5)
ani.save(r"D:\slide_app_presentation\spike_times.mp4", writer="ffmpeg", fps=5)
# %%  create list of tuples (start, end, rate) for generate_poisson_spike_train function from df
starts = df_test.select(f"breaks_{trigger}")[0].item().to_numpy()[:-1]
ends = df_test.select(f"breaks_{trigger}")[0].item().to_numpy()[1:]
rates = df_test.select(f"slopes_{trigger}")[0].item().to_numpy()

# rates = rates * bin_freq / fff_repeats
segments = list(zip(starts, ends, rates))

# %%
bin_size = 0.001
max_window = 0.2
nr_bins = int(max_window / bin_size)
hist_before = np.empty((10, nr_bins - 1))
hist_after = np.empty((10, nr_bins - 1))
hist_before.fill(np.nan)
hist_after.fill(np.nan)
bins = np.arange(bin_size, max_window + bin_size, bin_size)
for rep in range(10):
    df_sub = df_test.filter(pl.col("repeat") == rep)
    spikes = df_sub.select(f"{trigger}_fs").item()[1:-1].to_numpy()
    spikes_before = np.diff(spikes[spikes <= 0])
    spikes_after = np.diff(spikes[spikes > 0])
    # bin spike diffs
    hist_before[rep, :], _ = np.histogram(spikes_before, bins=bins)
    hist_after[rep, :], _ = np.histogram(spikes_after, bins=bins)
# generate synthetic data
hist_before_synthetic = np.empty((10, nr_bins - 1))
hist_before_synthetic.fill(np.nan)
hist_after_synthetic = np.empty((10, nr_bins - 1))
hist_after_synthetic.fill(np.nan)
for i in range(10):
    spike_train = generate_poisson_spike_train(segments)
    spikes_before = np.diff(spike_train[spike_train <= 0])
    spikes_after = np.diff(spike_train[spike_train > 0])
    hist_before_synthetic[i, :], _ = np.histogram(spikes_before, bins=bins)
    hist_after_synthetic[i, :], _ = np.histogram(spikes_after, bins=bins)

hist_before_mean = np.nanmean(hist_before, axis=0)
hist_after_mean = np.nanmean(hist_after, axis=0)
hist_before_synthetic_mean = np.nanmean(hist_before_synthetic, axis=0)
hist_after_synthetic_mean = np.nanmean(hist_after_synthetic, axis=0)
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(15, 10))
ax[0].plot(bins[:-1], hist_before_mean, color="black", label="Before")
ax[1].plot(bins[:-1], hist_after_mean, color="black", label="After")
ax[0].plot(bins[:-1], hist_before_synthetic_mean, color="red", label="Before synthetic")
ax[1].plot(bins[:-1], hist_after_synthetic_mean, color="red", label="After synthetic")
# remove top and right spines
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_ylim([0, np.max([hist_before_mean, hist_after_mean])])

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Spike count")
ax[1].legend()

fig.show()

# %% plot real spiketrain and synthetic spiketrain
rep = 2
fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(15, 10))
spikes = df_sub.select(f"{trigger}_fs").item()[1:-1].to_numpy()
ax.scatter(spikes, np.ones_like(spikes) + 0.2, marker="|", s=100, color="black")
spike_train = generate_poisson_spike_train(segments)
ax.scatter(spike_train, np.ones_like(spike_train), marker="|", s=100, color="red")
# fig.set_yrange([-0.2, 0.2])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# set y range
ax.set_ylim([0.6, 1.5])
ax.set_xlim([-0.2, 0.2])
ax.set_xlabel("Time (s)", fontsize=15)
ax.set_ylabel("Spike train", fontsize=15)
# remove yticks
ax.set_yticks([])
# make xticks larger
ax.tick_params(axis="x", labelsize=12)
fig.show()

# %%
x_before = []
y_before = []
x_after = []
y_after = []
x_before_synthetic = []
y_before_synthetic = []
x_after_synthetic = []
y_after_synthetic = []
for rep in range(10):
    spikes = (
        df_test.filter(pl.col("repeat") == rep)
        .select(f"{trigger}_fs")
        .item()[1:-1]
        .to_numpy()
    )
    spikes_before = np.diff(spikes[spikes <= 0])
    spikes_after = np.diff(spikes[spikes > 0])
    x_before.extend(spikes[spikes <= 0][1:])
    y_before.extend(spikes_before)
    x_after.extend(spikes[spikes > 0][1:])
    y_after.extend(spikes_after)
    #
    spike_synthetic = generate_poisson_spike_train(segments)
    spikes_before_synthetic = np.diff(spike_synthetic[spike_synthetic <= 0])
    spikes_after_synthetic = np.diff(spike_synthetic[spike_synthetic > 0])
    x_before_synthetic.extend(spike_synthetic[spike_synthetic <= 0][1:])
    y_before_synthetic.extend(spikes_before_synthetic)
    x_after_synthetic.extend(spike_synthetic[spike_synthetic > 0][1:])
    y_after_synthetic.extend(spikes_after_synthetic)

# %%


sns.set_theme(style="ticks")

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharey="col", sharex="col")

# First plot
sns.scatterplot(x=x_before, y=y_before, ax=ax[0, 0], marker="+")

# sns.histplot(x=x_before, y=y_before, bins=50, pthresh=0.1, cmap="mako", ax=ax[0, 0])
ax[0, 0].set_xlabel("Time (s)", fontsize=15)
ax[0, 0].set_ylabel("ISI (s)", fontsize=15)
ax[0, 0].set_title("Joint Plot of Time vs ISI", fontsize=18)

# Second plot (placeholder for another plot)
sns.scatterplot(x=x_after, y=y_after, ax=ax[0, 1], marker="+")

# sns.histplot(x=x_after, y=y_after, bins=10, pthresh=0.1, cmap="mako", ax=ax[0, 1])
ax[0, 1].set_xlabel("Time (s)", fontsize=15)
ax[0, 1].set_ylabel("ISI (s)", fontsize=15)
ax[0, 1].set_title("Joint Plot of Time vs ISI", fontsize=18)

# Third plot
sns.scatterplot(x=x_before_synthetic, y=y_before_synthetic, ax=ax[1, 0], marker="+")

ax[1, 0].set_xlabel("Time (s)", fontsize=15)
ax[1, 0].set_ylabel("ISI (s)", fontsize=15)
ax[1, 0].set_title("Joint Plot of Time vs ISI", fontsize=18)

# Fourth plot
sns.scatterplot(x=x_after_synthetic, y=y_after_synthetic, ax=ax[1, 1], marker="+")

ax[1, 1].set_xlabel("Time (s)", fontsize=15)
ax[1, 1].set_ylabel("ISI (s)", fontsize=15)
ax[1, 1].set_title("Joint Plot of Time vs ISI", fontsize=18)

#  set y axis to log scale
# for a in ax.flatten():
#     a.set_yscale("log")

fig.tight_layout()
fig.show()

# %% single repeat plotting
df_test = test[30]  # .collect().sample(1)  # fff_df fff_df.filter(
#     (pl.col("recording") == "chicken_19_07_2024_p0") & (pl.col("cell_index") == 432)
# ).collect()

# df_test = df_test.filter(pl.col("repeat") == 4)
trigger = 1
df_test = df_test.with_columns(
    pl.col(f"{trigger}_fs").list.eval(pl.element() + window_neg)
)


fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(15, 20))
ax[0].plot(
    df_test.select(f"{trigger}_fs").item()[1:-1],
    df_test.select(f"posterior_{trigger}").item().to_numpy()
    / np.nanmax(df_test.select(f"posterior_{trigger}").item().to_numpy()),
)
ax[0].scatter(
    df_test.select(f"{trigger}_fs").item(),
    np.ones(df_test.select(f"{trigger}_fs").item().len()),
    marker="|",
    color="black",
    s=100,
)
# ax[0].vlines(
#     df_test.select(f"{trigger}_fs")
#     .item()[1:-1]
#     .to_numpy()[np.argmax(df_test.select(f"posterior_{trigger}").item().to_numpy())],
#     0,
#     1,
# )
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
# add ylabels
ax[0].set_ylabel("Posterior", fontsize=15)
ax[1].set_ylabel("L1", fontsize=15)
ax[2].set_ylabel("L2", fontsize=15)
ax[3].set_ylabel("Combined", fontsize=15)
ax[4].set_ylabel("Prior", fontsize=15)
# remove splines
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    # increase tick size
    a.tick_params(axis="both", which="major", labelsize=12)
# plot vlines at 0s
for a in ax:
    a.axvline(0, color="black", linestyle="--")
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[4].set_xlim([window_neg, window_pos])
fig.show()
# %%
rates = df_test.select(f"slopes_{trigger}")[0].item().to_numpy()
rates = rates * fff_repeats
x_line, y_line = generate_line_data(starts, rates, ends - starts)
fig, ax = plt.subplots(nrows=4, figsize=(15, 10))
width = ends - starts
bin_size_plotting = 0.05
ax[0].bar(
    starts + width / 2,
    rates / (1 / bin_size_plotting),
    width=ends - starts,
    color="red",
    alpha=0.5,
    label="Spline rates",
    edgecolor="black",
)
all_spikes = (
    df_test.select(f"{trigger}_fs")
    .explode(f"{trigger}_fs")
    .sort(f"{trigger}_fs")
    .to_numpy()
    .flatten()
)
values, bins_plotting = np.histogram(
    all_spikes,
    bins=np.arange(window_neg, window_pos + bin_size_plotting, bin_size_plotting),
)
ax[1].bar(
    bins_plotting[1:] - (bin_size_plotting / 2),
    values,
    width=bin_size_plotting,
    color="black",
    alpha=0.5,
    label="PSTH",
    # show bar outline
    edgecolor="black",
)
values, bins_plotting = np.histogram(
    all_spikes,
    bins=np.arange(window_neg, window_pos + 0.001, 0.001),
)
values_cum_sum = np.cumsum(values)

ax[2].bar(
    bins_plotting[1:] - (0.001 / 2),
    values_cum_sum,
    width=0.001,
    color="black",
    alpha=0.5,
    label="Cumulative PSTH",
)
ax[2].plot(x_line[1:], y_line[1:], color="red", label="estimated rate")
ax[3].scatter(all_spikes, np.ones(all_spikes.shape), color="black", marker="|")
ax[3].set_xlabel("Time (s)")
ax[1].set_ylabel("Spike count")
ax[0].legend()
ax[1].legend()
ax[2].legend()
# remove splines
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.show()
# %% plot each repeat as above
fig, ax = plt.subplots(nrows=fff_repeats, figsize=(15, 25))
for rep in range(fff_repeats):
    all_spikes = (
        df_test.filter(pl.col("repeat") == rep)
        .select(f"{trigger}_fs")
        .explode(f"{trigger}_fs")
        .sort(f"{trigger}_fs")
        .to_numpy()
        .flatten()
    )
    values, bins_plotting = np.histogram(
        all_spikes,
        bins=np.arange(window_neg, window_pos + 0.001, 0.001),
    )
    values_cum_sum = np.cumsum(values)
    values_cum_sum = values_cum_sum / np.max(values_cum_sum)
    ax[rep].bar(
        bins_plotting[1:] - (0.001 / 2),
        values_cum_sum,
        width=0.001,
        color="black",
        alpha=0.5,
        label="Cumulative PSTH",
    )
    ax[rep].plot(
        x_line[1:], y_line[1:] / np.max(y_line[1:]), color="red", label="estimated rate"
    )
    # rsquared
    y_measured_interp = np.interp(x_line[1:], bins_plotting[1:], values_cum_sum)
    r2 = r2_score(y_line[1:] / np.max(y_line[1:]), y_measured_interp)
    ax[rep].set_title(f"Repeat {rep}, R2: {r2}")
for rep in range(fff_repeats - 1):
    # remove x ticks
    plt.setp(ax[rep].get_xticklabels(), visible=False)
fig.show()
# %% compare all different colours on average
fff_results_split = fff_df.collect().partition_by(["recording", "cell_index"])
x_interp = np.arange(window_neg, window_pos, 0.05)
all_slopes = np.zeros(
    (len(fff_results_split), x_interp.shape[0], fff_cum_triggers.shape[0] - 1)
)
for trigger_idx, trigger in enumerate(fff_cum_triggers[:-1]):
    for e_idx, entry in enumerate(fff_results_split):
        starts = entry.select(f"breaks_{trigger_idx}")[0].item().to_numpy()[:-1]
        ends = entry.select(f"breaks_{trigger_idx}")[0].item().to_numpy()[1:]
        rates = entry.select(f"slopes_{trigger_idx}")[0].item().to_numpy()
        # rates = rates * bin_freq / fff_repeats
        segments = list(zip(starts, ends, rates))
        x_line, y_line = generate_line_data(starts, rates, ends - starts)
        y_interp = np.interp(x_interp, x_line, y_line)
        all_slopes[e_idx, :, trigger_idx] = y_interp
# %%
fig, ax = plt.subplots(
    nrows=1,
    ncols=(fff_cum_triggers.shape[0] - 1) // 2,
    figsize=(25, 5),
    sharey=True,
    sharex=True,
)
row = 0
for trigger_idx, trigger in enumerate(fff_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0 and trigger_idx != 0:
        row = row + 1
    # for e_idx, entry in enumerate(fff_results_split):
    #     ax[trigger_idx].plot(
    #         x_interp,
    #         all_slopes[e_idx, :, trigger_idx],
    #         color=CT_fff.colours[trigger_idx],
    #         alpha=0.1,
    #     )
    ax[row].plot(
        x_interp,
        np.nanmean(all_slopes[:, :, trigger_idx], axis=0)
        / np.max(np.nanmean(all_slopes[:, :, trigger_idx], axis=0)),
        color=CT_fff.colours[trigger_idx],
        linewidth=4,
    )
    ax[row].set_title(f"Trigger {trigger_idx}")
    # set y axis to log scale
ax[0].set_ylabel("Normalised rate")
ax[0].set_xlabel("Time (s)")
# remove ticks from subplots 1 - end
# for a in ax[1:]:
#     a.set_yticks([])
#     a.set_xticks([])
# remove spine
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.show()
# %%
contrast_results_split = contrast_df.collect().partition_by(["recording", "cell_index"])
x_interp = np.arange(window_neg, window_pos, 0.05)
all_slopes = np.zeros(
    (len(contrast_results_split), x_interp.shape[0], contrast_cum_triggers.shape[0] - 1)
)
all_scores = np.zeros((len(contrast_results_split), contrast_cum_triggers.shape[0] - 1))
for trigger_idx, trigger in enumerate(contrast_cum_triggers[:-1]):
    for e_idx, entry in enumerate(contrast_results_split):
        starts = entry.select(f"breaks_{trigger_idx}")[0].item().to_numpy()[:-1]
        ends = entry.select(f"breaks_{trigger_idx}")[0].item().to_numpy()[1:]
        rates = entry.select(f"slopes_{trigger_idx}")[0].item().to_numpy()
        all_scores[e_idx, trigger_idx] = entry.select(f"ssr_score_{trigger_idx}")[
            0
        ].item()

        # rates = rates * bin_freq / contrast_repeats
        segments = list(zip(starts, ends, rates))
        x_line, y_line = generate_line_data(starts, rates, ends - starts)
        y_interp = np.interp(x_interp, x_line, y_line)
        all_slopes[e_idx, :, trigger_idx] = y_interp
# %%
fig, ax = plt.subplots(
    nrows=1,
    ncols=(contrast_cum_triggers.shape[0] - 1) // 2,
    figsize=(40, 5),
    sharey=True,
    sharex=True,
)
row = 0
for trigger_idx, trigger in enumerate(contrast_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0 and trigger_idx != 0:
        row = row + 1
    # for e_idx, entry in enumerate(contrast_results_split):
    #     ax[trigger_idx].plot(
    #         x_interp,
    #         all_slopes[e_idx, :, trigger_idx],
    #         color=CT_contrast.colours[trigger_idx],
    #         alpha=0.1,
    #     )
    ax[row].plot(
        x_interp,
        np.nanmean(all_slopes[:, :, trigger_idx], axis=0)
        / np.max(np.nanmean(all_slopes[:, :, trigger_idx], axis=0)),
        color=CT_contrast.colours[trigger_idx],
        linewidth=4,
    )
    ax[row].set_title(f"Trigger {trigger_idx}")
    # set y axis to log scale
ax[0].set_ylabel("Normalised rate")
ax[0].set_xlabel("Time (s)")
# remove ticks from subplots 1 - end

# remove spine
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.show()
fig.show()
# %% plot slope before and after for all contrast steps
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharey=True)

for trigger_idx, trigger in enumerate(contrast_cum_triggers[:-1]):
    if np.mod(trigger_idx, 2) == 0:
        row = 0
    else:
        row = 1
    slope_before = contrast_df.select(f"slope_before{trigger_idx}").to_numpy().flatten()
    slope_after = contrast_df.select(f"slope_after{trigger_idx}").to_numpy().flatten()
    first_spike_posterior = (
        contrast_df.select(f"first_spike_posterior_{trigger_idx}").to_numpy().flatten()
    )
    first_spike_posterior[np.isnan(first_spike_posterior)] = 0
    slope_diff = slope_after  # - slope_before
    slope_diff[slope_diff > np.median(slope_diff) * 10] = 0
    slope_diff[slope_diff < -np.median(slope_diff) * 10] = 0
    slope_diff_mean = np.average(slope_diff, weights=first_spike_posterior)
    ax[row].bar(trigger_idx, slope_diff_mean, color=CT_contrast.colours[trigger_idx])
# light blue background for axes
ax[0].set_facecolor("lightblue")
ax[1].set_facecolor("lightblue")
ax[1].set_ylabel("Spike rate")
ax[1].set_xlabel("Contrast step")
fig.show()
# %%
recordings.dataframes["csteps"] = recordings.spikes_df.query(
    "stimulus_name =='csteps560' & qi > 0.3"
)
spikes = recordings.get_spikes_df("csteps")

# %%
psth, bins = histograms.psth(spikes, bin_size=2, end=42)
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].bar(np.arange(0, 10, 1), psth[::2] / 1005, color="black")
ax[1].bar(np.arange(0, 10, 1), psth[1::2] / 1005, color="black")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Spike count")
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.show()
