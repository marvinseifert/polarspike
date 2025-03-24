import numpy as np
from polarspike import Overview
from polarspike import histograms
from polarspike import Opsins
from polarspike import colour_template
from polarspike import stimulus_spikes
from polarspike import spiketrain_plots
from polarspike import stimulus_dfs
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from functools import reduce
import operator
import ruptures as rpt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from scipy.spatial.distance import euclidean, cdist
import plotly.graph_objects as go


# %%
def piecewise_linear_regression(x, y, min_points=10, plot=False):
    """
    Performs piecewise linear regression with two segments on the provided data.

    Parameters:
        x (array-like): 1D array of x-values.
        y (array-like): 1D array of y-values.
        min_points (int): Minimum number of points required in each segment.
        plot (bool): If True, plots the data and fitted piecewise linear function.

    Returns:
        breakpoint_x (float): x value at which the optimal breakpoint occurs.
        model1 (LinearRegression): Fitted linear model for the left segment.
        model2 (LinearRegression): Fitted linear model for the right segment.
        best_error (float): Total sum of squared errors for the optimal segmentation.
    """

    # Ensure numpy arrays and proper shapes
    x = np.array(x)
    y = np.array(y)
    X = x.reshape(-1, 1)  # sklearn expects a 2D array for predictors

    best_error = np.inf
    best_index = None
    best_model1 = None
    best_model2 = None

    # Loop over possible split indices, ensuring both segments have at least min_points.
    for i in range(min_points, len(x) - min_points):
        # Split data at index i: left segment uses indices [0, i), right uses [i, end)
        X1, y1 = X[:i], y[:i]
        X2, y2 = X[i:], y[i:]

        # Fit linear regression on the left segment
        model1 = LinearRegression()
        # create sample weights that increase towards the breakpoint
        sample_weights = np.logspace(0, 20, i)
        model1.fit(X1, y1, sample_weights)
        y1_pred = model1.predict(X1)
        error1 = np.sum((y1 - y1_pred) ** 2)

        # Fit linear regression on the right segment
        model2 = LinearRegression()
        # create sample weights that increase towards the breakpoint
        sample_weights = np.logspace(20, 0, len(X2))
        model2.fit(X2, y2, sample_weights)
        y2_pred = model2.predict(X2)
        error2 = np.sum((y2 - y2_pred) ** 2)

        total_error = error1 + error2

        # Check if this is the best split so far
        if total_error < best_error:
            best_error = total_error
            best_index = i
            best_model1 = model1
            best_model2 = model2

    # Compute the x-value corresponding to the best split index.
    breakpoint_x = x[best_index]

    # Optional plotting of the results
    if plot:
        y_fit = np.empty_like(y)
        y_fit[:best_index] = best_model1.predict(X[:best_index])
        y_fit[best_index:] = best_model2.predict(X[best_index:])

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color="gray", label="Data")
        plt.plot(x, y_fit, color="red", lw=2, label="Piecewise Linear Fit")
        plt.axvline(
            x=breakpoint_x,
            color="blue",
            linestyle="--",
            label=f"Breakpoint: {breakpoint_x:.2f}",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Piecewise Linear Regression (Two Segments)")
        plt.legend()
        plt.show()

    return breakpoint_x, best_model1, best_model2, best_error


# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

# %%
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

# %%
recordings.dataframes["sub_df"] = recordings.dataframes["fff_filtered"].sample(20)
spikes = recordings.get_spikes_df("sub_df", stimulus_df="fff_stim", pandas=False)


# %% plot the spiketrain
fig, ax = spiketrain_plots.whole_stimulus(spikes, indices=["cell_index", "repeat"])
# fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()

# %%
# find mean trigger times
mean_trigger_times = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, [1])
cum_triggers = np.cumsum(mean_trigger_times)
cum_triggers_extended = np.copy(cum_triggers)
# add another 9 trigger instances
for t in range(9):
    cum_triggers_extended = np.append(
        cum_triggers_extended, cum_triggers + cum_triggers[-1] * (t + 1)
    )
# %%
# Define some parameters
window_neg = -0.1
window_pos = 0.3
shuffle_window = 5
max_trigger = 12
# %%
first_spike_potential = []
# For each trigger time, I need to find the spike times which are > trigger time + window_neg and < trigger time + window_pos
spikes_lazy = spikes.lazy()
first_spike_potential.extend(
    spikes_lazy.filter(
        (pl.col("times_triggered") > trigger + window_neg)
        & (pl.col("times_triggered") < trigger + window_pos)
    ).with_columns(pl.col("times_triggered").sub(trigger).alias("fs_time"))
    for trigger in cum_triggers
)
# %% random shuffle
first_spike_random = []
n_rand = 1000
random_triggers = np.random.choice(cum_triggers, n_rand) + np.random.uniform(
    -shuffle_window, shuffle_window, n_rand
)

first_trigger_random = []
first_trigger_random.extend(
    spikes_lazy.filter(
        (pl.col("times_triggered") > trigger + window_neg)
        & (pl.col("times_triggered") < trigger + window_pos)
    ).with_columns(pl.col("times_triggered").sub(trigger).alias("fs_time"))
    for trigger in random_triggers
)

spikes_random = pl.concat(first_trigger_random)
spikes_random = spikes_random.collect()

# %%
spikes_filtered = pl.concat(first_spike_potential)
spikes_filtered = spikes_filtered.collect()
# %% plot all fs times
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(spikes_filtered["fs_time"], bins=1000, cumulative=True, density=True)
ax.hist(spikes_random["fs_time"], bins=1000, cumulative=True, alpha=0.5, density=True)
fig.show()

# %% plot all fs times without bining
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=spikes_filtered["fs_time"], y=np.ones(len(spikes_filtered)), mode="markers"
    )
)
fig.show(renderer="browser")

# %% ON vs OFF
on_triggers = np.arange(0, max_trigger, 2)
off_triggers = np.arange(1, max_trigger, 2)
bins = np.arange(window_neg, window_pos, 0.001)
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
ax[0].hist(
    spikes_filtered.filter(pl.col("trigger").is_in(on_triggers))["fs_time"],
    bins=bins,
    alpha=0.5,
    density=True,
    label="ON",
)
ax[0].hist(
    spikes_filtered.filter(pl.col("trigger").is_in(off_triggers))["fs_time"],
    bins=bins,
    alpha=0.5,
    density=True,
    label="OFF",
)
ax[1].hist(
    spikes_filtered.filter(pl.col("trigger").is_in(on_triggers))["fs_time"],
    bins=bins,
    alpha=0.5,
    density=True,
    label="ON",
    cumulative=True,
)
ax[1].hist(
    spikes_filtered.filter(pl.col("trigger").is_in(off_triggers))["fs_time"],
    bins=bins,
    alpha=0.5,
    density=True,
    label="OFF",
    cumulative=True,
)
# add subplot labels
ax[0].set_title("Density")
ax[1].set_title("Cumulative")
ax[0].legend()
fig.show()
# %%
cum_times, cum_bins = np.histogram(
    spikes_filtered["fs_time"], bins=np.arange(window_neg, window_pos, 0.001)
)
random_times, _ = np.histogram(spikes_random["fs_time"], bins=cum_bins)
# %%
cum_times = np.cumsum(cum_times) / cum_times.sum()
random_times = np.cumsum(random_times) / random_times.sum()

# %%
breakpoint_x, model1, model2, best_error = piecewise_linear_regression(
    cum_bins[:-1], cum_times, min_points=10, plot=True
)
# %% split bins according to breakpoint_x
bins_before = cum_bins[:-1][cum_bins[:-1] < breakpoint_x]
bins_after = cum_bins[:-1][cum_bins[:-1] >= breakpoint_x]
# %% calculate the mean squared error for the two segments
error1 = root_mean_squared_error(
    cum_times[cum_bins[:-1] < breakpoint_x],
    model1.predict(bins_before.reshape(-1, 1)),
)
error2 = root_mean_squared_error(
    cum_times[cum_bins[:-1] >= breakpoint_x],
    model2.predict(bins_after.reshape(-1, 1)),
)

# %% ON vs OFF analysis
cum_times_on, _ = np.histogram(
    spikes_filtered.filter(pl.col("trigger").is_in(on_triggers))["fs_time"],
    bins=np.arange(window_neg, window_pos, 0.001),
)
cum_times_off, _ = np.histogram(
    spikes_filtered.filter(pl.col("trigger").is_in(off_triggers))["fs_time"],
    bins=np.arange(window_neg, window_pos, 0.001),
)
cum_times_on = np.cumsum(cum_times_on) / cum_times_on.sum()
cum_times_off = np.cumsum(cum_times_off) / cum_times_off.sum()

breakpoint_x_on, model1_on, model2_on, best_error_on = piecewise_linear_regression(
    cum_bins[:-1], cum_times_on, min_points=10, plot=True
)
breakpoint_x_off, model1_off, model2_off, best_error_off = piecewise_linear_regression(
    cum_bins[:-1], cum_times_off, min_points=10, plot=True
)


# %% plot the two segments
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(bins_before, random_times[cum_bins[:-1] < breakpoint_x], label="real")
ax.plot(bins_before, model1.predict(bins_before.reshape(-1, 1)), label="fit")
ax.plot(bins_after, random_times[cum_bins[:-1] >= breakpoint_x], label="real")
ax.plot(bins_after, model2.predict(bins_after.reshape(-1, 1)), label="fit")
ax.legend()
fig.show()
# %%
euc = euclidean(cum_times, random_times)
max_distance = max(
    max(cum_times) - min(random_times), max(random_times) - min(cum_times)
)
argmax_distance = np.argmax(np.abs(cum_times - random_times))
cum_bins[argmax_distance]

# %% add breakpoint to spikes df
spikes = spikes.with_columns(
    pl.when(pl.col("times_relative") < bins_before[-1] + 4)
    .then(0)
    .otherwise(1)
    .alias("segment")
)
# %% Inter spike intervals
isi = spikes.with_columns(pl.col("times_relative").diff().alias("isi"))
isi = isi.filter(pl.col("isi") > 0)

# %% plot isi
bins = np.arange(0, 0.5, 0.001)
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(
    isi.filter(pl.col("segment") == 0)["isi"],
    bins=bins,
    alpha=0.5,
    density=True,
    label="before",
)
ax.hist(
    isi.filter(pl.col("segment") == 1)["isi"],
    bins=bins,
    alpha=0.5,
    density=True,
    label="after",
)
ax.legend()
fig.show()
# %%
import pwlf
from scipy.optimize import minimize

# %%
my_pwlf = pwlf.PiecewiseLinFit(cum_bins[:-1], cum_times)
res = my_pwlf.fitfast(4, pop=3)
x_new = np.linspace(cum_bins[0], cum_bins[-1], 100)
y_pred = my_pwlf.predict(x_new)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(cum_bins[:-1], cum_times, "o")
ax.plot(x_new, y_pred, "-")
fig.show()
# %%
# %%
from GPyOpt.methods import BayesianOptimization

# initialize piecewise linear fit with your x and y data
x = cum_bins[:-1]
y = cum_times
# %%

# Initialize variables to store the best results
best_score = np.inf
best_n_segments = None

# Loop over the range of possible segments
for n_segments in range(2, 10):
    # Create a new PiecewiseLinFit instance for each run (if needed)
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # Fit the model with the current number of segments
    breaks = my_pwlf.fitfast(n_segments)

    # Compute the penalized error
    ssr_score = my_pwlf.ssr

    score = np.sum(np.abs(np.log(np.diff(breaks))) * ssr_score)
    # if any(np.diff(breaks) < 0.05):
    #     score = np.inf
    print(f"Segments: {n_segments}, Score: {score}")

    # Check if this is the best score so far
    if score < best_score:
        best_score = score
        best_n_segments = n_segments

print("Best number of segments:", best_n_segments)
# %%
breaks = my_pwlf.fitfast(best_n_segments)
# find bins near the breaks
breaks_pos = np.array([np.argmin(np.abs(cum_bins[:-1] - b)) for b in breaks])

# %% plot the results
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x, y, "o")
ax.plot(x, my_pwlf.predict(x), "-")
ax.plot(breaks, cum_times[breaks_pos], "ro")
fig.show()
# %%

response_start = breaks[np.argmax(my_pwlf.slopes)]
print(response_start, my_pwlf.calc_slopes())
print(my_pwlf.ssr)
