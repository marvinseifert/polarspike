from polarspike import (
    Overview,
    spiketrain_plots,
    moving_bars,
    quality_tests,
    stimulus_spikes,
    binarizer,
)
from bokeh.plotting import show, curdoc
import panel as pn
import matplotlib.pyplot as plt
import polars as pl
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.express as px


# %%
# Functions
def jitter_angle_within_bin(angle):
    """Randomly distribute an angle within its 45-degree bin."""
    jittered_angle = angle + np.random.uniform(-22.5, 22.5)

    if jittered_angle < 0:
        jittered_angle = jittered_angle + 360
    return jittered_angle


def bin_angle(angle):
    """Bins an angle into 45 degree intervals."""
    bin_centers = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    bin_edges = np.concatenate(
        [(bin_centers - 22.5) % 360, [337.5]]
    )  # Include the upper bound for the last bin
    bin_edges.sort()  # Make sure bin edges are sorted
    idx = np.digitize(angle, bin_edges) - 1
    idx = np.clip(idx, 0, len(bin_centers) - 1)  # Ensure index is within valid range
    return bin_centers[idx]


# %%
recording = Overview.Recording.load(r"A:\Marvin\chicken_04_09_2024\Phase_00\overview")


# %%

spikes = recording.get_spikes_triggered([[0]], [["all"]], pandas=False)


# # %%
# spikes = spikes.with_columns(recording=pl.lit(recording.name))
#
# # %%
# df_temp = recording.spikes_df.query("stimulus_index==0").set_index("cell_index")
# df_temp["qi"] = 0
# mean_trigger_times = stimulus_spikes.mean_trigger_times(recording.stimulus_df, [0])
# binary_df = binarizer.timestamps_to_binary_multi(
#     spikes,
#     0.001,
#     np.sum(mean_trigger_times),
#     10,
# )
# qis = binarizer.calc_qis(binary_df)
# cell_ids = binary_df["cell_index"].unique().to_numpy()
#
# df_temp.loc[cell_ids, "qi"] = qis

# %%
all_df = spikes.partition_by("cell_index")
MB = moving_bars.Moving_stats(frames_per_direction=300)

results = []
for df in all_df:
    results.append(MB.calc_circ_stats(df))

results_df = pl.concat(results).to_pandas()
# results_df["cluster_id"] = 0
# results_df["cluster_id"] = results_df.apply(update_cluster_id, axis=1)
# %%

spikes_single = recording.get_spikes_triggered([[0]], [[186]], pandas=False)
directions = np.array([0, 180, 270, 90, 315, 225, 45, 135])
directions = np.repeat(directions, 300)
trigger = spikes_single["trigger"].to_numpy()
directions_spikes = directions[trigger]

trigger = trigger * (0.5 / 60)

plt.ioff()
# Vectorize the jitter function for efficient application on the entire array
vectorized_jitter_angle = np.vectorize(jitter_angle_within_bin)

# Jitter the direction angles within their bins
jittered_directions = vectorized_jitter_angle(directions_spikes)

# Plot the jittered data
fig, axs = plt.subplots(
    1, 2, figsize=(15, 5), subplot_kw={"projection": "polar", "projection": "polar"}
)
axs[0].scatter(np.radians(jittered_directions), trigger, alpha=0.4, s=10)

# Set title and labels
axs[0].set_title("Jittered Spikes by Direction and time after direction begin")
axs[0].set_rlabel_position(-22.5)
axs[0].set_theta_zero_location("N")
axs[0].set_theta_direction(-1)

width = np.pi * 0.1

direction_counts = np.unique(directions_spikes, return_counts=True)
axs[1].bar(
    np.radians(direction_counts[0]),
    direction_counts[1],
    width=width,
    bottom=0.0,
    alpha=0.5,
)
axs[1].set_title("Spike counts per direction")
axs[1].set_rlabel_position(-22.5)
axs[1].set_theta_zero_location("N")
axs[1].set_theta_direction(-1)


fig.show()
# %%

fig, ax = spiketrain_plots.whole_stimulus(
    spikes_single.to_pandas(), indices=["cell_index", "repeat"]
)
fig.show()

# %%
results_df = pl.from_pandas(results_df)
results_cluster = results_df.group_by("cluster_id").agg(pl.col("p_val").mean())

# %%
cluster_sig = results_df  # .filter(pl.col("p_val") < 0.2)

fig = px.histogram(
    cluster_sig.to_pandas(),
    x="mean_deg",
    color="cluster_id",
    opacity=0.5,
    marginal="rug",
)
fig.update_layout(barmode="overlay")
fig.show(renderer="browser")


# %%
rf_df = pl.read_parquet(
    r"A:\Marvin\chicken_30_08_2024\Phase_00\noise_analysis\noise_df"
)
# %%
# results_df = pl.from_pandas(results_df)
results_df = results_df.with_columns(pl.col("cell_index").cast(pl.UInt32))
# %%
combined_df = rf_df.join(results_df, on=["cell_index"])

# %%
colors = ["grey", "red", "green", "blue"]
fig, ax = plt.subplots(figsize=(10, 5))
for idx, channel in enumerate(["white", "630", "560", "460"]):
    p_vals = combined_df.filter(pl.col("channel") == channel)["p_val"].to_numpy()
    rf_std = (
        combined_df.filter(pl.col("channel") == channel)["rf_std"].to_numpy().copy()
    )
    rf_std[np.isnan(rf_std)] = 0
    ax.scatter(rf_std, p_vals, color=colors[idx], label=channel, alpha=0.5)
    ax.scatter(np.nanmean(rf_std), np.nanmean(p_vals), color=colors[idx], s=100)
ax.legend()
fig.show()
