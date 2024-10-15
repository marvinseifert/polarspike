from polarspike import Overview, histograms, Opsins, colour_template, spiketrain_plots
import numpy as np
import plotly.graph_objects as go
from scipy.stats import zscore
from tslearn.matrix_profile import MatrixProfile
import matplotlib.pyplot as plt
from tslearn.metrics import dtw


def norm_01(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# %%
bin_size = 0.05
recording = Overview.Recording.load(
    r"B:\Marvin\chicken_02_13_2024_2nd\Phase_01\overview"
)
# %%
spikes = recording.get_spikes_triggered([[0]], [["all"]], pandas=False)
psth, bins = histograms.psth(spikes, end=24, bin_size=bin_size)
psth_norm = norm_01(psth)

# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(bins[:-1], psth, c="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Number of spikes")
fig.show()

# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(bins[:-1], psth_norm, c="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Z-score")
fig.show()

# %%
average_firing_rate = np.median(psth)
random_spikes = np.random.poisson(average_firing_rate, size=psth.shape[0])

specific_snippet = psth[int(4 / bin_size) : int(4.5 / bin_size)]
specific_snippet = norm_01(specific_snippet)

snippets = np.lib.stride_tricks.sliding_window_view(psth, specific_snippet.shape[0])


dtw_result = []
for snippet in snippets:
    snippet = norm_01(snippet)
    dtw_result.append(
        dtw(
            specific_snippet,
            snippet,
        )
    )

# Compute the Euclidean distances
# distances = np.linalg.norm(snippets - specific_snippet, axis=1)
distances = np.vstack(dtw_result)
# %%
fig, ax = plt.subplots(figsize=(20, 10))
# ax.plot(bins[:-1], psth_norm, c="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Z-score")
ax.plot(
    bins[int(specific_snippet.shape[0] / 2) : -int(specific_snippet.shape[0] / 2)],
    distances,
    c="blue",
)
# ax.hlines(np.std(norm_01(distances)), xmin=bins[40], xmax=bins[-10], color="red")
fig.show()

# %%
hist, edges = np.histogram(distances, bins=100)
fig, ax = plt.subplots()
ax.plot(edges[:-1], hist)
ax.set_xlabel("Distance")
ax.set_ylabel("Frequency")
fig.show()
