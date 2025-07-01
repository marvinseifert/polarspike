from polarspike import (
    Overview,
    spiketrain_plots,
    colour_template,
    histograms,
)
import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%
recording = Overview.Recording.load(r"A:\Marvin\chicken_30_08_2024\Phase_00\overview")

spikes = recording.get_spikes_triggered([[27]], [["all"]])

# %%
recording_top = Overview.Recording.load(
    r"A:\Marvin\chicken_05_09_2024\Phase_00\overview_top"
)
spikes_top = recording_top.get_spikes_triggered([[2]], [["all"]])
spikes_top = spikes_top.query("repeat<8")
# %%


fig, ax = spiketrain_plots.whole_stimulus(
    spikes,
    indices=["stimulus_index", "cell_index"],
    cmap=["Greys", "Reds"],
    single_psth=False,
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)

fig.show()

# %%

fig, ax = spiketrain_plots.whole_stimulus(
    spikes_top,
    indices=["stimulus_index", "repeat"],
    cmap=["Greys"],
    single_psth=False,
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)

fig.show()

# %%
psth, bins, cell_index = histograms.psth_by_index(
    spikes, index=["cell_index"], return_idx=True, window_end=24
)

psth_top, bins, cell_index_top = histograms.psth_by_index(
    spikes_top, index=["cell_index"], return_idx=True, window_end=24
)

# %%
all_cells = np.zeros((recording.nr_cells, (len(bins) - 1) * 2))
all_cells[cell_index.astype(int).flatten(), : len(bins) - 1] = psth
all_cells[cell_index_top.astype(int).flatten(), len(bins) - 1 :] = psth_top

# %% scaling
scaler = StandardScaler()
all_cells_scaled = scaler.fit_transform(all_cells)

# %%
clustering = AffinityPropagation(random_state=5, max_iter=1000).fit(all_cells)
labels = clustering.labels_
print(np.max(labels) + 1)

# %%
clustering = AgglomerativeClustering(n_clusters=20, linkage="complete")
labels = clustering.fit_predict(all_cells_scaled)
# %% Plot cluster means
fig, axs = plt.subplots(nrows=21, ncols=1, figsize=(10, 10), sharex=True)
for cluster in range(20):
    mean = np.mean(all_cells[labels == cluster], axis=0)
    axs[cluster].plot(bins[:-1], mean[: len(bins) - 1], label=f"Cluster {cluster}")
    axs[cluster].plot(bins[:-1], mean[len(bins) - 1 :], label=f"Cluster {cluster}")
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()
