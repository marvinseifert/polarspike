import numpy as np
import matplotlib.pyplot as plt
from polarspike import colour_template, Overview, histograms, spiketrains
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from cdb_clustering.clustering import CompressionBasedDissimilarity

CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%
root = Path(r"A:\Marvin\fff_clustering")

recordings = Overview.Recording_s.load(root / "records")
spikes = recordings.get_spikes_df("fff_filtered", pandas=False)

# %%
spikes_arrays = spiketrains.collect_as_arrays(
    spikes, ["recording", "cell_index"], "times_triggered", "spikes"
)

# %%
spiketimes = np.stack(spikes_arrays["spikes"].to_numpy())

# %%
spike_list = []
for sp in spiketimes:
    spike_list.append(np.round(sp, 2))

# %%
psths, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["recording", "cell_index"],
    return_idx=True,
    window_end=24.1,
    bin_size=0.05,
)

# %%
psths_scaled = StandardScaler().fit_transform(psths)
# %%
clustering = CompressionBasedDissimilarity(
    spike_list, bins=np.linspace(0, 24.1, 4200), compression_factor=20
)
clustering.pair_wise_distances()
# %%
fig, ax = clustering.dendrogram()
fig.show()
# %%
labels = clustering.get_clusters(cut=1)
print(np.max(labels))
# plot means
# %%
fig, ax = plt.subplots(np.max(labels), 1, figsize=(30, 15))
for i in range(np.max(labels)):
    ax[i].plot(bins[:-1], psths[labels == i].mean(axis=0))
fig.show()
