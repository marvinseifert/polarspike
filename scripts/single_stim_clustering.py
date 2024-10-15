from polarspike import (
    Overview,
    spiketrain_plots,
    moving_bars,
    quality_tests,
    stimulus_spikes,
    binarizer,
    colour_template,
    histograms,
)
from bokeh.io import show
import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans

stimulus_name = "csteps"
max_time = 40
repeats = 10
# %%

CT = colour_template.Colour_template()
CT.pick_stimulus("Contrast_Step")
# %%
recordings = Overview.Recording_s.load_from_single(
    r"D:\fff_clustering",
    "chicken_fff",
    "" r"A:\Marvin\chicken_30_08_2024\Phase_00\overview",
)
recordings.add_from_saved(r"A:\Marvin\chicken_04_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_05_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_09_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_17_07_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_18_07_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_19_07_2024\Phase_00\overview")
# %%

recordings.dataframes[stimulus_name] = recordings.spikes_df.query(
    "stimulus_name == @stimulus_name"
)
# %%
spikes = recordings.get_spikes_df(stimulus_name)

# %%
qis = quality_tests.spiketrain_qi(spikes, max_window=max_time, max_repeat=repeats)
# %%
recordings.dataframes[stimulus_name] = recordings.dataframes[stimulus_name].set_index(
    ["cell_index", "recording"]
)
# %%
recordings.dataframes[stimulus_name].update(qis)
recordings.dataframes[stimulus_name] = recordings.dataframes[
    stimulus_name
].reset_index()
# %% plot qi as histogram
fig = px.histogram(
    recordings.dataframes[stimulus_name], x="qi", nbins=500, color="recording"
)
fig.show(renderer="browser")

# %%
recordings.dataframes[f"{stimulus_name}_filtered"] = recordings.dataframes[
    stimulus_name
].query("qi>0.5")
# %%
spikes = recordings.get_spikes_df(f"{stimulus_name}_filtered")
# %%
psth, bins, cell_index = histograms.psth_by_index(
    spikes, index=["recording", "cell_index"], return_idx=True, window_end=max_time
)
# %% Clustering
scaler = StandardScaler()
psth_scaled = scaler.fit_transform(psth)

clustering = AffinityPropagation(random_state=5, max_iter=1000).fit(psth)
labels = clustering.labels_
print(np.max(labels) + 1)

# %%
clustering = AgglomerativeClustering(n_clusters=20, linkage="complete")
labels = clustering.fit_predict(psth)
print(np.max(labels) + 1)

# %%
sdtw_km = TimeSeriesKMeans(
    n_clusters=15,
    metric="dtw",
    # metric_params={"gamma": 0.01},
    verbose=True,
    random_state=5,
    n_jobs=-1,
)
y_pred = sdtw_km.fit_predict(psth)
# %%
labels = sdtw_km.labels_
print(np.max(labels) + 1)
# %%
fig, ax = plt.subplots(nrows=np.max(labels) + 2, ncols=1, figsize=(10, 20), sharex=True)
for cluster in range(np.max(labels) + 1):
    mean_psth = np.mean(psth[labels == cluster], axis=0)
    ax[cluster].plot(bins[:-1], mean_psth, c="k")
    ax[cluster].set_title(
        f"Cluster {cluster}, n={np.sum(labels == cluster)}", position=(1, 0.8)
    )
fig = CT.add_stimulus_to_plot(fig, [2] * 20)
fig.show()
