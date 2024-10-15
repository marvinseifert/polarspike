from polarspike import Overview, histograms, colour_template, spiketrain_plots
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly_templates
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
from quality_tests import spiketrain_qi


# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

index = (
    recordings.dataframes["fff_filtered"].set_index(["recording", "cell_index"]).index
)
# %%
cstep_names = [
    "csteps",
    "csteps610",
    "csteps535",
    "csteps460",
    "csteps413",
    "csteps365",
]

recordings.dataframes["csteps"] = recordings.spikes_df.query(
    "stimulus_name in @cstep_names"
).copy()

# %%
intersect_index = index.intersection(
    recordings.dataframes["csteps"].set_index(["recording", "cell_index"]).index
)
recordings.dataframes["csteps"] = (
    recordings.dataframes["csteps"]
    .set_index(["recording", "cell_index"])
    .loc[intersect_index]
    .reset_index()
)

# %%
spikes = recordings.get_spikes_df("csteps", carry=["stimulus_name", "qi"])
# %%
dfs = []
psths = []
for cstep in cstep_names:
    spikes_sub = spikes.query(f"stimulus_name == '{cstep}'")
    psth, bins, cell_index = histograms.psth_by_index(
        spikes_sub,
        index=["recording", "cell_index"],
        return_idx=True,
        window_end=40.1,
        bin_size=0.1,
    )
    psths.append(psth)
    index = pd.MultiIndex.from_arrays([cell_index[:, 0], cell_index[:, 1]])
    quality = (
        recordings.dataframes["csteps"]
        .set_index(["recording", "cell_index"])
        .loc[index]
    )
    quality = quality.query("stimulus_name == @cstep")["qi"].values
    epoch_lenght = 20
    epoch_max = np.zeros((psth.shape[0], 20))
    epoch_argmax = np.zeros_like(epoch_max)
    epoch_sparsity = np.zeros_like(epoch_max)
    epoch_mean_firing_rate = np.zeros_like(epoch_max)
    epoch_medium_firing_rate = np.zeros_like(epoch_max)
    epoch_sum = np.zeros_like(epoch_max)
    epoch_sustained = np.zeros_like(epoch_max)

    for i in range(20):
        epoch_max[:, i] = np.max(
            psth[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
        )
        epoch_max[:, i] = epoch_max[:, i] * quality
        epoch_argmax[:, i] = np.argmax(
            psth[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
        )
        epoch_sparsity[:, i] = (
            np.sum(psth[:, i * epoch_lenght : (i + 1) * epoch_lenght] > 0, axis=1)
            / epoch_lenght
        )
        # epoch_mean_firing_rate[:, i] = np.mean(
        #     psth_z[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
        # )
        epoch_medium_firing_rate[:, i] = np.median(
            psth[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
        )
        epoch_sum[:, i] = np.sum(
            psth[:, i * epoch_lenght : (i + 1) * epoch_lenght], axis=1
        )
        transient_sum = np.sum(
            psth[:, i * epoch_lenght : (i + 2) * epoch_lenght], axis=1
        )
        sustained_sum = np.sum(
            psth[:, i * epoch_lenght + 2 : (i + 18) * epoch_lenght], axis=1
        )
        epoch_sustained[:, i] = (transient_sum - sustained_sum) / (
            transient_sum + sustained_sum
        )

    epoch_sustained[np.isnan(epoch_sustained)] = 0
    temp_df = pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [np.repeat(cell_index[:, 0], 20), np.repeat(cell_index[:, 1], 20)],
            names=["recording", "cell_index"],
        )
    )
    temp_df["maxima"] = epoch_max.flatten()
    temp_df["argmax"] = epoch_argmax.flatten()
    temp_df["sparsity"] = epoch_sparsity.flatten()
    temp_df["medium_firing_rate"] = epoch_medium_firing_rate.flatten()
    temp_df["sum"] = epoch_sum.flatten()
    temp_df["sustained"] = epoch_sustained.flatten()
    temp_df["step"] = ["ON", "OFF"] * 10 * len(cell_index)
    temp_df["stimulus_name"] = cstep
    temp_df["contrast"] = [
        1,
        1,
        0.9,
        0.9,
        0.8,
        0.8,
        0.7,
        0.7,
        0.6,
        0.6,
        0.5,
        0.5,
        0.4,
        0.4,
        0.3,
        0.3,
        0.2,
        0.2,
        0.1,
        0.1,
    ] * len(cell_index)
    dfs.append(temp_df)

# %%
results_df = pd.concat(dfs)
# %%
fig = px.histogram(
    results_df,
    x="contrast",
    y="maxima",
    color="stimulus_name",
    color_discrete_sequence=["gray"] + CT.colours[[0, 4, 6, 8, 10]].tolist(),
    facet_col="step",
    facet_row="stimulus_name",
    nbins=20,
)
fig.update_layout(template="scatter_template", width=600)
fig.show(renderer="browser")

# %%
contrasts = np.linspace(0.1, 1, 10)[::-1]
fig, ax = plt.subplots(6, 2, figsize=(6, 10), sharex=True, sharey=True)
for stim_idx, stimulus in enumerate(cstep_names):
    for step_idx, step in enumerate(["ON", "OFF"]):
        temp_df = results_df.query(f"stimulus_name == '{stimulus}' & step == '{step}'")
        # melt df to get contrast as x axis
        temp_df = temp_df.melt(id_vars=["contrast"], value_vars=["maxima"])
        # extract maxima for each contrast value
        maxima = temp_df["value"].values.reshape((len(temp_df) // 10, 10))
        maxima_mean = np.nanmean(maxima, axis=0)
        maxima_mean = maxima_mean / np.max(maxima_mean)

        ransac = linear_model.RANSACRegressor()
        ransac.fit(contrasts.reshape(-1, 1), maxima_mean)

        x = np.linspace(0.1, 1, 100)
        y = ransac.predict(x.reshape(-1, 1))
        ax[stim_idx, step_idx].plot(contrasts, maxima_mean, "o")
        ax[stim_idx, step_idx].plot(x, y, c="r")
        ax[stim_idx, step_idx].set_title(f"{stimulus} {step}")
# remove spines
for axes in ax.flatten():
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_xlim(0, 1.2)
    axes.set_ylim(0, 1.2)

fig.show()
# %%
CT.pick_stimulus("FFF_6_MC")
colours = ["gray"] + CT.colours[[0, 4, 6, 8, 10]].tolist()

fig, ax = plt.subplots(7, 1, figsize=(20, 30), sharex=True)
for c_idx, c in enumerate(psths):
    ax[c_idx].plot(bins[:-1], c.mean(axis=0) / np.max(c.mean(axis=0)), c=colours[c_idx])
CT.pick_stimulus("Contrast_Step")
fig = CT.add_stimulus_to_plot(fig, [2] * 20)
fig.show()
# %%
cmaps = ["Greys", "Reds", "Greens", "Blues", "Purples", "Purples"]
figs = []
CT.pick_stimulus("Contrast_Step")
for idx, cstep in enumerate(cstep_names):
    print(idx)
    spikes_sub = spikes.query(f"stimulus_name == '{cstep}'&qi>0.2").copy()
    # add unique index per recording and cell_index
    spikes_sub["unique"] = spikes_sub.groupby(["recording", "cell_index"]).ngroup()
    fig, ax = spiketrain_plots.whole_stimulus(
        spikes_sub,
        indices=["stimulus_name", "unique"],
        width=20,
        bin_size=0.05,
        cmap=cmaps[idx],
        norm="eq_hist",
    )
    fig = CT.add_stimulus_to_plot(fig, [2] * 20)
    figs.append(fig)

# %%
for i, fig in enumerate(figs):
    fig.savefig(rf"A:\Marvin\csteps\{cstep_names[i]}_quality_filtered", dpi=300)
