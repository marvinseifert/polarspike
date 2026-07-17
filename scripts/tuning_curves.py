"""
This script is used to calculate the tuning curves for responses. Two different methods are used:
1. Summing the responses over a window of time.
2. Finding the peaks in the response.

The script will calculate the correlation coefficient between the tuning curves and a diagonal line which indicates
the ideal linear response.

Differences between the methods:
- The first method is more sensitive to the overall response of the cell. Particularly, if cells have sustained responses
this method might more accurately reflect the response.
- The second method is more sensitive to the peaks in the response. Particularly, if cells have transient responses
this method might more accurately reflect the response.

"""
from polarspike import (
    Overview,
    histograms,
    Opsins,
    colour_template,
    spike_loader,
    spiketrain_plots,
)
from polarspike.analysis import response_peaks, count_spikes
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Input parameters
# recording = Overview.Recording.load(
#     r"B:\Marvin\chicken_02_13_2024_2nd\Phase_01\overview"
# )
window = 1  # This is the window size in seconds over which the response will be summed
stimulus_id = 0
use_opsin = True  # If contrast steps = False, if chromatic stimuli = True
colour_name = "FFF_8_MC"  # Which colour template to use
animal_name = ["Chicken"]
bin_size = 0.05  # Binsize for findpeaks method

# %%
recordings = Overview.Recording_s(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/combined_analysis",
    "fff_analysis",
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_11_02_2026/Phase_01/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_12_02_2026/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_13_11_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_17_11_2025/Phase_00/overview"
)
# recordings.add_from_saved(
#     r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_18_11_2025/Phase_00/overview"
# )
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_01_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_05_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_07_05_2025/Phase_00/overview"
)
recordings.add_from_saved(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_24.local,share=root/Marvin/chicken_14_05_2025/Phase_00/overview"
)

# %%
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
# %%
fff_ids = recordings.stimulus_df.query("stimulus_name == 'fff'").index.to_list()
# # %%
mean_trigger = spike_loader.mean_trigger_times(recordings.stimulus_df, fff_ids)

# %%
spikes = recordings.get_spikes_triggered(
    [{"stimulus_name": ["scf_13nd", "scf(1.3ND)"]}],
    pandas=False,
    carry=["stimulus_name", "qi"],
)
spikes = spikes.filter(pl.col("qi") >= 0.3)

# %%
psth, bins = histograms.psth(spikes)
# %%
fig, ax = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [10, 1]}, sharex=True)
ax[0].plot(bins[:-1], psth)
fig = ct.add_stimulus_to_plot(fig, [2] * 16)
fig.show()
# %%
spikes_summed, cell_indices = count_spikes.sum_spikes(
    spikes,
    mean_trigger,
    window=window,
    group_by=["recording", "cell_index", "repeat"],
)
# %% transfer to polars dataframe
summed_df = pl.DataFrame(
    {
        "recordings": cell_indices[:, 0].tolist(),
        "cell_index": cell_indices[:, 1].astype(int),
        "repeat": cell_indices[:, 2].astype(int),
        "spikes_summed": spikes_summed,
    },
    schema={
        "recordings": pl.String,
        "cell_index": pl.UInt32,
        "repeat": pl.UInt8,
        "spikes_summed": pl.List,
    },
)
unique_pairs = summed_df.select(["recordings", "cell_index"]).unique()
all_repeats = pl.DataFrame({"repeat": pl.Series(range(5), dtype=pl.UInt8)})

# Cross join to get all expected combinations
expected = unique_pairs.join(all_repeats, how="cross")

# Left join back to the actual data, filling missing with zeros
zero_array = [0.0] * 16

summed_df = (
    expected.join(summed_df, on=["recordings", "cell_index", "repeat"], how="left")
    .sort(["recordings", "cell_index", "repeat"])
    .with_columns(
        pl.col("spikes_summed").fill_null(
            pl.lit(zero_array, dtype=pl.Array(pl.Float64, 16))
        )
    )
)
spikes_per_cell = (
    spikes.group_by(["recording", "cell_index", "repeat"], maintain_order=True)
    .agg(pl.col("times_triggered").count().alias("spike_counts"))
    .group_by(pl.col(["recording", "cell_index"]), maintain_order=True)
    .agg(pl.col("spike_counts").mean().alias("mean_spike_count"))
    .with_columns(pl.col("mean_spike_count") / (1 / window * np.sum(mean_trigger)))
)
spikes_per_cell = spikes_per_cell.rename({"recording": "recordings"})
spikes_summed = summed_df.join(spikes_per_cell, on=["recordings", "cell_index"])
spikes_summed_norm = np.mean(np.stack(spikes_summed["spikes_summed"]), axis=0)
spikes_summed_norm = (spikes_summed_norm / np.max(spikes_summed_norm))[np.newaxis, :]
# %% Findpeaks method

all_heights, peak_locations = response_peaks.find_peaks(
    spikes, mean_trigger, bin_size=bin_size
)

spikes_heights_norm = all_heights / np.max(all_heights)
spikes_heights_norm = np.atleast_2d(spikes_heights_norm).T
# %% Plot the results
CT = colour_template.Colour_template()
CT.pick_stimulus(colour_name)
if use_opsin:
    Ops = Opsins.Opsin_template()
    fig = Ops.plot_overview(animal_name)
else:
    fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[::2],
        y=spikes_summed_norm[0, ::2],
        mode="lines",
        line=dict(color="grey"),
        name="Summed spikes ON",
    )
)
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[1::2],
        y=spikes_summed_norm[0, 1::2],
        mode="lines",
        line=dict(color="black"),
        name="Summed spikes OFF",
    )
)
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[::2],
        y=spikes_heights_norm[::2].flatten(),
        mode="lines",
        line=dict(color="grey", dash="dash"),
        name="Peaks ON",
    )
)
fig.add_trace(
    go.Scatter(
        x=CT.wavelengths[1::2],
        y=spikes_heights_norm[1::2].flatten(),
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Peaks OFF",
    )
)
# Add diagonal line
fig.add_trace(
    go.Scatter(
        x=[CT.wavelengths[-1], CT.wavelengths[0]],
        y=[0, 1],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Diagonal",
    )
)
fig.update_layout(height=700, width=1400)
fig.show(renderer="browser")
# %% plot this with the oil droplet sensitivity curves
double_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_double")
single_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_oil")
combined_df = pd.concat([double_ab_df, single_ab_df])
combined_df = combined_df.sort_values(["cone", "wavelength"])
combined_df["absorption"] = combined_df["absorption"] * np.e

# %% Or get animal specific opsins
Ops = Opsins.Opsin_template()
templates = Ops.govardovskii_animal("Zebrafish").T
# this returns an array 400 x 4 for wavelength 300-700 nm and 4 opsins need to export it to a dataframe with
# one column for wavelength and one column for absorption and one column with the cone names
cone_names = ["LWS"] * 400 + ["MWS"] * 400 + ["SWS2"] * 400 + ["SWS1"] * 400
combined_df = pd.DataFrame(
    data={
        "wavelength": np.tile(np.arange(300, 700, 1), 4),
        "absorption": templates.flatten(),
        "cone": cone_names,
    }
)
# %% log transform the absorption
combined_df = combined_df.sort_values(["cone", "wavelength"])
combined_df["absorption"] = combined_df["absorption"] * np.e
combined_df.loc[combined_df["absorption"] < 0.05, "absorption"] = 0.05
combined_df.loc[combined_df["absorption"] == np.NaN, "absorption"] = 0.05
combined_df["absorption"] = np.log(combined_df["absorption"])
# combined_df.loc[combined_df["absorption"] <= np.log(10e-10), "absorption"] = -np.inf
# renormalize to range 0-1
combined_df["absorption"] = (
                                    combined_df["absorption"] - np.nanmin(combined_df["absorption"])
                            ) / (np.nanmax(combined_df["absorption"]) - np.nanmin(combined_df["absorption"]))

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_8_MC")
colours = CT.colours[::2]
cone_colours = [
    colours[0],
    colours[2],
    colours[4],
    colours[5],
    "darkgoldenrod",
    "orange",
    "saddlebrown",
]
cone_color_dict = {
    "LWS": colours[0],
    "MWS": colours[3],
    "SWS2": colours[6],
    "SWS1": colours[7],
    "principal": "orange",
    "accessory": "darkgoldenrod",
}

# %%
fig, ax = plt.subplots(nrows=1, figsize=(20, 10))
for idx, cone in enumerate(combined_df["cone"].unique()):
    cone_df = combined_df.query(f"cone == '{cone}'")
    ax.fill_between(
        cone_df["wavelength"],
        cone_df["absorption"],
        color=cone_color_dict[cone],
        label=cone,
        linewidth=3,
        alpha=0.2,
    )
    # add summed spikes
ax.plot(
    CT.wavelengths[::2],
    spikes_summed_norm[0, ::2] / np.max(spikes_summed_norm[0, ::2]),
    color="grey",
    linestyle="--",
    label="Summed spikes ON",
    linewidth=3,
)
ax.plot(
    CT.wavelengths[1::2],
    spikes_summed_norm[0, 1::2] / np.max(spikes_summed_norm[0, 1::2]),
    color="black",
    linestyle="--",
    label="Summed spikes OFF",
    linewidth=3,
)
# remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# set axis labels
ax.set_xlabel("Wavelength (nm)", fontsize=20)
ax.set_ylabel("Normalized response", fontsize=20)
# increase tick marker size to 14
ax.tick_params(axis="both", which="major", labelsize=14)

fig.legend()
fig.show()
# %%
spikes_grouped = recordings.get_spikes_df("fff_filtered", carry=["stimulus_name"])
spikes_grouped["unique"] = spikes_grouped.groupby(["recording", "cell_index"]).ngroup()
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_grouped, indices=["stimulus_name", "unique"], height=15, width=25
)
fig = CT.add_stimulus_to_plot(fig, [2] * 12)
fig.show()
# save figure as vector graphic
# %%
fig.savefig(
    r"C:\Users\Marvin\Documents\ISER_presentation\spiketrain_fff_filtered_zf.svg",
    format="svg",
)
