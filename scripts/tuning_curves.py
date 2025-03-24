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
    stimulus_spikes,
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
window = (
    0.5  # This is the window size in seconds over which the response will be summed
)
stimulus_id = 0
use_opsin = True  # If contrast steps = False, if chromatic stimuli = True
colour_name = "FFF_6_MC"  # Which colour template to use
animal_name = ["Zebrafish"]
bin_size = 0.05  # Binsize for findpeaks method

# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering_zf\records")

# %%
fff_ids = recordings.stimulus_df.query("stimulus_name == 'fff'").index.to_list()
# # %%
mean_trigger = stimulus_spikes.mean_trigger_times(recordings.stimulus_df, fff_ids)

# %%
spikes = recordings.get_spikes_df("fff_filtered", pandas=False)


# %%
# Create a list of possible windows
spikes_summed = count_spikes.sum_spikes(
    spikes, mean_trigger, window=window, group_by="stimulus_index"
)[0]

spikes_summed = np.mean(spikes_summed, axis=0)
# %%
# Normalize and plotting
spikes_summed_norm = spikes_summed / np.max(spikes_summed)
spikes_summed_norm = np.atleast_2d(spikes_summed_norm)

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
fig.update_layout(height=700, width=700)
fig.show(renderer="browser")
# %% plot this with the oil droplet sensitivity curves
double_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_double")
single_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_oil")
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
CT.pick_stimulus("FFF_6_MC")
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
    "MWS": colours[2],
    "SWS2": colours[4],
    "SWS1": colours[5],
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
