import pandas as pd
import spiketrain_plots
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
from sklearn.metrics import euclidean_distances, pairwise_distances
import matplotlib.pyplot as plt
from scipy.signal import correlate

# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
OT = Opsins.Opsin_template()

double_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_double")
single_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_oil")


# %%
recordings = Overview.Recording_s.load_from_single(
    r"A:\Marvin\fff_clustering",
    "oil_analysis",
    r"A:\Marvin\chicken_04_09_2024\Phase_00\overview",
)
# %%
recordings.add_from_saved(r"A:\Marvin\chicken_11_09_2024\Phase_00\overview")
recordings.add_from_saved(r"A:\Marvin\chicken_11_09_2024\Phase_00\overview_top")

# %%
recordings.dataframes["spikes_df"] = recordings.spikes_df.replace(
    "chicken_11_09_2024_p0_top", "chicken_11_09_2024_p0"
)
recordings.stimulus_df.loc[35, "recording"] = "chicken_11_09_2024_p0"
# %%

window = (
    0.5  # This is the window size in seconds over which the response will be summed
)
stimulus_name = "fff_combined"
use_opsin = True  # If contrast steps = False, if chromatic stimuli = True
colour_name = "FFF_6_MC"  # Which colour template to use
animal_name = ["Chicken"]
bin_size = 0.05  # Binsize for findpeaks method


# %%
# Get the stimulus id
fff_stimulus = recordings.stimulus_df.query(
    "stimulus_name == 'fff' | stimulus_name == 'fff_top'"
)
recordings.dataframes[f"{stimulus_name}_stimulus"] = fff_stimulus
mean_trigger = stimulus_spikes.mean_trigger_times(fff_stimulus, ["all"])


# %%
recordings.dataframes[stimulus_name] = recordings.spikes_df.query(
    "stimulus_name == 'fff' | stimulus_name == 'fff_top'"
)
recordings.dataframes[f"{stimulus_name}_filtered"] = recordings.dataframes[
    stimulus_name
].query("qi>0.3")

# %%
unique_index = (
    recordings.dataframes[f"{stimulus_name}_filtered"]
    .set_index(["recording", "cell_index"])
    .index.unique()
)
# %%
recordings.dataframes[f"{stimulus_name}_filtered"] = (
    recordings.dataframes[stimulus_name]
    .set_index(["recording", "cell_index"])
    .loc[unique_index]
    .reset_index()
)

# %%
spikes = recordings.get_spikes_df(
    f"{stimulus_name}_filtered", carry=["stimulus_name"], pandas=False
)
# %%
spikes = spikes.sort(["recording", "stimulus_name", "times_triggered"])
# %%
spikes_rec = spikes.partition_by(["recording", "stimulus_name"])

# %%
summed_spikes = np.empty(4, dtype=object)


for r_idx, recording in enumerate(spikes_rec):
    summed_spikes_temp = np.zeros((recording["cell_index"].unique().max() + 1, 12))
    summed_spikes_temp[:] = np.nan
    r_spikes, r_cell_indices = count_spikes.sum_spikes(
        recording, mean_trigger, window=window, group_by="cell_index"
    )
    r_spikes = r_spikes / np.nanmax(r_spikes, axis=1)[:, None]
    summed_spikes_temp[r_cell_indices.astype(int)] = r_spikes
    summed_spikes[r_idx] = summed_spikes_temp

# %%
all_zeros = ~np.logical_and(
    np.all(np.isnan(summed_spikes[3][:, ::2]), axis=1),
    np.all(np.isnan(summed_spikes[2][:, ::2]), axis=1),
)
diff_spikes = summed_spikes[3][all_zeros, ::2] - summed_spikes[2][all_zeros, ::2]
# %% Plot all cells
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(CT.wavelengths[::2], diff_spikes.T, c="k", alpha=0.1)
ax.set_ylabel("Response difference oil-no oil")
ax.set_xlabel("Wavelength (nm)")
# title
ax.set_title("Oil - No oil ON")
fig.show()
# %%
np.where(np.min(diff_spikes, axis=1) == -1)


# %%
summed_spikes
