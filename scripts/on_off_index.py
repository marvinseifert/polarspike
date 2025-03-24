from polarspike import (
    Overview,
    histograms,
    Opsins,
    colour_template,
    stimulus_spikes,
)
from polarspike.analysis import response_peaks, count_spikes
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# %%
# Input parameters
recording = Overview.Recording.load(r"B:\Marvin\Zebrafish_04_23_2024\Phase_01\overview")
window = (
    0.2  # This is the window size in seconds over which the response will be summed
)
stimulus_id = 16
use_opsin = True  # If contrast steps = False, if chromatic stimuli = True
colour_name = "FFF_6_MC"  # Which colour template to use
animal_name = ["Chicken"]
bin_size = 0.05  # Binsize for findpeaks method


mean_trigger = stimulus_spikes.mean_trigger_times(recording.stimulus_df, [stimulus_id])

spikes = recording.get_spikes_triggered([[stimulus_id]], [["all"]], pandas=False)

# %% Sum method
spikes_summed, cell_indices = count_spikes.sum_spikes(
    spikes, mean_trigger, window=window, group_by="cell_index"
)


# %%
# Sum On
spikes_all_on = np.sum(spikes_summed[:, :2], axis=1)
spikes_all_off = np.sum(spikes_summed[:, 1::2], axis=1)

# Calculate polarity index
on_off_index = (spikes_all_on - spikes_all_off) / (spikes_all_on + spikes_all_off)

# %%
new_df = pd.DataFrame(
    index=pd.Index(cell_indices, name="cell_index"),
    data=on_off_index,
    columns=["on_off_index"],
)
temp_df = recording.dataframes["tuning_df"].query("stimulus_index == @stimulus_id")
temp_df = temp_df.set_index("cell_index")
temp_df.update(new_df)
temp_df = temp_df.reset_index()
recording.dataframes["tuning_df"] = recording.dataframes["tuning_df"].set_index(
    ["cell_index", "stimulus_index"]
)
recording.dataframes["tuning_df"].update(
    temp_df.set_index(["cell_index", "stimulus_index"])
)
recording.dataframes["tuning_df"] = recording.dataframes["tuning_df"].reset_index()
recording.save_save()
