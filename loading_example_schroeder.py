import numpy as np
from pathlib import Path
from polarspike import stimulus_dfs
from polarspike import Extractors
from polarspike import Overview
from polarspike import recording_overview

recording_name = "FG002_2022_10_25"

# Experiment 1:
stimulus_folder = Path(r"D:\Florencia_data\StimulusInfo_Experiment_1")
starts = np.load(stimulus_folder / "stimuli_start.npy")
ends = np.load(stimulus_folder / "stimuli_end.npy")
random_ids = np.load(stimulus_folder / "stimuli_id.npy")

# Add the stimulus to the stimulus_df
df_creator = stimulus_dfs.Stimulus_df_schroeder(
    recording_name="FG002_2022_10_25", sampling_freq=30000
)
df_creator.add_stimulus(
    "Gratings",
    starts=starts,
    ends=ends,
    stimulus_repeat_logic=24,
    stimulus_repeat_sublogic=1,
    random_ids=random_ids,
)

# Experiment 2:
stimulus_folder = Path(r"D:\Florencia_data\StimulusInfo_Experiment_2")
starts = np.load(stimulus_folder / "stimuli_start.npy")
ends = np.load(stimulus_folder / "stimuli_end.npy")
random_ids = np.load(stimulus_folder / "stimuli_id.npy")

# I wasn't sure if the stimulus_repeat_logic and stimulus_repeat_sublogic should be the same as in experiment 1
df_creator.add_stimulus(
    "stimulus_2",
    starts=starts,
    ends=ends,
    stimulus_repeat_logic=1,
    stimulus_repeat_sublogic=1,
    random_ids=random_ids,
)

# You can add as many experiments as you want
# Experiment 3:

# Now its time to load the spikes:
extr = Extractors.Extractor_KS(
    r"D:\Florencia_data\sorting\spike_times.npy", df_creator.stimulus_df
)
spikes = extr.get_spikes()
# Here we create the dataframe with the cells:
spikes_df = extr.load(recording_name=recording_name)

# Now we can create the recording object:
recording = Overview.Recording(
    extr.file_parquet,
    dataframes={"spikes_df": spikes_df, "stimulus_df": df_creator.stimulus_df},
    sampling_freq=df_creator.sampling_freq,
)

# Save the recording object:
recording.save(r"D:\Florencia_data\sorting\recording")

# Plot the recording overview:
fig, ax = recording_overview.spiketrains_from_file(
    recording.parquet_path, freq=recording.sampling_freq
)
fig = recording_overview.add_stimulus_df(fig, recording.stimulus_df)
fig.show()
