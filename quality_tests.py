from polarspike import binarizer
from polarspike import stimulus_spikes
import polars as pl
import numpy as np
import pandas as pd


def spiketrain_qi(spikes, max_window=1, bin_size=0.001, max_repeat=24):
    if len(spikes["recording"].unique()) > 1:
        spikes_by_recording = spikes.partition_by("recording")
    else:
        spikes_by_recording = [spikes]
    rec_result = []
    for recording in spikes_by_recording:
        binary_df = binarizer.timestamps_to_binary_multi(
            recording, max_window=max_window, bin_size=0.001, max_repeat=max_repeat
        )
        cell_indices = binary_df["cell_index"].unique().to_numpy()
        qis = binarizer.calc_qis(binary_df)
        result_df = pd.DataFrame(
            data=qis,
            columns=["qi"],
            index=pd.Index(data=cell_indices, name="cell_index"),
        )
        result_df["recording"] = recording["recording"][0]
        result_df = result_df.set_index("recording", append=True)
        rec_result.append(result_df)
    return pd.concat(rec_result)
