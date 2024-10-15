from polarspike import Overview, histograms
import polars as pl
import numpy as np
import pandas as pd

# %%
if __name__ == "__main__":
    stimulus_name = "fff_top"
    bin_size = 0.05
    window_end = 24.1
    recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

    recordings.dataframes["sub_df"] = recordings.dataframes["spikes_df"].query(
        "stimulus_name == @stimulus_name"
    )

    spikes = recordings.get_spikes_df("sub_df", pandas=False, carry=["qi"])

    spikes_rec = spikes.partition_by("recording")
    # %%
    psths = []
    rec_indices = []
    cell_indices = []
    qis = []
    for recording in spikes_rec:
        psth, bins, cell_idx = histograms.psth_by_index(
            recording,
            index=["cell_index", "qi"],
            window_end=window_end,
            return_idx=True,
        )
        psths.append(psth)
        rec_indices = rec_indices + [recording["recording"][0]] * len(cell_idx)
        cell_indices = cell_indices + cell_idx[:, 0].flatten().tolist()
        qis = qis + cell_idx[:, 1].flatten().tolist()

    # %%
    psths = np.concatenate(psths, axis=0)
    psths_list = [psths[i, :] for i in range(psths.shape[0])]

    # %%
    store_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(zip(rec_indices, cell_indices))
    )
    store_df["psth"] = psths_list
    store_df["qi"] = qis

    # %% save
    store_df.to_pickle(rf"A:\Marvin\psth_exports\{stimulus_name}_df.pkl")
