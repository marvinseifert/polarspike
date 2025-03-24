import pandas as pd
import h5py
import numpy as np

# %%
fff_df = pd.read_pickle(r"A:\Marvin\psth_exports\fff_df.pkl")
fff_top_df = pd.read_pickle(r"A:\Marvin\psth_exports\fff_top_df.pkl")

# %% Save psths to hdf5
intersection = fff_df.index.intersection(fff_top_df.index)

fff_subset = fff_df.loc[intersection]
# %%
fff_subset.sort_index(inplace=True)
fff_top_df.sort_index(inplace=True)
# %% export to hdf5
for df, name in zip([fff_subset, fff_top_df], ["fff", "fff_top"]):
    with h5py.File(rf"A:\Marvin\psth_exports\{name}_psths.h5", "a") as f:
        f.create_dataset("psths", data=np.stack(df["psth"].values))
        f.create_dataset(
            "recording", data=np.array(df.index.get_level_values(0).values, dtype="S")
        )
        f.create_dataset("cell_index", data=df.index.get_level_values(1).values)
        f.create_dataset("qi", data=df["qi"].values)
