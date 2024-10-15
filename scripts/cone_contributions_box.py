import pandas as pd
import matplotlib.pyplot as plt
from polarspike import (
    Overview,
    histograms,
    Opsins,
    colour_template,
    stimulus_spikes,
    spiketrain_plots,
)
import numpy as np

# %%
feature_df = pd.read_pickle(r"A:\Marvin\fff_clustering\feature_df_corrected_coeffs")
# %%
lasso_columns = [
    "ON_Lasso_LWS",
    "ON_Lasso_MWS",
    "ON_Lasso_SWS2",
    "ON_Lasso_SWS1",
    "ON_Lasso_principal",
    "ON_Lasso_accessory",
    "OFF_Lasso_LWS",
    "OFF_Lasso_MWS",
    "OFF_Lasso_SWS2",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_principal",
    "OFF_Lasso_accessory",
]
cone_colours = [
    "#7c86fe",
    "#7cfcfe",
    "#8afe7c",
    "#fe7c7c",
    "orange",
    "darkorange",
]
# %% extract the cone contributions
cone_contributions = np.asarray(np.abs(feature_df[lasso_columns].copy()))
# %% box plot for on off lasso
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
for idx, cone in enumerate(lasso_columns[:6]):
    ax[0].boxplot(
        cone_contributions[:, idx],
        positions=[idx],
        patch_artist=True,
        boxprops=dict(facecolor=cone_colours[idx], color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )
    ax[0].set_xticks(range(len(lasso_columns[:6])))
    ax[0].set_xticklabels(lasso_columns[:6], rotation=45)
    ax[0].set_title("ON Lasso")
    ax[0].set_ylabel("Contribution")
    ax[0].set_xlabel("Cone type")
# %% box plot for on off lasso
for idx, cone in enumerate(lasso_columns[6:]):
    ax[1].boxplot(
        cone_contributions[:, idx],
        positions=[idx],
        patch_artist=True,
        boxprops=dict(facecolor=cone_colours[idx], color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )
    ax[1].set_xticks(range(len(lasso_columns[6:])))
    ax[1].set_xticklabels(lasso_columns[6:], rotation=45)
    ax[1].set_title("OFF Lasso")
    ax[1].set_ylabel("Contribution")
    ax[1].set_xlabel("Cone type")
# %%
fig.show()
