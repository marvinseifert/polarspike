import pandas as pd
import matplotlib.pyplot as plt

sys.path.extend(["/home/marvin-seifert/PycharmProjects"])
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
feature_df = pd.read_pickle(
    r"/home/marvin-seifert/Marvin/fff_clustering_zf/features_coeffs_pca_clustered"
)
# %%
lasso_columns = [
    "ON_Lasso_SWS1",
    "ON_Lasso_SWS2",
    "ON_Lasso_MWS",
    "ON_Lasso_LWS",
    # "ON_Lasso_principal",
    # "ON_Lasso_accessory",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_SWS2",
    "OFF_Lasso_MWS",
    "OFF_Lasso_LWS",
    # "OFF_Lasso_principal",
    # "OFF_Lasso_accessory",
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
# %%
all_contributions = np.mean(np.abs(cone_contributions), axis=0)
# peak normalize the contributions
all_contributions = all_contributions / np.max(all_contributions)
# %% Barplots
nr_cones = len(lasso_columns) // 2
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
for idx, cone in enumerate(lasso_columns[:nr_cones]):
    ax[0].bar(
        idx,
        all_contributions[idx],
        yerr=np.std(np.abs(cone_contributions[:, idx])),
        color=cone_colours[idx],
    )
    ax[0].set_title("ON")
    ax[0].set_xticks(range(len(lasso_columns[:nr_cones])))
    ax[0].set_xticklabels(lasso_columns[:nr_cones], rotation=45)
# %% box plot for on off lasso
for idx, cone in enumerate(lasso_columns[nr_cones:]):
    ax[1].bar(
        idx,
        all_contributions[idx],
        yerr=np.std(np.abs(cone_contributions[:, idx])),
        color=cone_colours[idx],
    )
    ax[1].set_xticks(range(len(lasso_columns[nr_cones:])))
    ax[1].set_xticklabels(lasso_columns[nr_cones:], rotation=45)
    ax[1].set_title("OFF")
    ax[1].set_ylabel("Contribution")
    ax[1].set_xlabel("Cone type")

# %% remove splines of all plots and x and y tick values from the top plot
for idx in range(2):
    ax[idx].spines["top"].set_visible(False)
    ax[idx].spines["right"].set_visible(False)
ax[0].set_yticks([])
ax[0].set_xticks([])
# increase margins at bottom
plt.subplots_adjust(bottom=0.2)

# Change font size of x and y labels to 20
for idx in range(2):
    ax[idx].set_xlabel(ax[idx].get_xlabel(), fontsize=20)
    ax[idx].set_ylabel(ax[idx].get_ylabel(), fontsize=20)
# Only show 0 and 1 on y axis and inceased tick label font size to 14
for idx in range(2):
    ax[idx].set_yticks([0, 1])
    ax[idx].tick_params(axis="y", labelsize=14)
ax[0].tick_params(axis="x", labelsize=14)
# %%
fig.show()
