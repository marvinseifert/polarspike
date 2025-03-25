import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

from scripts.heatmap_plots import lasso_columns

sys.path.extend(["/home/marvin-seifert/PycharmProjects"])
from scripts.moving_bars_analysis import results

# %%
cone_colours = [
    "#7c86fe",
    "#7cfcfe",
    "#8afe7c",
    "#fe7c7c",
    "orange",
    "darkorange",
]
# %%
results_df = pd.read_pickle(
    r"/home/marvin-seifert/Marvin/chirps/chirp_analysis_df_mean.pkl"
)
feature_df = pd.read_pickle(
    r"/home/marvin-seifert/Marvin/fff_clustering/features_coeffs_pca_clustered"
)
#
# %% lasso columns
lasso_columns = [
    "ON_Lasso_principal",
    "ON_Lasso_LWS",
    "ON_Lasso_MWS",
    "ON_Lasso_SWS1",
    "ON_Lasso_SWS2",
    "OFF_Lasso_principal",
    "OFF_Lasso_LWS",
    "OFF_Lasso_MWS",
    "OFF_Lasso_SWS1",
    "OFF_Lasso_SWS2",
]
lasso_df = feature_df[lasso_columns].copy()
# %% normalize lasso values to max value
lasso_df = lasso_df.div(lasso_df.max(axis=1), axis=0)
feature_df[lasso_columns] = lasso_df
# %%
results_df = results_df.query("threshold_freq>0 & threshold_freq<60")

# %%
feature_df["labels"] = feature_df["labels"] + 1
# results_df = results_df.set_index(["recording", "cell_index"])
results_df["labels"] = 0
results_df.update(feature_df)
# %% find the combined index of cells that are in both dataframes

# %%
chirp_df = results_df.query("stimulus_name=='chirp'")
combined_index = chirp_df.index.intersection(feature_df.index)
combined_df = pd.concat([chirp_df, feature_df.loc[combined_index]], axis=1)
# %% plot regression plot
log_x = combined_df["frequency"].values
log_y = combined_df["OFF_Lasso_principal"].values
log_x[np.isnan(log_x)] = 0
log_y[np.isnan(log_y)] = 0
log_x[np.isinf(log_x)] = 0
log_y[np.isinf(log_y)] = 0
log_x = log_x[log_y != 0]
log_y = log_y[log_y != 0]
fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(
    x=log_x,
    y=log_y,
    ax=ax,
)
ax.set_xlabel("log(frequency)")
ax.set_ylabel("log(test)")
fig.show()

# %%

# %%

combination = [log_x, log_y]

max_x = combination[0].max()
max_y = combination[1].max()
min_x = combination[0].min()
min_y = combination[1].min()
# fill 0 values with nan and drop them
# sub_df = sub_df.replace(0, np.nan).dropna()
# fitt RANSAC regression
ransac = linear_model.RANSACRegressor()
ransac.fit(combination[0].reshape(-1, 1), combination[1])
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.linspace(min_x, max_x, 100)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))
fig, axs = plt.subplots(ncols=2, figsize=(20, 10), sharex=True, sharey=True)
axs[0].scatter(
    combination[0],
    combination[1],
    c="black",
    label="Data",
    s=5,
)
axs[0].plot(
    line_X,
    line_y_ransac,
    color="blue",
    label="RANSAC regressor",
    linewidth=2,
)
# Only plot the inliers
axs[1].scatter(
    combination[0][inlier_mask],
    combination[1][inlier_mask],
    c="blue",
    label="Inliers",
    s=5,
)
axs[1].scatter(
    combination[0][outlier_mask],
    combination[1][outlier_mask],
    c="black",
    label="Outliers",
    s=5,
)
axs[1].plot(
    line_X,
    line_y_ransac,
    color="blue",
    label="RANSAC regressor",
    linewidth=2,
)
# print the ransac score
axs[1].text(
    0.05,
    0.95,
    f"RANSAC score: {ransac.score(combination[0].reshape(-1, 1), combination[1])}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=axs[1].transAxes,
)
# print the score only for the inliers
axs[1].text(
    0.05,
    0.9,
    f"RANSAC score inliers: {ransac.score(combination[0][inlier_mask].reshape(-1, 1), combination[1][inlier_mask])}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=axs[1].transAxes,
)

axs[0].legend()
fig.show()
# %%
fast_cells = combined_df.query("threshold_freq>15")
# create a bar plot showing cone contributions of fast cells
cone_contributions = np.asarray(np.abs(fast_cells[lasso_columns].copy()))
# %%
all_contributions = np.nanmean(np.abs(cone_contributions), axis=0)
# peak normalize the contributions
all_contributions = all_contributions / np.nanmax(all_contributions)
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
