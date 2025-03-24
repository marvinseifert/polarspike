from polarspike import Overview, spiketrain_plots, colour_template, histograms
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# %%
CT = colour_template.Colour_template()

# %%

recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
feature_df = pl.from_pandas(
    pd.read_pickle(r"A:\Marvin\fff_clustering\features_coeffs_pca_clustered")
)

# %%
feature_stats = feature_df.group_by("labels").mean().to_pandas()

# %%
fig = px.box(feature_df.to_pandas(), x="labels", y="maxima")
fig.show(renderer="browser")
# %%
figs = []
for column in feature_df.to_pandas().columns:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(data=feature_df.to_pandas(), x="labels", y=column, ax=ax)
    ax.set_title(column)
    fig.savefig(
        r"A:\Marvin\fff_clustering\cluster_stats\boxplots\{}_boxplot.png".format(column)
    )

# %%
for column in feature_df.to_pandas().columns:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(feature_df[column].to_numpy(), feature_df["labels"].to_numpy())
    ax.set_title(column)
    fig.savefig(
        r"A:\Marvin\fff_clustering\cluster_stats\scatter\{}_scatter.png".format(column)
    )
