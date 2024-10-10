from polarspike import (
    Overview,
    colour_template,
    spiketrain_plots,
    Opsins,
    spiketrains,
    histograms,
)
from group_pca import GroupPCA
import panel as pn

pn.extension("tabulator")
from bokeh.io import show
from importlib import reload
import numpy as np
import polars as pl
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Import normalizer
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # %%
    recording = Overview.Recording.load(
        r"A:\Marvin\chicken_19_07_2024\Phase_00\overview"
    )
    clusters = np.load(r"A:\Marvin\chicken_19_07_2024\Phase_00\rf_clusters.npy")
    recording.spikes_df
