from Neuroelectropy import electrical_imaging as elpy
from polarspike import (
    Overview,
    spiketrain_plots,
    colour_template,
    stimulus_spikes,
    recording_overview,
)

from bokeh.plotting import figure, show
import panel as pn
import polars as pl
import numpy as np

pn.extension()

if __name__ == "__main__":
    recordings = Overview.Recording_s.load_from_single(
        r"B:\Marvin\combined_analysis",
        "test_analysis",
        r"B:\Marvin\Zebrafish_20_11_23\ks_sorted\overview",
    )
    recordings.add_from_saved("B:\Marvin\Zebrafish_21_11_23\ks_sorted\overview")
    recordings.dataframes["test"] = recordings.spikes_df.query("stimulus_name == 'FFF'")
    spikes = recordings.get_spikes_triggered([["all"]], [[[0]]], [[["all"]]])
