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
    recordings = Overview.Recording_s.load("D:\\combined_analysis\\all_recordings")
    spikes = pl.scan_parquet(recordings.recordings["zebrafish_14_11_23"].parquet_path)
    spikes = spikes.filter(pl.col("cell_index") == 270).collect()
    trigger = np.arange(0, np.max(spikes["times"].to_numpy()), 1)
    sta = elpy.sta_main_loop(
        spikes["times"].to_numpy(),
        trigger,
        "D:\\Zebrafish_14_11_23\\ks_sorted\\alldata.dat",
        nr_pre_bins=10,
        nr_post_bins=40,
        nr_boxes=252,
        mea="MCS",
    )
