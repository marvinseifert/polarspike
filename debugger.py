import pandas as pd

import Extractors

stimulus_df = pd.read_pickle("stimulus_df")

SC = Extractors.Extractor_SPC(
    r"D:\zebrafish_02_10_2023\Phase_00\ks_sorted\alldata.dat", stimulus_df
)


SC.get_spikes()
df = SC.load(recording_name="test")
