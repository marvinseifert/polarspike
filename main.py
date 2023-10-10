import Overview

single_rec = Overview.Recording.load("test")
# %%
single_rec.get_spikes_triggered([50], 1, time="seconds")