from polarspike import Overview, quality_tests

recording = Overview.Recording.load(r"A:\Marvin\toy_example\Phase_0\overview")

spikes = recording.get_spikes_triggered([[0]], [["all"]], pandas=False)
# %%
test = quality_tests.spiketrain_qi(spikes, 24, max_repeat=10)
