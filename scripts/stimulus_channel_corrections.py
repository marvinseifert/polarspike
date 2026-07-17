from polarspike.stimulus_trace import Stimulus_Extractor

# %%

extractor = Stimulus_Extractor(
    r"/mnt/nas_b/Marvin/chicken_29_05_2026/Phase_00/analog/2026-05-29T12-36-09McsRecording_aux.dat",
    freq=20000,
    second_trigger=False,
)
