from polarspike.stimulus_trace import Stimulus_Extractor

# %%

extractor = Stimulus_Extractor(
    r"/home/mawa/nas_a/Marvin/chicken_13_05_2025/Phase_00/analog/2025-05-13T12-38-31McsRecording_aux.dat",
    freq=20000,
    second_trigger=True,
)
