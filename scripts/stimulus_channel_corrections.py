from polarspike import stimulus_trace
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# %%
stimulus_file = r"A:\Marvin\chicken_04_09_2024\Phase_00\analog\2024-09-04T12-40-49McsRecording_aux.dat"
channel = np.fromfile(stimulus_file, dtype=np.int16)
channel = channel[::2]
# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(channel[20000 * 40 : 20000 * 60])
fig.show()

# %%
fig = px.line(y=channel[20000 * 50 : 20000 * 60])
fig.show(renderer="browser")
