from polarspike import Overview, histograms
import ppseq
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from ppseq.plotting import plot_model, color_plot, sort_neurons, plot_sorted_neurons
from ppseq.model import PPSeq

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Could not find a GPU. Defaulting to CPU instead.")


# %%
recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")

recordings.dataframes["sub_df"] = recordings.dataframes["fff_filtered"].sample(1)
spikes = recordings.get_spikes_df("sub_df", pandas=False)
# %%
psth, bins, cell_index = histograms.psth_by_index(
    spikes,
    index=["cell_index", "repeat"],
    return_idx=True,
    window_end=24.1,
    bin_size=0.1,
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(bins[:-1], psth[0], color="black")
fig.show()
# %%
num_neurons = psth.shape[0]
data = torch.tensor(psth, device=device)
# %%
torch.manual_seed(0)
model = PPSeq(
    num_templates=2,
    num_neurons=num_neurons,
    template_duration=20,
    alpha_a0=1.5,
    beta_a0=0.2,
    alpha_b0=1,
    beta_b0=0.1,
    alpha_t0=1.2,
    beta_t0=0.1,
)

# %%
lps, amplitudes = model.fit(data, num_iter=100)
# %%
fig = plot_model(model.templates.cpu(), amplitudes.cpu(), data.cpu(), figsize=(20, 10))
fig.show()
# %%
color_plot(
    data.cpu(),
    model,
    amplitudes.cpu(),
)
# %%
fig, ax = plt.subplots()
ax.plot(lps.cpu())
ax.set_ylabel("Log Probability")
ax.set_xlabel("Iteration")
fig.show()
