from polarspike.analysis import response_peaks, count_spikes
from polarspike import Opsins, colour_template
import numpy as np
import plotly.graph_objects as go


def for_ui(spikes, bin_size, mean_trigger, window, colour_name, animal_name):
    if animal_name is None:
        use_opsin = False
    else:
        use_opsin = True
    spikes_summed = count_spikes.sum_spikes(
        spikes, mean_trigger, window=window, group_by="stimulus_index"
    )[0]
    spikes_summed_mean = np.mean(spikes_summed, axis=0)
    spikes_summed_norm = spikes_summed_mean / np.max(spikes_summed_mean)

    # %% Findpeaks method

    all_heights, peak_locations = response_peaks.find_peaks(
        spikes, mean_trigger, bin_size=bin_size
    )

    spikes_heights_norm = all_heights / np.max(all_heights)
    # %% Plot the results
    CT = colour_template.Colour_template()
    CT.pick_stimulus(colour_name)
    if use_opsin:
        Ops = Opsins.Opsin_template()
        fig = Ops.plot_overview(animal_name)
    else:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=CT.wavelengths[::2],
            y=spikes_summed_norm[::2],
            mode="lines+markers",
            line=dict(color="grey"),
            name="Summed spikes ON",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=CT.wavelengths[1::2],
            y=spikes_summed_norm[1::2],
            mode="lines+markers",
            line=dict(color="black"),
            name="Summed spikes OFF",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=CT.wavelengths[::2],
            y=spikes_heights_norm[::2],
            mode="lines+markers",
            line=dict(color="grey", dash="dash"),
            name="Peaks ON",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=CT.wavelengths[1::2],
            y=spikes_heights_norm[1::2],
            mode="lines+markers",
            line=dict(color="black", dash="dash"),
            name="Peaks OFF",
        )
    )
    # Add diagonal line
    if not use_opsin:
        fig.add_trace(
            go.Scatter(
                x=[CT.wavelengths[-1], CT.wavelengths[0]],
                y=[0, 1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Diagonal",
            )
        )
    fig.update_layout(height=700, width=700)
    return fig
    # %%
