from polarspike import Overview, histograms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import polars as pl
import multiprocessing as mp
import plotly.express as px
from functools import partial
from scipy.signal import butter, sosfilt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# %%
def chirp_analysis(spikes_list, c_freqs):
    c_freqs_positions = np.load(r"A:\Marvin\chirps\c_freqs_positions.npy")
    sos = butter(10, 30, "low", fs=300, output="sos")
    wavelet = "cmor3.0-1.5"
    # logarithmic scale for scales, as suggested by Torrence & Compo:
    widths = np.geomspace(1, 1024, num=300)
    sampling_period = 0.01 / 3
    all_results = []
    for spikes in spikes_list:
        results = np.zeros((4, 6))
        cell_index = spikes["cell_index"].to_numpy()[0]
        recording_index = spikes["recording"].to_numpy()[0]
        stimulus_name = spikes["stimulus_name"].to_numpy()[0]
        psth, bins, repeat = histograms.psth_by_index(
            spikes,
            index=["repeat"],
            bin_size=0.01 / 3,
            window_end=35,
            return_idx=True,
        )
        if psth.shape[0] != 1:
            try:
                psth = psth[np.sort(repeat.flatten().astype(int))]
            except IndexError:
                print(f"Error in recording {recording_index} cell {cell_index}")
                continue
        for rep in range(psth.shape[0]):
            # wavelet transform
            # perform CWT
            cwtmatr, freqs = pywt.cwt(
                psth[rep, :] - psth[rep, :].mean(),
                widths,
                wavelet,
                sampling_period=sampling_period,
            )

            cwtmatr = np.abs(cwtmatr[:-1, :-1])

            # find the most powerfull power trace
            max_power = np.zeros(cwtmatr.shape[1])
            max_positions = np.zeros(cwtmatr.shape[1], dtype=int)
            power = -np.inf
            for i in range(150):
                temp_positions = c_freqs_positions + i
                temp_positions[temp_positions >= 299] = 0
                power_trace = cwtmatr[temp_positions, np.arange(cwtmatr.shape[1])]
                if np.sum(power_trace) > power:
                    max_power = power_trace
                    power = np.sum(power_trace)
                    max_positions = temp_positions
                temp_positions = c_freqs_positions - i
                temp_positions[temp_positions < 0] = 0
                power_trace = cwtmatr[temp_positions, np.arange(cwtmatr.shape[1])]
                if np.sum(power_trace) > power:
                    max_power = power_trace
                    power = np.sum(power_trace)
                    max_positions = temp_positions
            try:
                filtered_signal = sosfilt(sos, np.abs(max_power))
                power_max = np.max(filtered_signal)
                power_max_position = np.argmax(filtered_signal)
                power_max_frequency = c_freqs[power_max_position]
                averaged_signal = moving_average(filtered_signal, 400)
                # threshold = power_max * 0.10
                # threshold_position = np.where(np.abs(max_power) > threshold)[0][-1]
                threshold = power_max * 0.20
                threshold_position = np.where(averaged_signal > threshold)[0][-1]
                threshold_power = np.sum(
                    averaged_signal[threshold_position - 10 : threshold_position + 10]
                )
                threshold_frequency = c_freqs[threshold_position - 400]
                power_ratio = (
                    np.sum(max_power[:power_max_position]) / threshold_power
                ) / np.sum(max_power)
                results[rep, :] = [
                    power_max,
                    power_max_frequency,
                    power_ratio,
                    threshold_frequency,
                    threshold_power,
                    rep,
                ]
            except Exception as e:
                print(f"Error in recording {recording_index} cell {cell_index}")
                print(e)
                continue
        result_df = pd.DataFrame(
            results,
            columns=[
                "power_max",
                "frequency",
                "power_ratio",
                "threshold_freq",
                "threshold_power",
                "repeat",
            ],
        )
        result_df["cell_index"] = cell_index
        result_df["recording"] = recording_index
        result_df["stimulus_name"] = stimulus_name

        all_results.append(result_df)
    return pd.concat(all_results)

    # %%


def normalize_0_1(data):
    return (data - data.min()) / (data.max() - data.min())


if __name__ == "__main__":
    wavelet = "cmor3.0-1.5"
    # logarithmic scale for scales, as suggested by Torrence & Compo:
    widths = np.geomspace(1, 1024, num=300)
    sampling_period = 0.01 / 3

    # %%
    # Define chirp
    start_freq = 1  # Start frequency in Hz
    end_freq = 30  # End frequency in Hz
    duration = 30  # Duration in seconds
    refresh_rate = 300  # Refresh rate in Hz
    num_cycles = duration * refresh_rate

    chirp_real = []
    time = []
    freq = []
    for i in range(num_cycles):
        t = i / refresh_rate
        f = np.power(end_freq / start_freq, t / duration) * start_freq
        # Intended instantaneous frequency (exponential chirp)
        k = np.log(end_freq / start_freq) / duration
        f_intended = start_freq * np.exp(k * t)

        # Actual instantaneous frequency due to incorrect phase
        f_actual = f_intended * (1 + k * t)
        value = np.sin(2 * np.pi * f * t)
        chirp_real.append(int((value + 1) * 4095 / 2))
        time.append(t)
        freq.append(f_actual)

    chirp_real = np.asarray(chirp_real)

    chirp_real = normalize_0_1(chirp_real)
    chirp_complete = np.zeros(35 * 300 - 1)
    chirp_complete[300 * 3 : 3300 * 3] = chirp_real
    c_freqs = np.zeros_like(chirp_complete)
    c_freqs[300 * 3 : 3300 * 3] = np.asarray(freq)
    time = np.arange(0, 35 - 1 / 300, 1 / 300)
    # find closest position to actual frequency

    # %%
    recordings = Overview.Recording_s.load(r"A:\Marvin\fff_clustering\records")
    recordings.dataframes["chirps"] = recordings.spikes_df.query(
        "stimulus_name == 'chirp'"
        "| stimulus_name == 'chirp413'|stimulus_name == 'chirp460'|stimulus_name == 'chirp535'|stimulus_name == 'chirp610'"
    ).copy()
    # %%
    labels_df = pd.read_pickle(r"A:\Marvin\fff_clustering\coeff_label_df")
    # labels_df = labels_df.set_index(["recording", "cell_index"])

    recordings.dataframes["chirps"]["labels"] = -1

    recordings.dataframes["chirps"] = recordings.dataframes["chirps"].set_index(
        ["recording", "cell_index"]
    )
    recordings.dataframes["chirps"].update(labels_df)

    recordings.dataframes["chirps"] = recordings.dataframes["chirps"].reset_index()
    # recordings.dataframes["chirps"] = recordings.dataframes["chirps"].query("qi>0.35")

    spikes = recordings.get_spikes_df(
        "chirps", carry=["stimulus_name", "labels"], pandas=False
    )
    # %%
    spikes_stimuli = spikes.partition_by(["stimulus_name", "recording", "cell_index"])

    # %% split list into chunks
    spikes_stimuli = np.array(spikes_stimuli, dtype=object)

    spikes_stimuli = np.array_split(spikes_stimuli, mp.cpu_count())
    # %%
    par_func = partial(chirp_analysis, c_freqs=c_freqs)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(par_func, spikes_stimuli)

    results_df = pd.concat(results)
    results_df.to_pickle(r"A:\Marvin\chirps\chirp_analysis_df.pkl")

# %%
results_pf = pl.from_pandas(results_df)
# %%
results_pf_mean = results_pf.group_by(["recording", "cell_index", "stimulus_name"]).agg(
    pl.col("threshold_freq").mean().alias("frequency_mean")
)
# %%

fig = px.histogram(
    results_pf_mean.to_pandas(), x="frequency_mean", color="stimulus_name", opacity=0.5
)
fig.update_layout(barmode="overlay")
fig.show(renderer="browser")
# %%
results_df_mean = results_pf_mean.to_pandas().set_index(
    ["recording", "cell_index", "stimulus_name"]
)
results_df_mean["qi"] = 0
results_df_mean.update(
    recordings.dataframes["chirps"].set_index(
        ["recording", "cell_index", "stimulus_name"]
    )
)
# %%
fig = px.scatter(results_df_mean, x="frequency_mean", y="qi")
fig.show(renderer="browser")
# %%
results_df_filtered = results_df_mean.query("qi>0.2").reset_index()
results_df_filtered = results_df_filtered.sort_values("stimulus_name")
# %%
fig = px.histogram(
    results_df_filtered,
    x="frequency_mean",
    color="stimulus_name",
    facet_row="stimulus_name",
    color_discrete_sequence=["grey", "magenta", "blue", "green", "red"],
    nbins=100,
    histnorm="percent",
)
fig.update_layout(barmode="overlay", template="simple_white")
fig.show(renderer="browser")

# %% example cell
spikes_cell = spikes.filter(
    (pl.col("cell_index") == 256)
    & (pl.col("recording") == "chicken_30_08_2024_p0")
    & (pl.col("stimulus_name") == "chirp")
)
# %%
psth, bins, repeat = histograms.psth_by_index(
    spikes_cell, index=["repeat"], bin_size=0.01 / 3, window_end=35, return_idx=True
)
# %%
fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
for idx, trace in enumerate(psth):
    ax[idx].plot(bins[:-1], trace, color="black")
fig.show()

# %%

fig = px.histogram(results_df, x="frequency", color="stimulus_name")
fig.show(renderer="browser")

# %%
results_df = results_df.sort_values("stimulus_name")
fig = px.scatter(
    results_df,
    x="threshold_freq",
    y="threshold_power",
    color="stimulus_name",
    color_discrete_sequence=["grey", "magenta", "blue", "green", "red"],
    hover_data=["recording", "cell_index"],
)
fig.update_layout(template="simple_white")
fig.show(renderer="browser")

# %%
results_pf_filtered = results_pf.filter(
    (pl.col("threshold_freq") < 105) & (pl.col("threshold_power") > 2)
)
results_pf_filtered = results_pf_filtered.sort("stimulus_name")
fig = px.scatter(
    results_pf_filtered.to_pandas(),
    x="threshold_freq",
    y="threshold_power",
    color="stimulus_name",
    color_discrete_sequence=["grey", "magenta", "blue", "green", "red"],
    hover_data=["recording", "cell_index"],
    opacity=0.5,
    marginal_x="histogram",
)
fig.update_layout(template="simple_white")
fig.show(renderer="browser")
fig.write_html(r"A:\Marvin\chirps\threshold_freq_power.html")

# %%
fig = px.box(
    results_pf_filtered.to_pandas(),
    x="repeat",
    y="threshold_freq",
    color="stimulus_name",
    color_discrete_sequence=["grey", "magenta", "blue", "green", "red"],
    hover_data=["recording", "cell_index"],
)
fig.show(renderer="browser")
# %% aggregate every recording, cell_index, stimulus_name
results_pf_mean = results_pf_filtered.group_by(
    ["recording", "cell_index", "stimulus_name"]
).agg(
    pl.col("threshold_freq").median().alias("threshold_freq"),
    pl.col("threshold_power").mean().alias("threshold_power"),
    pl.col("power_max").mean().alias("power_max"),
    pl.col("power_ratio").mean().alias("power_ratio"),
    pl.col("frequency").mean().alias("frequency"),
)
results_pf_mean = results_pf_mean.sort("stimulus_name")
# %%
fig = px.scatter(
    results_pf_mean.to_pandas(),
    x="threshold_freq",
    y="frequency",
    color="stimulus_name",
    color_discrete_sequence=["grey", "magenta", "blue", "green", "red"],
    hover_data=["recording", "cell_index"],
    opacity=0.5,
    marginal_x="histogram",
    log_x=True,
    log_y=True,
)

fig.update_layout(template="simple_white")
fig.show(renderer="browser")

# %%
chirp_460 = results_pf_mean.filter(pl.col("stimulus_name") == "chirp460")
chirp_610 = results_pf_mean.filter(pl.col("stimulus_name") == "chirp610")

combined = chirp_460.join(
    chirp_610,
    on=["recording", "cell_index"],
    suffix="_610",
)
combined = combined.with_columns(index=np.arange(combined.height))

# %%

fig = px.scatter(
    combined.to_pandas(),
    x="frequency",
    y="frequency_610",
    color_discrete_sequence=["grey", "magenta", "blue", "green", "red"],
    hover_data=["recording", "cell_index"],
    opacity=0.5,
    marginal_x="histogram",
    log_x=True,
    log_y=True,
)

fig.update_layout(template="simple_white")
fig.show(renderer="browser")

# %%
import seaborn as sns

# %%
g = sns.lmplot(
    data=combined.to_pandas(),
    x="frequency",
    y="frequency_610",
    scatter_kws={"alpha": 0.5},
    height=10,
)
g.fig.show()
# %%
results_df_mean = results_pf_mean.to_pandas().set_index(["recording", "cell_index"])
pd.to_pickle(results_df_mean, r"A:\Marvin\chirps\chirp_analysis_df_mean.pkl")
