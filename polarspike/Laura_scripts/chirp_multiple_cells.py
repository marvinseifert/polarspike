import numpy as np
import pandas as pd
import pywt
from matplotlib import pyplot as plt

from polarspike import Overview, histograms
from tqdm import tqdm

# --- your helpers ---
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def generate_chirp_data(start_freq, end_freq, duration, refresh_rate, max_power=4095):
    num_cycles = int(duration * refresh_rate)
    beta = np.log(end_freq / start_freq) / duration
    t = np.linspace(0, duration, num_cycles, endpoint=False)
    exp_bt = np.exp(beta * t)
    freqs = start_freq * exp_bt
    phase = 2 * np.pi * (start_freq / beta) * (exp_bt - 1)
    raw_sine = np.sin(phase)
    chirp_signal = (raw_sine + 1.0) * max_power / 2.0
    chirp_signal = chirp_signal.astype(int)
    return t, chirp_signal, freqs


# --- build chirp c_freqs once (used for all cells/stimuli if chirp definition is same) ---
start_freq, end_freq, duration, refresh_rate = 1, 30, 30, 300
time, chirp_real, freq_actual = generate_chirp_data(start_freq, end_freq, duration, refresh_rate)

total_length_sec = 35
total_samples = int(total_length_sec * refresh_rate) - 1

c_freqs = np.zeros(total_samples)
offset = 300 * 3
end_idx = offset + len(freq_actual)
limit = min(len(c_freqs), end_idx)
insert_len = limit - offset
if insert_len > 0:
    c_freqs[offset:limit] = freq_actual[:insert_len]

# --- analysis function for one (cell, stimulus) ---
def analyze_cell_stimulus(spikes, cell_index, stimulus_index,
                          c_freqs, window_end=35,
                          bin_size=0.01/3,
                          wavelet="cmor3.0-1.5",
                          n_freqs=300,
                          widths_min=1, widths_max=1024,
                          repeat_to_use=2,
                          search_shift=150,
                          smooth_n=400,
                          threshold_frac=0.10):
    # slice spikes
    cell_spikes = spikes.query("cell_index == @cell_index & stimulus_index == @stimulus_index")
    if len(cell_spikes) == 0:
        return None  # no data for this combo

    # PSTH (by repeat)
    psth, bins, repeat = histograms.psth_by_index(
        cell_spikes, index=["repeat"], bin_size=bin_size, window_end=window_end, return_idx=True
    )

    # guard: ensure requested repeat exists
    if psth.shape[0] <= repeat_to_use:
        return None

    # Wavelet transform
    widths = np.geomspace(widths_min, widths_max, num=n_freqs)
    sampling_period = bin_size
    signal = psth[repeat_to_use] - psth[repeat_to_use].mean()

    cwtmatr, freqs = pywt.cwt(signal, widths, wavelet, sampling_period=sampling_period)
    cwtmatr = np.abs(cwtmatr[:-1, :-1])  # match your trimming

    # Map each time point to closest stimulus frequency bin
    c_freqs_positions = np.zeros(c_freqs.shape[0] - 1, dtype=int)
    freqs_trim = freqs[:-1]
    for i, f in enumerate(c_freqs[:-1]):
        c_freqs_positions[i] = np.argmin(np.abs(f - freqs_trim))

    # Find best shifted path (max total power)
    max_power = np.zeros(cwtmatr.shape[1])
    power_best = -np.inf

    # IMPORTANT: cwtmatr time length might differ from c_freqs_positions length.
    # Align to common length.
    T = min(cwtmatr.shape[1], len(c_freqs_positions))
    base_pos = c_freqs_positions[:T]

    for i in range(search_shift):
        # upward shift
        temp = base_pos + i
        temp[temp >= (n_freqs - 1)] = 0
        trace = cwtmatr[temp, np.arange(T)]
        s = np.sum(trace)
        if s > power_best:
            power_best = s
            max_power = trace.copy()

        # downward shift
        temp = base_pos - i
        temp[temp < 0] = 0
        trace = cwtmatr[temp, np.arange(T)]
        s = np.sum(trace)
        if s > power_best:
            power_best = s
            max_power = trace.copy()

    # Mask padded regions (no chirp)
    c_freqs_use = c_freqs[:T]
    max_power[c_freqs_use == 0] = 0

    # Smooth + stats
    abs_power = np.abs(max_power)
    if len(abs_power) < smooth_n:
        return None

    filtered = moving_average(abs_power, smooth_n)
    power_max = np.max(abs_power)
    power_max_position = np.argmax(abs_power)
    power_max_frequency = c_freqs_use[power_max_position]

    threshold = power_max * threshold_frac

    # filtered is shorter by smooth_n-1, so threshold_position lives in filtered index space
    idxs = np.where(np.abs(filtered) > threshold)[0]
    if len(idxs) == 0:
        return None
    threshold_position = idxs[-1]
    threshold_frequency = c_freqs_use[threshold_position]

    threshold_power = np.sum(abs_power[threshold_position:])
    if threshold_power == 0:
        power_ratio = np.nan
    else:
        power_ratio = (np.sum(abs_power[:threshold_position]) / threshold_power) / np.sum(abs_power)

    return {
        "cell_index": cell_index,
        "stimulus_index": stimulus_index,
        "power_max_frequency": float(power_max_frequency),
        "power_ratio": float(power_ratio),
        "threshold_frequency": float(threshold_frequency),
        "power_max": float(power_max),
        "power_max_position": int(power_max_position),
        "threshold_position": int(threshold_position),
        "n_spikes": int(len(cell_spikes)),
    }

# %%
# --- load recording + spikes  ---
recordings = Overview.Recording.load(r"F:\Laura\zebrafish_22_04_2026\Phase_00\overview")
stimulus_id = [16]
recordings.dataframes["chirps"] = recordings.spikes_df.query("stimulus_index in @stimulus_id").copy()
spikes = recordings.get_spikes_df("chirps", carry=["stimulus_name"])

df = pd.read_csv(r"C:\Users\Laura Steel\PycharmProjects\polarspike\polarspike\Laura_scripts\good_cells_list.csv")
good_cells_list = df.iloc[:, 0].tolist()
print(good_cells_list)
# %%
# --- choose what to loop over ---
cells_to_run = good_cells_list        #or e.g. [270, 271, 272]

stimuli_to_run = stimulus_id

# --- run loops and collect results ---
rows = []

total_iterations = len(cells_to_run) * len(stimuli_to_run)

with tqdm(total=total_iterations, desc="Processing cells × stimuli") as pbar:
    for cell_idx in cells_to_run:
        for stim_idx in stimuli_to_run:
            out = analyze_cell_stimulus(spikes, cell_idx, stim_idx, c_freqs)
            if out is not None:
                rows.append(out)
            pbar.update(1)

results_df = pd.DataFrame(rows)

# A simple NumPy array output too (cell, stim, max_freq, power_ratio, thresh_freq)
results_array = results_df[
    ["cell_index", "stimulus_index", "power_max_frequency", "threshold_frequency", "power_ratio"]
].to_numpy()

# %%
# Max frequency distribution
plt.hist(results_df["power_max_frequency"], bins=100, alpha=0.5, color="tab:green", edgecolor="tab:blue", linewidth = 0.5)
plt.xlabel("max frequency")
plt.ylabel("Number of Cells")
plt.title("Distribution of Max Power Frequency (15_01_2026p0, chirp_1)")
plt.show()

# Threshold frequency distribution
plt.hist(results_df["threshold_frequency"], bins=100, alpha=0.5, color="tab:orange", edgecolor="tab:red", linewidth = 0.5)
plt.xlabel("threshold frequency")
plt.ylabel("Number of Cells")
plt.title(f"Distribution of Threshold Frequency (15_01_2026p0, chirp_1)")
plt.show()

mean_value = results_df["power_max_frequency"].mean()
print(f"power max frequency {mean_value} Hz")
mean_value = results_df["power_ratio"].mean()
print(f"power ratio at {mean_value} Hz")
mean_value = results_df["threshold_frequency"].mean()
print(f"frequency threshold at {mean_value} Hz")
# %%
from scipy.stats import skew

data = results_df["threshold_frequency"]  # your array

mean = np.mean(data)
median = np.median(data)
std = np.std(data)
iqr = np.percentile(data, 75) - np.percentile(data, 25)
skewness = skew(data)
max = np.max(data)
min = np.min(data)

print(mean, median, std, iqr, skewness, max, min)