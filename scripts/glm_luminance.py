from pandas._libs.tslibs.offsets import OffsetMeta

from polarspike import Overview, spiketrain_plots, colour_template, binarizer
from polarspike.histograms import psth_by_index, psth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from polarspike import spike_loader
import polars_ds as pds
import polars as pl
from scipy.signal.windows import gaussian
from scipy.optimize import minimize
from matplotlib.collections import LineCollection

# %%
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
colours = ct.colours
# %%
luminance_adaptation = 3000
luminance_interval = 1000
luminance_min = 1000
luminance_max = 4095
a_step_on = np.array([1100, 1250, 1400, 1550, 1700, 1850])
b_step_on = np.array([1375, 1562, 1750, 1937, 2125, 2312])
a_step_off = np.array([3685, 3071, 2457, 1842, 1228, 614])
b_step_off = np.array([2763, 2303, 1842, 1381, 921, 460])
a_step_on_c = (a_step_on - luminance_min) / luminance_min
b_step_on_c = (b_step_on - luminance_min) / luminance_min
a_step_over_background = a_step_on_c.copy()
b_step_over_a = (b_step_on - a_step_on) / a_step_on
a_luminance_over_background = a_step_on - luminance_min
a_luminance_over_background = a_luminance_over_background / np.max(
    a_luminance_over_background
)
b_luminance_over_background = b_step_on - luminance_min
b_luminance_over_background = b_luminance_over_background / np.max(
    b_luminance_over_background
)
times_a_on = np.arange(3, 30, 5)
times_b_on = np.arange(4, 30, 5)
# %%
a_step_off_c = (a_step_off - luminance_max) / luminance_max
b_step_off_c = (b_step_off - luminance_max) / luminance_max
a_step_off_over_background = a_step_off_c.copy()
b_step_off_over_a_off = (b_step_off - a_step_off) / a_step_off
a_off_luminance_over_background = a_step_off - luminance_max
a_off_luminance_over_background = -a_off_luminance_over_background / np.max(
    -a_off_luminance_over_background
)
b_off_luminance_over_background = b_step_off - luminance_max
b_off_luminance_over_background = -b_off_luminance_over_background / np.max(
    -b_off_luminance_over_background
)
times_a_off = np.arange(3, 30, 5)
times_b_off = np.arange(4, 30, 5)
# %%
luminance_off = []
for step in range(6):
    luminance_off.append([luminance_max] * 3)
    luminance_off.append([a_step_off[step]])
    luminance_off.append([b_step_off[step]])
luminance_off = np.concatenate(luminance_off)
luminance_off_weber = (luminance_off[1:] - luminance_off[:-1]) / luminance_off[:-1]
weber_for_colour_off = luminance_off_weber.copy()
luminance_off_weber = np.concatenate([luminance_off_weber, [0]])
luminance_off_weber = np.repeat(luminance_off_weber, 20)
# %%
luminance_on = []
for step in range(6):
    luminance_on.append([luminance_min] * 3)
    luminance_on.append([a_step_on[step]])
    luminance_on.append([b_step_on[step]])
luminance_on = np.concatenate(luminance_on)
luminance_on_weber = (luminance_on[1:] - luminance_on[:-1]) / luminance_on[:-1]
weber_for_colour_on = luminance_on_weber.copy()
luminance_on_weber = np.concatenate([luminance_on_weber, [0]])
luminance_on_weber = np.repeat(luminance_on_weber, 20)
# %%
luminance_on = luminance_on / 1000 - 1
# %%
recording = Overview.Recording.load(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_20_04_2026/Phase_00/overview"
)
# %%
spikes_on = recording.get_spikes_triggered(
    [{"stimulus_name": "c_luminance_on"}], pandas=False
)
# %%
cell = 10  # (
#     recording.spikes_df.query("stimulus_name=='fff'")
#     .sort_values("qi", ascending=False)["cell_index"]
#     .values[27]
# )
binsize = 0.05
spikes_binned, bins = psth_by_index(
    spikes_on.filter(pl.col("cell_index") == cell),
    index=["cell_index"],
    to_bin="times_relative",
    window_end=(3 * 6 + 6 * 2) * 5 + binsize,
    bin_size=binsize,
)
spikes_binned = spikes_binned.squeeze()
luminance_on_ext = np.repeat(luminance_on, int(1 / binsize))
luminance_on_ext = np.tile(luminance_on_ext, 5)
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].plot(bins[:-1], spikes_binned, c="black")
ax[1].plot(bins[:-1], luminance_on_ext, c="black")
ax[0].set_ylabel("Spike rate")
ax[1].set_ylabel("Luminance")
fig.show()
# %%
spikes_binned_single, bins_single = psth_by_index(
    spikes_on.filter(pl.col("cell_index") == cell),
    index=["cell_index"],
    to_bin="times_triggered",
    window_end=(3 * 6 + 6 * 2) + binsize,
    bin_size=binsize,
)
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].plot(bins_single[:-1], spikes_binned_single.squeeze(), c="black")
ax[1].plot(bins_single[:-1], np.repeat(luminance_on, int(1 / binsize)), c="black")
ax[0].set_ylabel("Spike rate")
ax[1].set_ylabel("Luminance")
fig.show()


# %%
def make_design_matrix(stim, d=25):
    """Create time-lag design matrix from stimulus intensity vector.

    Args:
      stim (1D array): Stimulus intensity at each time point.
      d (number): Number of time lags to use.

    Returns
      X (2D array): GLM design matrix with shape T, d

    """
    # Create version of stimulus vector with zeros before onset
    padded_stim = np.concatenate([np.zeros(d - 1), stim])

    # Construct a matrix where each row has the d frames of
    # the stimulus preceding and including timepoint t
    T = len(stim)  # Total number of timepoints (hint: number of stimulus frames)
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = padded_stim[t: t + d]

    return X


# Make design matrix
X = make_design_matrix(luminance_on_ext)


# %%
def neg_log_lik_lnp(theta, X, y):
    """Return -loglike for the Poisson GLM model.

    Args:
      theta (1D array): Parameter vector.
      X (2D array): Full design matrix.
      y (1D array): Data values.

    Returns:
      number: Negative log likelihood.

    """
    # Compute the Poisson log likelihood
    rate = np.exp(X @ theta)
    log_lik = y @ np.log(rate) - rate.sum()
    return -log_lik


def fit_lnp(stim, spikes, d=25):
    """Obtain MLE parameters for the Poisson GLM.

    Args:
      stim (1D array): Stimulus values at each timepoint
      spikes (1D array): Spike counts measured at each timepoint
      d (number): Number of time lags to use.

    Returns:
      1D array: MLE parameters

    """

    # Build the design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim, d=d)])  # pass d through

    x0 = np.random.normal(0, 0.1, d + 1)
    res = minimize(neg_log_lik_lnp, x0, args=(X, y))
    return res["x"]


def predict_spike_counts_lnp(stim, spikes, theta=None, d=25):
    """Compute a vector of predicted spike counts given the stimulus.

    Args:
      stim (1D array): Stimulus values at each timepoint
      spikes (1D array): Spike counts measured at each timepoint
      theta (1D array): Filter weights; estimated if not provided.
      d (number): Number of time lags to use.

    Returns:
      yhat (1D array): Predicted spikes at each timepoint.

    """
    y = spikes
    constant = np.ones_like(spikes)
    X = np.column_stack([constant, make_design_matrix(stim, d=d)])  # same fix here
    if theta is None:
        theta = fit_lnp(stim, y, d=d)  # and here — was passing X instead of stim
    yhat = np.exp(X @ theta)
    return yhat, theta


# %%
def make_full_design_matrix(stim, spikes, d_stim=25, d_hist=50):
    """
    Add spike history as a predictor to capture adaptation.
    """
    X_stim = make_design_matrix(stim, d=d_stim)
    X_hist = make_design_matrix(spikes, d=d_hist)  # past spikes as regressor
    constant = np.ones(len(stim))
    return np.column_stack([constant, X_stim, X_hist])


def fit_lnp_general(X, spikes):
    """
    Fit LNP GLM to a pre-built design matrix.
    Separating design matrix construction from fitting
    makes it easy to swap in any X.
    """
    y = spikes
    n_params = X.shape[1]
    x0 = np.random.normal(0, 0.1, n_params)
    res = minimize(neg_log_lik_lnp, x0, args=(X, y))
    return res["x"]


def predict_lnp_general(X, theta):
    return np.exp(X @ theta)


def make_raised_cosine_basis(d, n_basis=10, stretch=5):
    """
    Basis functions densely packed at recent lags,
    sparse at distant lags — log-compressed timescale.

    Args:
        d: total number of lags
        n_basis: number of basis vectors
        stretch: controls log compression (higher = more compression)
    """
    peaks = np.linspace(np.log(1 + stretch), np.log(d + stretch), n_basis)
    lags = np.arange(d)
    log_lags = np.log(lags + stretch)

    spacing = peaks[1] - peaks[0]
    basis = np.array(
        [
            np.maximum(0, np.cos(np.pi * (log_lags - p) / spacing / 2)) ** 2
            for p in peaks
        ]
    ).T  # shape (d, n_basis)

    return basis


def make_design_matrix_basis(stim, d=200, n_basis=10):
    """Project design matrix onto raised cosine basis."""
    X_raw = make_design_matrix(stim, d=d)
    basis = make_raised_cosine_basis(d, n_basis)
    return X_raw @ basis  # shape (T, n_basis) — massively reduced


# %%
def neg_log_lik_lnp_regularised(theta, X, y, lambda_l2=1.0):
    rate = np.exp(X @ theta)
    log_lik = y @ np.log(rate) - rate.sum()
    penalty = lambda_l2 * (theta[1:] @ theta[1:])  # don't penalise bias
    return -log_lik + penalty


def fit_lnp_regularised(X, spikes, lambda_l2=1.0):
    y = spikes
    x0 = np.random.normal(0, 0.1, X.shape[1])
    res = minimize(neg_log_lik_lnp_regularised, x0, args=(X, y, lambda_l2))
    return res["x"]


# %%
X_combined = make_full_design_matrix(luminance_on_ext, spikes_binned, 10, 10)
# %%
theta_combined = fit_lnp_regularised(X, spikes_binned, lambda_l2=300)
# %%
yhat_combined = predict_lnp_general(X_combined, theta_combined)

# %%
X_fast = make_full_design_matrix(luminance_on_ext, spikes_binned)
# %%
fig, ax = plt.subplots()
ax.plot(np.arange(-theta_combined.shape[0] * binsize, 0, binsize), theta_combined)
fig.show()

# %%
fig, ax = plt.subplots(nrows=2)
ax[0].plot(
    np.arange(-theta_combined[-10:].shape[0] * binsize, 0, binsize),
    theta_combined[-10:],
)
ax[1].plot(
    np.arange(
        np.round(-theta_combined.shape[0] * binsize, 4),
        0 - theta_combined[-10:].shape[0] * binsize - binsize,
        binsize,
    ),
    theta_combined[1:-10],
)
fig.show()
# %%%
fig, ax = plt.subplots(nrows=3, figsize=(20, 10))
ax[0].plot(bins[:-1], spikes_binned)
ax[1].plot(bins[:-1], spikes_binned)
ax[1].plot(bins[:-1], yhat_combined, c="red")
ax[2].plot(bins[:-1], luminance_on_ext)

fig.show()
# %%
theta_lnp = fit_lnp(luminance_on_ext, spikes_binned, d=10)

# %%
# Predict spike counts
yhat, theta = predict_spike_counts_lnp(luminance_on_ext, spikes_binned, theta_lnp, d=10)

# %%
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].plot(bins[:-1], spikes_binned)
ax[0].plot(bins[:-1], yhat, c="red")
ax[1].plot(bins[:-1], luminance_on_ext)

fig.show()
# %%
fig, ax = plt.subplots()
ax.plot(np.arange(-theta.shape[0] * binsize, 0, binsize), theta)
fig.show()


# %%
def poisson_deviance(y, yhat):
    # Avoid log(0)
    mask = y > 0
    d = 2 * (y[mask] @ np.log(y[mask] / yhat[mask]) - (y[mask] - yhat[mask]).sum())
    return d


dev = poisson_deviance(spikes_binned, yhat)
rng = np.random.default_rng()
dev_random = poisson_deviance(rng.permuted(spikes_binned, axis=0), yhat)

# %%
n_shuffles = 1000
dev_null = np.array(
    [
        poisson_deviance(rng.permuted(spikes_binned, axis=0), yhat_full)
        for _ in range(n_shuffles)
    ]
)

# Normalised score
dev_score = (dev_null.mean() - dev) / dev_null.std()

print(f"Model deviance:       {dev:.2f}")
print(f"Null mean ± std:      {dev_null.mean():.2f} ± {dev_null.std():.2f}")
print(f"Z-score:              {dev_score:.2f}")
print(f"p-value:              {(dev_null <= dev).mean():.4f}")

# %%
# Fast filter only
theta_fast = fit_lnp_regularised(X_fast, spikes_binned, lambda_l2=300)
yhat_fast = predict_lnp_general(X_fast, theta_fast)
dev_fast = poisson_deviance(spikes_binned, yhat_fast)

# Fast + slow filter
theta_full = fit_lnp_regularised(X_combined, spikes_binned, lambda_l2=300)
yhat_full = predict_lnp_general(X_combined, theta_full)
dev_full = poisson_deviance(spikes_binned, yhat_full)

print(f"Fast only deviance:     {dev_fast:.2f}")
print(f"Fast + slow deviance:   {dev_full:.2f}")
print(f"Improvement:            {dev_fast - dev_full:.2f}")
# %%
spikes_fff = recording.get_spikes_triggered([{"stimulus_name": "fff"}], pandas=False)
# %%
binsize = 0.05
spikes_binned_fff, bins_fff = psth_by_index(
    spikes_fff.filter(pl.col("cell_index") == cell),
    index=["cell_index"],
    to_bin="times_relative",
    window_end=(2 * 16) * 5 + binsize,
    bin_size=binsize,
)
spikes_binned_fff = spikes_binned_fff.squeeze()

# %%
# cell = (
#     recording.spikes_df.query("stimulus_name=='fff'")
#     .sort_values("qi", ascending=False)["cell_index"]
#     .values[17]
# )
fig, ax = spiketrain_plots.whole_stimulus(
    spikes_fff.filter(pl.col("cell_index") == cell), indices=["cell_index", "repeat"]
)
fig = ct.add_stimulus_to_plot(fig, [2] * 16)
fig.show()
# %%
fff_stim = np.tile(
    np.concatenate([np.ones(int(2 / binsize)) * 0.25, np.zeros(int(2 / binsize))]),
    8 * 5,
)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(bins_fff[:-1], spikes_binned_fff)
ax[1].plot(bins_fff[:-1], fff_stim)
fig.show()
# %%

# %%
X_fff = make_full_design_matrix(fff_stim, spikes_binned_fff, d_stim=10, d_hist=10)
# %%
fig, ax = plt.subplots(figsize=(10, 20))
ax.imshow(X_fff, aspect="auto")
fig.show()

# %%
# theta_fast_fff = fit_lnp_regularised(X_fff, spikes_binned_fff, lambda_l2=300)
yhat_fast_fff = predict_lnp_general(X_fff, theta_combined)
dev_fast_fff = poisson_deviance(spikes_binned_fff, yhat_fast_fff)

# %%
fig, ax = plt.subplots()
ax.plot(np.arange(-theta_combined.shape[0] * binsize, 0, binsize), theta_combined)
fig.show()

# %%
colours = np.tile(np.repeat(colours, int((2 / binsize))), 5)
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].plot(bins_fff[:-1], spikes_binned_fff)
ax[0].plot(bins_fff[:-1], yhat_fast_fff, c="red")
points = np.array([bins_fff[:-1][::20], fff_stim[::20]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, colors=colours[::20])
lc.set_linewidth(2)

# 4. Add it to your specific axes
ax[1].add_collection(lc)

fig.show()
