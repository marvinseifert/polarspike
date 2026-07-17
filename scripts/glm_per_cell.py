from sympy.printing.pretty.pretty_symbology import line_width

from polarspike import (
    Overview,
    spiketrain_plots,
    colour_template,
    binarizer,
    histograms,
)
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
import einops
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import KFold
from pathlib import Path
from scipy.stats import mstats
from sklearn.metrics import r2_score
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import grad
import numba as nb


def tukey_weighted_mean(traces, weights, c=4.685, max_iter=50, tol=1e-6):
    v = np.asarray(traces, dtype=float)  # (n_samples, n_time)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    mu = (w[:, None] * v).sum(axis=0)  # (n_time,)

    for _ in range(max_iter):
        r = v - mu
        mad = (w[:, None] * np.abs(r)).sum(axis=0)
        scale = np.where(mad < 1e-10, 1e-10, mad / 0.6745)

        u = r / (c * scale)
        rw = np.where(np.abs(u) <= 1.0, (1.0 - u ** 2) ** 2, 0.0)

        cw = w[:, None] * rw
        cw_sum = cw.sum(axis=0)
        cw /= np.where(cw_sum < 1e-10, 1e-10, cw_sum)

        mu_new = (cw * v).sum(axis=0)
        if np.max(np.abs(mu_new - mu)) < tol:
            break
        mu = mu_new

    return mu_new


ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
jax.config.update("jax_enable_x64", True)  # needed for numerical precision


# @nb.njit(parallel=True, cache=True)
# def leaky_iir_numba(delta_on, delta_off, decay_on, decay_off):
#     T, C = delta_on.shape
#     a_on = np.empty_like(delta_on)
#     a_off = np.empty_like(delta_off)
#     for c in nb.prange(C):
#         a_on[0, c] = delta_on[0, c]
#         a_off[0, c] = delta_off[0, c]
#         for t in range(1, T):
#             a_on[t, c] = decay_on * a_on[t - 1, c] + delta_on[t, c]
#             a_off[t, c] = decay_off * a_off[t - 1, c] + delta_off[t, c]
#     return a_on, a_off
#
#
# def leaky_iir(delta, decay):
#     """Pure numpy IIR — avoids scipy overhead for simple 1-pole filter."""
#     out = np.empty_like(delta)
#     out[0] = delta[0]
#     for t in range(1, len(delta)):
#         out[t] = decay * out[t - 1] + delta[t]
#     return out
#
#
# jax.config.update("jax_enable_x64", True)  # needed for numerical precision
#
#
# # def make_leaky_integrators_jax(stim, tau_on, tau_off):
# #     decay_on  = jnp.exp(-1.0 / tau_on)
# #     decay_off = jnp.exp(-1.0 / tau_off)
#
# #     delta = jnp.abs(jnp.diff(stim, axis=0, prepend=stim[0:1]))
# #     delta = delta / (delta.max(axis=0) + 1e-9)
#
# #     stim_centred = stim - stim.mean(axis=0)
# #     on_mask  = (stim_centred > jnp.mean(stim_centred)).astype(float)
# #     off_mask = (stim_centred <= jnp.mean(stim_centred)).astype(float)
#
# #     delta_on  = delta * on_mask
# #     delta_off = delta * off_mask
#
# #     def iir_scan(x, decay):
# #         def step(carry, xt):
# #             out = decay * carry + xt
# #             return out, out
# #         _, out = jax.lax.scan(step, jnp.zeros(x.shape[1]), x)
# #         return out
#
# #     a_on  = iir_scan(delta_on,  decay_on)
# #     a_off = iir_scan(delta_off, decay_off)
#
# #     def zscore(x):
# #         return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
#
# #     return jnp.column_stack([zscore(a_on), zscore(a_off)])
#
#
# def make_leaky_integrators_jax(stim, tau_on, tau_off):
#     decay_on = jnp.exp(-1.0 / tau_on)
#     decay_off = jnp.exp(-1.0 / tau_off)
#     delta = jnp.abs(jnp.diff(stim, axis=0, prepend=stim[0:1]))
#     delta = delta / (delta.max(axis=0) + 1e-9)
#     stim_centred = stim - stim.mean(axis=0)
#     on_mask = (stim_centred > jnp.mean(stim_centred)).astype(float)
#     off_mask = (stim_centred <= jnp.mean(stim_centred)).astype(float)
#     delta_on = delta * on_mask
#     delta_off = delta * off_mask
#
#     def iir_scan_forward_rising(x, decay):
#         """
#         At each transition (x > 0), reset the counter to 0 and let it grow.
#         Between transitions, accumulate upward. Flip at the end to get a dip.
#         """
#
#         def step(carry, xt):
#             # Reset to 0 at a transition, otherwise decay * carry + xt
#             triggered = xt > 0.01
#             new_carry = jnp.where(triggered, xt, decay * carry + xt)
#             return new_carry, new_carry
#
#         _, out = jax.lax.scan(step, jnp.zeros(x.shape[1]), x)
#         return out
#
#     a_on = iir_scan_forward_rising(delta_on, decay_on)
#     a_off = iir_scan_forward_rising(delta_off, decay_off)
#
#     def flip_and_normalise(x):
#         x = -x + x.max(axis=0)  # flip per cone
#         x = x / (x.max(axis=0) + 1e-9) - 1  # normalise per cone, not global max
#         return x
#
#     return jnp.column_stack([flip_and_normalise(a_on), flip_and_normalise(a_off)])
#
#
# def neg_log_lik_jax(
#         theta, X_no_adapt, stim, y, alpha=0.5, beta=0.5, n_cones=6, d=6, spike_weight=10.0
# ):
#     C = stim.shape[1]
#     tau_on = jnp.exp(jnp.clip(theta[-2], jnp.log(1.0), jnp.log(100.0)))
#     tau_off = jnp.exp(jnp.clip(theta[-1], jnp.log(1.0), jnp.log(100.0)))
#
#     adapt = make_leaky_integrators_jax(stim, tau_on, tau_off)
#     X = jnp.column_stack([X_no_adapt, adapt])
#
#     T = len(y)
#     z = X @ theta[:-2]
#     rate = jnp.exp(jnp.clip(z, -10, 10))
#
#     w = jnp.where(y > 0, spike_weight, 1.0)
#     w = w / w.mean()
#     nll = -(w @ (y * jnp.log(rate + 1e-9) - rate)) / T
#
#     lag_weights = theta[1: n_cones * d + 1].reshape(n_cones, d)
#     group_pen = jnp.sqrt(jnp.sum(lag_weights ** 2, axis=1) + 1e-8).sum() / n_cones
#     smooth_pen = jnp.sum(jnp.diff(lag_weights, axis=1) ** 2) / (n_cones * (d - 1))
#
#     adapt_weights = theta[n_cones * d + 1: n_cones * d + 1 + C]
#     adapt_pen = jnp.sum(adapt_weights ** 2) * 0.01
#
#     return nll + alpha * group_pen + beta * smooth_pen + adapt_pen
#
#
# def centre_stimulus(stim: np.ndarray) -> np.ndarray:
#     """
#     Centre each cone channel to [-1, 1] based on its actual min/max.
#     This ensures OFF = negative, ON = positive, baseline = 0
#     regardless of how much time was spent in each state.
#     """
#     s_min = stim.min(axis=0)
#     s_max = stim.max(axis=0)
#     midpoint = (s_max + s_min) / 2.0
#     half_range = (s_max - s_min) / 2.0 + 1e-9
#     return (stim - midpoint) / half_range
#
#
# # def make_leaky_integrators_per_cone(
# #     stim: np.ndarray,
# #     tau_on: float = 25.0,
# #     tau_off: float = 10.0,
# # ) -> np.ndarray:
# #     """
# #     Two adaptation signals per cone: one for ON periods, one for OFF periods.
# #     Each decays independently with its own tau.
# #     Returns shape (T, 2*C) — [on_signals | off_signals]
# #     """
# #     from scipy.signal import lfilter
#
# #     decay_on  = np.exp(-1.0 / tau_on)
# #     decay_off = np.exp(-1.0 / tau_off)
#
# #     # Detect transitions per cone
# #     delta = jnp.abs(jnp.diff(stim, axis=0, prepend=stim[0:1]))
# #     delta /= (delta.max(axis=0) + 1e-9)
#
# #     # Determine ON vs OFF bins from stimulus state (not hardcoded indices)
# #     stim_centred = stim - stim.mean(axis=0)
# #     on_mask  = (stim_centred > np.mean(stim_centred)).astype(float)   # (T, C)
# #     off_mask = (stim_centred <= np.mean(stim_centred)).astype(float)  # (T, C)
#
# #     # Inject step energy only during the relevant period
# #     delta_on  = delta * on_mask   # only ON transitions drive ON adaptation
# #     delta_off = delta * off_mask  # only OFF transitions drive OFF adaptation
#
# #     a_on  = lfilter([1.0], [1.0, decay_on],  delta_on,  axis=0)  # (T, C)
# #     a_off = lfilter([1.0], [1.0, decay_off], delta_off, axis=0)  # (T, C)
# #     return np.column_stack([a_on, a_off])
# #     # # Z-score each signal independently
# #     # def zscore(x):
# #     #     return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
#
# #     #return np.column_stack([zscore(a_on), zscore(a_off)])  # (T, 2*C)
#
#
# def make_design_matrix(stim, d=6):
#     stim_c = centre_stimulus(stim)
#     T, C = stim_c.shape
#
#     pad = np.zeros((d - 1, C))
#     padded = np.vstack([pad, stim_c])
#     windows = sliding_window_view(padded, (d, C))
#     X_lag = windows.reshape(T, d, C)[:, ::-1, :].transpose(0, 2, 1).reshape(T, -1)
#
#     # C adaptation columns instead of 1 — no adapt here, handled in loss
#     return np.column_stack([np.ones(T), X_lag])
#
#
# def make_design_matrix_old(stim: np.ndarray, d: int = 25, threshold: float = 0.01):
#     """
#     Optimized design matrix for GLM:
#     - Standard lagged cone activations (Sustained drive)
#     - Non-linear adaptation signals (Fast/Slow suppression and Rest)
#     - Removed redundant linear 'difference' features.
#     """
#     T, C = stim.shape
#
#     # --- Feature 1: Lagged stimulus (Vectorized) ---
#     # Pad the beginning to handle the delay window
#     padding = np.zeros((d - 1, C))
#     padded_stim = np.vstack([padding, stim])
#
#     # Create windows of shape (d, C)
#     # sliding_window_view creates a (T, 1, d, C) array
#     windows = sliding_window_view(padded_stim, (d, C))
#
#     # Reshape and flip to match [t, t-1, t-2...] logic
#     # We flatten (d, C) so each row is [ch0_lag0, ch0_lag1... ch1_lag0, ch1_lag1...]
#     # stim_slice[::-1].T.flatten() logic:
#     X_lag = windows.reshape(T, d, C)[:, ::-1, :].transpose(0, 2, 1).reshape(T, -1)
#
#     return X_lag
#
#
# def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
#     """Predicted firing rate (spikes/bin)."""
#     return np.exp(np.clip(X @ theta, -100, 100))
#
#
# def neg_log_lik(
#         theta, X_no_adapt, stim, y, alpha=0.5, beta=0.5, n_cones=6, d=6, spike_weight=10.0
# ):
#     C = stim.shape[1]
#     tau_on = np.exp(np.clip(theta[-2], np.log(1.0), np.log(200.0)))
#     tau_off = np.exp(np.clip(theta[-1], np.log(1.0), np.log(200.0)))
#     adapt = make_leaky_integrators_jax(stim, tau_on=tau_on, tau_off=tau_off)
#
#     X = np.column_stack([X_no_adapt, adapt])  # bias + lags + C adapt cols
#
#     T = len(y)
#     z = X @ theta[:-2]  # all theta except log_tau
#     rate = np.exp(np.clip(z, -10, 10))
#
#     w = np.ones(T)
#     w[y > 0] = spike_weight
#     w /= w.mean()
#     nll = -(w @ (y * np.log(rate + 1e-9) - rate)) / T
#
#     lag_weights = theta[1: n_cones * d + 1].reshape(n_cones, d)
#     group_pen = np.sqrt(np.sum(lag_weights ** 2, axis=1) + 1e-8).sum() / n_cones
#     smooth_pen = np.sum(np.diff(lag_weights, axis=1) ** 2) / (n_cones * (d - 1))
#
#     # L2 penalty on adaptation weights to prevent any single cone dominating
#     adapt_weights = theta[n_cones * d + 1: n_cones * d + 1 + C]
#     adapt_pen = np.sum(adapt_weights ** 2) * 0.01
#
#     return nll + alpha * group_pen + beta * smooth_pen + adapt_pen
#
#
# # def fit_glm(X_no_adapt, spikes, stim, alpha=0.5, beta=0.5,
# #             n_cones=6, d=6, spike_weight=10.0):
#
# #     C = stim.shape[1]
# #     # theta layout: [bias, cone_filters(C*d), on_adapt(C), off_adapt(C), log_tau_on, log_tau_off]
# #     n_params = 1 + n_cones * d + 2 * C + 2
#
# #     x0 = np.zeros(n_params)
# #     x0[0] = -2.0
# #     x0[1 + n_cones * d : 1 + n_cones * d + C*2] = -0.1  # small negative adapt weights
# #     x0[-2] = np.log(2.0)  # log_tau_on
# #     x0[-1] = np.log(2.0)  # log_tau_off
# #     print(x0.shape, X_no_adapt.shape)
# #     res = minimize(
# #         neg_log_lik, x0,
# #         args=(X_no_adapt, stim, spikes, alpha, beta, n_cones, d, spike_weight),
# #         method="L-BFGS-B",
# #         options={"maxiter": 10000, "ftol": 1e-10, "gtol": 1e-7},
# #     )
#
# #     theta = res.x
# #     tau_learned = np.exp(theta[-1])
# #     adapt_weights = theta[1 + n_cones * d : 1 + n_cones * d + C]
#
# #     print(f"Learned tau:           {tau_learned:.1f} bins = {tau_learned * 50:.0f} ms")
# #     print(f"Adapt weights per cone: {adapt_weights.round(3)}")
# #     print(f"Dominant cone:          {np.argmax(np.abs(adapt_weights))}")
#
#
# #     return theta
# def fit_glm(
#         X_no_adapt, spikes, stim, alpha=0.5, beta=0.5, n_cones=6, d=6, spike_weight=10.0
# ):
#     C = stim.shape[1]
#     n_params = 1 + n_cones * d + 2 * C + 2
#
#     x0 = np.zeros(n_params)
#     x0[0] = -2.0
#     x0[1 + n_cones * d: 1 + n_cones * d + C * 2] = -0.1
#     x0[-2] = np.log(2.0)
#     x0[-1] = np.log(2.0)
#
#     # Convert fixed args to JAX arrays once
#     X_j = jnp.array(X_no_adapt)
#     S_j = jnp.array(stim)
#     y_j = jnp.array(spikes)
#
#     loss = lambda t: neg_log_lik_jax(
#         t, X_j, S_j, y_j, alpha, beta, n_cones, d, spike_weight
#     )
#     loss_jit = jax.jit(loss)
#     grad_fn = jax.jit(jax.grad(loss))
#
#     res = minimize(
#         fun=lambda t: np.array(loss_jit(jnp.array(t))),
#         x0=x0,
#         jac=lambda t: np.array(grad_fn(jnp.array(t))),
#         method="L-BFGS-B",
#         options={"maxiter": 10000, "ftol": 1e-10, "gtol": 1e-7},
#     )
#     return res.x


@nb.njit(parallel=True, cache=True)
def leaky_iir_numba(delta_on, delta_off, decay_on, decay_off):
    T, C = delta_on.shape
    a_on = np.empty_like(delta_on)
    a_off = np.empty_like(delta_off)
    for c in nb.prange(C):
        a_on[0, c] = delta_on[0, c]
        a_off[0, c] = delta_off[0, c]
        for t in range(1, T):
            a_on[t, c] = decay_on * a_on[t - 1, c] + delta_on[t, c]
            a_off[t, c] = decay_off * a_off[t - 1, c] + delta_off[t, c]
    return a_on, a_off


def leaky_iir(delta, decay):
    """Pure numpy IIR — avoids scipy overhead for simple 1-pole filter."""
    out = np.empty_like(delta)
    out[0] = delta[0]
    for t in range(1, len(delta)):
        out[t] = decay * out[t - 1] + delta[t]
    return out


jax.config.update("jax_enable_x64", True)  # needed for numerical precision


# def make_leaky_integrators_jax(stim, tau_on, tau_off):
#     decay_on  = jnp.exp(-1.0 / tau_on)
#     decay_off = jnp.exp(-1.0 / tau_off)

#     delta = jnp.abs(jnp.diff(stim, axis=0, prepend=stim[0:1]))
#     delta = delta / (delta.max(axis=0) + 1e-9)

#     stim_centred = stim - stim.mean(axis=0)
#     on_mask  = (stim_centred > jnp.mean(stim_centred)).astype(float)
#     off_mask = (stim_centred <= jnp.mean(stim_centred)).astype(float)

#     delta_on  = delta * on_mask
#     delta_off = delta * off_mask

#     def iir_scan(x, decay):
#         def step(carry, xt):
#             out = decay * carry + xt
#             return out, out
#         _, out = jax.lax.scan(step, jnp.zeros(x.shape[1]), x)
#         return out

#     a_on  = iir_scan(delta_on,  decay_on)
#     a_off = iir_scan(delta_off, decay_off)

#     def zscore(x):
#         return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)

#     return jnp.column_stack([zscore(a_on), zscore(a_off)])


def make_leaky_integrators_jax(stim, tau_on, tau_off):
    decay_on = jnp.exp(-1.0 / tau_on)
    decay_off = jnp.exp(-1.0 / tau_off)
    diff = jnp.diff(stim, axis=0, prepend=stim[0:1])
    delta_on = jnp.clip(diff, a_min=0.0)  # Increments only
    delta_off = jnp.clip(-diff, a_min=0.0)  # Decrements only

    # Normalize keeping the relative scale consistent per cone channel
    max_delta = jnp.maximum(delta_on.max(axis=0), delta_off.max(axis=0)) + 1e-9
    delta_on /= max_delta
    delta_off /= max_delta

    def iir_scan_forward_rising(x, decay):
        def step(carry, xt):
            # Direct 1-pole IIR accumulation matching your Numba implementation
            new_carry = decay * carry + xt
            return new_carry, new_carry

        _, out = jax.lax.scan(step, jnp.zeros(x.shape[1]), x)
        return out

    a_on = iir_scan_forward_rising(delta_on, decay_on)
    a_off = iir_scan_forward_rising(delta_off, decay_off)

    def flip_and_normalise(x):
        x = -x + x.max(axis=0)  # flip per cone
        x = x / (x.max(axis=0) + 1e-9) - 1  # normalise per cone, not global max
        return x

    return jnp.column_stack([flip_and_normalise(a_on), flip_and_normalise(a_off)])


def neg_log_lik_jax(
        theta, X_no_adapt, stim, y, alpha=0.5, beta=0.5, n_cones=6, d=6, spike_weight=10.0
):
    C = stim.shape[1]
    tau_on = jnp.exp(jnp.clip(theta[-2], jnp.log(1.0), jnp.log(100.0)))
    tau_off = jnp.exp(jnp.clip(theta[-1], jnp.log(1.0), jnp.log(100.0)))

    adapt = make_leaky_integrators_jax(stim, tau_on, tau_off)
    X = jnp.column_stack([X_no_adapt, adapt])

    T = len(y)
    z = X @ theta[:-2]
    rate = jnp.exp(jnp.clip(z, -10, 10))

    w = jnp.where(y > 0, spike_weight, 1.0)
    w = w / w.mean()
    nll = -(w @ (y * jnp.log(rate + 1e-9) - rate)) / T

    lag_weights = theta[1: n_cones * d + 1].reshape(n_cones, d)
    group_pen = jnp.sqrt(jnp.sum(lag_weights ** 2, axis=1) + 1e-8).sum() / n_cones
    smooth_pen = jnp.sum(jnp.diff(lag_weights, axis=1) ** 2) / (n_cones * (d - 1))

    adapt_weights = theta[n_cones * d + 1: n_cones * d + 1 + C]
    adapt_pen = jnp.sum(adapt_weights ** 2) * 0.01

    return nll + alpha * group_pen + beta * smooth_pen + adapt_pen


def centre_stimulus(stim: np.ndarray) -> np.ndarray:
    """
    Centre each cone channel to [-1, 1] based on its actual min/max.
    This ensures OFF = negative, ON = positive, baseline = 0
    regardless of how much time was spent in each state.
    """
    s_min = stim.min(axis=0)
    s_max = stim.max(axis=0)
    midpoint = (s_max + s_min) / 2.0
    half_range = (s_max - s_min) / 2.0 + 1e-9
    return (stim - midpoint) / half_range


# def make_leaky_integrators_per_cone(
#     stim: np.ndarray,
#     tau_on: float = 25.0,
#     tau_off: float = 10.0,
# ) -> np.ndarray:
#     """
#     Two adaptation signals per cone: one for ON periods, one for OFF periods.
#     Each decays independently with its own tau.
#     Returns shape (T, 2*C) — [on_signals | off_signals]
#     """
#     from scipy.signal import lfilter

#     decay_on  = np.exp(-1.0 / tau_on)
#     decay_off = np.exp(-1.0 / tau_off)

#     # Detect transitions per cone
#     delta = jnp.abs(jnp.diff(stim, axis=0, prepend=stim[0:1]))
#     delta /= (delta.max(axis=0) + 1e-9)

#     # Determine ON vs OFF bins from stimulus state (not hardcoded indices)
#     stim_centred = stim - stim.mean(axis=0)
#     on_mask  = (stim_centred > np.mean(stim_centred)).astype(float)   # (T, C)
#     off_mask = (stim_centred <= np.mean(stim_centred)).astype(float)  # (T, C)

#     # Inject step energy only during the relevant period
#     delta_on  = delta * on_mask   # only ON transitions drive ON adaptation
#     delta_off = delta * off_mask  # only OFF transitions drive OFF adaptation

#     a_on  = lfilter([1.0], [1.0, decay_on],  delta_on,  axis=0)  # (T, C)
#     a_off = lfilter([1.0], [1.0, decay_off], delta_off, axis=0)  # (T, C)
#     return np.column_stack([a_on, a_off])
#     # # Z-score each signal independently
#     # def zscore(x):
#     #     return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)

#     #return np.column_stack([zscore(a_on), zscore(a_off)])  # (T, 2*C)


def make_design_matrix(stim, d=6):
    stim_c = centre_stimulus(stim)
    T, C = stim_c.shape

    pad = np.zeros((d - 1, C))
    padded = np.vstack([pad, stim_c])
    windows = sliding_window_view(padded, (d, C))
    X_lag = windows.reshape(T, d, C)[:, ::-1, :].transpose(0, 2, 1).reshape(T, -1)

    # C adaptation columns instead of 1 — no adapt here, handled in loss
    return np.column_stack([np.ones(T), X_lag])


def make_design_matrix_old(stim: np.ndarray, d: int = 25, threshold: float = 0.01):
    """
    Optimized design matrix for GLM:
    - Standard lagged cone activations (Sustained drive)
    - Non-linear adaptation signals (Fast/Slow suppression and Rest)
    - Removed redundant linear 'difference' features.
    """
    T, C = stim.shape

    # --- Feature 1: Lagged stimulus (Vectorized) ---
    # Pad the beginning to handle the delay window
    padding = np.zeros((d - 1, C))
    padded_stim = np.vstack([padding, stim])

    # Create windows of shape (d, C)
    # sliding_window_view creates a (T, 1, d, C) array
    windows = sliding_window_view(padded_stim, (d, C))

    # Reshape and flip to match [t, t-1, t-2...] logic
    # We flatten (d, C) so each row is [ch0_lag0, ch0_lag1... ch1_lag0, ch1_lag1...]
    # stim_slice[::-1].T.flatten() logic:
    X_lag = windows.reshape(T, d, C)[:, ::-1, :].transpose(0, 2, 1).reshape(T, -1)

    return X_lag


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predicted firing rate (spikes/bin)."""
    return np.exp(np.clip(X @ theta, -100, 100))


def neg_log_lik(
        theta, X_no_adapt, stim, y, alpha=0.5, beta=0.5, n_cones=6, d=6, spike_weight=10.0
):
    C = stim.shape[1]
    tau_on = np.exp(np.clip(theta[-2], np.log(1.0), np.log(200.0)))
    tau_off = np.exp(np.clip(theta[-1], np.log(1.0), np.log(200.0)))
    adapt = make_leaky_integrators_jax(stim, tau_on=tau_on, tau_off=tau_off)

    X = np.column_stack([X_no_adapt, adapt])  # bias + lags + C adapt cols

    T = len(y)
    z = X @ theta[:-2]  # all theta except log_tau
    rate = np.exp(np.clip(z, -10, 10))

    w = np.ones(T)
    w[y > 0] = spike_weight
    w /= w.mean()
    nll = -(w @ (y * np.log(rate + 1e-9) - rate)) / T

    lag_weights = theta[1: n_cones * d + 1].reshape(n_cones, d)
    group_pen = np.sqrt(np.sum(lag_weights ** 2, axis=1) + 1e-8).sum() / n_cones
    smooth_pen = np.sum(np.diff(lag_weights, axis=1) ** 2) / (n_cones * (d - 1))

    # L2 penalty on adaptation weights to prevent any single cone dominating
    adapt_weights = theta[n_cones * d + 1: n_cones * d + 1 + 2 * C]
    adapt_pen = jnp.sum(adapt_weights ** 2) * 0.01

    return nll + alpha * group_pen + beta * smooth_pen + adapt_pen


# def fit_glm(X_no_adapt, spikes, stim, alpha=0.5, beta=0.5,
#             n_cones=6, d=6, spike_weight=10.0):

#     C = stim.shape[1]
#     # theta layout: [bias, cone_filters(C*d), on_adapt(C), off_adapt(C), log_tau_on, log_tau_off]
#     n_params = 1 + n_cones * d + 2 * C + 2

#     x0 = np.zeros(n_params)
#     x0[0] = -2.0
#     x0[1 + n_cones * d : 1 + n_cones * d + C*2] = -0.1  # small negative adapt weights
#     x0[-2] = np.log(2.0)  # log_tau_on
#     x0[-1] = np.log(2.0)  # log_tau_off
#     print(x0.shape, X_no_adapt.shape)
#     res = minimize(
#         neg_log_lik, x0,
#         args=(X_no_adapt, stim, spikes, alpha, beta, n_cones, d, spike_weight),
#         method="L-BFGS-B",
#         options={"maxiter": 10000, "ftol": 1e-10, "gtol": 1e-7},
#     )

#     theta = res.x
#     tau_learned = np.exp(theta[-1])
#     adapt_weights = theta[1 + n_cones * d : 1 + n_cones * d + C]

#     print(f"Learned tau:           {tau_learned:.1f} bins = {tau_learned * 50:.0f} ms")
#     print(f"Adapt weights per cone: {adapt_weights.round(3)}")
#     print(f"Dominant cone:          {np.argmax(np.abs(adapt_weights))}")

from scipy.optimize import basinhopping


#     return theta
def fit_glm(
        X_no_adapt, spikes, stim, alpha=0.5, beta=0.5, n_cones=6, d=6, spike_weight=10.0
):
    C = stim.shape[1]
    n_params = 1 + n_cones * d + 2 * C + 2

    x0 = np.zeros(n_params)
    x0[0] = -2.0
    x0[1 + n_cones * d: 1 + n_cones * d + C * 2] = -0.1
    x0[-2] = np.log(2.0)
    x0[-1] = np.log(2.0)

    # Convert fixed args to JAX arrays once
    X_j = jnp.array(X_no_adapt)
    S_j = jnp.array(stim)
    y_j = jnp.array(spikes)

    loss = lambda t: neg_log_lik_jax(
        t, X_j, S_j, y_j, alpha, beta, n_cones, d, spike_weight
    )
    loss_jit = jax.jit(loss)
    grad_fn = jax.jit(jax.grad(loss))

    # res = minimize(
    #     fun=lambda t: np.array(loss_jit(jnp.array(t))),
    #     x0=x0,
    #     jac=lambda t: np.array(grad_fn(jnp.array(t))),
    #     method="L-BFGS-B",
    #     options={"maxiter": 10000, "ftol": 1e-10, "gtol": 1e-7},
    # )
    # return res.x
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "jac": lambda t: np.array(grad_fn(jnp.array(t))),
        "options": {"maxiter": 1000, "ftol": 1e-9},
    }

    res = basinhopping(
        func=lambda t: np.array(loss_jit(jnp.array(t))),
        x0=x0,
        niter=50,  # Number of times it will jump out of local minima
        T=1.0,  # "Temperature" parameter controlling jump acceptance
        minimizer_kwargs=minimizer_kwargs,
    )

    return res.x


def shape_r2(predicted, actual):
    # z-score both — removes mean and amplitude
    p = (predicted - predicted.mean()) / (predicted.std() + 1e-9)
    a = (actual - actual.mean()) / (actual.std() + 1e-9)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum(a ** 2)  # ss_tot of z-scored signal = T-1 ≈ T
    return 1 - ss_res / (ss_tot + 1e-9)


def spectral_shape_r2(predicted, actual, top_k=None):
    P = np.abs(np.fft.rfft(predicted))
    A = np.abs(np.fft.rfft(actual))

    if top_k:
        P, A = P[:top_k], A[:top_k]

    # Z-score the spectra so R² is scale-invariant (matches shape_r2's approach)
    P = (P - P.mean()) / (P.std() + 1e-9)
    A = (A - A.mean()) / (A.std() + 1e-9)

    ss_res = np.sum((A - P) ** 2)
    ss_tot = np.sum(A ** 2)  # A is zero-mean after z-scoring, so this is correct

    return 1 - ss_res / (ss_tot + 1e-9)


def combined_r2(predicted, actual):
    return (shape_r2(predicted, actual) + spectral_shape_r2(predicted, actual)) / 2


def zscore(data):
    return data / np.std(data)


# %% Cone preparations
ct_cone = colour_template.Colour_template()
ct_cone.pick_stimulus("FFF_6_MC")
colours = ct_cone.colours[::2]
cone_colours = [
    colours[5],
    colours[4],
    colours[2],
    colours[0],
    "orange",
    "darkgoldenrod",
    "saddlebrown",
]
double_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_double")
single_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_oil")
double_ab_df.loc[double_ab_df["absorption"] < 0.01, "absorption"] = np.nan
single_ab_df.loc[single_ab_df["absorption"] < 0.01, "absorption"] = np.nan

# single_ab_df["absorption"] = np.log(single_ab_df["absorption"])
# double_ab_df["absorption"] = np.log(double_ab_df["absorption"])

# normalize the absorption values to 0-1 per cone
for cone in single_ab_df["cone"].unique():
    single_ab_df.loc[single_ab_df["cone"] == cone, "absorption"] = (
                                                                           single_ab_df.loc[single_ab_df[
                                                                                                "cone"] == cone, "absorption"]
                                                                           - np.nanmin(single_ab_df["absorption"])
                                                                   ) / (np.nanmax(
        single_ab_df["absorption"]) - np.nanmin(single_ab_df["absorption"]))
for cone in double_ab_df["cone"].unique():
    double_ab_df.loc[double_ab_df["cone"] == cone, "absorption"] = (
                                                                           double_ab_df.loc[double_ab_df[
                                                                                                "cone"] == cone, "absorption"]
                                                                           - np.nanmin(double_ab_df["absorption"])
                                                                   ) / (np.nanmax(
        double_ab_df["absorption"]) - np.nanmin(double_ab_df["absorption"]))
# %% Extract individual opsins
wavelengths = [365, 416, 460, 500, 535, 560, 610, 660]
wavelengths = np.flip(wavelengths)
# need to extract the absorption values for each wavelength
# for each cone type
cone_names = []
X = np.zeros((6, 8))
for i, cone in enumerate(single_ab_df["cone"].unique()):
    cone_names.append(cone)
    for j, wavelength in enumerate(wavelengths):
        X[i, j] = single_ab_df.query("cone == @cone & wavelength == @wavelength")[
            "absorption"
        ]
# add double cone
for i, d_cone in enumerate(double_ab_df["cone"].unique()):
    cone_names.append(d_cone)
    for j, wavelength in enumerate(wavelengths):
        X[i + 4, j] = double_ab_df.query("cone == @d_cone & wavelength == @wavelength")[
            "absorption"
        ]
# X[-1, :] = X[4, :] + X[5, :]
# cone_names.append("both_double")
# Max normalize X per cone
X = X / np.nanmax(X, axis=1)[:, None]
X[np.isnan(X)] = 0.001

# %% Parameters
nr_steps = 16
time_per_step = 2
time_total = time_per_step * nr_steps
bin_size = 0.05
bins_per_step = int(time_per_step / bin_size)
bin_ms = 1 / bin_size
nr_cones = 6
filter_d = 6  # 300ms filter
tau_adapt = 10  # ~1.25s adaptation time constant
d = 7
# %%
all_psth = np.load(r"/media/mawa/fast_data/swav_embedding/psths.npy")
all_psth_mean = np.mean(all_psth, axis=1)
nr_repeats = all_psth.shape[1]
# %%
indices_repeats = []
starts = np.arange(0, nr_steps, time_per_step)
for start in starts:
    indices_repeats.append(np.arange(0, bins_per_step, 1) + start * bins_per_step)
indices_repeats = np.array(indices_repeats).flatten().astype(int)

stim = np.repeat(X, int(bins_per_step * 2), axis=1)
stim[:, indices_repeats + 40] = 0
X = make_design_matrix(stim.T, d=d)

# %% fit all cells
on_taus = np.zeros(all_psth_mean.shape[0])
off_taus = np.zeros_like(on_taus)
combined_errors = np.zeros((all_psth_mean.shape[0], nr_repeats))
cone_weights = np.zeros((all_psth.shape[0], nr_cones, d))
biases = np.zeros_like(on_taus)
cell_predictions = np.zeros((all_psth_mean.shape[0], all_psth_mean.shape[1]))
total_error = np.zeros_like(on_taus)
for cell in tqdm(range(all_psth_mean.shape[0])):
    theta_hat = fit_glm(
        X,
        zscore(all_psth_mean[cell]),
        stim.T,
        alpha=0.1,
        beta=0.5,
        n_cones=nr_cones,
        d=d,
        spike_weight=50.0,
    )
    # extract learned adaptation taus
    on_taus[cell], off_taus[cell] = np.exp(theta_hat[-2:])
    # predict response
    adapt = make_leaky_integrators_jax(
        stim.T, tau_on=on_taus[cell], tau_off=off_taus[cell]
    )
    X_full = np.column_stack([X, adapt])
    y_pred = predict(X_full, theta_hat[:-2])
    total_error[cell] = combined_r2(y_pred, zscore(all_psth_mean[cell]))
    y_pred_norm = y_pred / np.max(y_pred)
    biases[cell] = theta_hat[0]
    weights = theta_hat[1: nr_cones * d + 1]  # The actual cone filters
    cone_weights[cell] = weights.reshape(nr_cones, d)
    cell_predictions[cell] = y_pred
    for repeat in range(nr_repeats):
        repeat_data = zscore(all_psth[cell, repeat, :])
        repeat_data = repeat_data / np.max(repeat_data)
        combined_errors[cell, repeat] = combined_r2(y_pred_norm, repeat_data)

# %%
from scipy.stats import pearsonr

combined_pearson = np.zeros_like(combined_errors)
corr_error = np.zeros_like(total_error)
for cell in tqdm(range(all_psth_mean.shape[0])):
    y_pred = cell_predictions[cell]
    corr_error[cell], _ = pearsonr(y_pred, zscore(all_psth_mean[cell]))
    y_pred_norm = y_pred / np.max(y_pred)
    for repeat in range(nr_repeats):
        repeat_data = zscore(all_psth[cell, repeat, :])
        repeat_data = repeat_data / np.max(repeat_data)
        combined_pearson[cell, repeat], _ = pearsonr(y_pred_norm, repeat_data)
# %%

fig, ax = plt.subplots(figsize=(7, 6))
ax.hist(
    np.max(combined_pearson, axis=1),
    bins=np.arange(-1, 1, 0.01),
    density=True,
    alpha=0.3,
    label="best fit",
    color="green",
)

ax.hist(
    np.min(combined_pearson, axis=1),
    bins=np.arange(-1, 1, 0.01),
    density=True,
    alpha=0.3,
    label="worst fit",
    color="red",
)
ax.hist(
    np.mean(combined_pearson, axis=1),
    bins=np.arange(-1, 1, 0.01),
    density=True,
    alpha=0.3,
    label="average fit",
    color="blue",
)
ax.vlines(np.nanmean(np.min(combined_pearson, axis=1)), 0, 3, color="red", zorder=10)
ax.vlines(np.nanmean(np.max(combined_pearson, axis=1)), 0, 3, color="green", zorder=10)
ax.vlines(np.nanmean(np.mean(combined_pearson, axis=1)), 0, 3, color="blue", zorder=10)
ax.legend()
ax.set_xlabel("Pearson Correlation")
ax.set_ylabel("Density count (area norm. to 1)")
fig.show()
# %%
fig, ax = plt.subplots()
ax.hist(combined_errors)
fig.show()
# %%
corr_error[np.isnan(corr_error)] = 0
# %%
fig, ax = plt.subplots(nrows=2, figsize=(10, 10), sharey=True)
ax[0].hist(
    on_taus,
    bins=np.linspace(0, 21, 100),
    density=True,
    weights=corr_error,
    color="grey",
    label="ON",
)
ax[1].hist(
    off_taus,
    bins=np.linspace(0, 21, 100),
    density=True,
    weights=corr_error,
    color="black",
    label="OFF",
)
ax[0].legend()
ax[1].legend()
ax[1].set_ylabel("count")
ax[1].set_xlabel("adaptation tau")
fig.show()
# %%
fig, ax = plt.subplots(nrows=1, figsize=(10, 10), sharey=True)
ax.hist(
    np.exp(biases) / (1 / bin_ms),
    bins=np.linspace(0, np.max(np.exp(biases) / (1 / bin_ms)), 100),
    density=True,
    weights=corr_error,
)
ax.vlines(
    np.average(np.exp(biases) / (1 / bin_ms), weights=corr_error), 0, 0.2, color="red"
)
fig.show()
# %% plot some examples
# best prediction
bins = np.arange(0, nr_steps * time_per_step, bin_size)
sorted_indices = np.argsort(corr_error)[::-1]

# %%
id = 55
print(corr_error[sorted_indices[id]])
w_reshaped = cone_weights[sorted_indices[id]]
fig, ax = plt.subplots(
    figsize=(10, 7),
    nrows=8,
    gridspec_kw={"height_ratios": [2] * 6 + [10] + [1]},
)

for i in range(6):
    ax[i].plot(
        np.arange(0, d * bin_size, bin_size),
        np.flipud(w_reshaped[i]),
        label=f"Cone {cone_names[i]}",
        c=cone_colours[i],
    )
    ax[i].set_ylim((np.min(w_reshaped), np.max(w_reshaped)))
    ax[i].hlines(
        0, 0, np.max(np.arange(0, d * bin_size, bin_size)), color="black", lw=0.5
    )
    ax[i].set_axis_off()
    ax[i].legend()
    ax[i].set_xlim((0, 5))
ax[-2].fill_between(
    np.arange(0, nr_steps * time_per_step, bin_size),
    zscore(all_psth_mean[sorted_indices[id]]),
    color="black",
    label="cluster mean",
)
ax[-2].plot(
    np.arange(0, nr_steps * time_per_step, bin_size),
    cell_predictions[sorted_indices[id]],
    c="red",
    label="predicted",
)
ax[-2].hlines(
    np.exp(biases[sorted_indices[id]]) / bin_ms,
    0,
    nr_steps * time_per_step,
    color="blue",
    label="predicted base firing rate",
)
ax[0].set_title("Cone filters")
ax[-2].set_title("real vs predicted")
ax[-2].legend()
fig = ct.add_stimulus_to_plot(fig, [time_per_step] * nr_steps, names=False)
ax[-1].set_xlim((0, nr_steps * time_per_step))
ax[-2].set_xlim((0, nr_steps * time_per_step))
fig.show()
# %%

adapt = make_leaky_integrators_jax(
    stim.T, on_taus[sorted_indices[id]], off_taus[sorted_indices[id]]
)
adapt_combined = adapt[:, :nr_cones] + adapt[:, nr_cones: nr_cones * 2]
# %%
fig, ax = plt.subplots(nrows=nr_cones + 2, sharex=True)
for c in range(nr_cones):
    ax[c].plot(adapt_combined[:, c])
ax[c + 1].plot(np.sum(adapt_combined, axis=1))
fig = ct.add_stimulus_to_plot(fig, [40] * 16)
fig.show()
# %%
cone_weights_averaged = np.average(cone_weights, axis=0, weights=corr_error)

fig, ax = plt.subplots(
    figsize=(10, 7),
    nrows=6,
)

for i in range(6):
    ax[i].plot(
        np.arange(0, d * bin_size, bin_size),
        np.flipud(cone_weights_averaged[i]),
        label=f"Cone {cone_names[i]}",
        c=cone_colours[i],
    )
    ax[i].set_ylim((np.min(cone_weights_averaged), np.max(cone_weights_averaged)))
    ax[i].hlines(
        0, 0, np.max(np.arange(0, d * bin_size, bin_size)), color="black", lw=0.5
    )
    ax[i].set_axis_off()
    ax[i].legend()
    ax[i].set_xlim((0, 5))
fig.show()
# %%
import matplotlib.colors as mcolors


def _darken(hex_color: str, factor: float = 0.15) -> str:
    """Return a near-black version of hex_color (factor=0 → black, 1 → original)."""
    c = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(tuple(v * factor for v in c))


_cmaps = {
    "rose_div": "#fe7cfe",
    "periwinkle_div": "#7c86fe",
    "lime_div": "#8afe7c",
    "salmon_div": "#fe7c7c",
    "orange_div": "orange",
    "goldenrod_div": "darkgoldenrod",
}

for name, colour in _cmaps.items():
    dark = _darken(colour, factor=0.6)
    cmap = mcolors.LinearSegmentedColormap.from_list(name, [dark, "#ffffff", colour])
    plt.colormaps.register(cmap, force=True)

# %%
import numpy as np
from sklearn.cluster import AgglomerativeClustering

cone_weights_norm = cone_weights - np.median(cone_weights, axis=2, keepdims=True)
cone_weights_norm = cone_weights_norm / np.max(
    np.abs(cone_weights_norm), axis=2, keepdims=True
)
num_traces = cone_weights_norm.shape[0]
flattened_cones = cone_weights_norm.reshape(num_traces, -1)

# Group them into, say, 10 distinct style groups based on cosine distance
# 'cosine' metric requires 'average' or 'complete' linkage in sklearn
clusterer = AgglomerativeClustering(n_clusters=30, metric="cosine", linkage="average")
cluster_labels = clusterer.fit_predict(flattened_cones)
unique_labels, nr_cells_in_cluster = np.unique(cluster_labels, return_counts=True)
nr_sorted_labels = unique_labels[np.argsort(nr_cells_in_cluster)[::-1]]
nr_cells_sorted = np.sort(nr_cells_in_cluster)[::-1]
# Group your traces by their assigned cluster
sorted_indices = np.argsort(cluster_labels)
cones_sorted = cone_weights[sorted_indices]
sorted_cluster_groups = cluster_labels[sorted_indices]
# %%
cmaps_to_use = [
    "rose_div",
    "periwinkle_div",
    "lime_div",
    "salmon_div",
    "orange_div",
    "goldenrod_div",
]
fig, ax = plt.subplots(figsize=(5, 10), ncols=nr_cones)
for cone in range(nr_cones):
    cone_weight = cones_sorted[:, cone, :] - np.median(
        cones_sorted[:, cone, :], axis=1, keepdims=True
    )
    cone_weight = cone_weight / np.max(np.abs(cone_weight), axis=1, keepdims=True)
    ax[cone].imshow(
        cone_weight, cmap=cmaps_to_use[cone], aspect="auto", interpolation="kaiser"
    )
fig.show()
# %%
from matplotlib.patches import Rectangle

unique_clusters = np.unique(sorted_cluster_groups)
num_clusters = unique_clusters.shape[0]

# 1. Double the nrows because each cluster now takes 2 vertical grid slots
fig, ax = plt.subplots(
    nrows=num_clusters * 2,
    ncols=7,
    figsize=(20, 30),
    gridspec_kw={"width_ratios": [10, 1, 1, 1, 1, 1, 1]},
)
step_positions = np.arange(0, nr_steps * time_per_step, time_per_step)

for idx, cluster in enumerate(nr_sorted_labels):
    # Calculate the grid row indices for this specific cluster block
    row_top = idx * 2
    row_bottom = row_top + 1

    # --- Step 1: Combine the two rows in the first column for the PSTH ---
    # We remove the bottom axis and combine it with the top one using subplotspec
    gs = ax[row_top, 0].get_subplotspec().get_gridspec()
    ax[row_top, 0].remove()
    ax[row_bottom, 0].remove()

    # Create a new, joined axis that spans from row_top to row_bottom
    psth_ax = fig.add_subplot(gs[row_top: row_bottom + 1, 0])

    # Plot the cluster mean PSTH
    psth_ax.plot(
        bins,
        np.mean(all_psth_mean[cluster_labels == cluster], axis=0)
        / np.max(np.mean(all_psth_mean[cluster_labels == cluster], axis=0)),
        color="black",
        label=f"Cluster {cluster}, n={nr_cells_sorted[idx]}",
    )
    psth_ax.plot(
        bins,
        np.mean(
            (
                    cell_predictions[cluster_labels == cluster]
                    / np.max(
                cell_predictions[cluster_labels == cluster], axis=1, keepdims=True
            )
            ),
            axis=0,
        )
        / np.max(
            np.mean(
                (
                        cell_predictions[cluster_labels == cluster]
                        / np.max(
                    cell_predictions[cluster_labels == cluster],
                    axis=1,
                    keepdims=True,
                )
                ),
                axis=0,
            )
        ),
        color="red",
        linestyle="--",
    )
    for step in range(nr_steps):
        psth_ax.add_patch(
            Rectangle(
                (step_positions[step], 0),
                time_per_step,
                1,
                color=ct.colours[step],
                alpha=0.2,
            )
        )
    psth_ax.legend()
    psth_ax.set_axis_off()
    interpolation_method = (
        "blackman" if np.sum(cluster_labels == cluster) > 20 else "nearest"
    )
    # --- Step 2: Plot the two rows of Cone Weights ---
    for cone in range(nr_cones):
        cluster_mask = cluster_labels == cluster

        # Row 1 (Top): Your standard normalized weights
        ax[row_top, cone + 1].plot(
            np.flipud(np.mean(cone_weights_norm[cluster_mask, cone, :], axis=0)),
            color=cone_colours[cone],
        )
        ax[row_top, cone + 1].set_ylim(
            (np.min(cone_weights_norm), np.max(cone_weights_norm))
        )
        ax[row_top, cone + 1].hlines(
            0, 0, cone_weights_norm.shape[2], "black", "--", linewidth=0.2
        )

        # Row 2 (Bottom): Your second set of weights
        # (Replace 'cone_weights_norm' with whatever alternative weights array you want)]

        ax[row_bottom, cone + 1].imshow(
            np.flip(
                cone_weights_norm[cluster_mask, cone, :], axis=1
            ),  # <-- Change data source here if needed
            cmap=cmaps_to_use[cone],
            aspect="auto",
            interpolation=interpolation_method,
        )

        ax[row_bottom, cone + 1].set_axis_off()
        ax[row_top, cone + 1].set_axis_off()
ax[0, 0].set_axis_on()

fig.show()
fig.savefig(r"/media/mawa/fast_data/glm_results/overview_fig.svg")

# %%
fig, ax = plt.subplots(nrows=num_clusters, figsize=(20, 20))
for idx, cluster in enumerate(nr_sorted_labels):
    ax[idx].plot(
        bins,
        np.mean(all_psth_mean[cluster_labels == cluster], axis=0)
        / np.max(np.mean(all_psth_mean[cluster_labels == cluster], axis=0)),
        color="black",
        label=f"Cluster {cluster}, n={nr_cells_sorted[idx]}",
    )
    ax[idx].plot(
        bins,
        np.mean(
            (
                    cell_predictions[cluster_labels == cluster]
                    / np.max(
                cell_predictions[cluster_labels == cluster], axis=1, keepdims=True
            )
            ),
            axis=0,
        )
        / np.max(
            (
                    cell_predictions[cluster_labels == cluster]
                    / np.max(
                cell_predictions[cluster_labels == cluster], axis=1, keepdims=True
            )
            )
        ),
        color="red",
        label=f"Cluster {cluster}, n={nr_cells_sorted[idx]}",
    )
    for step in range(nr_steps):
        ax[idx].add_patch(
            Rectangle(
                (step_positions[step], 0),
                time_per_step,
                1,
                color=ct.colours[step],
                alpha=0.2,
            )
        )
fig.show()
# %%
for idx, cluster in enumerate(nr_sorted_labels):
    print(np.max(corr_error[cluster_labels == cluster]))
# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(
    (
            cell_predictions[cluster_labels == 0]
            / np.max(cell_predictions[cluster_labels == 0], axis=1, keepdims=True)
    ).T,
    alpha=0.5,
)
ax.plot(
    np.mean(
        (
                cell_predictions[cluster_labels == 0]
                / np.max(cell_predictions[cluster_labels == 0], axis=1, keepdims=True)
        ),
        axis=0,
    ),
    c="black",
)
ax.set_ylim(0, 1.5)
fig.show()
# %%
psth_norm = all_psth / np.std(all_psth, axis=2, keepdims=True)
std_repeats = np.std(psth_norm, axis=1)
# %%
bins = np.arange(0, 32, 0.05)
example_cell = [11]  # np.random.choice(all_psth.shape[0], 1)
combined_errors[example_cell]
example_psth = all_psth[example_cell].squeeze()
fig, ax = plt.subplots(figsize=(20, 10), nrows=2)
ax[0].fill_between(
    bins,
    np.clip(np.mean(example_psth, axis=0) - np.std(example_psth, axis=0), 0, 100),
    np.mean(example_psth, axis=0),
    color="blue",
    alpha=0.2,
)
ax[0].fill_between(
    bins,
    np.mean(example_psth, axis=0) + np.std(example_psth, axis=0),
    np.max(example_psth, axis=0),
    color="blue",
    alpha=0.2,
)
ax[0].plot(bins, np.mean(example_psth, axis=0), color="black")
ax[1].plot(
    bins,
    np.cumsum(
        np.min(example_psth, axis=0),
    ),
    color="blue",
)
ax[0].plot(bins, cell_predictions[example_cell[0]], color="red")
ax[1].plot(bins, np.cumsum(np.mean(example_psth, axis=0)), color="black")
ax[1].plot(bins, np.cumsum(np.max(example_psth, axis=0)), color="blue")
ax[1].plot(bins, np.cumsum(cell_predictions[example_cell[0]]), color="red")
fig.show()
print(np.sum(example_psth, axis=1) / np.max(np.sum(example_psth, axis=1)))
# %%

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(bins, np.std(example_psth, axis=0))
fig.show()
