# Imports
import numpy as np


def changepoint_posterior(
    event_times, lambda1, lambda2, tau_guess=None, prior_std=None
):
    """
    Compute the posterior distribution over changepoint times given the event times,
    using a Gaussian prior centered at tau_guess if provided.

    Parameters:
      event_times : array-like
          Sorted event times (e.g. when events occurred).
      lambda1 : float
          Rate parameter of the first process.
      lambda2 : float
          Rate parameter of the second process.
      tau_guess : float, optional
          A prior guess for the changepoint time.
      prior_std : float, optional
          The standard deviation of the Gaussian prior. Lower values imply higher confidence
          in the guess. If either tau_guess or prior_std is None, a uniform prior is used.

    Returns:
      candidate_times : np.ndarray
          Array of candidate changepoint times (each candidate is the time after an event).
      posterior : np.ndarray
          Posterior probability for each candidate changepoint.
    """
    # Ensure event_times is a numpy array.
    event_times = np.array(event_times)

    # Assume the process starts at time 0.
    full_times = np.concatenate(([0.0], event_times))
    dt = np.diff(full_times)  # interarrival times
    N = len(dt)  # total number of events

    # Cumulative event times: t_i = sum_{j=0}^{i-1} dt[j]
    cum_t = np.cumsum(dt)

    # Candidate changepoints: changepoint is assumed to occur immediately after an event.
    indices = np.arange(1, N)  # candidate indices: 1, 2, ..., N-1
    candidate_times = cum_t[:-1]  # candidate changepoint times

    # Compute log likelihoods:
    # For the first segment (first i events): L1 = lambda1^i * exp(-lambda1 * t_candidate)
    # For the second segment (remaining events): L2 = lambda2^(N-i) * exp(-lambda2 * (t_N - t_candidate))
    L1_log = indices * np.log(lambda1) - lambda1 * candidate_times
    L2_log = (N - indices) * np.log(lambda2) - lambda2 * (cum_t[-1] - candidate_times)

    log_likelihood = L1_log + L2_log

    # Compute likelihood in a numerically stable way.
    max_log = np.max(log_likelihood)
    likelihood = np.exp(log_likelihood - max_log)

    # Incorporate the prior:
    if tau_guess is None or prior_std is None:
        # Uniform prior over candidate changepoints.
        prior = np.ones_like(candidate_times)
    else:
        # Gaussian prior centered at tau_guess with standard deviation prior_std.
        prior = np.exp(-0.5 * ((candidate_times - tau_guess) / prior_std) ** 2)

    # The unnormalized posterior is likelihood times prior.
    unnormalized_posterior = likelihood * prior

    # Normalize to sum to 1.
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

    return candidate_times, posterior


def credible_interval(candidate_times, posterior, credibility=0.95):
    """
    Compute a credible interval (e.g., 95%) from the posterior distribution.

    Parameters:
      candidate_times : np.ndarray
          Array of candidate changepoint times.
      posterior : np.ndarray
          Posterior probabilities corresponding to candidate_times.
      credibility : float (default 0.95)
          The desired credibility level.

    Returns:
      lower, upper : floats
          The lower and upper bounds of the credible interval.
    """
    cdf = np.cumsum(posterior)
    lower = candidate_times[np.searchsorted(cdf, (1 - credibility) / 2)]
    upper = candidate_times[np.searchsorted(cdf, 1 - (1 - credibility) / 2)]
    return lower, upper


def detect_changepoint_single_rate(event_times, lambda2, epsilon=1e-6, threshold=0.9):
    """
    Detects the changepoint in a stream of events when the first process has an
    extremely low rate (approximated by epsilon) and the second process has rate lambda2.

    For each interarrival interval, we compute the probability that it came from the
    lambda2 process:

        f1(dt) = epsilon * exp(-epsilon * dt)   (background process)
        f2(dt) = lambda2 * exp(-lambda2 * dt)       (lambda2 process)

    Assuming equal prior probability for either hypothesis for each interval,
    the probability that the interval dt comes from lambda2 is:

        p = f2(dt) / (f1(dt) + f2(dt))

    The function then identifies the first event (based on the interarrival interval)
    for which p exceeds the given threshold.

    Parameters:
        event_times : array-like
            Sorted event times (e.g. when events occurred).
        lambda2 : float
            Rate of the second process.
        epsilon : float, optional
            A very small rate to represent the nearly zero rate of the first process.
            Default is 1e-6.
        threshold : float, optional
            The probability threshold to decide that an event is likely from the lambda2 process.
            Default is 0.9 (i.e. 90% chance).

    Returns:
        changepoint_time : float or None
            The time of the first event for which the probability that it comes from lambda2
            exceeds the threshold. Returns None if no such event is found.
        p_lambda2 : np.ndarray
            The array of computed probabilities for each interarrival interval.
    """
    event_times = np.asarray(event_times)

    # Include t = 0 as the start time.
    full_times = np.concatenate(([0.0], event_times))

    # Compute interarrival times.
    dt = np.diff(full_times)

    # Compute probability densities for each interval.
    f1 = epsilon * np.exp(
        -epsilon * dt
    )  # likelihood under the low-rate (background) model
    f2 = lambda2 * np.exp(-lambda2 * dt)  # likelihood under the lambda2 model

    # Compute the probability that each interval comes from lambda2.
    p_lambda2 = f2 / (f1 + f2)

    # Identify the first interval where p_lambda2 exceeds the threshold.
    indices = np.where(p_lambda2 > threshold)[0]

    if len(indices) == 0:
        # No interval met the threshold.
        return None, p_lambda2
    else:
        # The changepoint is assumed to occur immediately after this interval.
        changepoint_index = indices[0]
        changepoint_time = full_times[
            changepoint_index + 1
        ]  # +1 because full_times includes t=0
        return changepoint_time, p_lambda2
