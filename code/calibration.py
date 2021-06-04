import numpy as np
import scipy as sp
import scipy.stats

from sklearn.isotonic import IsotonicRegression
from logger import logger


def compute_pred_reliability(class_probs, y_true, bins=30, min_obs_per_bin=10):
    assert len(class_probs) == len(y_true), "class_probs and y_true must have same length."

    # Use log spacing of bin sizes at the edges of interval [0,1].
    # We do that because after enough training, all predictions tend to be either close to 0 or 1
    # and we need more granular buckets to estimate the true accuracy.
    # Note going too granular (e.g. bin centered at 0.99999) can introduce a bias
    # since the accuracy can be estimated only in discrete steps of 1/val_sample_size.

    N, C = class_probs.shape
    log_bins = bins//3 + int((bins%3==2))
    lin_bins = bins//3 + int((bins%3==1))
    bin_centers = np.concatenate((np.logspace(-3, -1, log_bins),
                                 np.linspace(0.1, 0.9, lin_bins+2)[1:-1],
                                 1.0 - np.logspace(-1, -3, log_bins)))

    correct_count = np.zeros_like(bin_centers)
    count = np.zeros_like(bin_centers)
    confidence = np.zeros_like(bin_centers)

    for i in range(N):
        # Aggregate confidence across all classes
        for j in range(C):
            # Find nearest bucket
            b = np.abs(class_probs[i][j] - bin_centers).argmin()
            confidence[b] += class_probs[i][j]

            if y_true[i] == j:
                correct_count[b] += 1

            count[b] += 1

    acc = np.zeros_like(bin_centers)
    for b in range(bins):
        if count[b] <= min_obs_per_bin:
            # Remove accuracy estimates with too few observations
            acc[b] = np.nan
            confidence[b] = np.nan
        else:
            acc[b] = correct_count[b] / count[b]
            # Apply "rule of three" if none or all observations are positive
            significance_level = 0.1
            thresh = np.exp(np.log(significance_level) / count[b])
            acc[b] = np.clip(acc[b], a_min=1. - thresh, a_max=thresh)

            confidence[b] = confidence[b] / count[b]

    return bin_centers, count, confidence, acc


def compute_expected_calibration_error(class_probs, y_true, bins=30, min_obs_per_bin=10):
    _, count, confidence, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    ece = np.nansum(np.abs(confidence - acc)*count)/count.sum()

    return ece


def compute_empirical_cdf(observations):
    if observations.size == 0:
        return np.empty(0)

    observations_sorted = np.copy(observations)
    observations_sorted.sort()
    N = len(observations)
    empirical_cdf = np.empty((N,))
    prev_value = observations_sorted[0]
    write_idx = 0
    for i in range(N):
        if observations_sorted[i] != prev_value:
            empirical_cdf[write_idx:i] = i
            write_idx = i
            prev_value = observations_sorted[i]

    empirical_cdf[write_idx:] = N

    empirical_cdf /= N

    return empirical_cdf


def compute_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape=None, calibration_model=None, bins=30, min_obs_per_bin=10):
    assert len(class_probs) == len(posterior_params[0]) == len(posterior_params[1]), "class_probs and posterior_params must have same length."
    assert dist_shape in ('unimodal', 'bimodal'), "dist_shape has to be either unimodal or bimodal."
    bin_centers, _, _, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    N, C = class_probs.shape
    alpha, beta = posterior_params
    obs_probs = np.empty((N, C))
    obs_probs[:] = np.nan

    bin_edges = np.concatenate(([0.0], 0.5*(bin_centers[:-1] + bin_centers[1:]), [1.0]), axis=0)
    # Compute a NxC bin matrix containing the corresponding bin for class_probs[i][j]
    bins = np.clip(np.searchsorted(bin_edges, class_probs, side='right'), 1, len(bin_centers)) - 1

    if dist_shape == 'unimodal':
        include_idx = (alpha >= 1.) | (beta >= 1.)
    if dist_shape == 'bimodal':
        include_idx = (alpha < 1.) & (beta < 1.)

    if not include_idx.any():
        logger.warning(f"No {dist_shape} samples.")

    # Since the Beta distribution is generally not symmetric but is mirrored when its parameters
    # alpha and beta are exchanged, we map all distributions to be left-tailed (where alpha >= beta) before
    # measuring and calibrating uncertainty quantiles.
    left_tailed_idx = (alpha >= beta)
    right_tailed_idx = (~left_tailed_idx)
    left_tailed_idx &= include_idx
    right_tailed_idx &= include_idx

    obs_probs[left_tailed_idx] = sp.stats.beta.cdf(acc[bins[left_tailed_idx]], alpha[left_tailed_idx], beta[left_tailed_idx])
    obs_probs[right_tailed_idx] = sp.stats.beta.cdf(1. - acc[bins[right_tailed_idx]], beta[right_tailed_idx], alpha[right_tailed_idx])

    obs_probs = obs_probs.flatten()
    obs_probs = obs_probs[~np.isnan(obs_probs)]  # Remove NaNs
    eps = 1e-10  # Note: Too small numbers cause problems with calibration model
    obs_probs[obs_probs < eps] = 0.0

    if obs_probs.size > 0 and calibration_model is not None:
        obs_probs_calibrated = calibration_model.predict(obs_probs)
        if calibration_model.out_of_bounds == 'nan':
            # Use uncalibrated values if observed probability outside of training data range.
            obs_probs = np.where(np.isnan(obs_probs_calibrated), obs_probs, obs_probs_calibrated)
        else:
            obs_probs = obs_probs_calibrated

    obs_probs.sort()  # sort in ascending order
    exp_cdf = obs_probs
    obs_cdf = compute_empirical_cdf(obs_probs)

    return exp_cdf, obs_cdf


def compute_wasserstein_dist(cdf_x, cdf_y):
    """Computes Wasserstein distance for two given CDFs."""
    bin_edges = np.concatenate(([0.0], 0.5*(cdf_x[:-1] + cdf_x[1:]), [1.0]), axis=0)
    dx = (bin_edges[1:] - bin_edges[:-1])
    wasserstein_dist = (np.abs(cdf_x-cdf_y) * dx).sum()

    return wasserstein_dist


def compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true, dist_shape=None, calibration_model=None, bins=30, min_obs_per_bin=10):
    exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape=dist_shape,
                                                           calibration_model=calibration_model,
                                                           bins=bins, min_obs_per_bin=min_obs_per_bin)
    return compute_wasserstein_dist(exp_cdf, obs_cdf)


def fit_calibration_model(calibration_data):
    """Fit calibration model using isotonic regression."""
    exp_cdf, obs_cdf = calibration_data
    X, y = exp_cdf, obs_cdf

    calibration_model = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibration_model = calibration_model.fit(X, y)
    
    return calibration_model
