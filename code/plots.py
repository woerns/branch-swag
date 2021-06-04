import scipy as sp
import scipy.stats
import sklearn
import sklearn.metrics

import matplotlib.pyplot as plt
from calibration import compute_pred_reliability, compute_uncertainty_reliability


def plot_pred_reliability(class_probs, y_true, bins=30, min_obs_per_bin=10):
    bin_centers, _, confidence, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    y = confidence - acc
    fig = plt.figure()
    plt.plot(bin_centers, y, marker='.', markersize=8)
    plt.axhline(0, xmin=0., xmax=1., color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([-1., 1.])
    plt.grid()
    plt.title('Reliability diagram')
    plt.xlabel('Predicted probability')
    plt.ylabel('Predicted probability - Accuracy')
    plt.close()

    return fig


def plot_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape=None, calibration_model=None, bins=30, min_obs_per_bin=10, return_data=False):
    # Note: If using a calibration model, set bins and min_obs_per_bin to the same config
    # that was used to estimate the model. Otherwise, the plot will not show actual calibration performance.
    exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape=dist_shape, calibration_model=calibration_model,
                                                           bins=bins, min_obs_per_bin=min_obs_per_bin)

    fig = plt.figure()
    plt.plot(exp_cdf, obs_cdf, color='C1', marker='.', markersize=8)
    plt.axline([0, 0], [1, 1], color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.grid()
    plt.title('P-P plot')
    plt.xlabel('Expected cdf')
    plt.ylabel('Observed cdf')
    plt.close()

    if return_data:
        return fig, (exp_cdf, obs_cdf)
    else:
        return fig


def plot_confidence_level_performance(class_probs, posterior_params, y_true,
                                 confidence_levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999],
                                 calibration_model=None):
    alpha, beta = posterior_params
    n_classes = class_probs.shape[-1]
    acc = []
    auc = []
    f1_score = []
    for confidence_level in confidence_levels:
        quantile = sp.stats.beta.ppf(1. - confidence_level, alpha, beta)

        if calibration_model is not None:
            # Calibrate quantile based on calibration model
            # Mirror quantiles of right-tailed distribution to left-tailed distribution for calibration model
            quantile[alpha < beta] = 1. - quantile[alpha < beta]
            is_bimodal = (alpha < 1.) & (beta < 1.)

            if 'unimodal' in calibration_model:
                quantile_unimodal = quantile[~is_bimodal]
                if quantile_unimodal.size > 0:
                    quantile_unimodal = calibration_model['unimodal'].predict(quantile_unimodal)
                    quantile[~is_bimodal] = quantile_unimodal

            if 'bimodal' in calibration_model:
                quantile_bimodal = quantile[is_bimodal]
                if quantile_bimodal.size > 0:
                    quantile_bimodal = calibration_model['bimodal'].predict(quantile_bimodal)
                    quantile[is_bimodal] = quantile_bimodal

            # B, C = quantile.shape  # B is batch size and C is number of classes
            # quantile = quantile.flatten()  # Note: calibration model only accepts 1D array
            # quantile = calibration_model.predict(quantile)
            # quantile = quantile.reshape(B, C)

            # Map recalibrated quantile back to right-tailed distribution
            quantile[alpha < beta] = 1. - quantile[alpha < beta]

        predicted = quantile.argmax(axis=-1)
        pred_probs = quantile/quantile.sum(axis=-1).reshape(-1, 1)

        acc.append((y_true == predicted).sum()/len(y_true))
        auc.append(sklearn.metrics.roc_auc_score(y_true, pred_probs, labels=list(range(n_classes)), multi_class='ovo'))
        f1_score.append(sklearn.metrics.f1_score(y_true, predicted, average='macro'))

    predicted_baseline = class_probs.argmax(axis=-1)
    acc_baseline = (y_true == predicted_baseline).sum()/len(y_true)
    fig_acc = plt.figure()
    plt.plot(confidence_levels, acc, color='C0', marker='.', markersize=8)
    plt.axhline(acc_baseline, color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.grid()
    plt.title('Accuracy vs confidence level')
    plt.xlabel('Confidence level')
    plt.ylabel('Accuracy')
    plt.legend(['Quantile', 'Mean prob'])
    plt.close()

    auc_baseline = sklearn.metrics.roc_auc_score(y_true, class_probs, labels=list(range(n_classes)), multi_class='ovo')
    fig_auc = plt.figure()
    plt.plot(confidence_levels, auc, color='C0', marker='.', markersize=8)
    plt.axhline(auc_baseline, color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.grid()
    plt.title('AUC vs confidence level')
    plt.xlabel('Confidence level')
    plt.ylabel('AUC')
    plt.legend(['Quantile', 'Mean prob'])
    plt.close()

    f1_score_baseline = sklearn.metrics.f1_score(y_true, predicted_baseline, average='macro')
    fig_f1_score = plt.figure()
    plt.plot(confidence_levels, f1_score, color='C0', marker='.', markersize=8)
    plt.axhline(f1_score_baseline, color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.grid()
    plt.title('F1-score vs confidence level')
    plt.xlabel('Confidence level')
    plt.ylabel('F1-score')
    plt.legend(['Quantile', 'Mean prob'])
    plt.close()

    return fig_acc, fig_auc, fig_f1_score