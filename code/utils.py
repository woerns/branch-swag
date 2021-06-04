import random

import sklearn
import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter

from logger import logger


def stratified_sample(X, y, n_samples):
    n_classes = len(set(y))
    indices = []
    for i in range(n_classes):
        indices.extend(random.sample([idx for idx, val in enumerate(y) if val == i], n_samples // n_classes))

    random.shuffle(indices)
    X_sampled = X[indices]
    y_sampled = [y[idx] for idx in indices]

    return X_sampled, y_sampled


def create_bs_train_loader(dataset, n_bootstrap, batch_size):
    bs_train_loader = []
    train_idx = list(range(len(dataset)))

    if n_bootstrap == 1:
        # If only one head, apply default case and do not bootstrap
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        bs_train_loader.append(train_loader)
    else:
        for _ in range(n_bootstrap):
            train_idx_bs = sklearn.utils.resample(train_idx, replace=True, n_samples=len(dataset))
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx_bs)

            train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=0)
            bs_train_loader.append(train_loader)

    return bs_train_loader


def fit_beta_distribution(pred_probs, dim=1):
    # Fit beta distribution using method-of-moments
    mean = pred_probs.mean(dim=dim)
    var = pred_probs.var(dim=dim)

    if (mean * (1 - mean) < var).any().item():
        logger.warning("Beta distribution parameters negative using unbiased variance. Using biased variance.")
        var_biased = pred_probs.var(dim=dim, unbiased=False)
        var = torch.where(mean * (1 - mean) < var, var_biased, var)

    v = (mean * (1 - mean) / var - 1.0)
    alpha = mean * v
    beta = (1 - mean) * v

    return alpha, beta
