import json
import os
import pickle
import random

import numpy as np
import scipy as sp
import scipy.stats
import sklearn
import sklearn.model_selection
import torch
from torch.utils.tensorboard import SummaryWriter

from calibration import compute_expected_calibration_error, compute_posterior_wasserstein_dist, fit_calibration_model
from data import ImageDataSet, load_data_transform
from logger import logger
from networks import NetEnsemble, BranchingNetwork
from networks import create_model, get_swag_branchout_layers
from plots import plot_pred_reliability, plot_uncertainty_reliability
from utils import fit_beta_distribution, create_bs_train_loader


def evaluate(model, val_loader, writer, step_num, epoch,
             confidence_level=None, calibration_model=None, tag='val', device='cpu'):
    criterion = torch.nn.NLLLoss()
    eval_results = {}

    total = correct = 0.0
    avg_loss = 0.0
    class_probs = []
    y_true = []
    y_pred = []
    posterior_params = []

    model.eval()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data

            # push tensors to set device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # outputs has shape (B, P, C) where B is batch size,
            # P is number of predictions and C is number of classes
            pred_probs = outputs.softmax(dim=-1)
            # need to average multiple predictions of all predictions
            mean_prob = pred_probs.mean(dim=-2)

            loss = criterion(mean_prob.log(), labels)
            avg_loss += loss.item()

            n_predictions = pred_probs.size(1)
            # Require more than 2 samples to estimate Beta distribution
            if n_predictions > 2:
                # To generalize confidence intervals for multiple classes, we fit a marginal distribution of the joint
                # Dirichlet distribution with parameters (alpha_0, alpha_1, ... alpha_C) for each class.
                # The marginal distribution for class j is a Beta distribution with parameters (alpha_j, sum_i,i!=j alpha_i).
                # We then take the class with the highest confidence quantile as the predicted class.
                alpha, beta = fit_beta_distribution(pred_probs, dim=-2)
                # Convert to numpy for subsequent scipy and sklearn functions
                alpha, beta = alpha.cpu().numpy(), beta.cpu().numpy()

                # Show warning if fitted Beta distribution is bimodal, i.e. alpha<1 and beta<1
                is_bimodal = (alpha < 1.) & (beta < 1.)
                if is_bimodal.any():
                    logger.info("Fitted Beta distribution is bimodal.")

                if confidence_level is not None:
                    # Uncertainty-based decision rule
                    # Run left tail statistical test
                    quantile = sp.stats.beta.ppf(1. - confidence_level, alpha, beta)
                    # quantile has shape (B, C), where B is batch size and C is number of classes

                    if calibration_model is not None:
                        # Calibrate quantile based on calibration model
                        # Mirror quantiles of right-tailed distribution to left-tailed distribution for calibration model
                        quantile[alpha < beta] = 1. - quantile[alpha < beta]

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

                        # Map recalibrated quantile back to right-tailed distribution
                        quantile[alpha < beta] = 1. - quantile[alpha < beta]

                    # Convert back to PyTorch tensor and move to device
                    quantile = torch.from_numpy(quantile).to(device)
                    predicted = quantile.argmax(dim=-1)
                else:
                    predicted = mean_prob.argmax(dim=-1)

                posterior_params.append((alpha, beta))
            else:
                predicted = mean_prob.argmax(dim=-1)

            # Compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            class_probs.append(mean_prob.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    avg_loss /= len(val_loader)
    acc = correct / total

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    class_probs = np.concatenate(class_probs)
    n_classes = class_probs.shape[-1]
    posterior_params = tuple(
        map(np.concatenate, zip(*posterior_params)))  # Change from list of tuples into tuple of numpy arrays.

    logger.info('[%d] {}_loss: %.3f'.format(tag) % (epoch, avg_loss))
    logger.info('[%d] {}_acc: %.1f %%'.format(tag) % (epoch, 100 * acc))

    # Write to tensorboard
    logger.info("Computing performance metrics...")
    writer.add_scalar('Epoch/{}'.format(tag), epoch, step_num)
    writer.add_scalar('Loss/{}'.format(tag), avg_loss, step_num)
    writer.add_scalar('Acc/{}'.format(tag), acc, step_num)
    writer.add_scalar('AUC/{}'.format(tag),
                      sklearn.metrics.roc_auc_score(y_true, class_probs, labels=list(range(n_classes)),
                                                    multi_class='ovo'), step_num)
    writer.add_scalar('F1/{}'.format(tag), sklearn.metrics.f1_score(y_true, y_pred, average='macro'), step_num)
    writer.add_scalar('ECE/{}'.format(tag), compute_expected_calibration_error(class_probs, y_true), step_num)
    logger.info("Generating prediction reliabiltiy plot...")
    writer.add_figure('Prediction reliability/{}'.format(tag), plot_pred_reliability(class_probs, y_true), step_num)

    if posterior_params:
        logger.info("Computing Wasserstein (unimodal) distance...")
        writer.add_scalar('Wasserstein dist (unimodal)/{}'.format(tag),
                          compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true,
                                                             dist_shape='unimodal'), step_num)
        logger.info("Computing Wasserstein (bimodal) distance...")
        writer.add_scalar('Wasserstein dist (bimodal)/{}'.format(tag),
                          compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true,
                                                             dist_shape='bimodal'), step_num)
        logger.info("Generating uncertainty reliability (unimodal) plot...")
        unimodal_fig, unimodal_calibration_data = plot_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                                               dist_shape='unimodal',
                                                                               calibration_model=None,
                                                                               return_data=True)
        writer.add_figure('Uncertainty reliability (unimodal)/{}'.format(tag), unimodal_fig, step_num)
        logger.info("Generating uncertainty reliability (bimodal) plot...")
        bimodal_fig, bimodal_calibration_data = plot_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                                             dist_shape='bimodal',
                                                                             calibration_model=None,
                                                                             return_data=True)
        writer.add_figure('Uncertainty reliability (bimodal)/{}'.format(tag), bimodal_fig, step_num)

        if calibration_model is not None:
            for name in calibration_model.keys():
                logger.info(f"Computing Wasserstein distance ({name}, calibrated)...")
                writer.add_scalar('Wasserstein dist ({}, calibrated)/{}'.format(name, tag),
                                  compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true,
                                                                     dist_shape=name,
                                                                     calibration_model=calibration_model[name]),
                                  step_num)
                logger.info(f"Generating uncertainty reliability ({name}, calibrated) plot...")
                writer.add_figure('Uncertainty reliability ({}, calibrated)/{}'.format(name, tag),
                                  plot_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape=name,
                                                               calibration_model=calibration_model[name]), step_num)

        if 'uncertainty_calibration_data' not in eval_results:
            eval_results['uncertainty_calibration_data'] = {}

        if unimodal_calibration_data[0].size > 0:
            eval_results['uncertainty_calibration_data']['unimodal'] = unimodal_calibration_data
        else:
            logger.warning("No unimodal calibration data. Could be caused if there are not enough samples per bin.")

        if bimodal_calibration_data[0].size > 0:
            eval_results['uncertainty_calibration_data']['bimodal'] = bimodal_calibration_data
        else:
            logger.warning("No bimodal calibration data. Could be caused if there are not enough samples per bin.")

    eval_results['y_pred'] = y_pred
    eval_results['class_probs'] = class_probs
    eval_results['y_true'] = y_true

    writer.flush()
    model.train()

    return eval_results


def train(model, train_loader, run_name, n_epochs=10, lr=0.0001, lr_hl=5,
          swag=True, swag_samples=10, swag_start=0.8, swag_lr=0.0001, swag_momentum=0.9,
          swag_interval=10, swag_branchout_layers=None, swag_bn_data_ratio=1.0,
          bootstrap=False, fold=None, confidence_level=None,
          val_loader=None, test_loader=None, eval_interval=5, ckpt_interval=None,
          checkpoint=None, log_dir=None, save=False, model_save_dir=None, device='cpu'):
    if bootstrap:
        assert isinstance(train_loader, list) and len(train_loader) > 0, \
            "Must pass in list of bootstrapped train loaders when applying bootstrap."
    else:
        if not isinstance(train_loader, list):
            train_loader = [train_loader]

    n_datasets = len(train_loader)
    steps_per_epoch = len(train_loader[0])
    calibration_model = None

    if isinstance(swag_start, float) and swag_start < 1:
        swag_start_epoch = int(n_epochs * swag_start)
    else:
        swag_start_epoch = swag_start

    results = {}
    # push model to set device (CPU or GPU)
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model

    if swag:
        if not isinstance(module, NetEnsemble) and isinstance(module.base_model, BranchingNetwork):
            swag_optimizer = torch.optim.SGD([
                {'params': module.trunk.parameters()},
                {'params': module.branches.parameters(), 'lr': swag_lr * module.n_branches}
            ], lr=swag_lr, momentum=swag_momentum, weight_decay=0.0)
        else:
            swag_optimizer = torch.optim.SGD(module.parameters(), lr=swag_lr, momentum=swag_momentum, weight_decay=0.0)

    if isinstance(module, BranchingNetwork) or (swag and isinstance(module.base_model, BranchingNetwork)):
        optimizer = torch.optim.Adam([
            {'params': module.trunk.parameters()},
            {'params': module.branches.parameters(), 'lr': lr * module.n_branches}
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_hl, gamma=0.5)

    if checkpoint is not None:
        if os.path.exists(checkpoint):
            logger.info(f"Loading checkpoint {checkpoint}...")
            ckpt = torch.load(checkpoint)
            init_epoch = ckpt['epoch']
            global_step_num = ckpt['global_step_num']
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            random.setstate(ckpt['random_state']['random'])
            np.random.set_state(ckpt['random_state']['numpy'])
            torch.random.set_rng_state(ckpt['random_state']['torch'])
            if swag:
                swag_optimizer.load_state_dict(ckpt['swag_optimizer_state'])
        else:
            logger.warning("Provided checkpoint path does not exist. Starting training from scratch.")
            init_epoch = 0
            global_step_num = 0
    else:
        init_epoch = 0
        global_step_num = 0

    if fold is not None:
        log_dir = os.path.join(log_dir, "fold{0:d}".format(fold))

    writer = SummaryWriter(log_dir)

    iterators = [iter(x) for x in train_loader]

    for epoch in range(init_epoch + 1, n_epochs + 1):  # loop over the datasets multiple times
        running_loss = 0.0

        for i in range(steps_per_epoch):
            k = random.randint(0, n_datasets - 1)

            try:
                # get the next item
                inputs, labels = next(iterators[k])
            except StopIteration:
                # restart if we reached the end of iterator
                iterators[k] = iter(train_loader[k])
                inputs, labels = next(iterators[k])

            # push tensors to set device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            # optimizer.zero_grad() # Note: this is slower
            for param in model.parameters():
                param.grad = None

            if bootstrap:
                outputs = model(inputs, branch_num=k)
                # outputs has shape (B, P, C) where B is batch size,
                # P is number of predictions and C is number of classes
                loss = criterion(outputs[:, 0, :], labels)
            else:
                outputs = model(inputs)
                # outputs has shape (B, P, C) where B is batch size,
                # P is number of predictions and C is number of classes

                n_predictions = outputs.size(1)
                for j in range(n_predictions):
                    if j == 0:
                        loss = criterion(outputs[:, j, :], labels)
                    else:
                        loss += criterion(outputs[:, j, :], labels)

                loss /= n_predictions

            loss.backward()

            if swag and epoch >= swag_start_epoch:
                swag_optimizer.step()
                if i % swag_interval == 0:
                    if bootstrap:
                        module.update_swag()  # Update SWAG params for trunk
                        module.update_swag(k)  # Update SWAG params for kth branch
                    else:
                        module.update_swag()  # Update SWAG params for trunk
                        for j in range(module.n_branches):
                            module.update_swag(j)  # Update SWAG params for all branches
            else:
                optimizer.step()

            global_step_num += 1

            # log statistics
            running_loss += loss.item()
            if i % 10 == 0:  # log every 10 mini-batches
                if i > 0:
                    logger.info('[%d, %3d] train_loss: %.5f' % (epoch, i, running_loss / 10))
                    writer.add_scalar('Loss/train', running_loss / 10, global_step_num)
                    running_loss = 0.0

                if swag and epoch >= swag_start_epoch:
                    writer.add_scalar('Learning rate/swag/train', swag_optimizer.param_groups[0]['lr'], global_step_num)
                else:
                    writer.add_scalar('Learning rate/main/train', optimizer.param_groups[0]['lr'], global_step_num)

        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            logger.info("Saving checkpoint...")
            ckpt = {
                'epoch': epoch,
                'global_step_num': global_step_num,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'random_state': {
                    'random': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.random.get_rng_state()
                }
            }
            if swag:
                ckpt['swag_optimizer_state'] = swag_optimizer.state_dict()

            if fold is not None:
                ckpt_file_name = f"{run_name}_fold{fold}_epoch{epoch}_ckpt.pth"
            else:
                ckpt_file_name = f"{run_name}_epoch{epoch}_ckpt.pth"

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            torch.save(ckpt, os.path.join(model_save_dir, ckpt_file_name))
            logger.info("Saving completed.")

        if epoch % eval_interval == 0:
            if swag and swag_branchout_layers is not None and epoch >= swag_start_epoch:
                for layer_name in swag_branchout_layers:
                    logger.info("Sampling SWAG models branching out at %s..." % layer_name)

                    # For Multi-SWAG model
                    module.sample(swag_samples, layer_name, swag_bn_data_ratio, device)

                    swag_writer = SummaryWriter(os.path.join(log_dir, 'branchout_{}'.format(layer_name)))

                    if val_loader is not None:
                        tag = 'val'
                        # tag = 'val/branchout_{}'.format(layer_name)
                        logger.info("Evaluating model on validation data...")
                        results[tag] = evaluate(model, val_loader, swag_writer, global_step_num, epoch,
                                                confidence_level=confidence_level,
                                                tag=tag,
                                                device=device)

                        if 'uncertainty_calibration_data' in results[tag]:
                            calibration_model = {}
                            if 'unimodal' in results[tag]['uncertainty_calibration_data']:
                                logger.info("Fitting unimodal calibration model...")
                                calibration_model['unimodal'] = fit_calibration_model(
                                    results[tag]['uncertainty_calibration_data']['unimodal'])
                            if 'bimodal' in results[tag]['uncertainty_calibration_data']:
                                logger.info("Fitting bimodal calibration model...")
                                calibration_model['bimodal'] = fit_calibration_model(
                                    results[tag]['uncertainty_calibration_data']['bimodal'])
                        else:
                            calibration_model = None

                        results['calibration_model/branchout_{}'.format(layer_name)] = calibration_model

                    if test_loader is not None:
                        tag = 'test'
                        logger.info("Evaluating model on test data...")
                        results[tag] = evaluate(model, test_loader, swag_writer, global_step_num, epoch,
                                                confidence_level=confidence_level,
                                                calibration_model=calibration_model,
                                                tag=tag,
                                                device=device)
                    swag_writer.close()
            else:
                if swag and epoch >= swag_start_epoch:
                    logger.info("Sampling SWAG models...")
                    module.sample(swag_samples, None, swag_bn_data_ratio, device)

                if val_loader is not None:
                    tag = 'val'
                    logger.info("Evaluating model on validation data...")
                    results[tag] = evaluate(model, val_loader, writer, global_step_num, epoch,
                                            confidence_level=confidence_level,
                                            tag=tag,
                                            device=device)

                    if 'uncertainty_calibration_data' in results[tag]:
                        logger.info("Fitting calibration model...")
                        calibration_model = {}
                        if 'unimodal' in results[tag]['uncertainty_calibration_data']:
                            calibration_model['unimodal'] = fit_calibration_model(
                                results[tag]['uncertainty_calibration_data']['unimodal'])
                        if 'bimodal' in results[tag]['uncertainty_calibration_data']:
                            calibration_model['bimodal'] = fit_calibration_model(
                                results[tag]['uncertainty_calibration_data']['bimodal'])
                    else:
                        calibration_model = None

                    results['calibration_model'] = calibration_model

                if test_loader is not None:
                    tag = 'test'
                    logger.info("Evaluating model on test data...")
                    results[tag] = evaluate(model, test_loader, writer, global_step_num, epoch,
                                            confidence_level=confidence_level,
                                            calibration_model=calibration_model,
                                            tag=tag,
                                            device=device)

        scheduler.step()

    writer.close()

    if save:
        # Save models
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        if calibration_model is not None:
            logger.info("Saving calibration model...")
            model_file_name = run_name + "_fold{0:d}_calibration.pkl".format(fold)
            with open(os.path.join(model_save_dir, model_file_name), 'wb') as file:
                pickle.dump(calibration_model, file)

        logger.info("Saving model...")
        if fold is not None:
            model_file_name = run_name + "_fold{}.pth".format(fold)
        else:
            model_file_name = run_name + '.pth'
        torch.save(model, os.path.join(model_save_dir, model_file_name))

    logger.info("Done.")

    return results


def crossvalidate(X, y, groups, args, X_test=None, y_test=None):
    """
        Function to cross-validate model on input data.
    """
    N_HEADS = args.heads
    CONFIDENCE_LEVEL = args.conf_level
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LEARNING_RATE = args.learning_rate
    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type
    USE_BOOTSTRAP = args.bootstrap
    USE_SWAG = args.swag
    RUN_NAME = args.run_name
    DEVICE = args.device
    CV_FOLDS = args.cv_folds
    TENSORBOARD_LOG_DIR = os.path.join("../runs", args.run_name, "seed{0}".format(args.seed))
    MODEL_SAVE_DIR = os.path.join("../models", args.run_name, "seed{0}".format(args.seed))

    logger.info("Run configuration:")
    logger.info('\n'.join(("{0}: {1}".format(k, v) for k, v in args.__dict__.items())))

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    with open(os.path.join(MODEL_SAVE_DIR, RUN_NAME + '.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if X_test is not None and y_test is not None:
        # Create test dataset
        test_data_transform = load_data_transform(train=False)
        test_dataset = ImageDataSet(X_test, y_test, transform=test_data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=0)

    # Create cross-validation splits
    group_kfold = sklearn.model_selection.GroupKFold(n_splits=CV_FOLDS)

    # NOTE: Infer number of classes from training data labels. This might not be ideal in some cases.
    n_classes = len(set(y))

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
        if fold < args.cv_fold_start:
            continue
        if fold > args.cv_fold_end:
            break

        logger.info("Running fold %d..." % fold)

        X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
        X_val, y_val = [X[i] for i in val_idx], [y[i] for i in val_idx]
        logger.info("Training samples: %d" % len(y_train))
        logger.info("Validation samples: %d" % len(y_val))

        # Create train dataset
        train_data_transform = load_data_transform(train=True)
        train_dataset = ImageDataSet(X_train, y_train, transform=train_data_transform)

        if USE_BOOTSTRAP:
            # Create bootstrap dataset
            train_loader = create_bs_train_loader(train_dataset, N_HEADS, batch_size=BATCH_SIZE)
        else:
            train_loader = [torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=0)]

        # Create validation datasets
        val_data_transform = load_data_transform(train=False)
        val_dataset = ImageDataSet(X_val, y_val, transform=val_data_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=0)

        if USE_SWAG:
            bn_update_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True,
                                                           num_workers=0)
            if args.swag_branchout:
                if args.swag_branchout_layers is None:
                    swag_branchout_layers = get_swag_branchout_layers(MODEL_NAME, args.branchout_layer_name)
                else:
                    swag_branchout_layers = args.swag_branchout_layers
            else:
                swag_branchout_layers = None
        else:
            bn_update_loader = None
            swag_branchout_layers = None

        # Create model
        model = create_model(MODEL_NAME, MODEL_TYPE, N_HEADS, n_classes=n_classes,
                             branchout_layer_name=args.branchout_layer_name,
                             swag=USE_SWAG, swag_rank=args.swag_rank,
                             swag_samples=args.swag_samples, bn_update_loader=bn_update_loader)

        # DataParallel
        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

        if args.ckpt_dir is not None:
            if fold is not None:
                matching_files = [x for x in os.listdir(args.ckpt_dir) if
                                  x.endswith(f'fold{fold}_epoch{args.ckpt_epoch}_ckpt.pth')]
            else:
                matching_files = [x for x in os.listdir(args.ckpt_dir) if
                                  x.endswith(f'epoch{args.ckpt_epoch}_ckpt.pth')]

            if len(matching_files) >= 1:
                if len(matching_files) > 1:
                    logger.warning("More than one matching checkpoint file found. Using first match.")
                ckpt_file_name = matching_files[0]
            else:
                raise ValueError("No checkpoint file found.")

            checkpoint = os.path.join(args.ckpt_dir, ckpt_file_name)
        else:
            checkpoint = None

        # Train model
        logger.info("Training model...")
        _ = train(model, train_loader, run_name=RUN_NAME, fold=fold, n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                  lr_hl=args.lr_halflife, swag=USE_SWAG, swag_samples=args.swag_samples,
                  swag_lr=args.swag_learning_rate,
                  swag_start=args.swag_start, swag_momentum=args.swag_momentum, swag_interval=args.swag_interval,
                  swag_branchout_layers=swag_branchout_layers, swag_bn_data_ratio=args.swag_bn_data_ratio,
                  confidence_level=CONFIDENCE_LEVEL, bootstrap=USE_BOOTSTRAP,
                  val_loader=val_loader, test_loader=test_loader,
                  eval_interval=args.eval_interval, ckpt_interval=args.ckpt_interval, checkpoint=checkpoint,
                  log_dir=TENSORBOARD_LOG_DIR, save=args.save, model_save_dir=MODEL_SAVE_DIR, device=DEVICE)
        logger.info("Training completed.")

    logger.info("Cross-validation completed.")

    return None
