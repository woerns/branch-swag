import copy
from collections import deque

import numpy as np
import torch

from logger import logger


class BranchSWAG(torch.nn.Module):
    def __init__(self, base_model, n_rank=10, bn_update_loader=None):
        # assert isinstance(self.base_model, BranchingNetwork), "Base model must be of type BranchingNetwork."
        super(BranchSWAG, self).__init__()
        self.base_model = base_model
        self.n_rank = n_rank
        self.param_names = [name for name, _ in self.base_model.named_parameters()]
        self.param_mean = [None for _ in range(self.n_branches+int(self.n_branches>1))]  # Length is number of branches + trunk
        self.param_second_mom = [None for _ in range(self.n_branches+int(self.n_branches>1))]
        self.param_buffer = [deque() for _ in range(self.n_branches+int(self.n_branches>1))]
        self.n_iter = [0] * (self.n_branches + 1)
        self.bn_update_loader = bn_update_loader
        self.sampled_branches = None
        self.swag_trunk = None
        self._swag_start_layer = None
        self._is_bn_calibrated = False
        self.min_var = 1e-30

    @property
    def n_branches(self):
        return self.base_model.n_branches

    @property
    def trunk(self):
        return self.base_model.trunk

    @property
    def branches(self):
        return self.base_model.branches

    def _get_params(self, branch_num=None):
        # TODO: Use torch Tensors instead of numpy arrays
        params = []
        if branch_num is None:
            for p in self.base_model.trunk.parameters():
                params.append(p.detach().cpu().numpy().flatten())
        else:
            for p in self.base_model.branches[branch_num].parameters():
                params.append(p.detach().cpu().numpy().flatten())

        params = np.concatenate(params, axis=0)

        return params

    def _sample_params(self, branch_num=None):
        assert branch_num is None or 0 <= branch_num < self.n_branches, "branch_num must be between 0 and {} or None.".format(
            self.n_branches)

        if branch_num is None:
            idx = self.n_branches
        else:
            idx = branch_num

        param_cov_diag = np.clip(self.param_second_mom[idx] - self.param_mean[idx] ** 2, self.min_var, None)

        z1 = np.random.normal(size=(len(self.param_mean[idx]),))
        z2 = np.random.normal(size=(len(self.param_buffer[idx]),))

        # NOTE: Compute dot product incrementally, since np.dot causes a memory limit error
        # for large matrices due to its internal workings of copying the input matrices.
        dot_prod = np.zeros_like(self.param_mean[idx])
        for i in range(len(self.param_buffer[idx])):
            # NOTE: Different from Maddox et. al. (https://arxiv.org/abs/1902.02476)
            # Subtracting current mean rather than running mean.
            dot_prod += (self.param_buffer[idx][i] - self.param_mean[idx])*z2[i]

        sampled_deviations = (1. / np.sqrt(2) * np.sqrt(param_cov_diag) * z1 +
                              1. / np.sqrt(2 * (self.n_rank - 1)) * dot_prod)

        params = self.param_mean[idx] + sampled_deviations

        return params

    def _set_params(self, model, params, device):
        #  TODO: Use torch Tensors instead of numpy arrays
        state_dict = copy.deepcopy(model.state_dict())
        idx = 0
        for name, _ in model.named_parameters():
            n = state_dict[name].numel()
            state_dict[name].copy_(torch.from_numpy(params[idx:idx + n]).reshape(state_dict[name].shape).to(device))
            idx += n

        model.load_state_dict(state_dict, strict=True)

    def update_bn(self, swag_bn_data_ratio=1.0, device=None):
        assert self.bn_update_loader is not None, "Must provide bn_update_loader to update BN layers."

        def _reset_bn(module):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)

        def _get_momenta(module, momenta):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                momenta[module] = module.momentum

        def _set_momenta(module, momenta):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

        # Save BN momenta and reset BN running mean and running var for all BN layers
        # was_training = []
        momenta = {}
        self.swag_trunk.train()
        self.swag_trunk.apply(_reset_bn)
        self.swag_trunk.apply(lambda module: _get_momenta(module, momenta))
        for j in range(self.n_branches):
            self.branches_fixed[j].train()
            self.branches_fixed[j].apply(_reset_bn)
            self.branches_fixed[j].apply(lambda module: _get_momenta(module, momenta))

            self.branches_swag[j].train()
            self.branches_swag[j].apply(_reset_bn)
            self.branches_swag[j].apply(lambda module: _get_momenta(module, momenta))

        # Only need to run if there are any BN layers
        if momenta:
            logger.info("Updating batch normalization layers.")
            n_batches_to_run = int(len(self.bn_update_loader) * swag_bn_data_ratio)
            n = 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(self.bn_update_loader):
                    if i >= n_batches_to_run:
                        break
                    b = inputs.size(0)
                    logger.debug("Processed samples: %d" % n)
                    momentum = b / float(n + b)
                    for module in momenta.keys():
                        module.momentum = momentum

                    if device is not None:
                        inputs = inputs.to(device)

                    self.swag_forward(inputs)
                    n += b
        else:
            logger.info("No batch normalization layers detected. No BN update required.")


    def update_swag(self, branch_num=None):
        assert branch_num is None or 0 <= branch_num < self.n_branches, "branch_num must be between 0 and {} or None.".format(self.n_branches)
        if self.n_branches == 1 and branch_num is None:
            # Return if we only have one single branch (i.e. no trunk)
            return None

        if branch_num is None:
            idx = self.n_branches
        else:
            idx = branch_num

        # Store and compute SWAG parameters for branch branch_num
        if len(self.param_buffer[idx]) >= self.n_rank:
            self.param_buffer[idx].popleft()
        self.param_buffer[idx].append(self._get_params(branch_num=branch_num))

        if self.n_iter[idx] == 0:
            self.param_mean[idx] = self.param_buffer[idx][-1]
            self.param_second_mom[idx] = self.param_buffer[idx][-1] ** 2
        else:
            self.param_mean[idx] = self.n_iter[idx] / (self.n_iter[idx] + 1) * self.param_mean[idx] + self.param_buffer[idx][-1] / (self.n_iter[idx] + 1)
            self.param_second_mom[idx] = self.n_iter[idx] / (self.n_iter[idx] + 1) * self.param_second_mom[idx] + self.param_buffer[idx][
                -1] ** 2 / (self.n_iter[idx] + 1)

        self.n_iter[idx] += 1

        # neg_var_count = (self.param_second_mom[branch_num] - self.param_mean[branch_num] ** 2 < 0).sum()
        # if neg_var_count > 0:
        #     print("Warning: %d negative variances." % neg_var_count)

    def sample(self, n_samples, swag_start_layer=None, swag_bn_data_ratio=1.0, device='cpu'):
        # Sample from Gaussian posterior
        assert all(x is not None for x in self.param_mean) and all(x is not None for x in self.param_second_mom), "First and second moment required. Call update_swag first."

        # Update SWAG trunk with new SWA
        self.swag_trunk = copy.deepcopy(self.base_model.trunk)
        if self.n_branches > 1:
            self._set_params(self.swag_trunk, self.param_mean[-1], device)  # Use SWA for trunk params

        # Create SWAG branches
        self.branches_fixed = torch.nn.ModuleList([torch.nn.Sequential() for _ in range(self.n_branches)])
        self.branches_swag = torch.nn.ModuleList([torch.nn.Sequential() for _ in range(self.n_branches)])

        for j in range(self.n_branches):
            swag = False
            for name, child in self.base_model.branches[j].named_children():
                if swag_start_layer is None or swag_start_layer in name:
                    swag = True
                if swag:
                    self.branches_swag[j].add_module(name, copy.deepcopy(child))
                else:
                    self.branches_fixed[j].add_module(name, copy.deepcopy(child))

        self.sampled_branches = torch.nn.ModuleList([torch.nn.ModuleList() for _ in range(self.n_branches)])

        for j in range(self.n_branches):
            # Set SWA params for fixed part of branches
            n_fixed_params = 0
            for p in self.branches_fixed[j].parameters():
                n_fixed_params += p.numel()
            self._set_params(self.branches_fixed[j], self.param_mean[j][:n_fixed_params], device)

            # Set sampled SWAG params for SWAG part of branches
            for i in range(n_samples):
                branch_params = self._sample_params(branch_num=j)
                sampled_branch_model = copy.deepcopy(self.branches_swag[j])
                # NOTE: Sampled params are for the entire branch, but we only need a subset of the params.
                n_swag_params = 0
                for p in sampled_branch_model.parameters():
                    n_swag_params += p.numel()
                self._set_params(sampled_branch_model, branch_params[-n_swag_params:], device)

                self.sampled_branches[j].append(sampled_branch_model)

        # Update BN for trunk and fixed branches and then reuse for all SWAG branches
        self.update_bn(swag_bn_data_ratio=swag_bn_data_ratio, device=device)

    def swag_forward(self, x):
        assert self.sampled_branches is not None, "No SWAG samples available. Must sample SWAG models first."
        sampled_outputs = []

        with torch.no_grad():
            trunk_features = self.swag_trunk(x)
            for j in range(self.n_branches):
                fixed_features = self.branches_fixed[j](trunk_features)
                for i in range(len(self.sampled_branches[j])):
                    sampled_outputs.append(self.sampled_branches[j][i](fixed_features))
        # Sampled outputs have shape (B, P, C) where B is batch size, P is number of predictions
        # and C is number of classes
        sampled_outputs = torch.stack(sampled_outputs, dim=1)

        return sampled_outputs

    def forward(self, x, **kwargs):
        if self.training:
            return self.base_model(x, **kwargs)
        else:
            if self.sampled_branches is not None:
                return self.swag_forward(x)
            else:
                return self.base_model(x, **kwargs)