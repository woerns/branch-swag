import copy

import torch
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG

from swag import BranchSWAG


class NetEnsemble(torch.nn.Module):
    def __init__(self, models):
        super(NetEnsemble, self).__init__()
        self.ensemble = torch.nn.ModuleList(models)

    def update_swag(self, k=None):
        if k is None:
            for k in range(len(self.ensemble)):
                self.ensemble[k].update_swag()
        else:
            self.ensemble[k].update_swag()

    def sample(self, k=None):
        if k is None:
            for k in range(len(self.ensemble)):
                self.ensemble[k].sample()
        else:
            self.ensemble[k].sample()

    def forward(self, x):
        ensemble_outputs = []
        for model in self.ensemble:
            ensemble_outputs.append(model(x))

        # Concatenate K model outputs (B, P, ...) to (B, K*P, ...). B is batch size.
        ensemble_outputs = torch.cat(ensemble_outputs, dim=1)

        return ensemble_outputs


class BranchingNetwork(torch.nn.Module):
    def __init__(self, model, n_classes, n_branches, branchout_layer_name=None):
        super(BranchingNetwork, self).__init__()
        self._n_classes = n_classes
        self._n_branches = n_branches
        self._branchout_layer_name = branchout_layer_name
        self._create_branchout(model)

    @property
    def n_branches(self):
        return self._n_branches

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_outputs(self):
        return self.n_branches * self.n_classes

    def _get_named_layers(self, model):
        if isinstance(model, DenseNet):
            named_layers = []
            for name, c in model.named_children():
                if name == 'features':
                    named_layers.extend(list(c.named_children()))
                elif name == 'classifier':
                    named_layers.append(('adapter', torch.nn.Sequential(
                        torch.nn.ReLU(inplace=True),
                        torch.nn.AdaptiveAvgPool2d((1, 1)),
                        torch.nn.Flatten(1)
                    )))
                    named_layers.append(('classifier', torch.nn.Linear(c.in_features, self.n_classes)))
                else:
                    raise ValueError("Unexpected layer name.")
        elif isinstance(model, ResNet):
            named_layers = []
            for name, c in model.named_children():
                if name == 'fc':
                    named_layers.append(('adapter', torch.nn.Flatten(1)))
                    named_layers.append(('classifier', torch.nn.Linear(c.in_features, self.n_classes)))
                else:
                    named_layers.append((name, c))
        elif isinstance(model, VGG):
            named_layers = []
            for name, c in model.named_children():
                if name == 'features':
                    layer_count = 0
                    for k, v in c.named_children():
                        named_layers.append((f'layer{layer_count}_{k}', v))
                        if isinstance(v, torch.nn.MaxPool2d):
                            layer_count += 1
                elif name == 'classifier':
                    named_layers.append(('adapter', torch.nn.Flatten(1)))
                    fc_layers = list(model.classifier.children())
                    named_layers.append(('fc', torch.nn.Sequential(*fc_layers[:-1])))
                    named_layers.append(('classifier', torch.nn.Linear(fc_layers[-1].in_features, self.n_classes)))
                else:
                    named_layers.append((name, c))
        else:
            raise NotImplementedError("Named layers undefined for model.")

        return named_layers

    def _create_branchout(self, model):
        named_layers = self._get_named_layers(model)

        # Branchout at first layer if not provided, i.e. we get a single branch, no trunk.
        if self._branchout_layer_name is None:
            self._branchout_layer_name = named_layers[0][0]

        branchout = False
        self.trunk = torch.nn.Sequential()
        self.branches = torch.nn.ModuleList()
        for _ in range(self.n_branches):
            self.branches.append(torch.nn.Sequential())

        for name, layer in named_layers:
            if self._branchout_layer_name in name:
                branchout = True

            if branchout:
                for b in self.branches:
                    layer_copy = copy.deepcopy(layer)
                    # Currently only re-initializing last classifier layer.
                    if name == 'classifier':
                        layer_copy.reset_parameters()

                    b.add_module(name, layer_copy)
            else:
                self.trunk.add_module(name, layer)

    def forward(self, x, branch_num=None):
        # Output has shape (B, P, C) where B is batch size, P is number of heads/predictions
        # and C is number of classes
        x = self.trunk(x)
        if branch_num is None:
            outputs = torch.stack([b(x) for b in self.branches], dim=1)
        else:
            outputs = self.branches[branch_num](x).unsqueeze(dim=1)

        return outputs


def create_branching_network(model_name, n_classes, n_heads=10, branchout_layer_name=None):
    if model_name.startswith('resnet'):
        # ResNet
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    elif model_name.startswith('densenet'):
        # DenseNet
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    elif model_name.startswith('vgg'):
        # VGG
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    else:
        raise ValueError("Unknown model name %s." % model_name)

    # Create branching network based on base model
    model = BranchingNetwork(model, n_classes, n_heads, branchout_layer_name=branchout_layer_name)

    return model


def get_swag_branchout_layers(model_name, branchout_layer_name=None):
    if model_name.startswith('densenet'):
        branchout_layers = [
            'conv0',
            'denseblock1',
            'denseblock2',
            'denseblock3',
            'denseblock4',
            'classifier'
        ]
    elif model_name.startswith('resnet'):
        branchout_layers = [
            'conv1',
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            'classifier'
        ]
    elif model_name.startswith('vgg'):
        branchout_layers = [
            'layer0',
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            'fc',
            'classifier'
        ]
    else:
        raise ValueError("Unknown model name %s." % model_name)

    if branchout_layer_name is not None:
        start_idx = branchout_layers.index(branchout_layer_name)
        branchout_layers = branchout_layers[start_idx:]

    return branchout_layers


def create_model(model_name, model_type, n_heads, n_classes, branchout_layer_name=None, swag=False, swag_rank=10,
                 swag_samples=10, bn_update_loader=None):
    if swag:
        assert bn_update_loader is not None, "Must provide training data loader for BN update when applying SWAG."

    if model_type == 'branching':
        model = create_branching_network(model_name, n_heads=n_heads, n_classes=n_classes,
                                         branchout_layer_name=branchout_layer_name)
        if swag:
            model = BranchSWAG(model, n_rank=swag_rank, bn_update_loader=bn_update_loader)
    elif model_type == 'ensemble':
        if swag:
            raise NotImplementedError("SWAG has not been implemented for ensembles.")
        models = [create_branching_network(model_name, n_heads=1, n_classes=n_classes) for _ in range(n_heads)]
        model = NetEnsemble(models)
    else:
        raise ValueError("Unknown model type %s." % model_type)

    return model
