import os

import torch
from PIL import Image
from torchvision import datasets, transforms


class ImageDataSet(torch.utils.data.Dataset):
    """Image datasets."""

    def __init__(self, X, y, transform=None):
        # Convert numpy array to PILImage
        self.x = list(map(lambda x: Image.fromarray(x).convert(mode='RGB'), X))  # Convert to RGB

        self.y = y
        self.n_samples = len(X)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x[idx]
        target = self.y[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def load_data_transform(train=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.5, 1.5)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return data_transform


def load_dataset(dataset_name, data_root_dir, dataset_version='full'):
    if dataset_name in ('cifar-10', 'cifar-100'):
        data_dir = os.path.join(data_root_dir, dataset_name)
        if dataset_name == 'cifar-10':
            if dataset_version == 'full':
                train = True
            elif dataset_version == 'test':
                train = False
            else:
                raise ValueError()

            dataset = datasets.CIFAR10(root=os.path.join(data_dir, dataset_version), train=train,
                                       download=True, transform=None)

            X, y, groups = dataset.data, dataset.targets, list(range(len(dataset.targets)))
        elif dataset_name == 'cifar-100':
            if dataset_version == 'full':
                train = True
            elif dataset_version == 'test':
                train = False
            else:
                raise ValueError()

            dataset = datasets.CIFAR100(root=os.path.join(data_dir, dataset_version), train=train,
                                        download=True, transform=None)

            X, y, groups = dataset.data, dataset.targets, list(range(len(dataset.targets)))
    else:
        raise ValueError("Unknown dataset name %s." % dataset_name)

    return X, y, groups
