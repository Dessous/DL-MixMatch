import torchvision
import os

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


def get_dataset(config, logger):
    assert config.name == 'cifar10' or config.name == 'cifar100', "Wrong dataset name"
    if config.name == 'cifar10':
        base_dataset = torchvision.datasets.CIFAR10(
            'data/',
            download=True)
    else:
        base_dataset = torchvision.datasets.CIFAR100(
            'data/',
            download=True)
