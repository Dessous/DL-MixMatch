import torchvision
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


def train_val_split(labels, n_labeled, n_classes, val_rate):
    idx_labeled = []
    idx_unlabeled = []
    idx_val = []
    for label in range(n_classes):
        idxs = np.where(labels == label)[0]
        idxs = idxs[np.random.permutation(len(idxs))]
        split = int(len(idxs) * val_rate)
        idx_train = idxs[split:]
        idx_val.extend(idxs[:split])
        assert n_labeled <= len(idx_train), f"{n_labeled} > {len(idx_train)}"
        idx_labeled.extend(idx_train[:n_labeled])
        idx_unlabeled.extend(idx_train[n_labeled:])
    return idx_labeled, idx_unlabeled, idx_val


def get_dataset(config, logger):
    assert config.dataset.name == 'cifar10' or config.dataset.name == 'cifar100', "Wrong dataset name"
    if config.dataset.name == 'cifar10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean['cifar10'], std['cifar10'])
        ])
        base_train = torchvision.datasets.CIFAR10(
            'data/', train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            'data/', train=False,
            transform=transform, download=True)
        n_classes = 10
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean['cifar100'], std['cifar100'])
        ])
        base_train = torchvision.datasets.CIFAR100(
            'data/', train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            'data/', train=False,
            transform=transform, download=True)
        n_classes = 100
    base_train.targets = np.array(base_train.targets)
    idx_labeled, idx_unlabeled, idx_val = train_val_split(
        base_train.targets,
        config.dataset.labeled_per_class,
        n_classes,
        config.dataset.val_rate
    )
    logger.add_row(f"train labeled size: {len(idx_labeled)}\n"
                   f"train unlabeled size: {len(idx_unlabeled)}\n"
                   f"validation size: {len(idx_val)}\n"
                   f"test size: {len(test_dataset.data)}")

    labeled_train_dataset = LabeledCIFAR(base_train.data[idx_labeled],
                                         base_train.targets[idx_labeled],
                                         transform)
    unlabeled_train_dataset = UnlabeledCIFAR(base_train.data[idx_unlabeled], transform)
    val_dataset = LabeledCIFAR(base_train.data[idx_val],
                               base_train.targets[idx_val],
                               transform)
    return labeled_train_dataset, unlabeled_train_dataset, val_dataset, test_dataset


def get_loaders(config, logger):
    train_labeled, train_unlabeled, val, test = get_dataset(config, logger)
    labeled_train_loader = DataLoader(
        train_labeled,
        batch_size=config.train.batch_size,
        shuffle=True
    )
    unlabeled_train_loader = DataLoader(
        train_unlabeled,
        batch_size=config.train.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val,
        batch_size=config.train.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test,
        batch_size=config.train.batch_size,
        shuffle=False
    )
    return labeled_train_loader, unlabeled_train_loader, val_loader, test_loader


class LabeledCIFAR(Dataset):
    def __init__(self, img, target, transform):
        super(LabeledCIFAR, self).__init__()
        self.img = img
        self.target = target
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.img[idx]), self.target[idx]

    def __len__(self):
        return len(self.img)


class UnlabeledCIFAR(Dataset):
    def __init__(self, img, transform):
        super(UnlabeledCIFAR, self).__init__()
        self.img = img
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.img[idx])

    def __len__(self):
        return len(self.img)
