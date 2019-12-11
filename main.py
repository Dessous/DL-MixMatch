import argparse
import yaml
import os
import torch
from logger import *
from dataset import *
from train_baseline import train_baseline
from train_mixmatch import train_mixmatch
from augmentations import Augmentor


class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key]) for key in data})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment info')
    parser.add_argument(
        '-e', '--experiment',
        help='Name of config from experiments/ folder. Option \'all\' runs all configs',
        required=True,
        metavar='NAME'
    )
    args = parser.parse_args()

    if not os.path.exists('results/' + args.experiment):
        os.mkdir('results/' + args.experiment)

    logger = Logger('results/' + args.experiment + '/')

    with open('experiments/' + args.experiment + '.yaml') as config_file:
        config = AttrDict.from_nested_dict(yaml.safe_load(config_file))
        logger.add_row(config_file.read())

    # fixing the seed
    np.random.seed(config.random.seed)
    torch.manual_seed(config.random.seed)
    torch.backends.cudnn.deterministic = True

    train_labeled_loader, train_unlabeled_loader, test_loader = \
        get_loaders(config, logger)
    augmentor = None
    if config.augmentations.use:
        augmentor = Augmentor(config)
    if config.train.use_mixmatch:
        train_mixmatch(train_labeled_loader, train_unlabeled_loader, test_loader, logger, augmentor, config)
    else:
        train_baseline(train_labeled_loader, test_loader, logger, augmentor, config.train.lr, config.train.num_epoch)
