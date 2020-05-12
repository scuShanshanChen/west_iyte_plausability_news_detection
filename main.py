import logging
import coloredlogs
import torch
import os
from torch import nn
from torch import optim

from data.plausible import read_files
from baselines.model_factory import model_maps
from configs.args import args
from utils.dl_runner import train_kfold, train_split

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

random_seed = 42


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_random_seeds(args.seed)

    args.target_path = './datasets/'
    args.plausible_path = os.path.join(args.target_path, 'plausible.tsv')
    args.implausible_path = os.path.join(args.target_path, 'implausible.tsv')
    args.pre_embeddings_path = os.path.join(args.target_path, 'GoogleNews-vectors-negative300.bin')

    logging.debug('Reading datasets...')
    train_data, test_data, text_field, label_field = read_files(args)

    args.vectors = text_field.vocab.vectors

    model = model_maps[args.model](args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_criterion = nn.BCEWithLogitsLoss()

    if 'kfold' == args.training_mode:
        train_kfold(model, train_data, text_field, label_field, optimizer, loss_criterion, args)
        # todo add cross validation score calculation
    elif 'random_split':
        train_split(model, train_data, text_field, label_field, optimizer, loss_criterion, args)
    else:
        logging.debug('Invalid option.')

    # todo evaluate with test data
