import logging
import argparse
import coloredlogs
import torch
import os
from torch import nn
from torch import optim

from data.plausible import read_files
from baselines.han import HAN, add_han_specific_parser
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
    parser = argparse.ArgumentParser(description='Experiments for Plausible Detection Models')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--target_class', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_vocab_size', default=100000, type=int)
    parser.add_argument('--sent_max_len', default=50, type=int)
    parser.add_argument('--word_max_len', default=50, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--checkpoint_dir', default='./datasets/model')
    parser.add_argument('--training_mode', choices=['kfold','random_split'])

    # add model specific params
    parser = add_han_specific_parser(parser)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_random_seeds(args.seed)

    args.target_path = './datasets/'
    args.plausible_path = os.path.join(args.target_path, 'plausible.tsv')
    args.implausible_path = os.path.join(args.target_path, 'implausible.tsv')
    args.pre_embeddings_path = os.path.join(args.target_path, 'GoogleNews-vectors-negative300.bin')

    logging.debug('Reading datasets...')
    train_data, test_data, text_field, label_field = read_files(args)

    args.vectors = text_field.vocab.vectors

    model = HAN(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_criterion = nn.BCEWithLogitsLoss()

    if 'kfold' == args.training_mode:
        train_kfold(model, train_data,text_field, label_field, optimizer, loss_criterion, args)
        # todo add cross validation score calculation
    elif 'random_split':
        train_split(model, train_data,text_field, label_field, optimizer, loss_criterion, args)
    else:
        logging.debug('Invalid option.')

    #todo evaluate with test data
