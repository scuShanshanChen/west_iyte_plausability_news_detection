import logging
import argparse
import coloredlogs
import torch
import os
from torchtext import data
from torch import nn
from torch import optim

from data.plausible import read_files
from baselines.han import HAN, add_han_specific_parser
from utils.dl_runner import train

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for Plausible Detection Models')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--target_class', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_vocab_size', default=25000, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--momentum', default=0.9)

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

    kfold_range = train_data.kfold(args.kfold)
    accuracys = []
    avg_losses = []

    for kfold_i, (train_range, test_range) in enumerate(kfold_range):
        torch.cuda.empty_cache()

        logging.info('Training {} th fold'.format(kfold_i + 1))

        train_data_k, dev_data_k = train_data.get_fold(fields=[('text', text_field), ('label', label_field)],
                                                       train_indexs=train_range,
                                                       test_indexs=test_range)

        train_iter, dev_iter = data.BucketIterator.splits((train_data_k, dev_data_k), device=args.device,
                                                          batch_sizes=(args.batch_size, args.batch_size),
                                                          sort_key=lambda x: len(x.text))

        logging.info("Number of training samples {train}, number of dev samples {dev}".format(train=len(train_iter),
                                                                                              dev=len(dev_iter)))

        model = HAN(args)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # add later
        loss_criterion = nn.BCEWithLogitsLoss()

        model.to(args.device)
        train(train_iter, dev_iter, model, optimizer, loss_criterion, args)
