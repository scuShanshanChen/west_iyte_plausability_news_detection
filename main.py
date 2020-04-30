import logging
import argparse
import coloredlogs
import torch
import numpy as np
from torchtext import data
from torch import nn
from data.plausible import read_files
from baselines.han import HAN, add_han_specific_parser
from utils.dl_runner import train, eval

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
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--kfold', default=5, type=int)

    # add model specific params
    parser = add_han_specific_parser(parser)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_random_seeds(args.seed)

    args.target_path = './datasets/'
    args.plausible_path = './dataset/plausible.tsv'
    args.implausible_path = './dataset/implausible.tsv'

    logging.debug('Reading datasets...')
    train, test, text_field, label_field = read_files(args)
    args.vectors = text_field.vocab.vectors

    kfold_range = train.kfold(args.kfold)
    accuracys = []
    avg_losses = []

    for train_range, test_range in kfold_range:
        train, dev = train.get_fold(fields=[('text', text_field), ('label', label_field)], train_indexs=train_range,
                                    test_indexs=test_range)

        train_iter, dev_iter = data.Iterator.splits((train, dev), device=args.device,
                                                    batch_sizes=(args.batch_size, len(dev)))

        model = HAN(args)
        optimizer = None  # add later
        loss_criterion = nn.BCEWithLogitsLoss()
        model = train(train_iter, dev_iter, model, optimizer, loss_criterion, args)

        accuracy, loss = eval(dev_iter, model)
        accuracys.append(accuracy)
        avg_losses.append(loss)

    print("avarage accuracy is %s, loss is %s".format(np.average(accuracys)), np.average(avg_losses))
