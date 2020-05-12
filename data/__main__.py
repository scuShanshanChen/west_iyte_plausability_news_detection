import logging
import os

from configs.args import args
from data.plausible import make_random_split, make_kfold_split

if __name__ == '__main__':
    target_path = './datasets/'
    plausible_path = os.path.join(target_path, 'plausible.tsv')
    implausible_path = os.path.join(target_path, 'implausible.tsv')

    if 'random_split' == args.training_mode:
        make_random_split(plausible_path, implausible_path, target_path, seed=args.seed)
    elif 'kfold' == args.training_mode:
        make_kfold_split(plausible_path, implausible_path, target_path, kfold=args.kfold, seed=args.seed)
    else:
        logging.error('Not valid mode')
