import logging
import argparse
import coloredlogs
import torch
from data.plausible import read_files

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for Plausible Detection Models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_path = './datasets/'
    plausible_path = './dataset/plausible.tsv'
    implausible_path = './dataset/implausible.tsv'

    logging.debug('Reading datasets...')
    train, test, text_field, label_field = read_files(plausible_path, implausible_path, target_path,
                                                      args.pre_embeddings_path)



