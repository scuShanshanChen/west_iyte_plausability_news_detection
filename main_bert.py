import logging
import argparse
import coloredlogs
import torch
import os
from torch import nn
from torch import optim
from transformers import BertForSequenceClassification, BertConfig, AdamW

from baselines.bert import add_bert_specific_parser,read_files, train_split

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main_bert.py')
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
    parser = add_bert_specific_parser(parser)

    args = parser.parse_args()
    #args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cpu'
    init_random_seeds(args.seed)

    args.target_path = './datasets/'

    logging.debug('Reading datasets...')
    train_data, dev_data, test_data = read_files(args)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,
                                                          output_attentions=False, output_hidden_states=False)
    model.cpu()
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    train_split(model, train_data, dev_data, optimizer, args)








    #todo evaluate with test data
