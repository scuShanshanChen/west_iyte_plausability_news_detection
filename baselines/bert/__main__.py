import argparse
import logging

import coloredlogs
import torch
from transformers import BertForSequenceClassification, AdamW
from configs.args import args

from baselines.bert.bert import add_bert_specific_parser, read_files, train_split

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('__main__.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

random_seed = 42


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for Plausible Detection Models')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_gpu', default=False, type=str2bool)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--checkpoint_dir', default='./datasets/model')

    # add model specific params
    parser = add_bert_specific_parser(parser)

    args = parser.parse_args()

    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    init_random_seeds(args.seed)

    args.target_path = './datasets/'

    logging.debug('Reading datasets...')
    train_data, dev_data, test_data = read_files(args)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,
                                                          output_attentions=False, output_hidden_states=False)
    model.cpu()
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    train_split(model, train_data, dev_data, optimizer, args)

    # todo evaluate with test data
