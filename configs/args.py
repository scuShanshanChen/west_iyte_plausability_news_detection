import argparse

RANDOM_STATE = 42


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


def add_han_specific_parser(parser):
    parser.add_argument('--word_num_hidden', type=int, default=50)
    parser.add_argument('--sentence_num_hidden', type=int, default=50)
    return parser


def data_sampling_parser(parser):
    parser.add_argument('--feature', choices=['headline', 'body', 'merged'], type=str)
    return parser


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
parser.add_argument('--training_mode', choices=['kfold', 'random_split'])
parser.add_argument('--model', choices=['han'])
parser.add_argument('--fine_tune', choices=[True, False], type=str2bool, default=False)
parser.add_argument('--clip', type=float)
parser.add_argument('--eps', type=float)
parser.add_argument('--weight_decay', type=float)

parser = data_sampling_parser(parser)
args = parser.parse_args()
