import argparse


def add_han_specific_parser(parser):
    parser.add_argument('--word_num_hidden', type=int, default=50)
    parser.add_argument('--sentence_num_hidden', type=int, default=50)
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

args = parser.parse_args()