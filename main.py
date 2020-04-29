import logging
import argparse
import coloredlogs
import torch
import numpy as np
from torchtext import data
from data.plausible import read_files
from baselines.han import HierarchicalAttentionNet
from utils.dl_runner import fit, eval

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for Plausible Detection Models')

    # pre_embedding path

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.target_path = './datasets/'
    args.plausible_path = './dataset/plausible.tsv'
    args.implausible_path = './dataset/implausible.tsv'

    logging.debug('Reading datasets...')
    train, test, text_field, label_field = read_files(args)
    args.embedding_vectors = text_field.vocab.vectors

    kfold_range = train.kfold(10)
    accuracys = []
    avg_losses = []

    for train_range, test_range in kfold_range:
        type(train_range)
        train, dev = train.get_fold(fields=[('text', text_field), ('label', label_field)], train_indexs=train_range,
                                    test_indexs=test_range)
        train_iter, dev_iter = data.Iterator.splits((train, dev), device=-1, batch_sizes=(args.batch_size, len(dev)))

        # model = HierarchicalAttentionNet(embedding_vectors=text_field.vocab.vectors, batch_size=args.batch_,
        #                                  retrain_emb=False,
        #                                  hidden_size=50)
        model = HierarchicalAttentionNet(args)
        model = fit(train_iter, dev_iter, model, args)
        accuracy, loss = eval(dev_iter, model)
        accuracys.append(accuracy)
        avg_losses.append(loss)

    print("avarage accuracy is %s, loss is %s".format(np.average(accuracys)), np.average(avg_losses))
