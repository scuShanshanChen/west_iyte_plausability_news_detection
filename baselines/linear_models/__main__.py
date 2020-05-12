import logging
import os
from collections import Counter

import coloredlogs
import pandas as pd

from baselines.linear_models.metrics import calculate_metrics
from baselines.linear_models.models import majority, nbsvm
from configs.args import args
from data.plausible import LABEL_MAP, clean_text

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

random_seed = 42

if __name__ == '__main__':
    args.target_path = './datasets/'
    stats_columns = '{0:>2} | {1:>2} | {2:>2} | {3:>2} | {4:>2} | {5:>2}'

    train_path = os.path.join(args.target_path, 'train.tsv')
    dev_path = os.path.join(args.target_path, 'dev.tsv')
    test_path = os.path.join(args.target_path, 'test.tsv')

    train = pd.read_csv(train_path, sep='\t')
    dev = pd.read_csv(dev_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

    train_X = train['text']
    train_y = train['label']
    dev_X = dev['text']
    dev_y = dev['label']
    test_X = test['text']
    test_y = test['label']

    train_stats = dict(Counter(train_y))
    train_stats = [''.join(LABEL_MAP(key).name + ' %d' % value) for key, value in train_stats.items()]
    logger.info('Train class distributions: {}'.format(train_stats))

    dev_stats = dict(Counter(dev_y))
    dev_stats = [''.join(LABEL_MAP(key).name + ' %d' % value) for key, value in dev_stats.items()]
    logger.info('Dev class distributions: {}'.format(dev_stats))

    test_stats = dict(Counter(test_y))
    test_stats = [''.join(LABEL_MAP(key).name + ' %d' % value) for key, value in test_stats.items()]
    logger.info('Test class distributions: {}'.format(test_stats))

    logger.info('Running majority prediction...')

    predictions_dev, predictions_test = majority(train_y=train_y, dev_y=dev_y, test_y=test_y)
    acc, f1, recall, prec = calculate_metrics(dev_y, predictions_dev)
    logger.info(stats_columns.format('Model', 'Mode', 'Acc', 'F1', 'Recall', 'Precision'))
    logger.info(stats_columns.format('majority', 'dev', acc, f1, recall, prec))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('majority', 'test', acc, f1, recall, prec))

    train_X = [clean_text(text) for text in train_X]
    test_X = [clean_text(text) for text in test_X]
    dev_X = [clean_text(text) for text in dev_X]

    logger.info('Running Multi-Class Naive Bayes SVM on unigrams')
    predictions_dev, predictions_test = nbsvm(train_X=train_X, train_y=train_y, dev_X=dev_X,
                                              test_X=test_X, ngram=(1, 1))
    logger.info(stats_columns.format('NBSVM-unigram', 'dev', acc, f1, recall, prec))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('NBSVM-unigram', 'test', acc, f1, recall, prec))

    logger.info('Running Multi-Class Naive Bayes SVM on unigrams + bigrams')
    predictions_dev, predictions_test = nbsvm(train_X=train_X, train_y=train_y, dev_X=dev_X,
                                              test_X=test_X, ngram=(1, 2))
    logger.info(stats_columns.format('NBSVM-unigram+bigram', 'dev', acc, f1, recall, prec))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('NBSVM-unigram+bigram', 'test', acc, f1, recall, prec))
