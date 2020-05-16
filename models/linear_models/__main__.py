import csv
import logging
import os
from collections import Counter

import coloredlogs
import numpy as np
import pandas as pd

from models.linear_models.metrics import calculate_metrics
from models.linear_models.models import majority, nbsvm
from configs.args import args
from data.plausible import LABEL_MAP, clean_text, get_feature

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)


def run_experiment_random_split(train_X, test_X, train_y, test_y, dev_X, dev_y, feature):
    stats_columns = '{0:>2} | {1:>2} | {2:>2} | {3:>2} | {4:>2} | {5:>2} | {6:2}'
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

    logger.info(stats_columns.format('Model', 'Feature', 'Mode', 'Acc', 'F1', 'Recall', 'Precision'))
    logger.info(stats_columns.format('majority', feature, 'dev', acc, f1, recall, prec))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('majority', feature, 'test', acc, f1, recall, prec))

    train_X = [clean_text(text) for text in train_X]
    test_X = [clean_text(text) for text in test_X]
    dev_X = [clean_text(text) for text in dev_X]
    logger.info('Running Multi-Class Naive Bayes SVM on unigrams')
    predictions_dev, predictions_test = nbsvm(train_X=train_X, train_y=train_y, dev_X=dev_X,
                                              test_X=test_X, ngram=(1, 1))

    logger.info(stats_columns.format('NBSVM-unigram', feature, 'dev', acc, f1, recall, prec))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)

    logger.info(stats_columns.format('NBSVM-unigram', feature, 'test', acc, f1, recall, prec))
    logger.info('Running Multi-Class Naive Bayes SVM on unigrams + bigrams')
    predictions_dev, predictions_test = nbsvm(train_X=train_X, train_y=train_y, dev_X=dev_X,
                                              test_X=test_X, ngram=(1, 2))

    logger.info(stats_columns.format('NBSVM-unigram+bigram', feature, 'dev', acc, f1, recall, prec))

    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('NBSVM-unigram+bigram', feature, 'test', acc, f1, recall, prec))


def run_experiment_kfold(train_X, test_X, train_y, test_y, feature, fold_idx, cw):
    logging.info('Running fold {}'.format(fold_idx))
    stats_columns = '{0:>2} | {1:>2} | {2:>2} | {3:>2} | {4:>2} | {5:>2} | {6:2}'
    train_stats = dict(Counter(train_y))
    train_stats = [''.join(LABEL_MAP(key).name + ' %d' % value) for key, value in train_stats.items()]
    logger.info('Train class distributions: {}'.format(train_stats))

    test_stats = dict(Counter(test_y))
    test_stats = [''.join(LABEL_MAP(key).name + ' %d' % value) for key, value in test_stats.items()]

    logger.info('Test class distributions: {}'.format(test_stats))
    logger.info('Running majority prediction...')

    _, predictions_test = majority(train_y=train_y, dev_y=None, test_y=test_y)
    logger.info(stats_columns.format('Model', 'Feature', 'Mode', 'Acc', 'F1', 'Recall', 'Precision'))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('majority', feature, 'test', acc, f1, recall, prec))
    cw.writerow([fold_idx, 'majority', feature, 'test', acc, f1, recall, prec])

    train_X = [clean_text(text) for text in train_X]
    test_X = [clean_text(text) for text in test_X]

    logger.info('Running Multi-Class Naive Bayes SVM on unigrams')
    _, predictions_test = nbsvm(train_X=train_X, train_y=train_y, dev_X=None,
                                test_X=test_X, ngram=(1, 1))
    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    cw.writerow([fold_idx, 'NBSVM-unigram', feature, 'test', acc, f1, recall, prec])
    logger.info(stats_columns.format('NBSVM-unigram', feature, 'test', acc, f1, recall, prec))

    logger.info('Running Multi-Class Naive Bayes SVM on unigrams + bigrams')
    _, predictions_test = nbsvm(train_X=train_X, train_y=train_y, dev_X=None,
                                test_X=test_X, ngram=(1, 2))

    acc, f1, recall, prec = calculate_metrics(test_y, predictions_test)
    logger.info(stats_columns.format('NBSVM-unigram+bigram', feature, 'test', acc, f1, recall, prec))
    cw.writerow([fold_idx, 'NBSVM-unigram+bigram', feature, 'test', acc, f1, recall, prec])


if __name__ == '__main__':
    logging.info('Run experiment in mode {}'.format(args.training_mode))
    target_dir = './datasets/'

    if 'kfold' == args.training_mode:
        target_folder = os.path.join(target_dir, 'kfold_random_seed_{}'.format(args.seed))
        assert (not os.path.exists(target_folder),
                'No such sets, first run python -m data --seed {seed} --training_mode kfold --kfold {kfold}'.format(
                    seed=args.seed, kfold=args.kfold))

        train_file = 'kfold_{}_train.tsv'
        test_file = 'kfold_{}_test.tsv'

        feature = args.feature
        results_file = os.path.join(target_folder, '{}_results.csv'.format(feature))

        with open(results_file, 'w') as output_file:
            cw = csv.writer(output_file, delimiter='\t')
            cw.writerow(['Fold ID', 'Model', 'Feature', 'Mode', 'Acc', 'F1', 'Recall', 'Precision'])
            for fold_idx in range(args.kfold):
                train = pd.read_csv(os.path.join(target_folder, train_file.format(fold_idx)), sep='\t')
                test = pd.read_csv(os.path.join(target_folder, test_file.format(fold_idx)), sep='\t')
                train_X = get_feature(train, feature=args.feature)
                train_y = train['label']
                test_X = get_feature(test, feature=args.feature)
                test_y = test['label']
                run_experiment_kfold(train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y,
                                     feature=feature,
                                     fold_idx=fold_idx, cw=cw)

        results = pd.read_csv(results_file, sep='\t')
        stats_columns = '{0:>2} | {1:>2} | {2:>2} | {3:>2} | {4:>2} | {5:>2}'
        logger.info(stats_columns.format('Model', 'Feature', 'Acc_CV', 'F1_CV', 'Recall_CV', 'Precision_CV'))
        groups = results.groupby('Model')
        for name, group in groups:
            acc = group['Acc'].values.tolist()
            f1 = group['F1'].values.tolist()
            recall = group['Recall'].values.tolist()
            prec = group['Precision'].values.tolist()
            logger.info(stats_columns.format(name, feature,
                                             '%0.2f' % np.mean(acc) + ' +/- %0.2f' % np.std(acc),
                                             '%0.2f' % np.mean(f1) + ' +/- %0.2f' % np.std(f1),
                                             '%0.2f' % np.mean(recall) + ' +/- %0.2f' % np.std(recall),
                                             '%0.2f' % np.mean(prec) + ' +/- %0.2f' % np.std(prec)))

    elif 'random_split' == args.training_mode:
        target_folder = os.path.join(target_dir, 'random_seed_{}'.format(args.seed))
        assert (not os.path.exists(target_folder),
                'No such sets, first run python -m data --seed {seed} --training_mode random_split'.format(
                    seed=args.seed))
        train_path = os.path.join(target_folder, 'train.tsv')
        dev_path = os.path.join(target_folder, 'dev.tsv')
        test_path = os.path.join(target_folder, 'test.tsv')

        train = pd.read_csv(train_path, sep='\t')
        dev = pd.read_csv(dev_path, sep='\t')
        test = pd.read_csv(test_path, sep='\t')

        train_X = get_feature(train, feature=args.feature)
        train_y = train['label']
        dev_X = get_feature(dev, feature=args.feature)
        dev_y = dev['label']
        test_X = get_feature(test, feature=args.feature)
        test_y = test['label']

        run_experiment_random_split(train_X=train_X, train_y=train_y, dev_X=dev_X, dev_y=dev_y, test_X=test_X,
                                    test_y=test_y, feature=args.feature)
