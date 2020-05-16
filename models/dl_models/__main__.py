import logging
import os

import coloredlogs
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from configs.args import args
from data.plausible import PlausibleDataset, collate_fn
from models.dl_models.bigru import biGRU
from utils.dl_runner import train, set_seed, load_model, inference

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

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

        # with open(results_file, 'w') as output_file:
        #     cw = csv.writer(output_file, delimiter='\t')
        #     cw.writerow(['Fold ID', 'Model', 'Feature', 'Mode', 'Acc', 'F1', 'Recall', 'Precision'])
        #     for fold_idx in range(args.kfold):
        #         train = pd.read_csv(os.path.join(target_folder, train_file.format(fold_idx)), sep='\t')
        #         test = pd.read_csv(os.path.join(target_folder, test_file.format(fold_idx)), sep='\t')
        #         train_X = get_feature(train, feature=args.feature)
        #         train_y = train['label']
        #         test_X = get_feature(test, feature=args.feature)
        #         test_y = test['label']
        # run_experiment_kfold(train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y,
        #                      feature=feature,
        #                      fold_idx=fold_idx, cw=cw)

        # results = pd.read_csv(results_file, sep='\t')
        # stats_columns = '{0:>2} | {1:>2} | {2:>2} | {3:>2} | {4:>2} | {5:>2}'
        # logger.info(stats_columns.format('Model', 'Feature', 'Acc_CV', 'F1_CV', 'Recall_CV', 'Precision_CV'))
        # groups = results.groupby('Model')
        # for name, group in groups:
        #     acc = group['Acc'].values.tolist()
        #     f1 = group['F1'].values.tolist()
        #     recall = group['Recall'].values.tolist()
        #     prec = group['Precision'].values.tolist()
        #     logger.info(stats_columns.format(name, feature,
        #                                      '%0.2f' % np.mean(acc) + ' +/- %0.2f' % np.std(acc),
        #                                      '%0.2f' % np.mean(f1) + ' +/- %0.2f' % np.std(f1),
        #                                      '%0.2f' % np.mean(recall) + ' +/- %0.2f' % np.std(recall),
        #                                      '%0.2f' % np.mean(prec) + ' +/- %0.2f' % np.std(prec)))

    elif 'random_split' == args.training_mode:
        set_seed(args.seed)
        target_folder = os.path.join(target_dir, 'random_seed_{}'.format(args.seed))
        assert (not os.path.exists(target_folder),
                'No such sets, first run python -m data --seed {seed} --training_mode random_split'.format(
                    seed=args.seed))
        train_path = os.path.join(target_folder, 'train.tsv')
        dev_path = os.path.join(target_folder, 'dev.tsv')
        test_path = os.path.join(target_folder, 'test.tsv')

        train_df = pd.read_csv(train_path, sep='\t')
        dev_df = pd.read_csv(dev_path, sep='\t')
        test_df = pd.read_csv(test_path, sep='\t')

        dataset_train = PlausibleDataset(train_df, embedding_model='googlenews',
                                         embedding_path='./datasets/GoogleNews-vectors-negative300.bin.gz',
                                         feature=args.feature)
        vocab = dataset_train.vocab
        weights_matrix = dataset_train.weights_matrix

        dataset_dev = PlausibleDataset(dev_df, vocab=vocab, feature=args.feature)

        train_data = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
        dev_data = DataLoader(dataset=dataset_dev, batch_size=32,
                              collate_fn=collate_fn, shuffle=False)

        model = biGRU(words_dim=300, hidden_size=100, weights_matrix=weights_matrix, num_output=1)
        params = [np[0] for np in model.named_parameters()]

        device = 'cpu'
        optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
        criterion = nn.BCEWithLogitsLoss()
        epoch = 5

        model_name = '{model_name}_best_{mode}.pth'.format(model_name=str(model.__class__.__name__),
                                                           mode=args.training_mode)
        checkpoint_dir = os.path.join('./datasets', model_name)
        train(epoch, model, train_data, dev_data, optimizer, criterion, device, checkpoint_dir)

        ## inference mode ##
        load_model(checkpoint_dir, model, optimizer=None)

        dataset_test = PlausibleDataset(test_df, vocab=vocab, feature=args.feature)
        test_data = DataLoader(dataset=dataset_test, batch_size=32,
                               collate_fn=collate_fn, shuffle=False)

        inference(test_data, model, device)
