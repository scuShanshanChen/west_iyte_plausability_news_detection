import logging
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from joblib import Memory
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import Dataset
from torchtext.vocab import Vectors

logger = logging.getLogger('data/plausible.py')
random.seed = 42
random_seed = 42
MEMORY = Memory(location="../datasets/cache", verbose=1)


class PlausibleDataset(data.TabularDataset):

    def splits(self, fields, dev_ratio=.2, shuffle=True, **kwargs):
        examples = self.examples

        if shuffle: random.shuffle(examples)

        dev_index = -1 * int(dev_ratio * len(examples))

        return (Dataset(fields=fields, examples=examples[:dev_index]),
                Dataset(fields=fields, examples=examples[dev_index:]))

    def kfold(self, k):
        """
        kfold using sklearn
        :param k:
        :return: index of kfolded
        """
        kf = StratifiedKFold(k, random_state=random_seed)
        examples = self.examples
        return kf.split(examples, [example.label for example in examples])

    def get_fold(self, fields, train_indexs, test_indexs):
        """
        get new batch
        :return:
        """
        examples = np.asarray(self.examples)

        return (Dataset(fields=fields, examples=examples[list(train_indexs)]),
                Dataset(fields=fields, examples=examples[list(test_indexs)]))


def word_tokenizer(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)
    text = re.sub(r"[\n\t\s]+", " ", text)
    text = text.strip()
    text = text.split()
    return text


def prepare_tsv(plausible_path, implausible_path, target_path, option='combined'):
    plausible = pd.read_csv(plausible_path, sep='\t')
    implausible = pd.read_csv(implausible_path, sep='\t')

    if 'combined' == option:
        data = combined_index(implausible, plausible)

    # split into 20% test, 80% train
    train, test = train_test_split(data, test_size=0.2, random_state=random_seed)

    train.to_csv(os.path.join(target_path, 'train.tsv'), sep='\t', index=False)
    test.to_csv(os.path.join(target_path, 'test.tsv'), sep='\t', index=False)


@MEMORY.cache
def combined_index(implausible, plausible):
    plausible['text'] = plausible['title'] + plausible['content']
    plausible['label'] = 1
    implausible['text'] = implausible['title'] + implausible['content']
    implausible['label'] = 0
    data = pd.concat([plausible[['text', 'label']], implausible[['text', 'label']]])
    train, test = train_test_split(data, test_size=0.2, random_state=random_seed)
    return train, test


@MEMORY.cache
def title_index(implausible, plausible):
    plausible['text'] = plausible['title']
    plausible['label'] = 1
    implausible['text'] = implausible['title']
    implausible['label'] = 0
    data = pd.concat([plausible[['text', 'label']], implausible[['text', 'label']]])
    train, test = train_test_split(data, test_size=0.2, random_state=random_seed)
    return train, test


@MEMORY.cache
def body_index(implausible, plausible):
    plausible['text'] = plausible['content']
    plausible['label'] = 1
    implausible['text'] = implausible['content']
    implausible['label'] = 0
    data = pd.concat([plausible[['text', 'label']], implausible[['text', 'label']]])
    train, test = train_test_split(data, test_size=0.2, random_state=random_seed)
    return train, test


def googlenews_wrapper(bin_file_path):
    if not os.path.exists('datasets/embedding_wrapper'):
        logging.debug('Wrapping Googlenews embedding ...')
        model = KeyedVectors.load_word2vec_format(bin_file_path, binary=True, encoding="ISO-8859-1",
                                                  unicode_errors='ignore')
        model.wv.save_word2vec_format('datasets/embedding_wrapper')
    vectors = Vectors(name='embedding_wrapper')
    return vectors


@MEMORY.cache
def read_files(args):
    target_path = args.target_path
    if args.is_from_scratch:
        plausible_path = args.plausible_path
        implausible_path = args.implausible_path
        prepare_tsv(plausible_path, implausible_path, target_path, option='combined')

    nesting_field = data.Field(batch_first=True, tokenize=word_tokenizer,
                               unk_token='<unk>', include_lengths=False, sequential=True, fix_length=args.word_max_len)
    text_field = data.NestedField(nesting_field, tokenize=sent_tokenize, fix_length=args.sent_max_len)
    label_field = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    fields = [('text', text_field), ('label', label_field)]

    train_path = os.path.join(target_path, 'train.tsv')
    logger.debug('Reading training samples from {}'.format(train_path))
    train = PlausibleDataset(path=train_path,
                             format='tsv',
                             skip_header=True,
                             fields=fields
                             )

    dev_path = os.path.join(target_path, 'dev.tsv')
    logger.debug('Reading dev samples from {}'.format(train_path))
    dev = PlausibleDataset(path=dev_path,
                           format='tsv',
                           skip_header=True,
                           fields=fields
                           )

    test_path = os.path.join(target_path, 'test.tsv')
    logger.debug('Reading test samples from {}'.format(test_path))
    test = PlausibleDataset(path=test_path,
                            format='tsv',
                            skip_header=True,
                            fields=fields
                            )

    logging.info('Initializing the vocabulary...')
    text_field.build_vocab(train, max_size=args.max_vocab_size,
                           vectors=get_embeddings(args.embedding_name), unk_init=torch.Tensor.normal_)

    return train, dev, test, text_field, label_field


def get_embeddings(embedding_name='conceptnet'):
    if 'conceptnet' == embedding_name:
        path = './datasets/numberbatch-en-17.02.txt'
        vector = Vectors(path, cache='./datasets/')
        return vector

    if 'glove' == embedding_name:
        return 'glove.6B.300d'

