import logging
import random
import re
import os
import torch
import numpy as np
import pandas as pd
from joblib import Memory
from torchtext import data
from sklearn.model_selection import train_test_split
from torchtext.data import Dataset
from torchtext.vocab import Vectors
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import KFold

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
        kf = KFold(k)
        examples = self.examples
        return kf.split(examples, random_state=random_seed)

    def get_fold(self, fields, train_indexs, test_indexs, shuffle=True):
        """
        get new batch
        :return:
        """
        examples = np.asarray(self.examples)

        if shuffle: random.shuffle(examples)
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
        plausible['text'] = plausible['title'] + plausible['content']
        plausible['label'] = 'plausible'
        implausible['text'] = implausible['title'] + implausible['content']
        implausible['label'] = 'implausible'
        data = pd.concat([plausible[['text', 'label']], implausible[['text', 'label']]])

    # split into 20% test, 80% train
    train, test = train_test_split(data, test_size=0.2, random_state=random_seed)

    train.to_csv(os.path.join(target_path, 'train.tsv'), index=False)
    test.to_csv(os.path.join(target_path, 'test.tsv'), index=False)


@MEMORY.cache
def read_files(plausible_path, implausible_path, target_path, pre_embeddings_path):
    prepare_tsv(plausible_path, implausible_path, target_path, option='combined')
    nesting_field = data.Field(batch_first=True, tokenize=word_tokenizer,
                               unk_token='<unk>', include_lengths=False, sequential=True)
    text_field = data.NestedField(nesting_field, tokenize=sent_tokenize)
    label_field = data.Field(sequential=False, use_vocab=True, batch_first=True, dtype=torch.float)
    fields = [('text', text_field), ('label', label_field)]

    train_path = os.path.join(target_path, 'train.tsv')
    logger.debug('Reading training samples from {}'.format(train_path))
    train = PlausibleDataset(path=train_path,
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
    pre_embeddings = Vectors(name=pre_embeddings_path)
    text_field.build_vocab(train, vectors=pre_embeddings)
    label_field.build_vocab(train)

    return train, test, text_field, label_field


