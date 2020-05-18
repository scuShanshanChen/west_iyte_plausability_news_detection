import logging
import os
import re
from enum import Enum

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger('data/plausible.py')

PAD = '<pad>'
EOS = '<eos>'
SOS = '<sos>'
UNK = '<unk>'


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)
    text = re.sub(r"[\n\t\s]+", " ", text)
    text = text.strip()
    return text


def word_tokenizer(text: str):
    text = clean_text(text)
    text = text.split()
    return text


def custom_tokenizer(text: str):
    sentences = sent_tokenize(text)
    data = []

    for idx, sentence in enumerate(sentences):
        data.append(SOS)
        tokens = word_tokenizer(sentence)
        for token in tokens:
            data.append(token)
        data.append(EOS)
    return data


class LABEL_MAP(Enum):
    PLAUSIBLE = 1
    IMPLAUSIBLE = 0


def get_embedding_model(model_name, fname):
    if 'googlenews' == model_name:
        model = KeyedVectors.load_word2vec_format(fname, limit=50000, binary=True)
    elif 'numberbatch' == model_name:
        model = KeyedVectors.load_word2vec_format(fname, limit=50000, binary=False)

    return model


class Vocab(object):
    def __init__(self, embedding_model, embedding_path):
        self.embedding_model = embedding_model
        self.embedding_path = embedding_path
        self.embed = get_embedding_model(embedding_model, embedding_path)
        self.vocab = self.build_vocab()
        self.weights_matrix = self.get_weights_matrix()

    def build_vocab(self):
        vocab = dict()
        vocab[PAD] = 0
        vocab[SOS] = 1
        vocab[EOS] = 2
        vocab[UNK] = 3
        idx = len(vocab)
        for word in self.embed.wv.vocab:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        logger.info('Vocab size {}'.format(len(vocab)))
        return vocab

    def write_vocab(self, filename, vocab):
        with open(filename, "w") as f:
            for i, word in enumerate(vocab):
                if i != len(vocab) - 1:
                    f.write("{}\n".format(word))
                else:
                    f.write(word)

    def get_weights_matrix(self):
        matrix_len = len(self.vocab) + 1  # add pad here
        emb_dim = self.embed.vector_size
        weights_matrix = np.zeros((matrix_len, emb_dim))
        for word in self.vocab:
            if word in self.embed:
                weights_matrix[self.vocab[word]] = self.embed[word]
            else:
                weights_matrix[self.vocab[word]] = np.random.uniform(-0.1, 0.1, (emb_dim,))
        return torch.FloatTensor(weights_matrix)  # make it as torch


class PlausibleDataset(Dataset):
    def __init__(self, df, vocab, tokenizer=custom_tokenizer, feature='headline'):
        texts = [tokenizer(text) for text in get_feature(df, feature)]
        labels = df['label'].values
        self.tokenizer = tokenizer
        self.vocab = vocab

        data = []
        for idx in range(len(labels)):
            data.append({'text': torch.LongTensor(self.get_word2ids(texts[idx])),
                         'label': torch.tensor(labels[idx], dtype=torch.float)})
        self.data = data

    def get_word2ids(self, sentence):
        word2ids = []
        for word in sentence:
            if word not in self.vocab:
                word2ids.append(self.vocab[UNK])
            else:
                word2ids.append(self.vocab[word])
        return word2ids

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch):
    text = [item['text'] for item in batch]
    label = [item['label'] for item in batch]

    # get length of each sentences
    text_lens = [len(item['text']) for item in batch]

    # pad sequence takes longest instance in batch and pads shorter instances with zero
    padded_text = pad_sequence(text, batch_first=True, padding_value=0)
    # we change the type of label to torch again
    new_batch = {'text': padded_text, 'text_len': torch.tensor(text_lens),
                 'label': torch.tensor(label, dtype=torch.float)}
    return new_batch


def make_random_split(plausible_path, implausible_path, target_path, seed):
    '''
    splits train, dev, test sets given seed. creates a folder random_seed_{seed} under target path
    :param plausible_path:
    :type plausible_path:
    :param implausible_path:
    :type implausible_path:
    :param target_path:
    :type target_path:
    :param seed:
    :type seed:
    :return:
    :rtype:
    '''
    logger.info('Preparing experimental data with the random split mode, seed {}'.format(seed))
    plausible = pd.read_csv(plausible_path, sep='\t')
    implausible = pd.read_csv(implausible_path, sep='\t')
    plausible['label'] = 1
    implausible['label'] = 0
    data = pd.concat([plausible[['title', 'content', 'label']], implausible[['title', 'content', 'label']]])
    # split into 40% test, 60% train
    train, test = train_test_split(data, test_size=0.4, random_state=seed)
    # split into 50% test, 50% dev
    test, dev = train_test_split(test, test_size=0.5, random_state=seed)
    # so final splits became 20% test, 20% dev, 60% train

    target_folder_name = 'random_seed_{}'.format(seed)

    target_folder_path = os.path.join(target_path, target_folder_name)

    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    train.to_csv(os.path.join(target_folder_path, 'train.tsv'), sep='\t', index=False)
    test.to_csv(os.path.join(target_folder_path, 'test.tsv'), sep='\t', index=False)
    dev.to_csv(os.path.join(target_folder_path, 'dev.tsv'), sep='\t', index=False)


def make_kfold_split(plausible_path, implausible_path, target_path, kfold, seed):
    '''
    splits kfold train test sets given seed. creates a folder kfold_random_seed_{seed} under target path
    :param plausible_path:
    :type plausible_path:
    :param implausible_path:
    :type implausible_path:
    :param target_path:
    :type target_path:
    :param n_fold:
    :type n_fold:
    :param seed:
    :type seed:
    :return:
    :rtype:
    '''
    logger.info(
        'Preparing experimental data with kfold {kfold} split mode, seed {seed}'.format(kfold=kfold, seed=seed))
    plausible = pd.read_csv(plausible_path, sep='\t')
    implausible = pd.read_csv(implausible_path, sep='\t')
    plausible['label'] = 1
    implausible['label'] = 0
    data = pd.concat([plausible[['title', 'content', 'label']], implausible[['title', 'content', 'label']]])

    skf = StratifiedKFold(kfold, random_state=seed, shuffle=True)
    targets = data.label
    for fold_idx, (train_index, test_index) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        train, test = data.iloc[train_index], data.iloc[test_index]
        target_folder_name = 'kfold_random_seed_{}'.format(seed)
        target_folder_path = os.path.join(target_path, target_folder_name)
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        train.to_csv(os.path.join(target_folder_path, 'kfold_{}_train.tsv'.format(fold_idx)), sep='\t', index=False)
        test.to_csv(os.path.join(target_folder_path, 'kfold_{}_test.tsv').format(fold_idx), sep='\t', index=False)


def get_feature(data, feature):
    if 'headline' == feature:
        data = data['title'].values
    elif 'body' == feature:
        data = data['content'].values
    else:
        data = data['title'].values + data['content'].values
    return data
