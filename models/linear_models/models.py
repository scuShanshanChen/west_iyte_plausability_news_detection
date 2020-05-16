import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from models.linear_models.nbsvm import NBSVM


def n_gram(train_X, train_y, dev_X, test_X, model, ngram):
    predictions_dev = None
    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, ngram))),
        ('clf', model)
    ])
    pipeline.fit(train_X, train_y)

    if dev_X:
        predictions_dev = pipeline.predict(dev_X)

    predictions_test = pipeline.predict(test_X)
    return predictions_dev, predictions_test


def nbsvm(train_X, train_y, dev_X, test_X, ngram):
    vect = CountVectorizer()
    classifier = NBSVM()
    predictions_dev = None
    # create pipeline
    clf = Pipeline([('vect', vect), ('nbsvm', classifier)])
    params = {
        'vect__token_pattern': r"\S+",
        'vect__ngram_range': ngram,
        'vect__binary': True
    }
    clf.set_params(**params)
    clf.fit(train_X, train_y)

    if dev_X:
        predictions_dev = clf.predict(dev_X)

    predictions_test = clf.predict(test_X)

    return predictions_dev, predictions_test


def majority(train_y, dev_y, test_y):
    '''
    Predicts always majority class
    '''
    majority_class = np.bincount(train_y).argmax()
    predictions_dev = None
    if dev_y:
        predictions_dev = np.full((len(dev_y),), majority_class)
    predictions_test = np.full((len(test_y),), majority_class)
    return predictions_dev, predictions_test
