

import random
import string
import numpy as np
from scipy.sparse import csr_matrix, vstack


def make_random_words(n):
    words = [random.choice(string.ascii_letters + string.digits)
             for i in range(n)]
    return ''.join(words)


def make_random_words_count(numusers, lam=0.3):
    words_count = [np.random.poisson(lam=lam) for i in range(numusers)]
    return np.array(words_count)


def create_synthetic_data(num_pos_users=200, num_neg_users=200, num_random_words=1000,
                          alpha=0.7, lam=0.001, shape=0.1, scale=2):

    pos_labels = [1] * num_pos_users
    neg_labels = [0] * num_neg_users

    # create synthetic corpus
    if lam is None:
        pos_corpus = [make_random_words_count(num_pos_users, lam=np.random.gamma(
            shape=shape, scale=scale)) for i in range(num_random_words)]
        neg_corpus = [make_random_words_count(num_neg_users, lam=np.random.gamma(
            shape=shape, scale=scale)) for i in range(num_random_words)]

    else:
        pos_corpus = [make_random_words_count(
            num_pos_users, lam=lam) for i in range(num_random_words)]
        neg_corpus = [make_random_words_count(
            num_neg_users, lam=lam) for i in range(num_random_words)]

    # convart to csr_matrix
    X = vstack([csr_matrix(pos_corpus).T, csr_matrix(neg_corpus).T])
    label = np.array(pos_labels + neg_labels)

    return X, label
