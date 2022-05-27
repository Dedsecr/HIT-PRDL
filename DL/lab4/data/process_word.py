import jieba
import pandas as pd
import re
from gensim.models import word2vec
import os
import numpy as np
import matplotlib.pyplot as plt


def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [
        line.strip()
        for line in open(filepath, 'r', encoding='utf-8').readlines()
    ]
    return stopwords


def word2vector(data: pd.DataFrame, attr: str, path: str, embedding_dim: int):
    path_model = path + 'word2vec_model_{}.model'.format(embedding_dim)
    if os.path.exists(path_model):
        print('Loading word2vec model from {}'.format(path_model))
        word2vec_model = word2vec.Word2Vec.load(path_model)
    else:
        words = [x.split() for x in data[attr].values]
        word2vec_model = word2vec.Word2Vec(words,
                                           vector_size=embedding_dim,
                                           min_count=1)
        word2vec_model.save(path_model)
    return word2vec_model


def formalize_data(x, max_length, embedding_dim):
    if len(x) == 0:
        x = np.zeros((max_length, embedding_dim))
    elif len(x) < max_length:
        # print('before:', x.shape, x)
        x = np.pad(x, ((0, max_length - len(x)), (0, 0)), 'constant')
        # print('after:', x.shape)
    elif len(x) > max_length:
        x = x[:max_length]
    return x


def process_word(data: pd.DataFrame, attr: str, path: str, max_length: int,
                 embedding_dim: int):
    print('Processing data...')
    attr_new = attr + '_processed'

    print('Removing punctuation...')
    data[attr_new] = data[attr].apply(remove_punctuation)

    print('Cutting words and removing stopwords...')
    stopwords = stopwordslist("./data/stopwords.txt")
    data[attr_new] = data[attr_new].apply(lambda x: " ".join(
        [w for w in list(jieba.cut(x)) if w not in stopwords]))

    print('Loading word2vec model...')
    word2vec_model = word2vector(data, attr_new, path, embedding_dim)
    data[attr + '_vec'] = data[attr_new].apply(
        lambda x: np.array([word2vec_model.wv[w] for w in x.split()]))

    # plot_length(data)

    print('Formalizing data...')
    data[attr + '_vec'] = data[attr + '_vec'].apply(
        formalize_data, max_length=max_length, embedding_dim=embedding_dim)

    return data


# output the length of the dataset
def plot_length(data):
    data_len = [len(x) for x in data['review' + '_vec']]
    plt.hist(data_len, range=(0, 60))
    plt.show()