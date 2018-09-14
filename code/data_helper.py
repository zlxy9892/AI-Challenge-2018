# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import gensim
from segment import Seg
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_text_and_label(filename, include_label=True, topn=None):
    # load data from file
    df = pd.read_csv(filename)
    content_list = list(df['content'])
    content_list = [s.strip('\n') for s in content_list]
    if include_label:
        labels = np.array(df[df.columns[2:]])
    else:
        labels = None

    seg = Seg(file_stopwords='./data/stopword.txt')

    data_size = len(content_list)
    x_text = []
    for i in range(data_size):
        word_list = seg.cut(content_list[i])
        x_text.append(list(word_list))
        if topn is not None and i >= topn-1:
            break
    if topn is not None:
        return x_text, labels[:topn]
    else:
        return x_text, labels


def load_origin_label(filename):
    df = pd.read_csv(filename)
    label = np.array(df[df.columns[2:]])
    return label


def extract_character_vocab(data):
    data = list(data)
    special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>']
    word_set = sorted(set([word for word_list in data for word in word_list]))
    id2word = {idx: word for idx, word in enumerate(special_words + word_set)}
    word2id = {word: idx for idx, word in id2word.items()}
    return word2id


def get_x_ids(texts=None, word2id=None):
    x_ids = [[word2id.get(word, word2id['<UNK>']) for word in word_list] for word_list in texts]
    return x_ids


def pad_sentence_batch(sequences, max_sentence_len=200, pad_id=0):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    '''
    sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
    return sequences


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generates a batch iterator for a dataset.
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield np.array(shuffled_data[start_index:end_index])


def load_word_vectors(word2id, pre_word_vectors_dict, embedding_size):
    vocab_size = len(list(word2id.keys()))
    embedding_matrix = np.zeros(shape=(vocab_size, embedding_size))  # shape: [vocab_size, embedding_size]
    not_include_count = 0
    for word, i in word2id.items():
        default_vector = np.random.randn(embedding_size)
        word_vector = pre_word_vectors_dict.get(word, None)
        if word_vector is None:
            word_vector = default_vector
            not_include_count += 1
        embedding_matrix[i] = word_vector
    embedding_matrix = embedding_matrix.astype(np.float32)
    include_prob = 1 - float(not_include_count) / vocab_size
    return embedding_matrix, include_prob

def transform_data(data):
    # the shape of the data is ? x 80, we need transform it to be ? x 20 (each item in one row is a number in set: {1,0,-1,-2})
    data_trans = np.zeros(shape=(data.shape[0],20), dtype=np.int32)
    for i in range(data.shape[0]):
        for j in range(20):
            prop_max = 0.0
            for k in range(4):
                index = j * 4 + k
                prop_tmp = data[i,index]
                if prop_max < prop_tmp:
                    best_id = k
                    prop_max = prop_tmp
            if best_id == 0:
                data_trans[i,j] = 1
            elif best_id == 1:
                data_trans[i, j] = 0
            elif best_id == 2:
                data_trans[i, j] = -1
            elif best_id == 3:
                data_trans[i, j] = -2
            else:
                data_trans[i, j] = -2
    return data_trans
