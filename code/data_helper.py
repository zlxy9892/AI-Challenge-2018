# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import re
# import gensim
from segment import Seg
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")  # only remain english characters, numbers and chinese words.
    string = rule.sub('', string)
    return string.strip().lower()


def label2id(label_mat):
    id_mat = np.zeros(shape=label_mat.shape)
    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            if label_mat[i, j] == 1:
                id_mat[i, j] = 0
            elif label_mat[i, j] == 0:
                id_mat[i, j] = 1
            elif label_mat[i, j] == -1:
                id_mat[i, j] = 2
            elif label_mat[i, j] == -2:
                id_mat[i, j] = 3
            else:
                pass
    return id_mat


def id2label(id_mat):
    label_mat = np.zeros(shape=id_mat.shape)
    for i in range(id_mat.shape[0]):
        for j in range(id_mat.shape[1]):
            if id_mat[i, j] == 0:
                label_mat[i, j] = 1
            elif id_mat[i, j] == 1:
                label_mat[i, j] = 0
            elif id_mat[i, j] == 2:
                label_mat[i, j] = -1
            elif id_mat[i, j] == 3:
                label_mat[i, j] = -2
            else:
                pass
    return label_mat.astype(np.int32)


def load_text_and_label(filename, include_label=True, topn=None):
    # load data from file
    df = pd.read_csv(filename)
    content_list = list(df['content'])
    content_list = [clean_str(s.strip('\n')) for s in content_list]
    if include_label:
        labels = np.array(df[df.columns[2:]])
        labels = label2id(labels)
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


def extract_character_vocab(data, min_frequency=2):
    data = list(data)
    special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>']

    # get word frequency
    frequency = defaultdict(int)
    for word_list in data:
        for word in word_list:
            frequency[word] += 1

    word_set = []
    for word_list in data:
        for word in word_list:
            if frequency[word] > min_frequency:
                word_set.append(word)
    word_set = sorted(set(word_set))
    # word_set = sorted(set([word for word_list in data for word in word_list]))
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


def get_word_dict_from_word_vector(f_word_vector='./data/merge_sgns_bigram_char300.txt', f_dict='./data/dict_for_cutword.txt'):
    word_list = []
    with open(f_word_vector, 'r', encoding='utf-8') as f_word_vec:
        lines = f_word_vec.readlines()
        for i in range(len(lines)):
            if i == 0:
                continue
            line = lines[i]
            line = line.strip('\n')
            line = line.strip(' ')
            tokens = line.split(' ')
            word = tokens[0]
            word_list.append(word)
    with open(f_dict, 'a', encoding='utf-8') as f:
        for word in word_list:
            f.write(word + '\n')
    return word_list


def get_pre_trained_word_vectors(f_word_vector='./data/sgns.zhihu.bigram-char'):
    word_vectors = {}
    with open(f_word_vector, encoding='utf-8') as f_word_vec:
        lines = f_word_vec.readlines()
        for i in range(len(lines)):
            if i == 0:
                continue
            line = lines[i]
            line = line.strip('\n')
            line = line.strip(' ')
            tokens = line.split(' ')
            word = tokens[0]
            vector = np.array(tokens[1:]).astype(np.float)
            word_vectors[word] = vector
    return word_vectors


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


def get_weight_matrix0(filename):
    df = pd.read_csv(filename)
    data_size = len(df)
    colnames = df.columns[2:]
    num_type = len(colnames)
    weight_matrix = np.ones(shape=(num_type, 4))
    for i in range(num_type):
        colname = colnames[i]
        sum_0 = np.sum(df[colname] == 1)
        sum_1 = np.sum(df[colname] == 0)
        sum_2 = np.sum(df[colname] == -1)
        sum_3 = np.sum(df[colname] == -2)
        weight_0 = 1 - sum_0 / data_size
        weight_1 = 1 - sum_1 / data_size
        weight_2 = 1 - sum_2 / data_size
        weight_3 = 1 - sum_3 / data_size
        weight_matrix[i, 0] = weight_0
        weight_matrix[i, 1] = weight_1
        weight_matrix[i, 2] = weight_2
        weight_matrix[i, 3] = weight_3
    return weight_matrix


def get_weight_matrix(filename):
    df = pd.read_csv(filename)
    data_size = len(df)
    colnames = df.columns[2:]
    num_type = len(colnames)
    weight_matrix = np.ones(shape=(num_type, 4))
    for i in range(num_type):
        colname = colnames[i]
        sum_0 = np.sum(df[colname] == 1)
        sum_1 = np.sum(df[colname] == 0)
        sum_2 = np.sum(df[colname] == -1)
        sum_3 = np.sum(df[colname] == -2)
        weight_0 = sum_0 / data_size
        weight_1 = sum_1 / data_size
        weight_2 = sum_2 / data_size
        weight_3 = sum_3 / data_size
        weight_matrix[i, 0] = weight_0
        weight_matrix[i, 1] = weight_1
        weight_matrix[i, 2] = weight_2
        weight_matrix[i, 3] = weight_3
    return weight_matrix


def get_sample_weights(ys, weight_matrix):
    sample_weights = np.ones(shape=ys.shape)
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            true_type = ys[i, j]
            if true_type == 0:
                sample_weights[i, j] = weight_matrix[j, 0]
            elif true_type == 1:
                sample_weights[i, j] = weight_matrix[j, 1]
            elif true_type == 2:
                sample_weights[i, j] = weight_matrix[j, 2]
            elif true_type == 3:
                sample_weights[i, j] = weight_matrix[j, 3]
            else:
                sample_weights[i, j] = 0.5
    return sample_weights


def resampling_data(x, y, weight_matrix, type_id):
    weights = weight_matrix[type_id]
    max_class_id = np.argmax(weights)
    duplicate_time_0 = int(weights[max_class_id] / weights[0])
    duplicate_time_1 = int(weights[max_class_id] / weights[1])
    duplicate_time_2 = int(weights[max_class_id] / weights[2])
    duplicate_time_3 = int(weights[max_class_id] / weights[3])

    duplicate_time_0 = max(int(duplicate_time_0*0.2), 1)
    duplicate_time_1 = max(int(duplicate_time_1*0.2), 1)
    duplicate_time_2 = max(int(duplicate_time_2*0.2), 1)
    duplicate_time_3 = max(int(duplicate_time_3*0.2), 1)

    data_size = len(x)
    x_add = []
    y_add = []
    for i in range(data_size):
        if y[i, type_id] == max_class_id:
            pass
        else:
            if y[i, type_id] == 0:
                for k in range(duplicate_time_0):
                    x_add.append(x[i])
                    y_add.append(y[i])
            elif y[i, type_id] == 1:
                for k in range(duplicate_time_1):
                    x_add.append(x[i])
                    y_add.append(y[i])
            elif y[i, type_id] == 2:
                for k in range(duplicate_time_2):
                    x_add.append(x[i])
                    y_add.append(y[i])
            elif y[i, type_id] == 3:
                for k in range(duplicate_time_3):
                    x_add.append(x[i])
                    y_add.append(y[i])
    x_add = np.array(x_add)
    y_add = np.array(y_add)
    x_new = np.concatenate([x, x_add], axis=0)
    y_new = np.concatenate([y, y_add], axis=0)
    data_size_new = len(x_new)
    shuffle_indices = np.random.permutation(np.arange(data_size_new))
    x_new = x_new[shuffle_indices]
    y_new = y_new[shuffle_indices]
    return x_new, y_new


def resampling_data_text(x_text, y, weight_matrix, type_id):
    x = x_text
    weights = weight_matrix[type_id]
    max_class_id = np.argmax(weights)
    duplicate_time_0 = int(weights[max_class_id] / weights[0])
    duplicate_time_1 = int(weights[max_class_id] / weights[1])
    duplicate_time_2 = int(weights[max_class_id] / weights[2])
    duplicate_time_3 = int(weights[max_class_id] / weights[3])
    data_size = len(x)
    x_add = []
    y_add = []
    for i in range(data_size):
        if y[i, type_id] == max_class_id:
            pass
        else:
            if y[i, type_id] == 0:
                for k in range(duplicate_time_0):
                    x_add.append(x[i])
                    y_add.append(y[i])
            elif y[i, type_id] == 1:
                for k in range(duplicate_time_1):
                    x_add.append(x[i])
                    y_add.append(y[i])
            elif y[i, type_id] == 2:
                for k in range(duplicate_time_2):
                    x_add.append(x[i])
                    y_add.append(y[i])
            elif y[i, type_id] == 3:
                for k in range(duplicate_time_3):
                    x_add.append(x[i])
                    y_add.append(y[i])
    y_add = np.array(y_add)
    x_new = x + x_add
    y_new = np.concatenate([y, y_add], axis=0)
    return x_new, y_new


def show_weight_matrix(y):
    data_size = len(y)
    num_type = y.shape[1]
    weight_matrix = np.ones(shape=(num_type, 4))
    for i in range(num_type):
        sum_0 = np.sum(y[:, i] == 0)
        sum_1 = np.sum(y[:, i] == 1)
        sum_2 = np.sum(y[:, i] == 2)
        sum_3 = np.sum(y[:, i] == 3)
        weight_0 = sum_0 / data_size
        weight_1 = sum_1 / data_size
        weight_2 = sum_2 / data_size
        weight_3 = sum_3 / data_size
        weight_matrix[i, 0] = weight_0
        weight_matrix[i, 1] = weight_1
        weight_matrix[i, 2] = weight_2
        weight_matrix[i, 3] = weight_3
    print(weight_matrix)
    return weight_matrix


def transform_data(data):
    # the shape of the data is ? x 80, we need transform it to be ? x 20 (each item in one row is a number in set: {1,0,-1,-2})
    data_trans = np.zeros(shape=(data.shape[0],20), dtype=np.int32)
    for i in range(data.shape[0]):
        for j in range(20):
            prop_max = 0.0
            for k in range(4):
                index = j * 4 + k
                prop_tmp = data[i, index]
                if prop_max < prop_tmp:
                    best_id = k
                    prop_max = prop_tmp
            if best_id == 0:
                data_trans[i, j] = 1
            elif best_id == 1:
                data_trans[i, j] = 0
            elif best_id == 2:
                data_trans[i, j] = -1
            elif best_id == 3:
                data_trans[i, j] = -2
            else:
                data_trans[i, j] = -2
    return data_trans
