# coding:utf-8

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import data_helper
import matplotlib.pyplot as plt
import pickle
from text_cnn import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # in {'0', '1', '2', '3'}
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


### hyper parameters
lr = 1e-3
num_epochs = 10
batch_size = 32
num_classes = 4
num_types = 1  # 20
embedding_size = 20
filter_sizes = [2, 3, 4, 5]
num_filters = 100
pooling_topk = 3
pooling_chunk_size = 3
dropout_keep_prob = 0.5
l2_reg_lambda = 3.0
device_name = '/cpu:0'
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 10
word_min_frequency = 2
max_sentence_len = None
use_pre_trained_model = False
pre_trained_model_path = './model/model-4100'
use_pre_trained_embedding = False
pre_trained_embedding_file = './data/merge_sgns_bigram_char300.txt'
use_pkl = True
type_slice_idx = 1
model_id = type_slice_idx
per_process_gpu_memory_fraction = None


### loading the train dataset
print('Preprocessing data set...')
if use_pkl:  # load data from pkl file (much faster)
    print('loading data from pkl file (much faster)...')
    with open('./data/x_text_train.pkl', 'rb') as f:
        x_text_train = pickle.load(f)
    with open('./data/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open('./data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('./data/word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
    with open('./data/x_dev.pkl', 'rb') as f:
        x_dev = pickle.load(f)
    with open('./data/y_dev.pkl', 'rb') as f:
        y_dev = pickle.load(f)
else:
    print('loading x_text and labels...\n')
    x_text_train, y_train = data_helper.load_text_and_label('./data/df_train.csv', topn=None)
    if max_sentence_len is None:
        max_sentence_len = max([len(s) for s in x_text_train])

    print('loading word2id...\n')
    word2id = data_helper.extract_character_vocab(x_text_train, min_frequency=word_min_frequency)

    print('loading padded x_ids...\n')
    x_train = data_helper.get_x_ids(x_text_train, word2id)
    x_train = data_helper.pad_sentence_batch(x_train, max_sentence_len=max_sentence_len)

    y_train = y_train.astype(np.int32)

    # save data by pickle
    with open('./data/x_text_train.pkl', 'wb') as f:
        pickle.dump(x_text_train, f)
    with open('./data/x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('./data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('./data/word2id.pkl', 'wb') as f:
        pickle.dump(word2id, f)

    # loading validation (dev) set
    print('loading validation set...\n')
    x_text_dev, y_dev = data_helper.load_text_and_label('./data/df_valid.csv', topn=None)
    x_dev = data_helper.get_x_ids(x_text_dev, word2id)
    x_dev = data_helper.pad_sentence_batch(x_dev, max_sentence_len=max_sentence_len)
    y_dev = y_dev.astype(np.int32)
    with open('./data/x_dev.pkl', 'wb') as f:
        pickle.dump(x_dev, f)
    with open('./data/y_dev.pkl', 'wb') as f:
        pickle.dump(y_dev, f)

# get the pre-trained word vectors
if not use_pre_trained_model and use_pre_trained_embedding:
    print('loading pre-trained word vectors...')
    pre_word_vectors = data_helper.get_pre_trained_word_vectors(pre_trained_embedding_file)
    pre_trained_embedding_size = len(list(pre_word_vectors.values())[0])
    pre_trained_embedding_matrix, include_prob = data_helper.load_word_vectors(word2id, pre_word_vectors, embedding_size)
    print('embedding_shape: {}, include_prob: {}'.format(pre_trained_embedding_matrix.shape, include_prob))
    with open('./data/pre_trained_embedding_matrix.pkl', 'wb') as f:
        pickle.dump(pre_trained_embedding_matrix, f)
    # with open('./data/pre_trained_embedding_matrix.pkl', 'rb') as f:
    #     pre_trained_embedding_matrix = pickle.load(f)
else:
    pre_trained_embedding_matrix = None

print('loading original labels...')
label_origin_dev = data_helper.load_origin_label('./data/df_valid.csv')

# get weight matrix of each type
weight_matrix = data_helper.get_weight_matrix('./data/df_train.csv')

# resampling the data
print('original data distribution:')
data_helper.show_weight_matrix(y_train)
x_train, y_train = data_helper.resampling_data(x_train, y_train, weight_matrix, type_slice_idx-1)
print('data distribution after resampling:')
data_helper.show_weight_matrix(y_train)

# get one type of data
y_train = y_train[:, type_slice_idx-1:type_slice_idx]
y_dev = y_dev[:, type_slice_idx-1:type_slice_idx]
label_origin_dev = label_origin_dev[:, type_slice_idx-1:type_slice_idx]

# show data description
max_sentence_len = x_train.shape[1]
vocab_size = len(word2id)
print('learning_rate:             {}'.format(lr))
print('max_sentence_len:          {}'.format(max_sentence_len))
print('vocab_size:                {}'.format(vocab_size))
print('embedding_size:            {}'.format(embedding_size))
print('filter_sizes:              {}'.format(filter_sizes))
print('pooling_topk:              {}'.format(pooling_topk))
print('pooling_chunk_size:        {}'.format(pooling_chunk_size))
print('num_epochs:                {}'.format(num_epochs))
print('x_train_shape:             {}'.format(x_train.shape))
print('y_train_shape:             {}'.format(y_train.shape))
print('num_types:                 {}'.format(num_types))
print('num_classes:               {}'.format(num_classes))
print('x_dev_shape:               {}'.format(x_dev.shape))
print('y_dev_shape:               {}'.format(y_dev.shape))
print('show data examples (top 2):')
print('x_train[0:2]:\n{}'.format(x_train[0:2]))
print('y_train[0:2]:\n{}'.format(y_train[0:2]))

# sys.exit(0)

### build the model and train
print('Start training...')
time.sleep(5)
text_cnn_model = TextCNN(
    model_id=model_id,
    lr=lr,
    num_epochs=num_epochs,
    batch_size=batch_size,
    num_classes=num_classes,
    num_types=num_types,
    sequence_length=max_sentence_len,
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    filter_sizes=filter_sizes,
    num_filters=num_filters,
    pooling_topk=pooling_topk,
    pooling_chunk_size=pooling_chunk_size,
    dropout_keep_prob=dropout_keep_prob,
    l2_reg_lambda=l2_reg_lambda,
    weight_matrix=None,
    pre_trained_embedding_matrix=pre_trained_embedding_matrix,
    device_name=device_name,
    evaluate_every=evaluate_every,
    checkpoint_every=checkpoint_every,
    num_checkpoints=num_checkpoints)

text_cnn_model.build_model()
text_cnn_model.train_model(x_train, y_train, x_dev, y_dev, label_origin_dev,
                           use_pre_trained_model, pre_trained_model_path, per_process_gpu_memory_fraction)
# text_cnn_model.train_model(x_train, y_train, x_dev[:1000], y_dev[:1000], label_origin_dev[:1000], use_pre_trained_model, pre_trained_model_path)
