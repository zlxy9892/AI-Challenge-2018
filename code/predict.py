# coding:utf-8

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
import data_helper
import pickle
from text_cnn import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # in {'0', '1', '2', '3'}
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


### hyper parameters
model_file = './model-best-20181016/model-best-1'
model_dir = './model-best-20181016/'

lr = 1e-3
num_epochs = 3
batch_size = 32
num_classes = 4
num_types = 1  # 20
embedding_size = 300
filter_sizes = [2, 3, 4, 5]
num_filters = 200
pooling_topk = 3
pooling_chunk_size = 3
dropout_keep_prob = 0.5
l2_reg_lambda = 1.0
device_name = '/cpu:0'
max_sentence_len = 1740

# load word2id dict
with open('./data/word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)
vocab_size = len(list(word2id.keys()))

# load validation (dev) set
print('loading validation set...\n')
with open('./data/x_dev.pkl', 'rb') as f:
    x_dev = pickle.load(f)
with open('./data/y_dev.pkl', 'rb') as f:
    y_dev = pickle.load(f)
print('loading original labels...')
label_origin_dev = data_helper.load_origin_label('./data/sentiment_analysis_validationset.csv')

# load testa dataset
print('loading testa dataset...')
# x_text_test, _ = data_helper.load_text_and_label('./data/df_testa.csv', include_label=False)
# print(len(x_text_test))
# print(x_text_test[0])
# x_test = data_helper.get_x_ids(x_text_test, word2id)
# x_test = data_helper.pad_sentence_batch(x_test, max_sentence_len=max_sentence_len)
# with open('./data/x_testa.pkl', 'wb') as f:
#     pickle.dump(x_test, f)
with open('./data/x_testa.pkl', 'rb') as f:
    x_test = pickle.load(f)
print('x_test data shape: {}'.format(x_test.shape))

# sys.exit(0)

print('Build the model...')
### build the model and train
text_cnn_model = TextCNN(
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
    pre_trained_embedding_matrix=None,
    device_name=device_name)

text_cnn_model.build_model()


# predict dev data by pre-trained model
def pred_dev_set(save_pred=False):
    print('Start predict dev set...')
    y_dev_pred = text_cnn_model.predict_batch(model_file=model_file, x_test=x_dev[:], batch_size=100)
    y_dev_pred = data_helper.id2label(y_dev_pred)
    print('y_dev_pred_shape: {}'.format(y_dev_pred.shape))

    # calc F1 score
    score_f1_list = []
    for col in range(y_dev_pred.shape[1]):
        score_f1_one = metrics.f1_score(label_origin_dev[:][:, col], y_dev_pred[:, col], average='macro')
        score_f1_list.append(score_f1_one)
    score_f1 = np.mean(score_f1_list)
    print('F1_score: {:.5f}'.format(score_f1))

    if save_pred:
        df_valid_pred = pd.read_csv('./data/sentiment_analysis_validationset.csv')
        colnames = df_valid_pred.columns
        df_valid_pred[colnames[2:]] = y_dev_pred
        filename = './data/df_valid_pred.csv'
        df_valid_pred.to_csv(path_or_buf=filename, header=True, index=False)  # write out predicted valid dataset
        print('write out predicted dev file: <{}> OK.'.format(filename))


# predict dev data by pre-trained model
def pred_test_set():
    print('Start predict test set...')
    y_test_pred = text_cnn_model.predict_batch(model_file=model_file, x_test=x_test[:], batch_size=100)
    y_test_pred = data_helper.id2label(y_test_pred)
    print('y_test_pred_shape: {}'.format(y_test_pred.shape))

    # transform the y_test_pred data to be the form of DataFrame
    df_test_pred = pd.read_csv('./data/sentiment_analysis_testa.csv')
    colnames = df_test_pred.columns
    df_test_pred[colnames[2:]] = y_test_pred
    filename = './data/df_testa_pred.csv'
    df_test_pred.to_csv(path_or_buf=filename, header=True, index=False)  # write out predicted testa dataset
    print('write out predicted testa file: <{}> OK.'.format(filename))


# predict dev data by pre-trained model (20 models)
def pred_dev_set_20(save_pred=False):
    print('Start predict dev set (20 models)...')
    y_dev_pred_all = []
    for i in range(1, 20 + 1):
        model_filename = './log/' + str(i) + '/checkpoints/best-model/model-best-' + str(i)
        print(model_filename)
        print('predicting type {}:'.format(i))
        y_dev_pred = text_cnn_model.predict_batch(model_file=model_filename, x_test=x_dev[:], batch_size=100)
        y_dev_pred = data_helper.id2label(y_dev_pred)
        y_dev_pred = y_dev_pred.reshape((-1))
        print('y_dev_pred_shape: {}'.format(y_dev_pred.shape))
        y_dev_pred_all.append(y_dev_pred)
    y_dev_pred_all = np.array(y_dev_pred_all).T
    print('y_dev_pred_all: {}'.format(y_dev_pred_all.shape))

    # calc F1 score
    score_f1_list = []
    for col in range(y_dev_pred_all.shape[1]):
        score_f1_one = metrics.f1_score(label_origin_dev[:][:, col], y_dev_pred_all[:, col], average='macro')
        score_f1_list.append(score_f1_one)
    score_f1 = np.mean(score_f1_list)
    print('F1_score: {:.5f}'.format(score_f1))

    if save_pred:
        df_valid_pred = pd.read_csv('./data/sentiment_analysis_validationset.csv')
        colnames = df_valid_pred.columns
        df_valid_pred[colnames[2:]] = y_dev_pred_all
        filename = './data/df_valid_pred.csv'
        df_valid_pred.to_csv(path_or_buf=filename, header=True, index=False)  # write out predicted valid dataset
        print('write out predicted dev file: <{}> OK.'.format(filename))


# predict dev data by pre-trained model (20 models)
def pred_test_set_20():
    print('Start predict test set (20 models)...')
    y_test_pred_all = []
    for i in range(1, 20 + 1):
        model_filename = './log/' + str(i) + '/checkpoints/best-model/model-best-' + str(i)
        print(model_filename)
        print('predicting type {}:'.format(i))
        y_test_pred = text_cnn_model.predict_batch(model_file=model_filename, x_test=x_test[:], batch_size=100)
        y_test_pred = data_helper.id2label(y_test_pred)
        y_test_pred = y_test_pred.reshape((-1))
        print('y_test_pred_shape: {}'.format(y_test_pred.shape))
        y_test_pred_all.append(y_test_pred)
    y_test_pred_all = np.array(y_test_pred_all).T
    print('y_test_pred_all: {}'.format(y_test_pred_all.shape))

    # transform the y_test_pred data to be the form of DataFrame
    df_test_pred = pd.read_csv('./data/sentiment_analysis_testa.csv')
    colnames = df_test_pred.columns
    df_test_pred[colnames[2:]] = y_test_pred_all
    filename = './data/df_testa_pred.csv'
    df_test_pred.to_csv(path_or_buf=filename, header=True, index=False)  # write out predicted testa dataset
    print('write out predicted testa file: <{}> OK.'.format(filename))


def main():
    #     pred_dev_set()
    #     pred_test_set()
    #     pred_dev_set_20(save_pred=True)
    pred_test_set_20()


if __name__ == '__main__':
    main()
