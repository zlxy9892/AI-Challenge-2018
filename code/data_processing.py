# coding:utf-8

import numpy as np
import pickle
import data_helper


def main():
    ### parameters
    word_min_frequency = 2
    embedding_size = 300
    max_sentence_len = None
    use_pre_trained_embedding = True
    pre_trained_embedding_file = './data/merge_sgns_bigram_char300.txt'

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
    with open('./data/x_text_dev.pkl', 'wb') as f:
        pickle.dump(x_text_dev, f)
    with open('./data/x_dev.pkl', 'wb') as f:
        pickle.dump(x_dev, f)
    with open('./data/y_dev.pkl', 'wb') as f:
        pickle.dump(y_dev, f)

    # loading testa set
    print('loading testa dataset...')
    x_text_testa, _ = data_helper.load_text_and_label('./data/df_testa.csv', include_label=False)
    print(len(x_text_testa))
    print(x_text_testa[0])
    x_testa = data_helper.get_x_ids(x_text_testa, word2id)
    x_testa = data_helper.pad_sentence_batch(x_testa, max_sentence_len=max_sentence_len)
    with open('./data/x_text_testa.pkl', 'wb') as f:
        pickle.dump(x_text_testa, f)
    with open('./data/x_testa.pkl', 'wb') as f:
        pickle.dump(x_testa, f)

    # get the pre-trained word vectors
    if use_pre_trained_embedding:
        print('loading pre-trained word vectors...')
        pre_word_vectors = data_helper.get_pre_trained_word_vectors(pre_trained_embedding_file)
        pre_trained_embedding_size = len(list(pre_word_vectors.values())[0])
        embedding_size = pre_trained_embedding_size
        pre_trained_embedding_matrix, include_prob = data_helper.load_word_vectors(word2id, pre_word_vectors, embedding_size)
        print('embedding_shape: {}, include_prob: {}'.format(pre_trained_embedding_matrix.shape, include_prob))
        with open('./data/pre_trained_embedding_matrix.pkl', 'wb') as f:
            pickle.dump(pre_trained_embedding_matrix, f)

    # show data description
    max_sentence_len = x_train.shape[1]
    vocab_size = len(word2id)
    print('max_sentence_len:          {}'.format(max_sentence_len))
    print('vocab_size:                {}'.format(vocab_size))
    print('embedding_size:            {}'.format(embedding_size))
    print('x_train_shape:             {}'.format(x_train.shape))
    print('y_train_shape:             {}'.format(y_train.shape))
    print('x_dev_shape:               {}'.format(x_dev.shape))
    print('y_dev_shape:               {}'.format(y_dev.shape))
    print('x_testa_shape:             {}'.format(x_testa.shape))
    print('show data examples (top 2):')
    print('x_train[0:2]:\n{}'.format(x_train[0:2]))
    print('y_train[0:2]:\n{}'.format(y_train[0:2]))


if __name__ == '__main__':
    main()
