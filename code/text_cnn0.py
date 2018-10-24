# coding:utf-8

import sys
import numpy as np
import time
import datetime
import os
import data_helper
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm


class TextCNN(object):
    """
    A cnn for text classification.
    Structure: embedding layer -> convolutional layer -> max-pooling layer -> softmax layer.
    """

    def __init__(self, model_id=1, num_epochs=100, batch_size=32, lr=1e-3, num_classes=4, num_types=20, sequence_length=1000, vocab_size=1000,
                 embedding_size=20, filter_sizes=None, num_filters=100, pooling_topk=3, pooling_chunk_size=10,
                 dropout_keep_prob=0.5, l2_reg_lambda=0.0, weight_matrix=None,
                 pre_trained_embedding_matrix=None, device_name='/cpu:0',
                 evaluate_every=100, checkpoint_every=100, num_checkpoints=10):
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        self.model_id = model_id
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_types = num_types
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.pooling_topk = pooling_topk
        self.pooling_chunk_size = pooling_chunk_size
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.weight_matrix = weight_matrix
        self.pre_trained_embedding_matrix = pre_trained_embedding_matrix
        self.device_name = device_name
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.num_checkpoints = num_checkpoints

    def build_model(self):
        # placeholder for the input, output and dropout
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.num_types], name='input_y')
        self.dropout_keep_prob_ph = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.weights = tf.placeholder(dtype=tf.float32, shape=[None, self.num_types], name='weights')

        # keep track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        with tf.device(self.device_name):

            with tf.name_scope('embedding'):
                # embedding layer
                # define W [vocab_size, embedding_size], W can put the vocabulary map to embedding (high dimension -> low dimension)
                if self.pre_trained_embedding_matrix is not None:
                    self.W_embed = tf.Variable(initial_value=self.pre_trained_embedding_matrix, name='W_embed')
                else:
                    self.W_embed = tf.Variable(initial_value=tf.random_uniform(shape=[self.vocab_size, self.embedding_size], minval=-1.0, maxval=1.0), name='W_embed')
                self.embedded_words = tf.nn.embedding_lookup(params=self.W_embed, ids=self.input_x)  # shape: [None, sequence_length, embedding_size]
                self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)  # add one dimension of channel. the final shape is 4 dimension: [None, sequence_length, embedding_size, 1]

            # create a convolutional + max-pooling layer for each filter size
            pooled_output = []
            for i, filter_size in enumerate(self.filter_sizes):
                temp_pooled_outputs = []
                with tf.name_scope('conv-%s' % filter_size):
                    # convolutional layer
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=0.0, stddev=0.1), name='W_conv')
                    b = tf.Variable(tf.constant(value=0.1, shape=[self.num_filters]), name='b_conv')
                    conv = tf.nn.conv2d(
                        input=self.embedded_words_expanded,
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    # max-pooling over outputs
                    # pooled = tf.nn.max_pool(
                    #     value=h,
                    #     ksize=[1, self.sequence_length-filter_size+1, 1, 1],
                    #     strides=[1, 1, 1, 1],
                    #     padding='VALID',
                    #     name='pool')
                    # pooled_output.append(pooled)  # the shape of the pooled output: [batch_size, 1, 1, num_filters]

                    # avg-pooling over outputs
                    pooled = tf.nn.avg_pool(
                        value=h,
                        ksize=[1, self.sequence_length-filter_size+1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool'
                    )
                    pooled_output.append(pooled)

                    # dynamic max pooling layer (k-max pooling)
                    # h = tf.transpose(h, perm=[0, 3, 2, 1])
                    # pooled = tf.nn.top_k(h, self.pooling_topk, sorted=False).values  # [batch_size, num_filters, 1, k]
                    # pooled = tf.transpose(pooled, perm=[0, 3, 2, 1], name='pool')
                    # num_filters_total = int(pooled.get_shape()[-1]) * int(pooled.get_shape()[-3])
                    # pooled = tf.reshape(pooled, [-1, num_filters_total])
                    # pooled_output.append(pooled)

                    # dynamic max pooling layer (chunck-max pooling)
            #         m = int(h.get_shape()[-3])
            #         m_d_p = int(m / self.pooling_chunk_size)
            #         m_bar = int(m_d_p * self.pooling_chunk_size)
            #         h = tf.slice(h, [0, 0, 0, 0], [-1, m_bar, -1, -1])
            #         m = int(h.get_shape()[-3])
            #         index_list = list(range(0, m, m_d_p))
            #         if m not in index_list:
            #             index_list.append(m)
            #         for j in range(len(index_list) - 1):
            #             start = index_list[j]
            #             slice_val = tf.slice(h, [0, start, 0, 0], [-1, m_d_p, -1, -1], name="slice_{}".format(j))
            #             pooled = tf.nn.max_pool(
            #                 slice_val,
            #                 ksize=[1, m_d_p, 1, 1],
            #                 strides=[1, 1, 1, 1],
            #                 padding='VALID',
            #                 name='pool_{}'.format(j))
            #             temp_pooled_outputs.append(pooled)
            #
            #     # Combine all the pooled features
            #     content_pool = tf.concat(temp_pooled_outputs, 3)
            #     num_filters_total = int(content_pool.get_shape()[-1]) * int(content_pool.get_shape()[-3])
            #     content_pool_flat = tf.reshape(content_pool, [-1, num_filters_total])
            #     pooled_output.append(content_pool_flat)

            # self.h_pool_flat = tf.concat(pooled_output, -1)
            # num_filter_total = int(self.h_pool_flat.get_shape()[-1])

            # combine all the pooled features
            num_filter_total = self.num_filters * len(self.filter_sizes)
            self.h_pool = tf.concat(values=pooled_output, axis=3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])  # shape: [batch_size, num_filter_total]

            # add bath normalization
            self.h_pool_flat_norm = tf.layers.batch_normalization(self.h_pool_flat)

            # add 1 FC layer
            with tf.name_scope('fc_layer'):
                h_size_fc = 1024
                W_fc_1 = tf.Variable(tf.truncated_normal(shape=[num_filter_total, h_size_fc], mean=0.0, stddev=0.1), name='W_fc_1')
                b_fc_1 = tf.Variable(tf.constant(value=0.1, shape=[h_size_fc]), name='b_fc_1')
                self.h_fc = tf.nn.xw_plus_b(x=self.h_pool_flat_norm, weights=W_fc_1, biases=b_fc_1, name='h_fc_1')

            # add batch normalization
            self.h_fc_norm = tf.layers.batch_normalization(self.h_fc)

            # add dropout
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_fc_norm, keep_prob=self.dropout_keep_prob_ph, name='dropout_keep_prob')

            # final output
            with tf.name_scope('output'):
                self.all_scores = []
                y_logits = []
                y_pred = []
                for i in range(self.num_types):
                    W_fc = tf.Variable(tf.truncated_normal(shape=[h_size_fc, self.num_classes], mean=0.0, stddev=0.1), name='W_fc_{}'.format(i))
                    b_fc = tf.Variable(tf.constant(value=0.1, shape=[self.num_classes]), name='b_fc_{}'.format(i))
                    # self.l2_loss += tf.nn.l2_loss(W_fc)
                    # self.l2_loss += tf.nn.l2_loss(b_fc)
                    scores = tf.nn.xw_plus_b(x=self.h_drop, weights=W_fc, biases=b_fc, name='scores_{}'.format(i))
                    self.all_scores.append(scores)
                    y_logits_sub = tf.nn.softmax(scores, name='y_logits_sub_{}'.format(i))
                    y_pred_sub = tf.argmax(input=scores, axis=1, name='y_pred_sub_{}'.format(i))
                    y_logits.append(y_logits_sub)
                    y_pred.append(y_pred_sub)
                self.y_logits = tf.transpose(y_logits, [1, 0, 2], name='y_logits_transpose')
                self.y_logits = tf.reshape(self.y_logits, [-1, self.num_types * self.num_classes], name='y_logits')
                self.y_pred = tf.transpose(y_pred, [1, 0], name='y_pred_transpose')
                self.y_pred = tf.cast(self.y_pred, 'int32', name='y_pred')

            # loss
            with tf.name_scope('loss'):
                losses = []
                for i in range(self.num_types):
                    if self.weight_matrix is None:
                        loss_sub = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y[:, i], logits=self.all_scores[i])
                    else:
                        loss_sub = tf.losses.sparse_softmax_cross_entropy(labels=self.input_y[:, i], logits=self.all_scores[i], weights=self.weights[:, i])
                    loss_sub = tf.reduce_mean(loss_sub)
                    losses.append(loss_sub)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

            # calculate accuracy
            with tf.name_scope('accuracy'):
                self.correct_preds = tf.equal(self.y_pred, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, 'float32'), name='accuracy')
                self.confusion_matrix_list = []
                # for i in range(self.num_types):
                #     confusion_matrix = tf.confusion_matrix(self.input_y[:, i], predictions=self.y_pred[:, i], num_classes=4, name='confusion_matrix_list')
                #     self.confusion_matrix_list.append(confusion_matrix)

        # optimize
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name='train_op')

    def train_model(self, x_train, y_train, x_dev, y_dev, label_origin_dev, use_pre_trained=False, pre_trained_model_path=None, per_process_gpu_memory_fraction=None):
        if per_process_gpu_memory_fraction is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        else:
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=session_conf) as sess:
            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.curdir, 'log', str(self.model_id), timestamp))
            print('Writing log to {}\n'.format(out_dir))

            # summary all the trainable variables
            for var in tf.trainable_variables():
                tf.summary.histogram(name=var.name, values=var)

            # summaries for loss and accuracy
            loss_summary = tf.summary.scalar('summary_loss', self.loss)
            acc_summary = tf.summary.scalar('summary_accuracy', self.accuracy)

            # train summaries
            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())

            # dev summaries
            dev_summary_op = tf.summary.merge_all()
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())

            # checkpointing, tensorflow assumes this directory already existed, so we need to create it
            checkpoint_dir = os.path.join(out_dir, 'checkpoints')
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            checkpoint_dir_best_model = os.path.join(checkpoint_dir, 'best-model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                os.makedirs(checkpoint_dir_best_model)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
            saver_best = tf.train.Saver(tf.global_variables())

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            # use pre-trained model to continue
            if use_pre_trained:
                print('reloading model parameters...')
                saver.restore(sess, pre_trained_model_path)

            # function for a single training step
            def train_step(x_batch, y_batch, writer=None):
                # sample_weights = data_helper.get_sample_weights(y_batch, self.weight_matrix)
                # sample_weights = sample_weights[:, 0:1]
                feed_dict = {
                    self.input_x: x_batch,
                    self.input_y: y_batch,
                    self.dropout_keep_prob_ph: self.dropout_keep_prob
                    # self.weights: sample_weights
                }
                _, step, summaries, loss, acc, y_logits, y_pred, input_y = sess.run(
                    [self.train_op, self.global_step, train_summary_op, self.loss, self.accuracy, self.y_logits, self.y_pred, self.input_y],
                    feed_dict)
                timestr = datetime.datetime.now().isoformat()
                num_batches_per_epoch = int((len(x_train) - 1) / self.batch_size) + 1
                epoch = int((step - 1) / num_batches_per_epoch) + 1
                print('\rtrain_step: {} => epoch {} | step {} | loss {:.5f} | acc {:.5f}'.format(timestr, epoch, step, loss, acc), end='')
                # print()
                # print('y_logits:\n{}'.format(y_logits[0]))
                # print('y_pred: {}'.format(y_pred[0]))
                # print('y_true: {}'.format(input_y[0]))
                # print('y_pred: {}'.format(np.reshape(y_pred, (-1))))
                # print('y_true: {}'.format(np.reshape(input_y, (-1))))
                # print('confusion_matrix:\n{}'.format(confusion_matrix))
                # print(sample_weights)

                if writer:
                    writer.add_summary(summaries, step)

            def dev_step0(x_batch, y_batch, label_origin_batch=None, writer=None):
                # sample_weights = data_helper.get_sample_weights(y_batch, self.weight_matrix)
                # sample_weights = sample_weights[:, 0:1]
                feed_dict = {
                    self.input_x: x_batch,
                    self.input_y: y_batch,
                    self.dropout_keep_prob_ph: 1.0
                    # self.weights: sample_weights
                }
                step, summaries, loss, accuracy, y_logits, y_pred, confusion_matrix_list = sess.run(
                    [self.global_step, dev_summary_op, self.loss, self.accuracy, self.y_logits, self.y_pred, self.confusion_matrix_list],
                    feed_dict)
                timestr = datetime.datetime.now().isoformat()
                num_batches_per_epoch = int((len(x_train) - 1) / self.batch_size) + 1
                epoch = int(step / num_batches_per_epoch) + 1
                y_pred_trans = data_helper.id2label(y_pred)

                # calc F1 score
                score_f1_list = []
                for col in range(y_pred_trans.shape[1]):
                    score_f1_one = metrics.f1_score(label_origin_batch[:, col], y_pred_trans[:, col], average='macro')
                    score_f1_list.append(score_f1_one)
                score_f1 = np.mean(score_f1_list)

                print('dev_step:   {} => epoch {} | step {} | loss {:.5f} | acc {:.5f} | F1_score: {:.5f}'.format(timestr, epoch, step, loss, accuracy, score_f1))
                print('confusion_matrix:\n{}'.format(confusion_matrix_list))

                if writer:
                    writer.add_summary(summaries, step)
                return score_f1

            def dev_step(x_batch, y_batch, label_origin_batch=None, batch_size=100, writer=None):
                all_y_pred = []
                batches = data_helper.batch_iter(list(x_batch), batch_size, 1, shuffle=False)
                iter = 1
                iter_sum = int((len(x_batch) - 1) / batch_size) + 1
                for x_dev_batch in batches:
                    progress_percent = iter / iter_sum * 100
                    print('\rpredicting: {:.0f}%'.format(progress_percent), end='')
                    feed_dict = {
                        self.input_x: x_dev_batch,
                        self.dropout_keep_prob_ph: 1.0
                    }
                    y_logits_batch, y_pred_batch = sess.run([self.y_logits, self.y_pred], feed_dict)
                    all_y_pred.append(y_pred_batch)
                    iter += 1
                print()
                all_y_pred = np.array(all_y_pred)
                all_y_pred = np.reshape(all_y_pred, (-1, self.num_types))
                y_pred_trans = data_helper.id2label(all_y_pred)

                # calc F1 score
                score_f1_list = []
                acc_list = []
                for col in range(y_pred_trans.shape[1]):
                    score_f1_one = metrics.f1_score(label_origin_batch[:, col], y_pred_trans[:, col], average='macro')
                    score_f1_list.append(score_f1_one)
                    acc_one = metrics.accuracy_score(label_origin_batch[:, col], y_pred_trans[:, col])
                    acc_list.append(acc_one)
                score_f1_dev = np.mean(score_f1_list)
                acc_dev = np.mean(acc_list)

                print('F1_score: {:.5}'.format(score_f1_dev))
                print('accuracy: {:.5}'.format(acc_dev))

                if writer:
                    pass
                return score_f1_dev

            ### training loop
            # generate batches
            batches = data_helper.batch_iter(
                data=list(zip(x_train, y_train)), batch_size=self.batch_size, num_epochs=self.num_epochs)
            score_f1_max = 0.0
            # train loop, for each batch
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch).astype(np.int32)
                y_batch = np.array(y_batch).astype(np.int32)
                train_step(x_batch, y_batch, writer=train_summary_writer)
                current_step = tf.train.global_step(sess, self.global_step)
                if current_step % self.evaluate_every == 0:
                    # print('\nEvaluation on dev set:')
                    print()
                    score_f1 = dev_step(x_dev, y_dev, label_origin_dev, writer=dev_summary_writer)
                    with open(os.path.join(checkpoint_dir_best_model, 'log_f1_score.txt'), 'a', encoding='utf-8') as f:
                        f.write(str(current_step) + ',' + str('{:.5f}'.format(score_f1)) + '\n')
                    if score_f1_max < score_f1:
                        score_f1_max = score_f1
                        path_best_model = os.path.join(checkpoint_dir_best_model, 'model-best-{}'.format(self.model_id))
                        # path_best_model = './model/model-best-{}'.format(current_step)
                        saver_best.save(sess, path_best_model)
                        print('>>> Saved best model to {}'.format(path_best_model))
                        with open(os.path.join(checkpoint_dir_best_model, 'log_best_f1_score.txt'), 'w',
                                  encoding='utf-8') as f:
                            f.write(str('{:.5f}'.format(score_f1_max)) + '\n')
                    print()
                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess=sess, save_path=checkpoint_prefix, global_step=self.global_step)
                    print('Saved model checkpoint to {}'.format(path))
                    print()

    def predict(self, model_file, x_test):
        # self.build_model()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('{}.meta'.format(model_file))
            saver.restore(sess, model_file)

            # Access and create placeholders variables and create feed-dict to feed new data
            graph = tf.get_default_graph()
            input_x_ph = graph.get_tensor_by_name('input_x:0')
            dropout_keep_prob_ph = graph.get_tensor_by_name('dropout_keep_prob:0')
            feed_dict = {
                input_x_ph: x_test,
                dropout_keep_prob_ph: 1.0
            }
            op_y_logits = graph.get_tensor_by_name('output/y_logits:0')
            op_y_pred = graph.get_tensor_by_name('output/y_pred:0')
            y_logits, y_pred = sess.run([op_y_logits, op_y_pred], feed_dict)
        return y_pred

    def predict_batch(self, model_file, x_test, batch_size=100):
        # self.build_model()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('{}.meta'.format(model_file))
            saver.restore(sess, model_file)

            # Access and create placeholders variables and create feed-dict to feed new data
            graph = tf.get_default_graph()
            input_x_ph = graph.get_tensor_by_name('input_x:0')
            dropout_keep_prob_ph = graph.get_tensor_by_name('dropout_keep_prob:0')

            op_y_logits = graph.get_tensor_by_name('output/y_logits:0')
            op_y_pred = graph.get_tensor_by_name('output/y_pred:0')
            all_y_pred = []
            batches = data_helper.batch_iter(list(x_test), batch_size, 1, shuffle=False)
            iter = 1
            iter_sum = int((len(x_test)-1)/batch_size)+1
            for x_test_batch in batches:
                progress_percent = iter/iter_sum*100
                print('\rpredicting: {:.0f}%'.format(progress_percent), end='')
                feed_dict = {
                    input_x_ph: x_test_batch,
                    dropout_keep_prob_ph: 1.0
                }
                y_logits_batch, y_pred_batch = sess.run([op_y_logits, op_y_pred], feed_dict)
                all_y_pred.append(y_pred_batch)
                iter += 1
            print()
            all_y_pred = np.array(all_y_pred)
            all_y_pred = np.reshape(all_y_pred, (-1, self.num_types))
        return all_y_pred
