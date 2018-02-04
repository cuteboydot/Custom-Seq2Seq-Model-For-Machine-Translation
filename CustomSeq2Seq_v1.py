"""
Implementation of Seq2Seq Model for Neural Machine Translation

Reference
1. https://arxiv.org/pdf/1409.0473.pdf
2. https://devblogs.nvidia.com/introduction-neural-machine-translation-gpus-part-3/
3. https://medium.com/datalogue/attention-in-keras-1892773a4f22

cuteboydot@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import numpy as np
import os
import datetime
import pickle
import random
from time import time
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
import matplotlib.font_manager as font_manager

np.core.arrayprint._line_width = 1000
np.set_printoptions(precision=3)

file_data = "./data/sentence_full.tsv"
file_model = "./data/model.ckpt"
file_dic_eng = "./data/dic_eng.bin"
file_rdic_eng = "./data/rdic_eng.bin"
file_dic_kor = "./data/dic_kor.bin"
file_rdic_kor = "./data/rdic_kor.bin"
file_data_list = "./data/data_list.bin"
file_data_idx_list = "./data/data_idx_list.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"
dir_summary = "./model/summary/"

pre_trained = 2
my_device = "/cpu:0"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

font_name = font_manager.FontProperties(fname="/Library/Fonts/AppleGothic.ttf").get_name()
rc('font', family=font_name)

print("+" * 70)
print("Custom Sequence to Sequence Start !!!")
print("+" * 70)

if pre_trained == 0:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)

    print("Load data file & make vocabulary...")

    data_list = []
    total_eng = ""
    total_kor = ""
    with open(file_data, "r", encoding="utf8") as tsv:
        for line in tsv:
            sep = line.split("\t")

            category = int(sep[0].replace("\ufeff", ""))
            sentence_english = sep[1].lower()
            sentence_english = sentence_english.replace("\n", "")
            sentence_korean = sep[2].lower()
            sentence_korean = sentence_korean.replace("\n", "")

            total_eng += sentence_english
            total_kor += sentence_korean
            data_list.append([list(sentence_english), list(sentence_korean), category])

    print("data_list example")
    print(data_list[0])
    print("data_list size = %d" % len(data_list))

    # make english char vocab
    symbols_eng = ["<PAD>", "<UNK>"]
    rdic_eng = symbols_eng + list(set(total_eng))
    dic_eng = {w: i for i, w in enumerate(rdic_eng)}
    voc_size_eng = len(rdic_eng)
    print("voc_size_eng size = %d" % voc_size_eng)
    print(dic_eng)

    # make korean char vocab
    symbols_kor = ["<PAD>", "<UNK>", "<GO>"]
    rdic_kor = symbols_kor + list(set(total_kor))
    dic_kor = {w: i for i, w in enumerate(rdic_kor)}
    voc_size_kor = len(rdic_kor)
    print("voc_size_kor size = %d" % voc_size_kor)
    print(dic_kor)

    data_idx_list = []
    eng_len_list = []
    kor_len_list = []
    for english, korean, category in data_list:
        idx_eng = []
        for eng in english:
            e = ""
            if eng in rdic_eng:
                e = dic_eng[eng]
            else:
                e = dic_eng["<UNK>"]
            idx_eng.append(e)

        idx_kor = []
        for kor in korean:
            k = ""
            if kor in rdic_kor:
                k = dic_kor[kor]
            else:
                k = dic_kor["<UNK>"]
            idx_kor.append(k)

        data_idx_list.append([idx_eng, idx_kor, category])
        eng_len_list.append(len(english))
        kor_len_list.append(len(korean))

    max_eng_len = max(eng_len_list)
    max_kor_len = max(kor_len_list)
    max_len = [max_eng_len, max_kor_len]
    print("max_eng_len = %d" % max_eng_len)
    print("max_kor_len = %d" % max_kor_len)
    print()

    padded_eng_len = max_eng_len + 1
    padded_kor_len = max_kor_len + 2
    print("padded_eng_len = %d" % padded_eng_len)
    print("padded_kor_len = %d" % padded_kor_len)

    # split data set
    SIZE_TEST_DATA = 100
    random.shuffle(data_idx_list)
    data_idx_list_test = data_idx_list[:SIZE_TEST_DATA]
    data_idx_list = data_idx_list[SIZE_TEST_DATA:]
    print("dataset for train = %d" % len(data_idx_list))
    print("dataset for test = %d" % len(data_idx_list_test))
    print()

    # save dictionary
    with open(file_data_list, 'wb') as handle:
        pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list, 'wb') as handle:
        pickle.dump(data_idx_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_data_idx_list_test, 'wb') as handle:
        pickle.dump(data_idx_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_dic_eng, 'wb') as handle:
        pickle.dump(rdic_eng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_rdic_eng, 'wb') as handle:
        pickle.dump(dic_eng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_dic_kor, 'wb') as handle:
        pickle.dump(rdic_kor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_rdic_kor, 'wb') as handle:
        pickle.dump(dic_kor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_max_len, 'wb') as handle:
        pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("data & vocabulary saved...")

else:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Load vocabulary from model file...")

    with open(file_data_list, 'rb') as handle:
        data_list = pickle.load(handle)
    with open(file_data_idx_list, 'rb') as handle:
        data_idx_list = pickle.load(handle)
    with open(file_data_idx_list_test, 'rb') as handle:
        data_idx_list_test = pickle.load(handle)
    with open(file_rdic_eng, 'rb') as handle:
        dic_eng = pickle.load(handle)
    with open(file_dic_eng, 'rb') as handle:
        rdic_eng = pickle.load(handle)
    with open(file_rdic_kor, 'rb') as handle:
        dic_kor = pickle.load(handle)
    with open(file_dic_kor, 'rb') as handle:
        rdic_kor = pickle.load(handle)
    with open(file_max_len, 'rb') as handle:
        max_len = pickle.load(handle)

    print("data_list example")
    print(data_list[0])
    print("data_list size = %d" % len(data_list))

    voc_size_eng = len(rdic_eng)
    print("voc_size_eng size = %d" % voc_size_eng)
    voc_size_kor = len(rdic_kor)
    print("voc_size_kor size = %d" % voc_size_kor)

    max_eng_len = max_len[0]
    max_kor_len = max_len[1]
    print("max_eng_len = %d" % max_eng_len)
    print("max_kor_len = %d" % max_kor_len)
    print()

    padded_eng_len = max_eng_len + 1
    padded_kor_len = max_kor_len + 2
    print("padded_eng_len = %d" % padded_eng_len)
    print("padded_kor_len = %d" % padded_kor_len)

    print("dataset for train = %d" % len(data_idx_list))
    print("dataset for test = %d" % len(data_idx_list_test))
    print()


'''''''''''''''''''''''''''''''''''''''''''''
BATCH GENERATOR
'''''''''''''''''''''''''''''''''''''''''''''
def generate_batch(size):
    np.random.seed(int(time()))
    assert size <= len(data_idx_list)

    data_x = np.zeros((size, padded_eng_len), dtype=np.int)
    data_y = np.zeros((size, padded_kor_len), dtype=np.int)
    data_t = np.zeros((size, padded_kor_len), dtype=np.int)
    len_x = np.zeros(size, dtype=np.int)
    len_y = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        x = data_idx_list[idx][0]
        len_x[a] = len(x)

        y = data_idx_list[idx][1]
        len_y[a] = len(y)

        x_ = x + [dic_eng["<PAD>"]] * (padded_eng_len - len(x))
        y_ = [dic_kor["<GO>"]] + y + [dic_kor["<PAD>"]] * (padded_kor_len - len(y) - 1)
        t_ = y + [dic_kor["<PAD>"]] * (padded_kor_len - len(y))
        assert len(x_) == padded_eng_len
        assert len(y_) == padded_kor_len
        assert len(t_) == padded_kor_len
        assert y_[-1] == dic_kor["<PAD>"]

        data_x[a] = x_
        data_y[a] = y_
        data_t[a] = t_

    return data_x, data_y, data_t, len_x, len_y


def generate_test_batch(size):
    np.random.seed(int(time()))
    assert size <= len(data_idx_list_test)

    data_x = np.zeros((size, padded_eng_len), dtype=np.int)
    data_y = np.zeros((size, padded_kor_len), dtype=np.int)
    data_t = np.zeros((size, padded_kor_len), dtype=np.int)
    len_x = np.zeros(size, dtype=np.int)
    len_y = np.zeros(size, dtype=np.int)

    index = np.random.choice(range(len(data_idx_list_test)), size, replace=False)
    for a in range(len(index)):
        idx = index[a]

        x = data_idx_list_test[idx][0]
        len_x[a] = len(x)

        y = data_idx_list_test[idx][1]
        len_y[a] = len(y)

        x_ = x + [dic_eng["<PAD>"]] * (padded_eng_len - len(x))
        y_ = [dic_kor["<GO>"]] + y + [dic_kor["<PAD>"]] * (padded_kor_len - len(y) - 1)
        t_ = y + [dic_kor["<PAD>"]] * (padded_kor_len - len(y))
        assert len(x_) == padded_eng_len
        assert len(y_) == padded_kor_len
        assert len(t_) == padded_kor_len
        assert y_[-1] == dic_kor["<PAD>"]

        data_x[a] = x_
        data_y[a] = y_
        data_t[a] = t_

    return data_x, data_y, data_t, len_x, len_y


with tf.Graph().as_default():
    '''''''''''''''''''''''''''''''''''''''''''''
    BUILD NETWORK
    '''''''''''''''''''''''''''''''''''''''''''''
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print("Build Graph...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        with tf.device(my_device):
            SIZE_EMBED_DIM = 80
            SIZE_RNN_STATE = 100
            SIZE_BiRNN_STATE = SIZE_RNN_STATE * 2
            SIZE_ATTN_DIM = 80
            SIZE_DENSE_DIM = 120
            LEARNING_RATE = 0.0003

            with tf.name_scope("input_placeholders"):
                enc_input = tf.placeholder(tf.int32, shape=[None, None], name="enc_input")
                enc_seq_len = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len")
                dec_input = tf.placeholder(tf.int32, shape=[None, None], name="dec_input")
                dec_seq_len = tf.placeholder(tf.int32, shape=[None, ], name="dec_seq_len")
                dec_pad_len = tf.placeholder(tf.int32, shape=[None, ], name="dec_pad_len")
                targets = tf.placeholder(tf.int32, shape=[None, None], name="targets")
                batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
                keep_prob = tf.placeholder(tf.float32, name="keep_prob")

                global_step = tf.Variable(0, name="global_step", trainable=False)

            with tf.name_scope("word_embedding"):
                embeddings_eng = tf.get_variable("embeddings_eng", [voc_size_eng, SIZE_EMBED_DIM])
                embed_enc = tf.nn.embedding_lookup(embeddings_eng, enc_input, name="embed_enc")
                embeddings_kor = tf.get_variable("embeddings_kor", [voc_size_kor, SIZE_EMBED_DIM])
                embed_dec = tf.nn.embedding_lookup(embeddings_kor, dec_input, name="embed_dec")

            with tf.variable_scope("encoder_layer"):
                output_enc, state_enc = bi_rnn(GRUCell(SIZE_RNN_STATE), GRUCell(SIZE_RNN_STATE),
                                               inputs=embed_enc, sequence_length=enc_seq_len, dtype=tf.float32)

                state_enc_last = tf.concat([state_enc[0], state_enc[1]], axis=1)  # [batch, state*2]

                output_enc = tf.concat(output_enc, axis=2)  # [batch, max_eng, state*2]
                output_enc = tf.nn.dropout(output_enc, keep_prob=keep_prob, name="output_enc")
                assert output_enc.get_shape()[2] == SIZE_BiRNN_STATE
                assert state_enc_last.get_shape()[1] == SIZE_BiRNN_STATE

            with tf.variable_scope("decoder_layer") as scope:
                # initial state matrix
                W_s = tf.get_variable("W_s", [SIZE_BiRNN_STATE, SIZE_RNN_STATE])

                # attention matrix
                W_a1 = tf.get_variable("W_a1", [SIZE_BiRNN_STATE, SIZE_RNN_STATE])
                W_a2 = tf.get_variable("W_a2", [SIZE_RNN_STATE, 1])
                b_a1 = tf.get_variable("b_a1", [SIZE_RNN_STATE])
                b_a2 = tf.get_variable("b_a2", [1])
                V_a = tf.get_variable("V_a", [SIZE_RNN_STATE, 1])
                W_a = tf.get_variable("W_a", [SIZE_RNN_STATE, SIZE_RNN_STATE])
                U_a = tf.get_variable("U_a", [SIZE_BiRNN_STATE, SIZE_RNN_STATE])
                b_a = tf.get_variable("b_a", [SIZE_RNN_STATE])

                # reset gate matrix
                C_r = tf.get_variable("C_r", [SIZE_BiRNN_STATE, SIZE_RNN_STATE])
                U_r = tf.get_variable("U_r", [SIZE_RNN_STATE, SIZE_RNN_STATE])
                W_r = tf.get_variable("W_r", [SIZE_EMBED_DIM, SIZE_RNN_STATE])
                b_r = tf.get_variable("b_r", [SIZE_RNN_STATE])

                # update gate matrix
                C_z = tf.get_variable("C_z", [SIZE_BiRNN_STATE, SIZE_RNN_STATE])
                U_z = tf.get_variable("U_z", [SIZE_RNN_STATE, SIZE_RNN_STATE])
                W_z = tf.get_variable("W_z", [SIZE_EMBED_DIM, SIZE_RNN_STATE])
                b_z = tf.get_variable("b_z", [SIZE_RNN_STATE])

                # new state matrix
                C_p = tf.get_variable("C_p", [SIZE_BiRNN_STATE, SIZE_RNN_STATE])
                U_p = tf.get_variable("U_p", [SIZE_RNN_STATE, SIZE_RNN_STATE])
                W_p = tf.get_variable("W_p", [SIZE_EMBED_DIM, SIZE_RNN_STATE])
                b_p = tf.get_variable("b_p", [SIZE_RNN_STATE])

                # output matrix
                C_o = tf.get_variable("C_o", [SIZE_BiRNN_STATE, voc_size_kor])
                U_o = tf.get_variable("U_o", [SIZE_RNN_STATE, voc_size_kor])
                W_o = tf.get_variable("W_o", [SIZE_EMBED_DIM, voc_size_kor])
                b_o = tf.get_variable("b_o", [voc_size_kor])

                attention_list = []
                y_list = []
                y = embed_dec[:, 0]  #  train mode, [batch, e_dim]
                y_next = embed_dec[:, 0]  # test mode, [batch, e_dim]
                s = tf.tanh(tf.matmul(state_enc_last, W_s))
                output_enc_time_major = tf.transpose(output_enc, [1, 0, 2])     # [pad_eng, batch, state*2]

                for t in range(padded_kor_len):
                    y_prev = y_next
                    s_prev = s

                    def func_get_e(h):
                        e = tf.matmul(s_prev, W_a) + tf.matmul(h, U_a) + b_a    # [batch, state]
                        e = tf.matmul(tf.tanh(e), V_a, name="e")                # [batch, 1]
                        return e

                    e_tot = tf.map_fn(lambda h: func_get_e(h), output_enc_time_major)   # [pad_eng, batch, 1]
                    e_tot = tf.transpose(e_tot, [1, 0, 2])                              # [batch, pad_eng, 1]

                    # align
                    exp = tf.exp(tf.reshape(e_tot, [-1, padded_eng_len]))               # [batch, pad_eng]
                    a = exp / tf.reshape(tf.reduce_sum(exp, 1), [-1, 1])
                    a = tf.reshape(a, [-1, padded_eng_len, 1], name="a")                # [batch, pad_eng, 1]
                    attention_list.append(a)

                    # context
                    c = tf.reduce_sum((output_enc * a), axis=1, name="c")  # [batch, state*2]

                    # reset gate
                    r = tf.matmul(y_prev, W_r) + tf.matmul(s_prev, U_r) + tf.matmul(c, C_r)
                    r = tf.sigmoid(r, name="r")  # [batch, state]

                    # update gate
                    z = tf.matmul(y_prev, W_z) + tf.matmul(s_prev, U_z) + tf.matmul(c, C_z)
                    z = tf.sigmoid(z, name="z")  # [batch, state]

                    # proposal state
                    p = tf.matmul(y_prev, W_p) + tf.matmul((s_prev * r), U_p) + tf.matmul(c, C_p)
                    p = tf.tanh(p, name="p")  # [batch, state]

                    # new state
                    s = (1 - z) * s_prev + z * p  # [batch, state]

                    # predict next y_t
                    y = tf.matmul(y_prev, W_o) + tf.matmul(s, U_o) + tf.matmul(c, C_o)

                    y_hat = tf.cast(tf.argmax(tf.nn.softmax(y), axis=1), dtype=tf.int32, name="y_hat")
                    y_next = tf.nn.embedding_lookup(embeddings_kor, y_hat, name="y_next")
                    y_list.append(y)

                    scope.reuse_variables()

                attention = tf.convert_to_tensor(attention_list)  # [pad_kor, batch, pad_eng, 1]
                attention = tf.squeeze(attention)  # [pad_kor, batch, pad_eng]
                attention = tf.transpose(attention, [1, 0, 2], name="attention")  # [batch, pad_eng, pad_kor]

                hypothesis = tf.convert_to_tensor(y_list)  # [pad_kor, batch, voc_kor]
                hypothesis = tf.transpose(hypothesis, [1, 0, 2], name="hypothesis")  # [batch, pad_kor, voc_kor]

            with tf.name_scope("predict_and_optimizer"):
                masks = tf.sequence_mask(dec_seq_len+1, padded_kor_len, dtype=tf.float32, name="masks")
                predict = tf.cast(tf.argmax(tf.nn.softmax(hypothesis), 2), dtype=tf.int32, name="predict")     # [batch, pad_kor]

                #loss = tf.contrib.seq2seq.sequence_loss(logits=hypothesis, targets=targets, weights=masks, name="loss")
                #accuracy = tf.contrib.metrics.accuracy(labels=targets, predictions=predict, weights=masks, name="accuracy")

                one_hot_targets = tf.one_hot(targets, voc_size_kor, dtype=tf.float32, name="one_hot_targets")
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_targets, logits=hypothesis)
                loss = tf.reduce_mean(loss, name="loss")

                correct = tf.equal(targets, predict)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

                train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        out_dir = os.path.abspath(os.path.join("./model", timestamp))
        print("LOGDIR = %s" % out_dir)
        print()

        # Summaries
        loss_summary = tf.summary.scalar("loss", loss)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_dir = os.path.join(out_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        test_summary_dir = os.path.join(out_dir, "summary", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model-step")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        attention_img_dir = os.path.join(out_dir, "attention_img")
        if not os.path.exists(attention_img_dir):
            os.makedirs(attention_img_dir)

        '''''''''''''''''''''''''''''''''''''''''''''
        TRAIN PHASE
        '''''''''''''''''''''''''''''''''''''''''''''
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if pre_trained == 2:
            print("Restore model file...")
            file_model = "./model/2018-02-02 01:29/checkpoints/"
            saver.restore(sess, tf.train.latest_checkpoint(file_model))

        BATCHS = 150
        BATCHS_TEST = 50
        EPOCHS = 250
        STEPS = int(len(data_idx_list) / BATCHS)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Train start!!")

        loop_step = 0
        for epoch in range(EPOCHS):
            for step in range(STEPS):
                data_x, data_y, data_t, len_x, len_y = generate_batch(BATCHS)

                feed_dict = {
                    enc_input: data_x,
                    enc_seq_len: len_x,
                    dec_input: data_y,
                    dec_seq_len: len_y,
                    targets: data_t,
                    batch_size: BATCHS,
                    keep_prob: 0.75
                }

                _, batch_loss, batch_acc, g_step, train_sum = \
                    sess.run([train_op, loss, accuracy, global_step, train_summary_op], feed_dict=feed_dict)

                if loop_step % 20 == 0:
                    train_summary_writer.add_summary(train_sum, g_step)

                if loop_step % 50 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("epoch[%03d] glob_step[%05d] - batch_loss:%.4f, batch_acc:%.4f (%s) " %
                          (epoch, g_step, batch_loss, batch_acc, now))

                    # test
                    data_x, data_y, data_t, len_x, len_y = generate_test_batch(BATCHS_TEST)

                    feed_dict = {
                        enc_input: data_x,
                        enc_seq_len: len_x,
                        dec_input: data_y,
                        dec_seq_len: len_y,
                        targets: data_t,
                        batch_size: BATCHS_TEST,
                        keep_prob: 1.0
                    }

                    pred, test_loss, test_acc, attend, g_step, test_sum = \
                        sess.run([predict, loss, accuracy, attention, global_step, test_summary_op], feed_dict=feed_dict)
                    test_summary_writer.add_summary(test_sum, g_step)

                    if loop_step % 100 == 0:
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print("epoch[%03d] glob_step[%05d] - test_loss:%.4f, test_acc:%.4f (%s)" %
                              (epoch, g_step, test_loss, test_acc, now))

                    if loop_step % 500 == 0:
                        for a in range(min(10, BATCHS_TEST)):
                            print("TEST[%03d]" % a)
                            encode = [rdic_eng[e] for e in data_x[a] if e != 0]
                            encode = "".join(encode)
                            decode = [rdic_kor[d] for d in data_y[a] if d != 0]
                            decode = "".join(decode)
                            predic = [rdic_kor[p] for p in pred[a] if p != 0]
                            predic = "".join(predic)
                            print("ENGLISH : %s " % encode)
                            print("KOREAN  : %s " % decode)
                            print("PREDICT : %s " % predic)

                            # visualize..
                            if loop_step % 500 == 0 and a == min(10, BATCHS_TEST) - 1:
                                att_map = attend[a]

                                from mpl_toolkits.axes_grid1 import make_axes_locatable

                                plt.clf()

                                fig = plt.figure(figsize=(8, 6))
                                ax = fig.add_subplot(111)
                                im = ax.imshow(att_map[:len(predic), :len(encode)], cmap="YlGnBu")

                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="5%", pad=0.1)
                                cbar = fig.colorbar(im, cax=cax)

                                ax.set_xticks(range(len(encode)))
                                ax.set_xticklabels(encode)

                                ax.set_yticks(range(len(predic)))
                                ax.set_yticklabels([p + " " for p in predic])

                                ax.grid()
                                # plt.show()
                                plt.savefig(os.path.join(attention_img_dir, "step%d.png"%g_step), bbox_inches="tight")

                loop_step += 1
            saver.save(sess, checkpoint_prefix, global_step=g_step)

print("+" * 70)
print("Custom Sequence to Sequence End !!!")
print("+" * 70)