import argparse
import os
import tensorflow.compat.v1 as tf
import warnings
import tensorflow.python.ops.nn_impl as mm
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=1978)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn", lineno=814)
tf.disable_v2_behavior()
import sys
sys.path.append('/Users/huwentao/Desktop/LaTeX/experiment/')
import time
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score,roc_curve
from data_generator import DataGenerator
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

class Model:
    def __init__(self, args):
        self.args = args

        input = tf.placeholder(shape=(args.batch_size, None), dtype=tf.float32, name='input')
        inputs = tf.placeholder(shape=(args.batch_size, None), dtype=tf.int32, name='inputs')
        mask = tf.placeholder(shape=(args.batch_size, None), dtype=tf.float32, name='inputs_mask')
        seq_length = tf.placeholder(shape=args.batch_size, dtype=tf.float32, name='seq_length')
        self.input_form = [inputs, mask, seq_length, input]

        encoder_inputs = inputs
        decoder_input = tf.concat([input, tf.zeros(shape=(args.batch_size, 1), dtype=tf.float32)], axis=1)
        decoder_inputs = tf.concat([tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32), inputs], axis=1)
        decoder_targets = tf.concat([inputs, tf.zeros(shape=(args.batch_size, 1), dtype=tf.int32)], axis=1)
        decoder_mask = tf.concat([mask, tf.zeros(shape=(args.batch_size, 1), dtype=tf.float32)], axis=1)
        self.decoder_input = decoder_input
        x_size = out_size = args.map_size[0] * args.map_size[1]
        embeddings = tf.Variable(tf.random_uniform([x_size, args.x_latent_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

        with tf.variable_scope("encoder"):
            # encoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size)
            # _, encoder_final_state = tf.nn.dynamic_rnn(
            #     encoder_cell, encoder_inputs_embedded,
            #     sequence_length=seq_length,
            #     dtype=tf.float32,
            # )
            mu_e_w = tf.get_variable("mu_e_w", [args.rnn_size, 1], tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.02))
            encoder_inputs_embedded = tf.reshape(encoder_inputs_embedded, [args.batch_size, args.rnn_size, args.rnn_size])
            encoder_final_state = tf.matmul(encoder_inputs_embedded, mu_e_w)
            encoder_final_state = tf.reshape(encoder_final_state, [args.batch_size, args.rnn_size])
            # encoder_final_state = tf.get_variable("encoder_final_state", [args.batch_size, args.rnn_size], tf.float32,
            #                          initializer=tf.random_normal_initializer(stddev=0.02))
            # mu_e_w2 = tf.get_variable("mu_e_w2", [args.rnn_size+1, args.rnn_size], tf.float32,
            #                          initializer=tf.random_normal_initializer(stddev=0.02))
            # mu_e_b2 = tf.get_variable("mu_e_b2", [args.rnn_size], tf.float32,
            #                          initializer=tf.constant_initializer(0.0))
            # encoder_final_state = tf.matmul(encoder_final_state, mu_e_w2) + mu_e_b2
            # encoder_final_state = tf.reshape(encoder_final_state, [args.batch_size, args.rnn_size])



        with tf.variable_scope("clusters"):

            mu_c = tf.get_variable("mu_c", [args.mem_num, args.rnn_size],
                                   initializer=tf.random_uniform_initializer(0.0, 1.0))
            log_sigma_sq_c = tf.get_variable("sigma_sq_c", [args.mem_num, args.rnn_size],
                                             initializer=tf.constant_initializer(0.0), trainable=False)
            log_pi_prior = tf.get_variable("log_pi_prior", args.mem_num,
                                           initializer=tf.constant_initializer(0.0), trainable=False)
            pi_prior = tf.nn.softmax(log_pi_prior)

            init_mu_c = tf.placeholder(shape=(args.mem_num, args.rnn_size), dtype=tf.float32, name='init_mu_c')
            init_sigma_c = tf.placeholder(shape=(args.mem_num, args.rnn_size), dtype=tf.float32, name='init_sigma_c')
            init_pi = tf.placeholder(shape=args.mem_num, dtype=tf.float32, name='init_pi')
            self.cluster_init = [init_mu_c, init_sigma_c, init_pi]
            self.init_mu_c_op = tf.assign(mu_c, init_mu_c)
            self.init_sigma_c_op = tf.assign(log_sigma_sq_c, init_sigma_c)
            self.init_pi_op = tf.assign(log_pi_prior, init_pi)

            self.mu_c = mu_c
            self.sigma_c = log_sigma_sq_c
            self.pi = pi_prior

            stack_mu_c = tf.stack([mu_c] * args.batch_size, axis=0)
            stack_log_sigma_sq_c = tf.stack([log_sigma_sq_c] * args.batch_size, axis=0)

        with tf.variable_scope("latent"):
            with tf.variable_scope("mu_z"):
                mu_z_w = tf.get_variable("mu_z_w", [args.rnn_size, args.rnn_size], tf.float32,
                                         initializer=tf.random_normal_initializer(stddev=0.02))
                mu_z_b = tf.get_variable("mu_z_b", [args.rnn_size], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
                mu_z = tf.matmul(encoder_final_state, mu_z_w) + mu_z_b
            with tf.variable_scope("sigma_z"):
                sigma_z_w = tf.get_variable("sigma_z_w", [args.rnn_size, args.rnn_size], tf.float32,
                                            initializer=tf.random_normal_initializer(stddev=0.02))
                sigma_z_b = tf.get_variable("sigma_z_b", [args.rnn_size], tf.float32,
                                            initializer=tf.constant_initializer(0.0))
                log_sigma_sq_z = tf.matmul(encoder_final_state, sigma_z_w) + sigma_z_b

            eps_z = tf.random_normal(shape=tf.shape(log_sigma_sq_z), mean=0, stddev=1, dtype=tf.float32)
            z = mu_z + tf.sqrt(tf.exp(log_sigma_sq_z)) * eps_z

            stack_mu_z = tf.stack([mu_z] * args.mem_num, axis=1)
            stack_log_sigma_sq_z = tf.stack([log_sigma_sq_z] * args.mem_num, axis=1)
            stack_z = tf.stack([z] * args.mem_num, axis=1)

            self.batch_post_embedded = z

        with tf.variable_scope("attention"):
            att_logits = - tf.reduce_sum(tf.square(stack_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c), axis=-1)
            att = tf.nn.softmax(att_logits) + 1e-10
            self.batch_att = att

        def generation(h):
            with tf.variable_scope("generation", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("decoder"):
                    decoder_init_state = h
                    # decoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size)
                    # decoder_outputs, _ = tf.nn.dynamic_rnn(
                    #     decoder_cell, decoder_inputs_embedded,
                    #     initial_state=decoder_init_state,
                    #     sequence_length=seq_length,
                    #     dtype=tf.float32,
                    # )
                    mu_d_w = tf.get_variable("mu_d_w", [args.rnn_size, 1 + args.rnn_size], tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.02))
                    decoder_init_state = tf.matmul(decoder_init_state, mu_d_w)
                    decoder_init_state = tf.reshape(decoder_init_state, [args.batch_size, 1 + args.rnn_size, 1])
                    mu_d_w2 = tf.get_variable("mu_d_w2", [1, args.rnn_size], tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.02))
                    decoder_outputs = tf.matmul(decoder_init_state, mu_d_w2)
                    decoder_outputs = tf.reshape(decoder_outputs, [args.batch_size, 1 + args.rnn_size, args.rnn_size])
                with tf.variable_scope("outputs"):
                    out_w = tf.get_variable("out_w", [out_size, args.rnn_size], tf.float32,
                                            tf.random_normal_initializer(stddev=0.02))
                    out_b = tf.get_variable("out_b", [out_size], tf.float32,
                                            initializer=tf.constant_initializer(0.0))

                    batch_rec_loss = tf.reduce_mean(
                        decoder_mask * tf.reshape(
                            tf.nn.sampled_softmax_loss(
                                weights=out_w,
                                biases=out_b,
                                labels=tf.reshape(decoder_input, [-1, 1]),
                                inputs=tf.reshape(decoder_outputs, [-1, args.rnn_size]),
                                num_sampled=args.neg_size,
                                num_classes=out_size
                            ), [args.batch_size, -1]
                        ), axis=-1
                    )


                    logs = tf.matmul(tf.reshape(decoder_outputs,[-1, args.rnn_size]), tf.transpose(out_w))
                    logs1 = tf.nn.bias_add(logs, out_b)
                    logs1 = tf.argmax(tf.nn.softmax(logs1), axis=1)
                    logs1 = tf.reshape(logs1, [20, -1])
                    # logs2 = abs(decoder_input - tf.cast(logs1, dtype=tf.float32))
                    # logs1 = tf.cast(logs2/decoder_input, dtype=tf.float32)
                    # logs1 = tf.cast(logs1 * decoder_input,dtype=tf.float32)
                    # logs1 = logs1[:args.batch_size, :args.x_latent_size]
                    # logs1 = decoder_input
                    # labels_one_hot = tf.one_hot(tf.reshape(decoder_inputs, [-1, 1]), out_size)
                    # loss = tf.nn.softmax_cross_entropy_with_logits(
                    #                             labels=labels_one_hot,
                    #                                 logits=logs1)
                    # logis,_= mm._compute_sampled_logits(
                    #             weights=out_w,
                    #             biases=out_b,
                    #             labels=tf.reshape(decoder_input, [-1, 1]),
                    #             inputs=tf.reshape(decoder_outputs, [-1, args.rnn_size]),
                    #             num_sampled=args.neg_size,
                    #             num_classes=out_size
                    #         )
                    # logs = tf.nn.softmax(logs)
                    # y_pred = tf.reshape(tf.argmax(logs, 1), [args.batch_size,-1])

                    target_out_w = tf.nn.embedding_lookup(out_w, decoder_targets)
                    target_out_b = tf.nn.embedding_lookup(out_b, decoder_targets)

                    # outputs = tf.get_variable("outputs", [args.batch_size, args.x_latent_size+1, args.rnn_size],
                    #                                initializer=tf.constant_initializer(0.0), trainable=False)
                    # self.outputs = decoder_outputs

                    w = tf.get_variable("w", [args.rnn_size+1, args.rnn_size], tf.float32,
                                            tf.random_normal_initializer(stddev=1))
                    b = tf.get_variable("b", [args.rnn_size+1], tf.float32,
                                            initializer=tf.constant_initializer(1.0))
                    # out_w1 = tf.stack([out_w1] * args.batch_size)
                    # out_b1 = tf.stack([out_b1]* args.batch_size)


                    logs = decoder_mask * (
                            tf.reduce_sum(decoder_outputs * w, -1) + b
                        )
                    # logs = logs[:args.batch_size, :args.x_latent_size]
                    # pre_loss = tf.get_variable("pre_loss", [1], tf.float32,
                    #                         initializer=tf.constant_initializer(0.0))
                    # y_pred = logs * out_w1 + out_b1
                    # y_pred = h
                    pre_loss = tf.reduce_mean(
                        tf.losses.absolute_difference(decoder_input,
                                                      logs)
                    )

                    logs2 = decoder_mask * tf.log_sigmoid(tf.reduce_sum(decoder_outputs * target_out_w, -1) + target_out_b)
                    # logs1 = logs1[:args.batch_size, :args.x_latent_size,:args.x_latent_size]


                    batch_likelihood = tf.reduce_mean(
                        decoder_mask * tf.log_sigmoid(
                            tf.reduce_sum(decoder_outputs * target_out_w, -1) + target_out_b
                        ), axis=-1, name="batch_likelihood")

                    batch_latent_loss = 0.5 * tf.reduce_sum(
                        att * tf.reduce_mean(stack_log_sigma_sq_c
                                             + tf.exp(stack_log_sigma_sq_z) / tf.exp(stack_log_sigma_sq_c)
                                             + tf.square(stack_mu_z - stack_mu_c) / tf.exp(stack_log_sigma_sq_c),
                                             axis=-1),
                        axis=-1) - 0.5 * tf.reduce_mean(1 + log_sigma_sq_z, axis=-1)
                    batch_cate_loss = tf.reduce_mean(tf.reduce_mean(att, axis=0) * tf.log(tf.reduce_mean(att, axis=0)))
                return batch_rec_loss, batch_latent_loss, batch_cate_loss, batch_likelihood, pre_loss, logs1, logs2
        if args.eval:
            results = tf.map_fn(fn=generation, elems=tf.stack([mu_c] * args.batch_size, axis=1),
                                dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.float32),
                                parallel_iterations=args.mem_num)
            self.batch_likelihood = tf.reduce_max(results[3], axis=0)
            self.outputs = results[5]
            self.probablity = results[6]


        else:
            results = generation(z)
            self.batch_likelihood = results[3]
            self.rec_loss = rec_loss = tf.reduce_mean(results[0])
            self.latent_loss = latent_loss = tf.reduce_mean(results[1])
            self.cate_loss = cate_loss = results[2]
            self.pre_loss = pre_loss = results[4]
            self.loss = loss =   latent_loss + 0.1 * cate_loss + rec_loss
            # self.loss = loss = 0.01 * pre_loss + latent_loss + 0.1 * cate_loss
            self.pretrain_loss = pretrain_loss = rec_loss
            self.pretrain_op = tf.train.AdamOptimizer(args.learning_rate).minimize(pretrain_loss)
            # self.generate_loss = tf.train.AdamOptimizer(args.learning_rate).minimize(pre_loss)
            self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

        # saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        saver = tf.train.Saver(max_to_keep=20)
        self.save, self.restore = saver.save, saver.restore

def filling_batch(batch_data):
    new_batch_data = []
    last_batch_size = len(batch_data[0])
    for b in batch_data:
        new_batch_data.append(
            np.concatenate([b, [np.zeros_like(b[0]).tolist()
                                for _ in range(args.batch_size - last_batch_size)]], axis=0))
    return new_batch_data

def compute_likelihood(sess, model, sampler, purpose):
    input_data = []
    all_likelihood = []
    rules_ = []
    all_likelihood1 = []
    for batch_data in sampler.iterate_all_data(args.batch_size,
                                               partial_ratio=args.partial_ratio,
                                               purpose=purpose):
        if len(batch_data[0]) < args.batch_size:
            last_batch_size = len(batch_data[0])
            batch_data = filling_batch(batch_data)
            feed = dict(zip(model.input_form, batch_data))
            batch_likelihood = sess.run(model.batch_likelihood, feed)[:last_batch_size]
            rules = sess.run(model.outputs, feed)
            probability = sess.run(model.probablity, feed)
        else:
            feed = dict(zip(model.input_form, batch_data))
            batch_likelihood = sess.run(model.batch_likelihood, feed)
            # rules = sess.run(model.rules, feed)
            rules = sess.run(model.outputs, feed)
            probability = sess.run(model.probablity, feed)
        rules_.append(rules)
        all_likelihood1.append(probability)
        all_likelihood.append(batch_likelihood)
        input_data.append(batch_data[0])
    return np.concatenate(all_likelihood), np.concatenate(input_data), all_likelihood1,rules_

def auc_score(y_true, y_score):
    precision, recall, t = precision_recall_curve(1-y_true, 1-y_score)
    f1 = 2 * (precision * recall) / (precision + recall)
    where_are_NaN = np.isnan(f1)
    f1[where_are_NaN] = 0

    f1_index = np.where(f1 == np.max(f1))[0][0]
    threshold = t[f1_index]


    a = 1 - y_true
    a = [i for i in a]
    tp = 1-y_score
    tp_label = []
    for i in tp:
        if i < threshold:
            tp_label.append(0)
        else:
            tp_label.append(1)

    import pandas as pd
    df = pd.DataFrame(a, tp_label)
    df.to_csv('./tp1.csv')

    return f1.max(), threshold

def generate_normal_range(normal_range):
    a = plt.hist(normal_range)
    mean = a[1][np.argmax(a[0])]
    dev = a[1][1] - a[1][0]
    n_r = [int(mean-dev), int(mean+dev)]
    n_r = np.array(n_r)
    return n_r



def computer_normal_range(sess, model, sampler,error_data,idx,min,max,tp_t):
    all_likelihood = []
    input_data = []
    for batch_data in sampler.iterate_error_data(args.batch_size,error_data,idx,min,max,
                                               partial_ratio=args.partial_ratio,
                                               ):
        if len(batch_data[0]) < args.batch_size:
            last_batch_size = len(batch_data[0])
            batch_data = filling_batch(batch_data)
            feed = dict(zip(model.input_form, batch_data))
            probability = sess.run(model.probablity, feed)[:last_batch_size]
        else:
            feed = dict(zip(model.input_form, batch_data))
            probability = sess.run(model.probablity, feed)
        all_likelihood.append(probability)
        input_data.append(batch_data[0])
    all_likelihood = np.concatenate(all_likelihood)
    input_data = np.concatenate(input_data)
    all_likelihood = np.reshape(all_likelihood, [len(input_data), -1]) * -1
    a_idx = all_likelihood[:,idx] < tp_t
    nomal_range = input_data[:,idx][a_idx]
    n_r = generate_normal_range(nomal_range)
    return n_r



def rules_generative(sess, model, sampler, error_data,error_pro,error_rules,threshold):
    error_data = np.array(error_data).astype(int)
    error_pro = np.array(error_pro)[:, :len(error_data[0])] * -1
    error_rules = np.array(error_rules[:, :len(error_data[0])])
    seq = len(error_data[0]) + 1
    threshold = seq * np.log(1 - threshold) * -1
    for i in range(len(error_data)):
        LS = [-1 for i in range(len(error_data[0]))]
        RS = [-1 for i in range(len(error_data[0]))]
        a_s = []
        for q in range(1, len(error_data[i])+1):
            idx = np.argpartition(-error_pro[i].ravel(), q)[:q]
            if sum(error_pro[i]) - sum(error_pro[i][idx]) < threshold:
                if len(idx) == 1:
                    tp_t = threshold - (sum(error_pro[i]) - sum(error_pro[i][idx]))
                    df = pd.read_csv('./datasets/wdbc_test.csv')
                    idx = idx[0]
                    corr = df.corr()
                    target_idx = np.argpartition(-corr.iloc[:, idx].ravel(), 1)[1]
                    min = int(df.min()[idx])
                    max = int(df.max()[idx])
                    n_r = computer_normal_range(sess, model, sampler, error_data[i], idx, min, max, tp_t)
                    n_r = n_r / error_data[i][target_idx]
                    n_r = n_r.round(2)
                    RS[idx] = list(n_r)
                    a_s = [idx, target_idx]
                else:
                    for k in range(len(idx) - 1):
                        m = idx[k]
                        RS[m] = error_rules[i][m]
                    tp_t = threshold - (sum(error_pro[i]) - sum(error_pro[i][idx]))
                    df = pd.read_csv('./datasets/wdbc_test.csv')
                    idx = idx[-1]
                    corr = df.corr()
                    target_idx = np.argpartition(-corr.iloc[:, idx].ravel(), 1)[1]
                    min = int(df.min()[idx])
                    max = int(df.max()[idx])
                    n_r = computer_normal_range(sess, model, sampler, error_data[i], idx, min, max, tp_t)
                    if corr.iloc[target_idx,idx] < 0.9:
                        RS[idx] = list(n_r)
                        a_s = []
                    else:
                        n_r = n_r / error_data[i][target_idx]
                        n_r = n_r.round(2)
                        RS[idx] = list(n_r)
                        a_s = [idx, target_idx]
                break
            else:
                continue
        for j in range(len(RS)):
            if RS[j] == -1:
                LS[j] = error_data[i][j]
        print(LS, RS, a_s)
        print(i + 1)


def evaluate():
    model = Model(args)
    sampler = DataGenerator(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, args.model_id)
        model.restore(sess, model_name)

        st = time.time()
        all_likelihood, input_data, all_likelihood1, rules_ = compute_likelihood(sess, model, sampler, "train")
        elapsed = time.time() - st

        all_prob = np.exp(all_likelihood)

        y_true = np.ones_like(all_prob)
        for idx in sampler.outliers:
            if idx < y_true.shape[0]:
                y_true[idx] = 0
        sd_auc = {}
        sd_index = sampler.sd_index
        for sd, tids in sd_index.items():
            sd_y_true = y_true[tids]
            sd_prob = all_prob[tids]
            if sd_y_true.sum() < len(sd_y_true):
                sd_auc[sd], threshold = auc_score(y_true=sd_y_true, y_score=sd_prob)

            # outliers = input_data[1-all_likelihood >=threshold]

        tp_a = np.reshape(all_likelihood1, [len(input_data), -1])
        # plt.scatter(tp_a,input_data[:,3])
        # plt.show()

        try:
            error_index = all_prob < (1 - threshold)
            input_data = input_data[:len(all_prob)]
            error_data = input_data[error_index]
            error_pro = np.reshape(all_likelihood1,[len(input_data),-1])[error_index]
            error_rules = np.reshape(rules_,[len(input_data), -1])[error_index]
            rules_generative(sess, model, sampler,error_data, error_pro, error_rules, threshold)
        except:
            print('No error data!')


        print("Average F1:", np.mean(list(sd_auc.values())), "Elapsed time:", elapsed)




def compute_loss(sess, model, sampler, purpose):
    all_loss = []
    if args.pt:
        loss_op = model.pretrain_loss
    else:
        loss_op = model.loss
    for batch_data in sampler.iterate_all_data(args.batch_size,
                                               partial_ratio=args.partial_ratio,
                                               purpose=purpose):
        if len(batch_data[0]) < args.batch_size:
            batch_data = filling_batch(batch_data)
            feed = dict(zip(model.input_form, batch_data))
            loss = sess.run(loss_op, feed)
        else:
            feed = dict(zip(model.input_form, batch_data))
            loss = sess.run(loss_op, feed)
        all_loss.append(loss)
    return np.mean(all_loss)

def find_mu(data1,data2):
    map_mu = {}
    for i in range(1, len(data2)-1):
        value1 = data2[i]
        value2 = data2[i+1]
        value3 = data2[i-1]
        if (value1 > value3 and value1 > value2) or value1 == value3 or value1 == value2:
            map_mu[data1[i][0]] = value1
    tp = sorted(map_mu.items(), key= lambda item:item[1], reverse=True)
    return tp


def kde_train(data): #data(1079,30)
    new_data = np.array(data)
    new_data_T = new_data.T
    all_mu = []
    for i in range(len(new_data_T)):
        tp = new_data_T[i].reshape([-1, 1])
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(tp)
        kde = grid.best_estimator_
        tp = kde.sample(10000)
        tp = np.unique(tp).reshape([-1,1])
        p_data = kde.score_samples(tp)
        tp = np.round(tp,5)
        p_data = np.exp(p_data)
        # plt.scatter(tp, p_data)
        # plt.show()
        map_mu = find_mu(tp, p_data)
        all_mu.append(map_mu)

    min_mu = 10000
    for j in range(len(all_mu)):
        tp_ = all_mu[j]
        if len(tp_) < min_mu:
            min_mu = len(tp_)
    mu = np.random.random([len(all_mu),min_mu])
    for k in range(len(all_mu)):
        temp_mu = []
        for m in range(min_mu):
            t_mu = round(all_mu[k][m][0],2)
            temp_mu.append(t_mu)
        mu[k] = temp_mu
    mu = mu.T
    sigma = np.zeros_like(mu)
    pi = np.zeros([min_mu])
    return mu, sigma, pi





def ppretrain():
    sampler = DataGenerator(args)
    all_val_loss = []
    model = Model(args)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        start = time.time()

        for epoch in range(args.num_epochs):
            for batch_idx in range(int(sampler.total_tup_num / args.batch_size)):
                batch_data = sampler.next_batch(args.batch_size)
                batch_data.append(batch_data[0])
                feed = dict(zip(model.input_form, batch_data))
                sess.run(model.pretrain_op, feed)

            val_loss = compute_loss(sess, model, sampler, "val")
            if len(all_val_loss) > 0 and val_loss >= all_val_loss[-1]:
                print("Early termination with val loss: {}:".format(val_loss))
                break
            all_val_loss.append(val_loss)

            end = time.time()
            print("pretrain epoch: {}\tval loss: {}\telapsed time: {}".format(
                epoch, val_loss, end - start))
            start = time.time()

            save_model_name = "./models/{}_{}_{}/{}_{}".format(
                args.model_type, args.x_latent_size, args.rnn_size, args.model_type, "pretrain")
            model.save(sess, save_model_name)



        # KDE-init
        # sample_num = 1000
        # x_embedded = []
        # for batch_idx in range(int(sample_num / args.batch_size)):
        #     batch_data = sampler.next_batch(args.batch_size)
        #     feed = dict(zip(model.input_form, batch_data))
        #     x_embedded.append(sess.run(model.batch_post_embedded, feed))
        # x_embedded = np.concatenate(x_embedded, axis=0)
        # mu, sigma, pi = kde_train(x_embedded)
        # print(len(sigma))
        #
        # feed_dict = dict(zip(model.cluster_init, [mu, sigma, pi]))
        # sess.run([model.init_mu_c_op, model.init_sigma_c_op, model.init_pi_op], feed_dict)
        #
        # save_model_name = "./models/{}_{}_{}/{}_{}".format(
        #     args.model_type, args.x_latent_size, args.rnn_size, args.model_type, "init")
        # model.save(sess, save_model_name)
        #
        # print("Init model saved.")


def train():
    model = Model(args)
    sampler = DataGenerator(args)
    all_val_loss = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        start = time.time()

        model_name = "./models/{}_{}_{}/{}_{}".format(
            args.model_type, args.x_latent_size, args.rnn_size, args.model_type, 'pretrain')
        model.restore(sess, model_name)




        for epoch in range(args.num_epochs):
            all_loss = []
            for batch_idx in range(int(sampler.total_tup_num / args.batch_size)):
                batch_data = sampler.next_batch(args.batch_size)
                batch_data.append(batch_data[0])
                feed = dict(zip(model.input_form, batch_data))
                rec_loss, cate_loss, latent_loss, _ = sess.run(
                    [model.rec_loss, model.cate_loss, model.latent_loss, model.train_op], feed)
                all_loss.append([rec_loss, cate_loss, latent_loss])

            val_loss = compute_loss(sess, model, sampler, "val")
            # if len(all_val_loss) > 0 and val_loss >= all_val_loss[-1]:
            #     print("Early termination with val loss: {}:".format(val_loss))
            #     break
            all_val_loss.append(val_loss)

            end = time.time()
            print("epoch: {}\tval loss: {}\telapsed time: {}".format(
                epoch, val_loss, end - start))
            print("loss: {}".format(np.mean(all_loss, axis=0)))
            print(sess.run(model.pi))
            start = time.time()
            save_model_name = "./models/{}_{}_{}/{}_{}".format(
                args.model_type, args.x_latent_size, args.rnn_size, args.model_type, epoch)
            model.save(sess, save_model_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_filename', type=str, default="./datasets/test.csv",
                        help='data file')
    parser.add_argument('--map_size', type=tuple, default=(8, 2),
                        help='size of map')
    # parser.add_argument('--map_size', type=tuple, default=(128, 64),
    #                     help='size of map')

    parser.add_argument('--model_type', type=str, default="Lym-RNN-FREE",
                        help='choose a model')

    parser.add_argument('--x_latent_size', type=int, default=18,
                        help='size of input embedding')
    parser.add_argument('--rnn_size', type=int, default=18,
                        help='size of RNN hidden state')
    parser.add_argument('--mem_num', type=int, default=1,
                        help='size of sd memory')

    parser.add_argument('--neg_size', type=int, default=16,
                        help='size of negative sampling')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')

    parser.add_argument('--model_id', type=str, default="18",
                        help='model id')
    parser.add_argument('--partial_ratio', type=float, default=1.0,
                        help='partial tuple evaluation')
    parser.add_argument('--eval', type=bool, default=True,
                        help='partial tuple evaluation')
    parser.add_argument('--pt', type=bool, default=False,
                        help='partial tuple evaluation')

    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.eval:
        evaluate()
    elif args.pt:
        ppretrain()
    else:
        train()