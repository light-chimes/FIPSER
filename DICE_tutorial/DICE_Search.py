import random
import shutil

import numpy as np
import csv
from itertools import product
import itertools

import tensorflow
import tensorflow.compat.v1 as tf

from fairness_improvement.data_process import read_data

tf.compat.v1.disable_eager_execution()
import sys, os

sys.path.append("../")
import copy
import pandas as pd
from tensorflow.python.platform import flags
from scipy.optimize import basinhopping
from scipy.stats import entropy
import time

from DICE_data.census import census_data
from DICE_data.credit import credit_data
from DICE_data.compas import compas_data
from DICE_data.default import default_data
from DICE_data.bank import bank_data
from DICE_data.heart import heart_data
from DICE_data.diabetes import diabetes_data
from DICE_data.students import students_data
from DICE_data.meps15 import meps15_data
from DICE_data.meps16 import meps16_data

from DICE_model.tutorial_models import dnn
from DICE_utils.utils_tf import model_prediction, model_argmax, layer_out
from DICE_utils.config import census, credit, bank, compas, default, heart, diabetes, students, meps15, meps16
from DICE_tutorial.utils import cluster, gradient_graph

FLAGS = flags.FLAGS

# step size of perturbation
perturbation_size = 1


def float_sec2hms(time_cost_sec):
    time_cost_sec = round(time_cost_sec)
    time_cost_min = time_cost_sec // 60
    time_cost_sec %= 60
    time_cost_hr = time_cost_min // 60
    time_cost_min %= 60
    return time_cost_hr, time_cost_min, time_cost_sec


def check_for_error_condition(conf, sess, x, preds, t, sens_params, input_shape, epsillon):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """

    t = [t.astype('int')]
    samples = m_instance(np.array(t), sens_params, conf)
    pred = pred_prob(sess, x, preds, samples, input_shape)
    partition = clustering(pred, samples, sens_params, epsillon)
    # entropy_min = np.log2(len(partition)-1)#sh_entropy(pred, ent_tresh)
    # entropy_sh = sh_entropy(pred, epsillon)

    return max(list(partition.keys())[1:]) - min(list(partition.keys())[1:]), \
           len(partition) - 1, conf  # (len(partition) -1),


def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)


def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input


class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, grad, x, n_values, sens_params, input_shape, conf):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_values = n_values
        self.input_shape = input_shape
        self.sens = sens_params
        self.conf = conf

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """

        # perturbation
        s = np.random.choice([1.0, -1.0]) * perturbation_size

        n_x = x.copy()
        for i in range(len(self.sens)):
            n_x[self.sens[i] - 1] = self.n_values[i]

        # compute the gradients of an individual discriminatory instance pairs
        ind_grad = self.sess.run(self.grad, feed_dict={self.x: np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x: np.array([n_x])})

        if np.zeros(self.input_shape).tolist() == ind_grad[0].tolist() and \
                np.zeros(self.input_shape).tolist() == n_ind_grad[0].tolist():

            probs = 1.0 / (self.input_shape) * np.ones(self.input_shape)

            for sens in self.sens:
                probs[sens - 1] = 0



        else:
            # nomalize the reciprocal of gradients (prefer the low impactful feature)
            grad_sum = 1.0 / (abs(ind_grad[0]) + abs(n_ind_grad[0]))

            for sens in self.sens:
                grad_sum[sens - 1] = 0

            probs = grad_sum / np.sum(grad_sum)
        probs = probs / probs.sum()
        if True in np.isnan(probs):
            probs = 1.0 / (self.input_shape) * np.ones(self.input_shape)

            for sens in self.sens:
                probs[sens - 1] = 0
            probs = probs / probs.sum()

        # randomly choose the feature for local perturbation
        index = np.random.choice(range(self.input_shape), p=probs)
        local_cal_grad = np.zeros(self.input_shape)
        local_cal_grad[index] = 1.0
        x = clip(x + s * local_cal_grad, self.conf).astype("int")
        return x


# --------------------------------------
def m_instance(sample, sens_params, conf):
    index = []
    m_sample = []
    for sens in sens_params:
        index.append([i for i in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens - 1][1] + 1)])

    for ind in list(product(*index)):
        temp = sample.copy()
        for i in range(len(sens_params)):
            temp[0][sens_params[i] - 1] = ind[i]
        m_sample.append(temp)
    return np.array(m_sample)


def global_sample_select(clus_dic, sens_params):
    leng = 0
    for key in clus_dic.keys():
        if key == 'Seed':
            continue
        if len(clus_dic[key]) > leng:
            leng = len(clus_dic[key])
            largest = key

    sample_ind = np.random.randint(len(clus_dic[largest]))
    n_sample_ind = np.random.randint(len(clus_dic[largest]))

    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i] - 1] = clus_dic[largest][sample_ind][i]
    # returns one sample of largest partition and its pair
    return np.array([sample]), clus_dic[largest][n_sample_ind]


def local_sample_select(clus_dic, sens_params):
    k_1 = min(list(clus_dic.keys())[1:])
    k_2 = max(list(clus_dic.keys())[1:])

    sample_ind = np.random.randint(len(clus_dic[k_1]))
    n_sample_ind = np.random.randint(len(clus_dic[k_2]))

    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i] - 1] = clus_dic[k_1][sample_ind][i]
    return np.array([sample]), clus_dic[k_2][n_sample_ind]


def clustering(probs, m_sample, sens_params, epsillon):
    cluster_dic = {}
    cluster_dic['Seed'] = m_sample[0][0]
    bins = np.arange(0, 1, epsillon)
    digitized = np.digitize(probs, bins) - 1
    for k in range(len(digitized)):

        if digitized[k] not in cluster_dic.keys():
            cluster_dic[digitized[k]] = [[m_sample[k][0][j - 1] for j in sens_params]]
        else:
            cluster_dic[digitized[k]].append([m_sample[k][0][j - 1] for j in sens_params])
    return cluster_dic


def pred_prob(sess, x, preds, m_sample, input_shape):
    probs = model_prediction(sess, x, preds, np.array(m_sample).reshape(len(m_sample),
                                                                        input_shape[1]))[:, 1:2].reshape(len(m_sample))
    return probs


def sh_entropy(probs, bin_thresh, base=2):
    bins = np.arange(0, 1, bin_thresh)
    digitized = np.digitize(probs, bins)
    value, counts = np.unique(digitized, return_counts=True)
    return entropy(counts, base=base)


def set_random_seed(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def dnn_fair_testing(dataset='census', sens_params=[9], model_path='../models/', cluster_num=4,
                     max_global=1000, max_local=1000, max_iter=10, epsillon=0.025, timeout=900, RQ=2,
                     preprocess_method='cluster', *args, **kwargs):
    """

    The implementation of ADF
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    args_dict = {'dataset': dataset, 'sensitive_param': sens_params, 'model_path': model_path,
                 'cluster_num': cluster_num,
                 'max_global': max_global, 'max_local': max_local, 'max_iter': max_iter, 'epsilon': epsillon,
                 'timeout': timeout,
                 'RQ': RQ, 'preprocess_method': preprocess_method}
    print('experiment args:', args_dict)
    data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data,
            "default": default_data, "heart": heart_data, "diabetes": diabetes_data,
            "students": students_data, "meps15": meps15_data, "meps16": meps16_data}
    data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas, "default": default,
                   "heart": heart, "diabetes": diabetes, "students": students, "meps15": meps15, "meps16": meps16}
    # prepare the testing data and model
    use_split_data = False
    X, Y, input_shape, nb_classes = data[dataset]()
    if use_split_data:
        train_x, train_y, test_x, test_y = read_data(dataset)
        X = train_x
        Y = train_y
        print('*' * 60, 'using split data', '*' * 60)
    else:
        print('*' * 60, 'using full data', '*' * 60)
    if 'random_seed' in kwargs:
        set_random_seed(kwargs['random_seed'])
    else:
        set_random_seed(1234)
    config = tf.ConfigProto(device_count={'GPU': 0})
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model, linear_weights = dnn(input_shape, nb_classes, get_weights=True)

    preds = model(x)
    saver = tf.train.Saver()
    model_path = model_path + dataset + "/test.model"
    saver.restore(sess, model_path)

    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)

    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]

    # store the result of fairness testing

    x_all_hidden, act_dict = model.get_all_activation(x, with_dict=True, only_hidden=False)
    linear_dict = {}
    for i in act_dict:
        if 'Linear' in i:
            linear_dict[i] = act_dict[i]

    global max_k
    global start_time
    global max_k_time

    print(dataset, sens_params)
    RQ2_table = []
    RQ1_table = []
    for trial in range(1):
        print('Trial', trial)
        if sess._closed:
            sess = tf.Session(config=config)
            sess = tf.Session(config=config)
            x = tf.placeholder(tf.float32, shape=input_shape)
            y = tf.placeholder(tf.float32, shape=(None, nb_classes))
            model = dnn(input_shape, nb_classes)
            preds = model(x)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            grad_0 = gradient_graph(x, preds)
        global_inputs = set()
        tot_inputs = set()
        global_inputs_list = []
        local_inputs = set()
        local_inputs_list = []
        seed_num = 0
        max_k_time = 0
        max_k_time_list = []
        init_k_list = []
        max_k_list = []
        pre_local_input_num = 0
        effective_seed_cnt = 0

        # -----------------------
        def evaluate_local(inp):

            """
            Evaluate whether the test input after local perturbation is an individual discriminatory instance
            :param inp: test input
            :return: whether it is an individual discriminatory instance
            """
            global max_k
            global max_k_time
            global start_time
            global time1
            result, K, conf = check_for_error_condition(data_config[dataset], sess, x, preds, inp,
                                                        sens_params, input_shape, epsillon)
            if K > max_k:
                max_k = K
                max_k_time = time.time() - start_time

            dis_sample = copy.deepcopy(inp.astype('int').tolist())
            for sens in sens_params:
                dis_sample[sens - 1] = 0
            if tuple(dis_sample) not in global_inputs and \
                    tuple(dis_sample) not in local_inputs:
                local_inputs.add(tuple(dis_sample))
                local_inputs_list.append(dis_sample + [time.time() - time1])

            return (-1 * result)

        # select the seed input for fairness testing
        inputs = seed_test_input(clusters, min(max_global, len(X)))
        if preprocess_method == 'outlier':
            from scipy.spatial import distance
            from scipy.stats import entropy
            def get_feature_weights(net, sample, linears_weight):
                max_label_indices = tf.argmax(preds, axis=1)
                max_label_indices = tf.reshape(max_label_indices, (-1, 1))
                lines = tf.range(0, len(sample), dtype=tf.int64)
                lines = tf.reshape(lines, (-1, 1))
                indices = tf.concat([lines, max_label_indices], axis=1)
                labels = tf.gather_nd(preds, indices)
                grads = tf.gradients(labels, x)

                bias_placeholders = []
                for i in model.layers:
                    if 'Linear' in i.name or 'logits' in i.name:
                        bias_placeholders.append(i.b)

                biases = net.run(bias_placeholders,
                                 feed_dict={x: sample}, )

                result, grads_t = net.run([linears_weight, grads],
                                          feed_dict={x: sample}, )

                acts = net.run([linear_dict[_i] for _i in linear_dict] + [preds], feed_dict={x: sample})

                acts_list = [sample] + acts
                act_mins = []
                act_maxs = []
                for i in acts_list:
                    act_mins.append(np.min(i, axis=0))
                    act_maxs.append(np.max(i, axis=0))
                sample_num = 3000
                if acts_list[0].shape[0] > sample_num:
                    idx = np.arange(0, acts_list[0].shape[0])
                    np.random.shuffle(idx)
                    idx = idx[:sample_num]
                    for i in range(len(acts_list)):
                        acts_list[i] = acts_list[i][idx]
                dropout_importance = [[] for i in range(len(acts_list))]
                dropout_importance[-1] = np.ones((acts[-1].shape[1],))
                for i in range(len(acts_list) - 2, -1, -1):
                    acts_min = act_mins[i]
                    acts_max = act_maxs[i]
                    for j in range(acts_list[i].shape[1]):
                        cur_neuron = np.copy(acts_list[i])

                        cur_neuron[:, j] = acts_max[j]
                        if i == len(acts_list) - 2:
                            max_neuron_result = net.run(tf.nn.softmax(cur_neuron @ result[i] + biases[i]))
                        else:
                            max_neuron_result = net.run(tf.nn.relu(cur_neuron @ result[i] + biases[i]))

                        cur_neuron[:, j] = acts_min[j]
                        if i == len(acts_list) - 2:
                            min_neuron_result = net.run(tf.nn.softmax(cur_neuron @ result[i] + biases[i]))
                        else:
                            min_neuron_result = net.run(tf.nn.relu(cur_neuron @ result[i] + biases[i]))

                        cur_importance = np.mean(
                            np.abs(max_neuron_result - min_neuron_result) * dropout_importance[i + 1], axis=1) / (
                                                 acts_max[j] - acts_min[j])

                        dropout_importance[i].append(np.mean(cur_importance))
                    dropout_importance[i] = np.array(dropout_importance[i])

                all_origin_features = np.concatenate([sample] + acts, axis=1)

                all_origin_features = all_origin_features[:, :-acts[-1].shape[1]]

                grad_potent = np.zeros_like(grads_t[0])
                short_grad_potent = np.zeros_like(grads_t[0])
                for attr_idx in range(sample.shape[1]):
                    decreases = np.where(grads_t[0][:, attr_idx] > 0)[0]
                    increases = np.where(grads_t[0][:, attr_idx] < 0)[0]
                    decrease_dist = sample[decreases, attr_idx] - \
                                    data_config[dataset].input_bounds[attr_idx][0]
                    increase_dist = -sample[increases, attr_idx] + \
                                    data_config[dataset].input_bounds[attr_idx][1]
                    short_decrease_dist = np.copy(decrease_dist)
                    short_decrease_dist[short_decrease_dist > 1] = 1
                    short_increase_dist = np.copy(increase_dist)
                    short_increase_dist[short_increase_dist > 1] = 1
                    grad_potent[decreases, attr_idx] = decrease_dist * grads_t[0][decreases, attr_idx]
                    grad_potent[increases, attr_idx] = increase_dist * -grads_t[0][increases, attr_idx]
                    short_grad_potent[decreases, attr_idx] = short_decrease_dist * grads_t[0][decreases, attr_idx]
                    short_grad_potent[increases, attr_idx] = short_increase_dist * -grads_t[0][increases, attr_idx]

                weighted_score = short_grad_potent[:, sens_params[0] - 1]
                sorted_idx = np.argsort(-weighted_score)
                print(sorted_idx)
                all_importance = np.ones_like(acts[-1].shape)
                importance = 1
                acts[-1] *= importance
                for __i in range(len(result) - 1, -1, -1):
                    result[__i] *= importance
                    importance = np.mean(np.abs(result[__i]), axis=1)
                    all_importance = np.concatenate([importance, all_importance])
                    if __i >= 1:
                        acts[__i - 1] *= importance
                weighted_sample = np.copy(sample) * importance
                acts = [weighted_sample] + acts
                acts = np.concatenate(acts, axis=1)
                return importance, sorted_idx, acts, short_grad_potent, all_importance, weighted_score, None, all_origin_features, dropout_importance

            feature_weight, sidx, acts, short_grad_potent, all_importance, potent_score, _, all_origin_features, dropout_importance = get_feature_weights(
                sess, X,
                linear_weights)

            all_importance = np.concatenate(dropout_importance)
            all_importance = all_importance[:-nb_classes]
            all_importance /= np.mean(all_importance)

            def standardize(mat):
                mins = np.min(mat, axis=0)
                maxs = np.max(mat, axis=0)
                mat = (mat - mins) / (maxs - mins)
                return mat
            acts = standardize(all_origin_features) * all_importance

            def column_entropy(column):
                unique_elements, counts = np.unique(column, return_counts=True)
                probabilities = counts / len(column)
                return entropy(probabilities, base=2)

            entropies = np.apply_along_axis(column_entropy, axis=0, arr=acts)
            entropy_thresh = 0.05
            acts = acts[:, np.where(entropies > entropy_thresh)[0]]
            all_importance = all_importance[np.where(entropies > entropy_thresh)[0]]

            def dist_cluster_columns(dist_thresh):
                col_distances = distance.cdist(acts.T, acts.T, 'euclidean')
                sorted_col_idx = np.argsort(col_distances, axis=1)[:, 1:]
                sorted_col_distances = np.sort(col_distances, axis=1)[:, 1:]
                # dist_thresh = 10
                col_cluster_dict = {}
                cluster_id = 0
                for col_idx in range(sorted_col_idx.shape[0]):
                    cluster_cols = sorted_col_idx[col_idx, sorted_col_distances[col_idx] < dist_thresh]
                    if cluster_cols.shape[0] > 0:
                        use_cluster_id = None
                        if col_idx in col_cluster_dict:
                            use_cluster_id = col_cluster_dict[col_idx]
                        for col in cluster_cols:
                            if col in col_cluster_dict:
                                if use_cluster_id is None or use_cluster_id == col_cluster_dict[col]:
                                    use_cluster_id = col_cluster_dict[col]
                                else:
                                    for key in col_cluster_dict.keys():
                                        if key in col_cluster_dict and col_cluster_dict[key] == use_cluster_id:
                                            col_cluster_dict[key] = col_cluster_dict[col]
                                    use_cluster_id = col_cluster_dict[col]
                        if use_cluster_id is None:
                            use_cluster_id = cluster_id
                            cluster_id += 1
                        for col in cluster_cols:
                            if col in col_cluster_dict and col_cluster_dict[col] != use_cluster_id:
                                raise Exception('Logic error!')
                            else:
                                col_cluster_dict[col] = use_cluster_id
                    else:
                        continue

                final_cluster_dict = {}
                for col in col_cluster_dict:
                    if col_cluster_dict[col] in final_cluster_dict:
                        final_cluster_dict[col_cluster_dict[col]].append(col)
                    else:
                        final_cluster_dict[col_cluster_dict[col]] = [col]
                return final_cluster_dict

            init_dist_rate = dist_rate = 0.015
            # dist_thresh = int(dist_rate)
            dist_thresh = dist_rate * acts.shape[0]
            dist_rate_decay_rate = 0.9
            cluster_importance_rate_thresh = 0.15
            final_cluster_dict = dist_cluster_columns(dist_thresh)
            cluster_decay = True
            while cluster_decay:
                cluster_decay = False
                for cluster_id in final_cluster_dict:
                    cluster_importance_rate = np.sum(
                        all_importance[np.array(final_cluster_dict[cluster_id], dtype=int)]) / np.sum(all_importance)
                    if cluster_importance_rate > cluster_importance_rate_thresh:
                        cluster_decay = True
                        break
                if cluster_decay:
                    dist_rate *= dist_rate_decay_rate
                    dist_thresh = dist_rate * acts.shape[0]
                    final_cluster_dict = dist_cluster_columns(dist_thresh)
            print(f'final dist rate:{dist_rate}')
            remove_mask = np.ones(acts.shape[1], dtype=bool)
            cluster_importance = None
            cluster_col = None
            print(f'column cluster num:{len(final_cluster_dict)}')
            for cur_cluster_id in final_cluster_dict:
                cluster_col_idx = np.array(final_cluster_dict[cur_cluster_id], dtype=int)
                if cluster_importance is None:
                    cluster_importance = np.sum(all_importance[cluster_col_idx], keepdims=True)
                else:
                    cluster_importance = np.concatenate(
                        [cluster_importance, np.sum(all_importance[cluster_col_idx], keepdims=True)])
                if cluster_col is None:
                    cluster_col = np.mean(acts[:, cluster_col_idx], axis=1).reshape((-1, 1))
                else:
                    cluster_col = np.concatenate(
                        [cluster_col, np.mean(acts[:, cluster_col_idx], axis=1).reshape((-1, 1))], axis=1)
                remove_mask[cluster_col_idx] = False
            if cluster_col is not None:
                acts = np.concatenate([acts[:, remove_mask], cluster_col], axis=1)
                all_importance = np.concatenate([all_importance[remove_mask], cluster_importance])

            # std = np.std(acts, axis=0)
            # std_idx = np.argsort(std)
            std_idx = np.argsort(-all_importance)
            sorted_std = np.std(acts, axis=0)[std_idx]
            sorted_mean = np.mean(acts, axis=0)[std_idx]

            all_importance = all_importance[std_idx]

            cum_importance = np.cumsum(all_importance) / np.sum(all_importance)
            cum_importance_rate = 0.3
            end_idx = np.where(cum_importance > cum_importance_rate)[0][0] + 1
            print(f'cumulative importance selected feature rate:{end_idx / acts.shape[1]}, end index:{end_idx}')
            # all_importance/=np.sum(all_importance)

            acts = acts[:, std_idx]
            # feature_mask = all_importance>(1/len(all_importance))
            # feature_rate = np.sum(feature_mask)/acts.shape[1]
            # print(f'selected feature num:{np.sum(feature_mask)}, rate:{np.sum(feature_mask)/acts.shape[1]}')
            # acts = acts[:,feature_mask]
            # min_select_rate = 0.03
            # if end_idx / acts.shape[1] < min_select_rate:
            #     acts = acts[:, :int(acts.shape[1] * min_select_rate)]
            # else:
            acts = acts[:, :end_idx]

            # higher_rate = np.sum(all_importance[all_importance>(1/len(all_importance))])/len(all_importance)
            #
            # # sorted_std = std[std_idx]
            # # entropies = np.apply_along_axis(column_entropy, axis=0, arr=acts)
            # # sorted_entropy = entropies[std_idx]
            # # sorted_mean = np.mean(acts,axis=0)[std_idx]
            #
            # feature_rate = higher_rate
            #
            # std_idx = std_idx[:int(len(std_idx) * feature_rate)]
            # acts = acts[:, std_idx]

            distances = distance.cdist(acts, acts, 'euclidean')
            # sorted_dist = np.sort(distances, axis=1)
            sorted_dist = distances
            k_dist = sorted_dist
            outlier_scores = np.mean(k_dist, axis=1)
            del k_dist
            del sorted_dist
            del distances
            potent_level = np.copy(potent_score)
            log_base = 32
            potent_level[potent_level > 0] = (np.log(potent_score[potent_score > 0]) / np.log(log_base)).astype(int)
            potent_level[potent_score == 0] = np.min(potent_level) - 1
            potent_level -= np.min(potent_level)
            unique_list = np.sort(np.unique(potent_level))

            for i in unique_list:
                print(len(np.where(potent_level == i)[0]))

            outlier_scores += (np.max(outlier_scores) + 1) * potent_level
            path_sorted_list = np.argsort(-outlier_scores)
            X = X[path_sorted_list]

            dump_sorted_data = False
            if dump_sorted_data:
                sorted_X = X[:200]
                sorted_Y = Y[path_sorted_list][:200]
                dump_dir = os.path.join('..', 'top200_sorted_dataset',
                                        f'{args_dict["dataset"]}_{args_dict["sensitive_param"][0]}')
                if os.path.exists(dump_dir):
                    shutil.rmtree(dump_dir)
                os.mkdir(dump_dir)
                np.save(os.path.join(dump_dir, 'X.npy'), sorted_X)
                np.save(os.path.join(dump_dir, 'Y.npy'), sorted_Y)
                return
                pass

            # X = X[sidx]
            # print(
            #     f'meanColSumImportCumImport{cum_importance_rate}Log{log_base}Entropy{entropy_thresh}MinSel{min_select_rate}InitDistRate{init_dist_rate}DistDecay{dist_rate_decay_rate}ClusterImportRate{cluster_importance_rate_thresh}')
            print(
                f'meanColSumImportCumImport{cum_importance_rate}Log{log_base}Entropy{entropy_thresh}InitDistRate{init_dist_rate}DistDecay{dist_rate_decay_rate}ClusterImportRate{cluster_importance_rate_thresh}')
        elif preprocess_method == 'random':
            np.random.shuffle(X)
        global time1
        time1 = time.time()
        for num in range(len(inputs)):

            # clear_output(wait=True)
            start_time = time.time()
            if time.time() - time1 > timeout:
                break
            print('Input ', seed_num)
            if preprocess_method == 'cluster':
                index = inputs[num]
            elif preprocess_method == 'outlier':
                index = num
            elif preprocess_method == 'random':
                index = num
            else:
                raise Exception('Unknown preprocessing method')
            sample = X[index: index + 1]

            # start global perturbation
            for iter in range(max_iter + 1):
                if time.time() - time1 > timeout:
                    break
                m_sample = m_instance(np.array(sample), sens_params, data_config[dataset])
                pred = pred_prob(sess, x, preds, m_sample, input_shape)
                clus_dic = clustering(pred, m_sample, sens_params, epsillon)

                if iter == 0:
                    init_k = len(clus_dic) - 1
                    max_k = init_k

                if len(clus_dic) - 1 > max_k:
                    max_k = len(clus_dic) - 1
                    max_k_time = round((time.time() - start_time), 4)

                sample, n_values = global_sample_select(clus_dic, sens_params)
                dis_sample = sample.copy()
                for sens in sens_params:
                    dis_sample[0][sens - 1] = 0

                if tuple(dis_sample[0].astype('int')) not in global_inputs and \
                        tuple(dis_sample[0].astype('int')) not in local_inputs:
                    dis_flag = True
                    global_inputs.add(tuple(dis_sample[0].astype('int')))
                    global_inputs_list.append(dis_sample[0].astype('int').tolist())

                else:
                    dis_flag = False

                if dis_flag and (len(clus_dic) - 1 >= 2):
                    loc_x, n_values = local_sample_select(clus_dic, sens_params)
                    minimizer = {"method": "L-BFGS-B"}
                    local_perturbation = Local_Perturbation(sess, grad_0, x, n_values,
                                                            sens_params, input_shape[1],
                                                            data_config[dataset])
                    basinhopping(evaluate_local, loc_x, stepsize=1.0,
                                 take_step=local_perturbation, minimizer_kwargs=minimizer,
                                 niter=max_local)

                if dis_flag:
                    global_inputs_list[-1] += [time.time() - time1]

                clus_dic = {}
                if iter == max_iter:
                    break

                # Making up n_sample
                n_sample = sample.copy()
                for i in range(len(sens_params)):
                    n_sample[0][sens_params[i] - 1] = n_values[i]

                    # global perturbation

                s_grad = sess.run(tf.sign(grad_0), feed_dict={x: sample})
                n_grad = sess.run(tf.sign(grad_0), feed_dict={x: n_sample})

                # find the feature with same impact
                if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                    g_diff = n_grad[0]
                elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                    g_diff = s_grad[0]
                else:
                    g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)
                for sens in sens_params:
                    g_diff[sens - 1] = 0

                cal_grad = s_grad * g_diff
                if np.zeros(input_shape[1]).tolist() == cal_grad.tolist()[0]:
                    index = np.random.randint(len(cal_grad[0]) - 1)
                    for i in range(len(sens_params) - 1, -1, -1):
                        if index == sens_params[i] - 1:
                            index = index + 1

                    cal_grad[0][index] = np.random.choice([1.0, -1.0])

                sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], data_config[dataset]).astype("int")
            seed_num += 1

            cur_time_cost = time.time() - time1
            ch, cm, cs = float_sec2hms(cur_time_cost)
            average_time_cost_per_seed = cur_time_cost / seed_num
            remain_seed_num = len(inputs) - seed_num
            remain_time_cost = remain_seed_num * average_time_cost_per_seed
            rh, rm, rs = float_sec2hms(remain_time_cost)
            if len(local_inputs) - pre_local_input_num != 0:
                effective_seed_cnt += 1
            print(len(local_inputs) - pre_local_input_num, len(local_inputs),
                  f'current time cost:{ch}h-{cm}m-{cs}s, remain time estimate:{rh}h-{rm}m-{rs}s, effective seed count:{effective_seed_cnt}, sum id per seed:{"nan" if effective_seed_cnt == 0 else (len(local_inputs) / effective_seed_cnt)}')
            pre_local_input_num = len(local_inputs)
            if max_k > 1:
                max_k_time_list.append(max_k_time)
                init_k_list.append(init_k)
                max_k_list.append(max_k)

        print('Search Done!')
        if RQ == 1:

            # create the folder for storing the fairness testing result
            if not os.path.exists('../results/'):
                os.makedirs('../results/')
            if not os.path.exists('../results/' + dataset + '/'):
                os.makedirs('../results/' + dataset + '/')
            if not os.path.exists('../results/' + dataset + '/DICE/'):
                os.makedirs('../results/' + dataset + '/DICE/')
            if not os.path.exists('../results/' + dataset + '/DICE/RQ1/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ1/')
            if not os.path.exists(
                    '../results/' + dataset + '/DICE/RQ1/' + ''.join(str(i) for i in sens_params) + '_10runs/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ1/' + ''.join(str(i) for i in sens_params) + '_10runs/')
            # storing the fairness testing result
            np.save('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/global_inputs_' + str(trial) + '.npy',
                    np.array(global_inputs_list))
            np.save('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/local_inputs_' + str(trial) + '.npy',
                    np.array(local_inputs_list))
            total_inputs = np.concatenate((local_inputs_list, global_inputs_list), axis=0)
            np.save('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/total_inputs_' + str(trial) + '.npy',
                    total_inputs)
            # RQ1 & RQ2

            local_sam = np.array(local_inputs_list).astype('int32')
            global_sam = np.array(global_inputs_list).astype('int32')
            # Storing result for RQ1 table
            print('Analyzing the search results....')

            with open('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                    str(i) for i in sens_params) + '_10runs' + '/global_inputs_90_' + str(trial) + '.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(global_inputs_list)):
                    m_sample = m_instance(np.array([global_inputs_list[ind][:input_shape[1]]]), sens_params,
                                          data_config[dataset])
                    rows = m_sample.reshape((len(m_sample), input_shape[1]))
                    writer.writerows(
                        np.append(rows, [[global_inputs_list[ind][-1]] for i in range(len(m_sample))], axis=1))

            with open('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                    str(i) for i in sens_params) + '_10runs' + '/local_inputs_90_' + str(trial) + '.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(local_inputs_list)):
                    m_sample = m_instance(np.array([local_inputs_list[ind][:input_shape[1]]]), sens_params,
                                          data_config[dataset])
                    rows = m_sample.reshape((len(m_sample), input_shape[1]))
                    writer.writerows(
                        np.append(rows, [[local_inputs_list[ind][-1]] for i in range(len(m_sample))], axis=1))

            df_l = pd.read_csv('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/local_inputs_90_' + str(trial) + '.csv', header=None)
            df_g = pd.read_csv('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/global_inputs_90_' + str(trial) + '.csv', header=None)

            df_g['label'] = model_argmax(sess, x, preds, df_g.to_numpy()[:, :input_shape[1]])
            df_l['label'] = model_argmax(sess, x, preds, df_l.to_numpy()[:, :input_shape[1]])
            g_pivot = pd.pivot_table(df_g, values="label", index=list(np.setxor1d(df_g.columns[:-1],
                                                                                  np.array(sens_params) - 1)),
                                     aggfunc=np.sum)
            l_pivot = pd.pivot_table(df_l, values="label", index=list(np.setxor1d(df_l.columns[:-1],
                                                                                  np.array(sens_params) - 1)),
                                     aggfunc=np.sum)

            g_time = g_pivot.index[
                np.where((g_pivot['label'] > 0) & (g_pivot['label'] < len(m_sample)))[0]].get_level_values(
                input_shape[1]).values
            l_time = l_pivot.index[
                np.where((l_pivot['label'] > 0) & (l_pivot['label'] < len(m_sample)))[0]].get_level_values(
                input_shape[1]).values
            tot_time = np.sort(np.concatenate((l_time, g_time), axis=0))

            g_dis = (len(m_sample) - g_pivot.loc[(g_pivot['label'] > 0) & \
                                                 (g_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()
            l_dis = (len(m_sample) - l_pivot.loc[(l_pivot['label'] > 0) & \
                                                 (l_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()

            tot_df = pd.DataFrame(total_inputs)
            tot_df.columns = [i for i in range(input_shape[1])] + ['time']

            k = []
            disc = []
            tot_df['sh_entropy'] = 0
            for sam_ind in range(total_inputs.shape[0]):

                m_sample = m_instance(np.array([total_inputs[:, :input_shape[1]][sam_ind]]), sens_params,
                                      data_config[dataset])
                pred = pred_prob(sess, x, preds, m_sample, input_shape)
                clus_dic = clustering(pred, m_sample, sens_params, epsillon)
                tot_df.loc[[sam_ind], 'sh_entropy'] = sh_entropy(pred, epsillon)
                if pred.max() > 0.5 and pred.min() < 0.5:
                    disc.append(1)
                else:
                    disc.append(0)
                k.append(len(clus_dic) - 1)

            tot_df['k'] = k
            tot_df['disc'] = disc
            tot_df['min_entropy'] = round(np.log2(tot_df['k']), 2)
            tot_dis = tot_df.loc[tot_df['disc'] == 1]
            tot_dis.to_csv('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/total_disc_' + str(trial) + '.csv')
            # reseting the TF graph for the next round
            sess.close()
            tf.reset_default_graph()
            haighest_k = np.sort(tot_df['k'].unique())[-3:]
            if len(haighest_k) > 2:
                IK1F = np.where(tot_df['k'] == haighest_k[2])[0].shape[0]
                IK2F = np.where(tot_df['k'] == haighest_k[1])[0].shape[0]
                IK3F = np.where(tot_df['k'] == haighest_k[0])[0].shape[0]

            else:
                IK1F = np.where(tot_df['k'] == haighest_k[1])[0].shape[0]
                IK2F = np.where(tot_df['k'] == haighest_k[0])[0].shape[0]
                IK3F = 0
            print('Global ID RQ1 =', g_dis)
            print('local  ID RQ1  =', l_dis)
            print('Total loc samples  = ', len(local_sam))
            print('Total glob samples = ', len(global_sam))
            print('Total ID = ', g_dis + l_dis)

            global_succ = round(g_dis / (len(global_sam) * \
                                         len(m_sample)) * 100, 1)
            local_succ = round(l_dis / (len(local_sam) * \
                                        len(m_sample)) * 100, 1)

            row = [len(total_inputs)] + [np.mean(init_k_list), np.mean(max_k_list), np.mean(max_k_time_list)] + list(
                tot_dis[['min_entropy', 'sh_entropy']].mean()) + [IK1F, IK2F, IK3F]
            RQ1_table.append(row)

        if RQ == 2:

            # create the folder for storing the fairness testing result
            if not os.path.exists('../results/'):
                os.makedirs('../results/')
            if not os.path.exists('../results/' + dataset + '/'):
                os.makedirs('../results/' + dataset + '/')
            if not os.path.exists('../results/' + dataset + '/DICE/'):
                os.makedirs('../results/' + dataset + '/DICE/')
            if not os.path.exists('../results/' + dataset + '/DICE/RQ2/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ2/')
            if not os.path.exists(
                    '../results/' + dataset + '/DICE/RQ2/' + ''.join(str(i) for i in sens_params) + '_10runs/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ2/' + ''.join(str(i) for i in sens_params) + '_10runs/')
            # storing the fairness testing result
            np.save('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/global_inputs_' + str(trial) + '.npy',
                    np.array(global_inputs_list))
            np.save('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/local_inputs_' + str(trial) + '.npy',
                    np.array(local_inputs_list))
            total_inputs = np.concatenate((local_inputs_list, global_inputs_list), axis=0)
            np.save('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/total_inputs_' + str(trial) + '.npy',
                    total_inputs)
            # RQ1 & RQ2

            local_sam = np.array(local_inputs_list).astype('int32')
            global_sam = np.array(global_inputs_list).astype('int32')
            # Storing result for RQ1 table
            print('Analyzing the search results....')

            with open('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                    str(i) for i in sens_params) + '_10runs' + '/global_inputs_90_' + str(trial) + '.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(global_inputs_list)):
                    m_sample = m_instance(np.array([global_inputs_list[ind][:input_shape[1]]]), sens_params,
                                          data_config[dataset])
                    rows = m_sample.reshape((len(m_sample), input_shape[1]))
                    writer.writerows(
                        np.append(rows, [[global_inputs_list[ind][-1]] for i in range(len(m_sample))], axis=1))

            with open('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                    str(i) for i in sens_params) + '_10runs' + '/local_inputs_90_' + str(trial) + '.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(local_inputs_list)):
                    m_sample = m_instance(np.array([local_inputs_list[ind][:input_shape[1]]]), sens_params,
                                          data_config[dataset])
                    rows = m_sample.reshape((len(m_sample), input_shape[1]))
                    writer.writerows(
                        np.append(rows, [[local_inputs_list[ind][-1]] for i in range(len(m_sample))], axis=1))

            df_l = pd.read_csv('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/local_inputs_90_' + str(trial) + '.csv', header=None)
            df_g = pd.read_csv('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/global_inputs_90_' + str(trial) + '.csv', header=None)

            df_g['label'] = model_argmax(sess, x, preds, df_g.to_numpy()[:, :input_shape[1]])
            df_l['label'] = model_argmax(sess, x, preds, df_l.to_numpy()[:, :input_shape[1]])
            g_pivot = pd.pivot_table(df_g, values="label", index=list(np.setxor1d(df_g.columns[:-1],
                                                                                  np.array(sens_params) - 1)),
                                     aggfunc=np.sum)
            l_pivot = pd.pivot_table(df_l, values="label", index=list(np.setxor1d(df_l.columns[:-1],
                                                                                  np.array(sens_params) - 1)),
                                     aggfunc=np.sum)

            g_time = g_pivot.index[
                np.where((g_pivot['label'] > 0) & (g_pivot['label'] < len(m_sample)))[0]].get_level_values(
                input_shape[1]).values
            l_time = l_pivot.index[
                np.where((l_pivot['label'] > 0) & (l_pivot['label'] < len(m_sample)))[0]].get_level_values(
                input_shape[1]).values
            tot_time = np.sort(np.concatenate((l_time, g_time), axis=0))

            if len(tot_time) >= 1000:
                time_1000 = tot_time[999]
            else:
                time_1000 = np.nan

            print('Time to 1st ID', tot_time[0])
            print('time to 1000 ID', time_1000)
            g_dis_adf = len(g_time)
            l_dis_adf = len(l_time)

            global_succ_adf = round((g_dis_adf / len(global_sam)) * 100, 1)
            local_succ_adf = round((l_dis_adf / len(local_sam)) * 100, 1)

            tot_df = pd.DataFrame(total_inputs)
            tot_df.columns = [i for i in range(input_shape[1])] + ['time']
            disc = []
            for sam_ind in range(total_inputs.shape[0]):

                m_sample = m_instance(np.array([total_inputs[:, :input_shape[1]][sam_ind]]), sens_params,
                                      data_config[dataset])
                pred = pred_prob(sess, x, preds, m_sample, input_shape)
                clus_dic = clustering(pred, m_sample, sens_params, epsillon)
                if pred.max() > 0.5 and pred.min() < 0.5:
                    disc.append(1)
                else:
                    disc.append(0)

            tot_df['disc'] = disc
            tot_dis = tot_df.loc[tot_df['disc'] == 1]
            tot_dis.to_csv('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/total_disc_' + str(trial) + '.csv')
            # reseting the TF graph for the next round
            sess.close()
            tf.reset_default_graph()
            print('Total ID RQ2  = ', g_dis_adf + l_dis_adf)
            print('Global ID RQ2  = ', g_dis_adf)
            print('Local  ID RQ2  = ', l_dis_adf)

            RQ2_table.append([g_dis_adf + l_dis_adf, local_succ_adf, tot_time[0], time_1000])

            print('Local search success rate  = ', local_succ_adf, '%')
            print('Global search success rate = ', global_succ_adf, '%')

    if RQ == 1:
        np.save('../results/' + dataset + '/DICE/RQ1/' + ''.join(
            str(i) for i in sens_params) + '_10runs' + '/QID_RQ1_10runs.npy',
                RQ1_table)

        with open('../results/' + dataset + '/DICE/RQ1/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/RQ1_table.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['#I', 'K_I', 'K_F', 'T_KF', 'Q_inf',
                             'Q_1', 'IK1F', 'IK2F', 'IK3F'])
            writer.writerow(np.mean(RQ1_table, axis=0))
            writer.writerow(np.std(RQ1_table, axis=0))

    elif RQ == 2:

        np.save('../results/' + dataset + '/DICE/RQ2/' + ''.join(
            str(i) for i in sens_params) + '_10runs' + '/QID_RQ2_10runs.npy',
                RQ2_table)

        with open('../results/' + dataset + '/DICE/RQ2/' + ''.join(
                str(i) for i in sens_params) + '_10runs' + '/RQ2_table.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['tot_adf_disc', 'local_succ_adf',
                             'time_to_first', 'time_to_1000'])
            writer.writerow(np.mean(RQ2_table, axis=0))
            writer.writerow(np.std(RQ2_table, axis=0))


def config_varies_from(**kwargs):
    dict_list = [{}]
    recursive_config_varies(dict_list, **kwargs)
    return dict_list


def recursive_config_varies(pub_list, **kwargs):
    if len(kwargs) == 0:
        return
    key = None
    value = None
    for i in kwargs:
        key = i
        value = kwargs[key]
        break
    del kwargs[key]
    if not isinstance(value, list):
        value = [value]
    for i in range(len(pub_list)):
        for j in range(1, len(value)):
            pub_list.append(copy.copy(pub_list[i]))
            pub_list[-1][key] = value[j]
        pub_list[i][key] = value[0]
    recursive_config_varies(pub_list, **kwargs)


def main(argv=None):
    experiment_configs = []
    experiment_configs += config_varies_from(sens_params=[[1], [8], [9]], dataset='census',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[1], [2], [3]], dataset='compas',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[1]], dataset='bank',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[9], [13]], dataset='credit',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[2], [5]], dataset='default',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[8]], dataset='diabetes',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[1], [2]], dataset='heart',
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[1], [2], [10]], dataset=['meps15', 'meps16'],
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    experiment_configs += config_varies_from(sens_params=[[2], [3]], dataset=['students'],
                                             preprocess_method=['outlier'], timeout=999999, RQ=2,
                                             max_global=100)
    print('experiment configs nums:', len(experiment_configs))
    for i in experiment_configs:
        dnn_fair_testing(**i)


if __name__ == '__main__':
    tf.app.run()
