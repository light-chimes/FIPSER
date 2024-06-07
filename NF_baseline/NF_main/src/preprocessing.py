import numpy as np
import tensorflow.compat.v1 as tf


def outlier(sess, x, linear_weights, linear_dict, preds, model, sens_param, sens_attr_lower_bound,
            sens_attr_upper_bound, X,
            nb_classes, Y=None):
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
            decrease_dist = sample[decreases, attr_idx] - sens_attr_lower_bound
            increase_dist = -sample[increases, attr_idx] + sens_attr_upper_bound
            short_decrease_dist = np.copy(decrease_dist)
            short_decrease_dist[short_decrease_dist > 1] = 1
            short_increase_dist = np.copy(increase_dist)
            short_increase_dist[short_increase_dist > 1] = 1
            grad_potent[decreases, attr_idx] = decrease_dist * grads_t[0][decreases, attr_idx]
            grad_potent[increases, attr_idx] = increase_dist * -grads_t[0][increases, attr_idx]
            short_grad_potent[decreases, attr_idx] = short_decrease_dist * grads_t[0][decreases, attr_idx]
            short_grad_potent[increases, attr_idx] = short_increase_dist * -grads_t[0][increases, attr_idx]

        weighted_score = short_grad_potent[:, sens_param - 1]

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
        del col_distances
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

    dist_rate = 0.015
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

    std_idx = np.argsort(-all_importance)

    all_importance = all_importance[std_idx]

    cum_importance = np.cumsum(all_importance) / np.sum(all_importance)
    cum_importance_rate = 0.3
    end_idx = np.where(cum_importance > cum_importance_rate)[0][0] + 1
    print(f'cumulative importance selected feature rate:{end_idx / acts.shape[1]}, end index:{end_idx}')

    acts = acts[:, std_idx]

    acts = acts[:, :end_idx]

    def classified_outlier_score(features, one_hot_labels, sample_num=100, sampling=True):
        mean_dist = np.zeros((features.shape[0],))
        accumulated_sample_cnt = 0
        class_list = []
        for i in range(one_hot_labels.shape[1] - 1):
            mask = one_hot_labels[:, i] == 1
            sample_cnt = round(np.sum(mask) / features.shape[0] * sample_num)
            accumulated_sample_cnt += sample_cnt
            mean_dist[mask] = np.mean(distance.cdist(features[mask, :], features[mask, :], 'euclidean'), axis=1)
            sorted_indices = np.argsort(mean_dist[mask], axis=0)
            if sampling:
                class_list.append(np.arange(0, features.shape[0])[mask][sorted_indices[:sample_cnt]])
            else:
                class_list.append(np.arange(0, features.shape[0])[mask][sorted_indices])
        i = one_hot_labels.shape[1] - 1
        mask = one_hot_labels[:, i] == 1
        sample_cnt = sample_num - accumulated_sample_cnt
        mean_dist[mask] = np.mean(distance.cdist(features[mask, :], features[mask, :], 'euclidean'), axis=1)
        sorted_indices = np.argsort(mean_dist[mask], axis=0)
        if sampling:
            class_list.append(np.arange(0, features.shape[0])[mask][sorted_indices[:sample_cnt]])
        else:
            class_list.append(np.arange(0, features.shape[0])[mask][sorted_indices])
        return class_list, mean_dist

    distances = distance.cdist(acts, acts, 'euclidean')

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
    return path_sorted_list
