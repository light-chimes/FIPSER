import os.path
import random

import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
import tqdm

from DICE_model.tutorial_models import dnn
from DICE_utils.utils_tf import model_eval
from DICE_utils.config import census, credit, bank, compas, default, heart, diabetes, students, meps15, meps16
from fairness_improvement.data_process import read_data


def set_random_seed(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def eval_fairness(dataset, sens_param, after_retrain=False, method='cluster', batch_size=int(1e5), random_seed=1234,
                  sample_num=int(1e8)):
    """
    :param dataset:
    :param sample_num: how many pairs of samples to use for fairness evaluation
    """
    tf.reset_default_graph()
    if isinstance(sens_param, list):
        pass
    elif isinstance(sens_param, int):
        sens_param = [sens_param]
    else:
        raise ValueError('Unknown sens_param type')
    sens_param = [i - 1 for i in sens_param]
    data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas, "default": default,
                   "heart": heart, "diabetes": diabetes, "students": students, "meps15": meps15, "meps16": meps16}
    cur_config = data_config[dataset]
    set_random_seed(random_seed)
    if after_retrain:
        path = os.path.join(os.path.split(__file__)[0], 'retrained_model', dataset, method, '499', 'test.model')
    else:
        path = os.path.join(os.path.split(__file__)[0], 'pretrained_model', dataset, '999', 'test.model')
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    train_x, train_y, test_x, test_y = read_data(dataset)
    x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]))
    model = dnn((None, train_x.shape[1]), train_y.shape[1])
    preds = model(x)
    saver = tf.train.Saver()
    saver.restore(sess, path)
    consist_num = 0
    for i in tqdm.tqdm(range(0, sample_num, batch_size)):
        batch_num = min(batch_size, sample_num - i)
        cols = []
        for attr_bound in cur_config.input_bounds:
            lower_bound, upper_bound = attr_bound[0], attr_bound[1]
            cols.append(np.random.randint(lower_bound, upper_bound + 1, size=(batch_num, 1)))
        generated_origin_data = np.concatenate(cols, axis=1)
        sens_idx = sens_param[random.randint(0, len(sens_param) - 1)]
        generated_perturb_data = np.copy(generated_origin_data)
        # generated_perturb_data[:, sens_idx] = generated_perturb_data[:, sens_idx] +
        if (cur_config.input_bounds[sens_idx][1] - cur_config.input_bounds[sens_idx][0]) == 1:
            perturb_val = np.ones((batch_num,))
        else:
            perturb_val = np.random.randint(
                1, cur_config.input_bounds[sens_idx][1] - cur_config.input_bounds[sens_idx][0], size=(batch_num,))
        generated_perturb_data[:, sens_idx] = generated_perturb_data[:, sens_idx] + perturb_val
        generated_perturb_data[:, sens_idx] = generated_perturb_data[:, sens_idx] % (
                cur_config.input_bounds[sens_idx][1] + 1)
        generated_origin_pred = sess.run(preds, feed_dict={x: generated_origin_data})
        generated_perturb_pred = sess.run(preds, feed_dict={x: generated_perturb_data})
        generated_origin_pred = np.argmax(generated_origin_pred, axis=1)
        generated_perturb_pred = np.argmax(generated_perturb_pred, axis=1)
        consist_num += np.sum(generated_origin_pred == generated_perturb_pred)
    return consist_num / sample_num


def eval_accuracy(dataset, after_retrain=False):
    tf.reset_default_graph()
    model_dir = 'retrained_model' if after_retrain else 'pretrained_model'
    path = os.path.join(os.path.split(__file__)[0], model_dir, dataset, '999', 'test.model')
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    eval_params = {'batch_size': 128}
    train_x, train_y, test_x, test_y = read_data(dataset)
    x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]))
    model = dnn((None, train_x.shape[1]), train_y.shape[1])
    preds = model(x)
    saver = tf.train.Saver()
    saver.restore(sess, path)
    accuracy = model_eval(sess, x, y, preds, test_x, test_y, args=eval_params)
    # print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    return accuracy


if __name__ == '__main__':
    # eval_accuracy('census', )
    # eval_fairness('census', [1, 8, 9])
    dataset_names = ["census", "credit", "bank", "compas", "default",
                     "heart", "diabetes", "students", "meps15", "meps16"]
    sens_params = [[1, 8, 9], [9, 13], [1], [1, 2, 3], [2, 5], [1, 2], [8], [2, 3], [1, 2, 10], [1, 2, 10]]
    dataset_sens = {j: sens_params[i] for i, j in enumerate(dataset_names)}
    for dataset_name in dataset_names:
        print(dataset_name, 'accuracy:', eval_accuracy(dataset_name), 'fairness:',
              eval_fairness(dataset_name, dataset_sens[dataset_name], False))
