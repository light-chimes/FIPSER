import os.path
import random

import numpy as np
import sys

from fairness_improvement.data_process import read_data

sys.path.append("../")

import tensorflow
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
from DICE_utils.utils_tf import model_train, model_eval
from DICE_model.tutorial_models import dnn


def set_random_seed(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_on_dataset(dataset, nb_epochs=1000, batch_size=128, learning_rate=0.01):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    # data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data,
    #         "default": default_data, "heart": heart_data}

    # prepare the data and model
    # X, Y, input_shape, nb_classes = data[dataset]()
    # tf.set_random_seed(1234)
    model_path = 'pretrained_model'
    train_x, train_y, test_x, test_y = read_data(dataset)
    set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]))
    model = dnn((None, train_x.shape[1]), train_y.shape[1])
    preds = model(x)

    # training parameters
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(os.path.split(__file__)[0], model_path, dataset),
        'filename': 'test.model'
    }

    # training procedure
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2024, 4, 10])
    model_train(sess, x, y, preds, train_x, train_y, args=train_params,
                rng=rng, save=True)

    # evaluate the accuracy of trained model
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, test_x, test_y, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))


def training():
    dataset_names = ["census", "credit", "bank", "compas", "default",
                     "heart", "diabetes", "students", "meps15", "meps16"]
    for dataset_name in dataset_names:
        train_on_dataset(dataset=dataset_name,
                         nb_epochs=1000,
                         batch_size=128,
                         learning_rate=0.01)


def main(argv=None):
    training()


if __name__ == '__main__':
    tf.app.run()
