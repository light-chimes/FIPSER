import os.path
import random
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from DICE_model.tutorial_models import dnn
from DICE_utils.utils_tf import model_train, model_eval
from fairness_improvement.data_process import read_data

import tensorflow.compat.v1 as tf

from fairness_improvement.eval import eval_fairness

tf.compat.v1.disable_eager_execution()

from DICE_utils.config import census, credit, bank, compas, default, heart, diabetes, students, meps15, meps16

dataset_shapes = {'census': (32561, 13), 'credit': (600, 20), 'bank': (45211, 16), 'compas': (7214, 12),
                  'default': (30000, 23), 'heart': (297, 13), 'diabetes': (768, 8), 'students': (1044, 32),
                  'meps15': (9668, 137), 'meps16': (7217, 137)}
dataset_names = ["census", "credit", "bank", "compas", "default",
                 "heart", "diabetes", "students", "meps15", "meps16"]
sens_params = [[1, 8, 9], [9, 13], [1], [1, 2, 3], [2, 5], [1, 2], [8], [2, 3], [1, 2, 10], [1, 2, 10]]
dataset_sens = {j: sens_params[i] for i, j in enumerate(dataset_names)}
data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas, "default": default,
               "heart": heart, "diabetes": diabetes, "students": students, "meps15": meps15, "meps16": meps16}


def set_random_seed(seed, reset_graph=True):
    if reset_graph:
        tf.reset_default_graph()
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_ensemble_classifiers(random_seed=1234):
    np.random.seed(random_seed)
    knn_clf = KNeighborsClassifier()
    mlp_clf = MLPClassifier()
    svm_clf = SVC(probability=True)
    rf_clf = RandomForestClassifier()
    nb_clf = GaussianNB()
    eclf = VotingClassifier(
        estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', nb_clf)],
        voting='soft')
    clf = Pipeline([('scaler', StandardScaler()),
                    ('ensemble', eclf)])
    path = 'ensemble_voter'
    path = os.path.join(os.path.split(__file__)[0], path)
    dataset_names = ["census", "credit", "bank", "compas", "default",
                     "heart", "diabetes", "students", "meps15", "meps16"]
    sens_params = [[1, 8, 9], [9, 13], [1], [1, 2, 3], [2, 5], [1, 2], [8], [2, 3], [1, 2, 10], [1, 2, 10]]
    for i, dataset_name in enumerate(dataset_names):
        train_x, train_y, test_x, test_y = read_data(dataset_name)
        train_y, test_y = np.argmax(train_y, axis=1), np.argmax(test_y, axis=1)
        if os.path.exists(os.path.join(path, dataset_name)):
            shutil.rmtree(os.path.join(path, dataset_name))
        os.mkdir(os.path.join(path, dataset_name))
        col_rmv = [idx - 1 for idx in sens_params[i]]
        narrow_train_x = np.delete(train_x, col_rmv, axis=1)
        narrow_test_x = np.delete(test_x, col_rmv, axis=1)
        model = clone(clf)
        model.fit(narrow_train_x, train_y)
        score = model.score(narrow_test_x, test_y)
        print(dataset_name, score)
        joblib.dump(model, os.path.join(path, dataset_name, 'voter.pkl'))


def read_idip(dataset_name, sens_param, method, line=None):
    path = 'generated_idip'
    path = os.path.join(os.path.split(__file__)[0], path, f'{dataset_name}_{sens_param}_{method}')
    for i in os.listdir(path):
        if 'runs' in i:
            for j in os.listdir(os.path.join(path, i)):
                if 'total_disc' in j:
                    df = pd.read_csv(os.path.join(path, i, j), nrows=line)
                    df = df.drop(df.columns[-2:], axis=1)
                    df = df.drop(df.columns[0], axis=1)
                    arr = np.array(df, dtype=int)
                    return arr
            return


def sample_idip(dataset_name, method, count, random_seed=1234):
    set_random_seed(random_seed, reset_graph=False)
    idip_list = []
    total_idip = 0
    for sens_param in dataset_sens[dataset_name]:
        idip_list.append(read_idip(dataset_name, sens_param, method))
        total_idip += idip_list[-1].shape[0]
    current_idip = 0
    for i, sens_param in enumerate(dataset_sens[dataset_name][:-1]):
        sample_num = round(idip_list[i].shape[0] / total_idip * count)
        current_idip += sample_num
        np.random.shuffle(idip_list[i])
        idip_list[i] = idip_list[i][:sample_num]
    np.random.shuffle(idip_list[-1])
    idip_list[-1] = idip_list[-1][:(count - current_idip)]
    return tuple(idip_list)


def retrain(dataset_name, method, nb_epochs=500, batch_size=128, learning_rate=0.01, corrected_idip_rate=0.5,
            random_seed=1234, merge_origin_data=True):
    set_random_seed(random_seed)
    if not os.path.exists(os.path.join(os.path.split(__file__)[0], 'retrained_model', dataset_name)):
        os.mkdir(os.path.join(os.path.split(__file__)[0], 'retrained_model', dataset_name))
    retrained_model_path = os.path.join(os.path.split(__file__)[0], 'retrained_model', dataset_name, method)
    ensemble_model_path = os.path.join(os.path.split(__file__)[0], 'ensemble_voter', dataset_name, 'voter.pkl')
    if os.path.exists(retrained_model_path):
        shutil.rmtree(retrained_model_path)
    os.mkdir(retrained_model_path)
    pretrained_model_path = os.path.join(os.path.split(__file__)[0], 'pretrained_model', dataset_name, '999',
                                         'test.model')

    train_x, train_y, test_x, test_y = read_data(dataset_name)
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, train_y.shape[1]))
    model = dnn((None, train_x.shape[1]), train_y.shape[1])
    preds = model(x)
    saver = tf.train.Saver()
    saver.restore(sess, pretrained_model_path)

    idips = sample_idip(dataset_name, method, round(train_x.shape[0] * corrected_idip_rate))
    total_idip_x = []
    for i in range(len(idips)):
        sens_param = dataset_sens[dataset_name][i]
        for j in range(data_config[dataset_name].input_bounds[sens_param - 1][0],
                       data_config[dataset_name].input_bounds[sens_param - 1][1] + 1):
            total_idip_x.append(np.copy(idips[i]))
            total_idip_x[-1][:, sens_param - 1] = j
    total_idip_x = np.concatenate(total_idip_x, axis=0)
    print(f'final idip:origin rate = {total_idip_x.shape[0] / train_x.shape[0]}')
    col_rmv = [idx - 1 for idx in dataset_sens[dataset_name]]
    sens_removed_x = np.delete(total_idip_x, col_rmv, axis=1)
    ensemble_voter = joblib.load(ensemble_model_path)
    corrected_y = ensemble_voter.predict(sens_removed_x)
    num_unique_labels = train_y.shape[1]
    corrected_y = np.eye(num_unique_labels)[corrected_y]
    if merge_origin_data:
        train_x = np.concatenate([train_x, total_idip_x], axis=0)
        train_y = np.concatenate([train_y, corrected_y], axis=0)
    else:
        train_x = total_idip_x
        train_y = corrected_y
    idx = np.arange(0, train_x.shape[0])
    np.random.shuffle(idx)
    train_x = train_x[idx]
    train_y = train_y[idx]
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': retrained_model_path,
        'filename': 'test.model'
    }
    rng = np.random.RandomState([2024, 4, 10])
    model_train(sess, x, y, preds, train_x, train_y, args=train_params,
                rng=rng, save=True)
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, test_x, test_y, args=eval_params)
    fairness = eval_fairness(dataset_name, dataset_sens[dataset_name], True, method)
    print(dataset_name, 'accuracy:', accuracy, 'fairness:', fairness, 'method:', method)


if __name__ == '__main__':
    for dataset_name in dataset_names:
        # for method in ['cluster', 'outlier','random']:
        for method in ['outlier']:
            retrain(dataset_name, method)
    # for dataset_name in dataset_names:
    #     total_idip_line = 0
    #     for sens_param in dataset_sens[dataset_name]:
    #         total_idip_line += read_idip(dataset_name, sens_param, 'cluster').shape[0]
    #     print(total_idip_line/ dataset_shapes[dataset_name][0])
