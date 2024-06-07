import os.path
import shutil

import numpy as np

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
from DICE_utils.config import census, credit, bank, compas, default, heart, diabetes, students, meps15, meps16

data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data,
        "default": default_data, "heart": heart_data, "diabetes": diabetes_data,
        "students": students_data, "meps15": meps15_data, "meps16": meps16_data}
data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas, "default": default,
               "heart": heart, "diabetes": diabetes, "students": students, "meps15": meps15, "meps16": meps16}


def create_dataset(train_size=0.8):
    path = 'split_dataset'
    path = os.path.join(os.path.split(__file__)[0], path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for dataset_name in data:
        cur_dataset_path = os.path.join(path, dataset_name)
        os.mkdir(cur_dataset_path)
        X, Y, input_shape, nb_classes = data[dataset_name]()
        concat_data = np.concatenate((X, Y), axis=1)
        np.random.shuffle(concat_data)
        train_data = concat_data[:int(train_size * len(concat_data))]
        test_data = concat_data[int(train_size * len(concat_data)):]
        train_x = train_data[:, :-Y.shape[1]]
        train_y = train_data[:, -Y.shape[1]:]
        test_x = test_data[:, :-Y.shape[1]]
        test_y = test_data[:, -Y.shape[1]:]
        np.save(os.path.join(cur_dataset_path, 'train_x.npy'), train_x)
        np.save(os.path.join(cur_dataset_path, 'train_y.npy'), train_y)
        np.save(os.path.join(cur_dataset_path, 'test_x.npy'), test_x)
        np.save(os.path.join(cur_dataset_path, 'test_y.npy'), test_y)


def read_data(dataset_name):
    path = 'split_dataset'
    path = os.path.join(os.path.split(__file__)[0], path)
    test_x = np.load(os.path.join(path, dataset_name, 'test_x.npy'))
    test_y = np.load(os.path.join(path, dataset_name, 'test_y.npy'))
    train_x = np.load(os.path.join(path, dataset_name, 'train_x.npy'))
    train_y = np.load(os.path.join(path, dataset_name, 'train_y.npy'))
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    dataset_names = ["census", "credit", "bank", "compas", "default",
                     "heart", "diabetes", "students", "meps15", "meps16"]
    for dataset_name in dataset_names:
        train_x, _, _, _ = read_data(dataset_name)
        print(len(train_x)/0.8,dataset_name)
    pass
