import numpy as np
import pandas as pd
import preprocess as o_pre
import random
import time

from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier


def read_dataset(train_prob):
    file_path = "/data/train.csv"
    backorders = pd.read_csv(file_path)
    print(backorders)

    npOrders = backorders.as_matrix()

    np.random.shuffle(npOrders)

    total = npOrders.shape[0]
    train_size = int(total * train_prob)
    train, valid = npOrders[:train_size, :], npOrders[train_size:, :]

    return (train, valid)


def train_svm_model(train):
    # raise NotImplementedError('Not implemented')
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train[:, 1:-1], train[:, -1])
    return clf


def train_mlp_model(train):
    # raise NotImplementedError('Not implemented')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(size,), random_state=1)
    clf.fit(train[:, 1:-1], train[:, -1])
    return clf


def train_rf_model(train):
    # raise NotImplementedError('Not implemented')
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(train[:, -1:-1], train[:, -1])
    return clf


def valid_model(valid, clf):
    # raise NotImplementedError('Not implemented')
    result = clf.fit(valid[:, 1:-1])
    return result


def count_pos_neg(data):
    pos = 0
    neg = 0
    for row in data:
        if(row[-1] == 0):
            neg += 1
        else:
            pos += 1
    print(pos, neg)

    return (pos, neg)


def count_results(data, result):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(0, result.shape[0]):
        if data[i, -1] == 1 and result[i] == 1:
            true_pos += 1
        elif data[i, -1] == 1 and result[i] == 0:
            false_neg += 1
        elif data[i, -1] == 0 and result[i] == 1:
            false_pos += 1
        elif data[i, -1] == 0 and result[i] == 0:
            true_neg += 1
    print(true_pos, true_neg, false_pos, false_neg)

    return (true_pos, true_neg, false_pos, false_neg)


def eval_result(true_pos, true_neg, false_pos, false_neg):
    # raise NotImplementedError('Not implemented')
    precision = true_pos * 1.0 / (true_pos + false_pos)
    recall = true_pos * 1.0 / (true_pos + false_neg)
    f_score = (precision * recall) * 2.0 / (precision + recall)
    print(precision, recall, f_score)
    return (precision, recall)


def eval_model(clf, data):
    # raise NotImplementedError('Not implemented')
    result = clf.predict(data[:, 1:22])
    print(result.shape)
    print(result)
    true_pos, true_neg, false_pos, false_neg = count_results(data, result)
    eval_result(true_pos, true_neg, false_pos, false_neg)


def predict(data, clf):
    raise NotImplementedError('Not implemented')


def write_model(clf, file_path):
    raise NotImplementedError('Not implemented')


def read_model(file_path):
    raise NotImplementedError('Not implemented')


def write_numpy(data, file_path):
    raise NotImplementedError('Not implemented')


def write_data(train, valid, train_path, valid_path):
    raise NotImplementedError('Not implemented')


def train_write(train, file_path):
    raise NotImplementedError('Not implemented')


def write_data():
    raise NotImplementedError('Not implemented')


def read_numpy(file_path):
    raise NotImplementedError('Not implemented')


def read_data(train_path, valid_path):
    raise NotImplementedError('Not implemented')


def sample_data(data, neg_prob):
    raise NotImplementedError('Not implemented')


def duplicate_data(data, pos_times):
    raise NotImplementedError('Not implemented')


def plot_roc(valid, clf, model_name):
    raise NotImplementedError('Not implemented')


def train_from_csv_data(data_file, model_file):
    raise NotImplementedError('Not implemented')


def build_classifier_sample():
    raise NotImplementedError('Not implemented')


def eval_result_sample(model_file, model_name):
    raise NotImplementedError('Not implemented')


def eval_result_rf():
    raise NotImplementedError('Not implemented')


if __name__ == '__main__':
    print()
    eval_result_rf()
