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
    raise NotImplementedError('Not implemented')


def train_svm_model(train):
    raise NotImplementedError('Not implemented')


def train_mlp_model(train):
    raise NotImplementedError('Not implemented')


def train_rf_model(train):
    raise NotImplementedError('Not implemented')


def valid_model(valid, clf):
    raise NotImplementedError('Not implemented')


def count_pos_neg(data):
    raise NotImplementedError('Not implemented')


def results(data, result):
    raise NotImplementedError('Not implemented')


def eval_result(true_pos, true_neg, false_pos, false_neg):
    raise NotImplementedError('Not implemented')
