import pandas as pd
import numpy as np
import order_classifier
import random


def read_data(file_path):
    raise NotImplementedError('Not implemented')


def convert_boolean(value):
    raise NotImplementedError('Not implemented')


def convert_instance(x):
    raise NotImplementedError('Not implemented')


def divide_dataset(data, train_prob):
    raise NotImplementedError('Not implemented')


def write_datafile():
    raise NotImplementedError('Not implemented')


def write_dataset(train, value, train_file, valid_file):
    raise NotImplementedError('Not implemented')


def pre_process():
    raise NotImplementedError('Not implemented')


if __name__ == '__main__':
    data = read_data('data/train.csv')
