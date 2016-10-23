#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import tensorflow

import argparse
import glob
import numpy as np
import re
import matplotlib.pyplot as plt

def remove_adjacent(train, test, runs):
    """
    removes runs adjacent to model runs in the test set from training set
    :param train: indices of training set
    :param test: indices of test set
    :param runs: list of model runs
    :returns: train, test set (as list of model runs)
    """
    set_test = list()
    for item in test:
        if item-1 in train:
            train = train[train != item -1]
        if item + 1 in train:
            train = train[train != item +1]
        set_test.append(runs[item])
    
    set_train = list()
    for item in train:
        set_train.append(runs[item])

    return set_train, set_test


def construct_np_arrays(runs):
    """
    Numpy files of the runs given are loaded and model/label sets are
    concatenated
    :param runs: list of model runs (e.g. of training set)
    :returns: model_data_np (X), labels_np (Y)
    """
    model_data_np = np.empty((0,28,28,21),int)
    labels_np = np.empty((0,2),int)

    for item in runs:
        # load model data
        cur_model = np.load(item)
        model_data_np = np.concatenate((model_data_np,cur_model),axis = 0)
            
        # load matching labels
        cur_index = re.match('.*([0-9]{8}_[0-9]{3}.*)',item).group(1)
        cur_label = np.load(datdir + 'training_labels_' + cur_index)
        labels_np = np.concatenate((labels_np,cur_label),axis = 0)
            
    return model_data_np, labels_np

def main(region):
    # Data loading & preprocessing
    datdir = '/home/silviar/Dokumente/Training_set/'
    model_files = sorted(glob.glob(datdir + 'training_data_*' + region +'*.npy'))

    # do k-folds
    kf = KFold(len(model_files),n_folds = 10, shuffle = True)
    scores = list()
    loop = 1
    for train,test in kf:
        with tensorflow.Graph().as_default():
            print('Performing loop ' + str(loop))
            # remove runs adjacent to test set
            training_set, test_set = remove_adjacent(train,test,model_files)

            # construct np arrays
            X,Y = construct_np_arrays(training_set)
            print('done with training set')

            testX,testY = construct_np_arrays(test_set)
            print('done with test set')

            print('Building network')
            # Building convolutional network (e.g. mnist tutorial)
            network = input_data(shape=[None, 28, 28, 21], name='input')
            network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
            network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)

            network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
            network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)

            network = fully_connected(network, 128, activation='tanh')
            network = dropout(network, 0.5)
            network = fully_connected(network, 256, activation='tanh')
            network = dropout(network, 0.5)
            network = fully_connected(network, 2, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
            model = tflearn.DNN(network, tensorboard_verbose=0)
            
            print('Starting training')
            # Training
            run_id = 'cnn_mnist_' + str(loop) + '_stratified'
            model.fit({'input': X}, {'target': Y}, n_epoch=100,
                   validation_set=({'input': testX}, {'target': testY}),
                   snapshot_step=500, show_metric=True, run_id=run_id)
            
            loop += 1 

if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("region")
    args = p.parse_args()

    main(int(args.region)) 
