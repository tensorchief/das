#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow
from sklearn.cross_validation import StratifiedShuffleSplit

import argparse
import glob
import numpy as np
import re
import matplotlib.pyplot as plt

    
def construct_np_arrays(datdir,runs):
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
    with open('../data_prep/training_set.txt', 'r') as infile:
        rundirs = infile.readlines()
    runs = [re.match('.*([0-9]{8}).*',item).group(1) for item in rundirs]
    model_files = [item for item in \
                    sorted(glob.glob(datdir + 'training_data_*'+ str(region) + '*'))\
                    if re.match('.*([0-9]{8}).*',item).group(1) in runs]
    print(len(model_files))

    X,y = construct_np_arrays(datdir,model_files)
    print("constructed initial arrays")
    y_list = [item[0] for item in y]
    print("prepared for stratification")

    print(X.shape)
    print(y.shape)
    print(len(y_list))
    # split into training & validation set
    indices = StratifiedShuffleSplit(y_list, n_iter=1, test_size = 0.1)
    
    for train,test in indices:
        
        print('preparing data set')
        X_train = X[train];Y_train = y[train]
        testX = X[test];testY = y[test]
     
        print('Building network')
        # Building convolutional network (e.g. mnist tutorial)
        network = input_data(shape=[None, 28, 28, 21], name='input')
        conv_1 = conv_2d(network, 32, 3, activation='relu', regularizer="L2",name="conv_1")
        network = max_pool_2d(conv_1, 2)

        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)

        network = fully_connected(network, 128, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 256, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
        model = tflearn.DNN(network, tensorboard_verbose=0)
        
        print('Starting training')
        # Training
        run_id = 'cnn_conv_labelled_' + str(region)
        model.fit({'input': X_train}, {'target': Y_train}, n_epoch=30,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=500, show_metric=True, run_id=run_id)
          
        model.save(run_id)

 
if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("region")
    args = p.parse_args()

    main(int(args.region)) 
