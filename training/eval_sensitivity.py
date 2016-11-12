#!/usr/bin/env python

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow

import argparse
import glob
import numpy as np
import re
import itertools
import matplotlib.pyplot as plt

def construct_np_arrays(datdir, runs, model_data_np):
    """
    Numpy files of the runs given are loaded and model/label sets are
    concatenated
    :param runs: list of model runs (e.g. of training set)
    :returns: model_data_np (X), labels_np (Y)
    """
    labels_np = np.empty((0,2),int)

    for item in runs:
        # load model data
        cur_model = np.load(item)
        model_data_np = np.concatenate((model_data_np,cur_model),axis = 0)

        # load matching labels
        cur_index = re.match('.*([0-9]{8}_[0-9]{3}.*)',item).group(1)
        cur_tag = re.match('.*/([a-z]+)_[a-z]+_[0-9]{8}_[0-9]{3}.*',item).group(1)
        cur_label = np.load(datdir + cur_tag + '_labels_' + cur_index)
        labels_np = np.concatenate((labels_np,cur_label),axis = 0)

    return model_data_np, labels_np

def replace_entries(np_array):
    """
    converts numpy array of shape (-1,2) to (-1,1)
    :param np_array: numpy_array (-1,2)
    :returns: list (-1,1)
    """
    output = list()
    for item in np_array:
        output.append(item[1])
    return output

def setup_cnn():
    """
    defines model architecture of a cnn
    :param void:
    :return: model
    """
    network = input_data(shape=[None, 28, 28, 21], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                 loss='categorical_crossentropy', name='target')
    return tflearn.DNN(network, tensorboard_verbose=0)

 
def main(region):

    # Data loading & preprocessing
    datdir = '/home/silviar/Dokumente/Training_set/'
    with open('../data_prep/test_set.txt', 'r') as infile:
        rundirs = infile.readlines()
    runs = [re.match('.*([0-9]{8}).*',item).group(1) for item in rundirs]
    test_files = [item for item in \
                    sorted(glob.glob(datdir + 'training_data_*'+ str(region) + '*'))\
                    if re.match('.*([0-9]{8}).*',item).group(1) in runs]

    X,Y = construct_np_arrays(datdir,test_files,np.empty((0,28,28,21),int))

    model = setup_cnn()

    # load weights
    model.load('/home/silviar/Dokumente/Abschlussarbeit/training/models/' \
                + 'cnn_mnist_stratified_' + str(region))

    zeroes = np.zeros((7,7,21))
    
    # get all fields with valid label 1
    indices_ones = np.where(Y[:,0] == 1)
    test_set = X[indices_ones,:][0,:]
    test_labels = Y[indices_ones,1]
    #pred_labels = dict()
    #new_labels = dict()
    diff_labels = dict()

    # go through snippets
    for i in xrange(0,27,7):
        for j in xrange(0,27,7):
            cur_label = 'i = ' + str(i) + ', j = ' + str(j)
            cur_pred = list()
            cur_new = list() 
            for item in test_set:
                cur_X = item[np.newaxis,...]
                # make deep copy
                cur_X_new = np.empty_like(cur_X)
                cur_X_new[:] = cur_X[np.newaxis,...]
                # set current snippet zero
                cur_X_new[0,i:i+7,j:j+7,:] = zeroes
                # do preditions
                cur_pred.append(model.predict(cur_X)[0])
                cur_new.append(model.predict(cur_X_new)[0])
            
            # store
            #pred_labels[cur_label] = cur_pred
            #new_labels[cur_label] = cur_new
            diff_labels[cur_label] = np.subtract(np.array(cur_new)[:,0],np.array(cur_pred)[:,0])

    #print(pred_labels)
    #print(new_labels)

    # plot differences
    outdir = '/home/silviar/Dokumente/Abschlussarbeit/training/models/'
    for index,key in enumerate(diff_labels):
        plt.figure()
        plt.hist(diff_labels[key])
        plt.title("P(p>10mm)* - P(p>10mm); " + key)
        cur_text = 'Mean = ' + str(np.mean(diff_labels[key])) + '\nStd = ' + str(np.std(diff_labels[key]))
        plt.text(0.99,0.99,cur_text,horizontalalignment='right',verticalalignment='top',transform=plt.gca().transAxes)
        plt.savefig(outdir + 'sensitivity_' + str(index) + '.png')

if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("region")
    args = p.parse_args()

    main(int(args.region))
