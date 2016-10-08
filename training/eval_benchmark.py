#!/usr/bin/env python

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import glob
import numpy as np
import re
import matplotlib.pyplot as plt

def construct_np_arrays(runs):
    """
    Numpy files of the runs given are loaded and model/label sets are
    concatenated
    :param runs: list of model runs (e.g. of training set)
    :returns: model_data_np (X), labels_np (Y)
    """
    model_data_np = np.empty((0,2),int)
    labels_np = np.empty((0,2),int)

    for item in runs:
        # load model data
        cur_model = np.load(item)
        model_data_np = np.concatenate((model_data_np,cur_model),axis = 0)

        # load matching labels
        cur_index = re.match('.*([0-9]{8}_[0-9]{3}.*)',item).group(1)
        cur_label = np.load(datdir + 'benchmark_labels_' + cur_index)
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


# Data loading & preprocessing
datdir = '/home/silviar/Dokumente/Training_set/'
benchmark_files = sorted(glob.glob(datdir + 'benchmark_data_*'))

X,Y = construct_np_arrays(benchmark_files)
X_pred = replace_entries(X)
Y_val = replace_entries(Y)

# calculate scores
print('Accuracy: ', accuracy_score(Y_val,X_pred))
print('Confusion matrix: ', confusion_matrix(Y_val,X_pred))

