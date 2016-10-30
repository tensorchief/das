#!/usr/bin/env python

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow

import glob
import numpy as np
import re
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

def construct_np_arrays(runs, model_data_np):
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
        cur_tag = re.match('.*/([a-z]+)_data_[0-9]{8}_[0-9]{3}.*',item).group(1)
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
    #network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    #network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                 loss='categorical_crossentropy', name='target')
    return tflearn.DNN(network, tensorboard_verbose=0)


region = 310

# Data loading & preprocessing
datdir = '/home/silviar/Dokumente/Test_set/'
test_files = sorted(glob.glob(datdir + 'training_data_*'))

X,Y = construct_np_arrays(test_files,np.empty((0,28,28,21),int))

model = setup_cnn()

# load weights
model.load('/home/silviar/Dokumente/Abschlussarbeit/training/models/' \
            + 'cnn_mnist_stratified_' + str(region))

# do predictions
pred = model.predict(X)
predicted_label = [item.index(max(item)) for item in pred]
valid_label = [item.argmax() for item in Y]
y_score = np.array(pred)

# get benchmark
benchmark_files = sorted(glob.glob(datdir + 'benchmark_data_*'))
pred_bench,Y_bench = construct_np_arrays(benchmark_files,np.empty((0,2),int))

predicted_bench = [item.argmax() for item in pred_bench]
valid_bench = [item.argmax() for item in Y_bench]

# do evaluations
#print('ROC/AUC: ',tflearn.objectives.roc_auc_score(pred,valid_label))
print('ROC/AUC: ',roc_auc_score(valid_label,predicted_label))

# calculate scores
print('Accuracy: ', accuracy_score(valid_label,predicted_label))
print('Confusion matrix: ', confusion_matrix(valid_label,predicted_label))

# calculate roc curve
fpr = list()
tpr = list()
#fpr, tpr, _ = roc_curve(Y.ravel(),y_score.ravel())
#roc_auc = auc(fpr,tpr)

fpr,tpr,_ = roc_curve(valid_label,predicted_label)
roc_auc = auc(fpr,tpr)

fpr_bench,tpr_bench,_ = roc_curve(valid_bench,predicted_bench)
roc_auc_bench = auc(fpr_bench,tpr_bench)

plt.figure()
plt.plot(fpr,tpr, color = 'deeppink', lw = 2, label = 'ROC curve (area = %0.2f)'\
        % roc_auc)
plt.plot(fpr_bench,tpr_bench, color = 'purple', lw = 2, \
        label = 'Benchmark (area = %0.2f)' % roc_auc_bench)
plt.plot([0,1],[0,1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for region ' + str(region))
plt.legend(loc='lower right')
plt.show()
