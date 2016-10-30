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

import argparse
import glob
import numpy as np
import re
import itertools
#import matplotlib
#matplotlib.use('Agg')
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


 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = 0.75*cm.max()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main(region):

    # Data loading & preprocessing
    datdir = '/home/silviar/Dokumente/Test_set/'
    test_files = sorted(glob.glob(datdir + 'training_data_*'))

    X,Y = construct_np_arrays(datdir,test_files,np.empty((0,28,28,21),int))

    model = setup_cnn()

    # load weights
    model.load('/home/silviar/Dokumente/Abschlussarbeit/training/models/' \
                + 'cnn_mnist_stratified_' + str(region))

    # do predictions
    pred = model.predict(X)
    y_score = np.array(pred)
    predicted_label = [round(item) for item in y_score[:,0]]
    valid_label = [item for item in Y[:,0]]
    print(valid_label)
    
    # calculate scores
    print('Accuracy: ', accuracy_score(valid_label,predicted_label))
    print('Confusion matrix: ', confusion_matrix(valid_label,predicted_label))

    # calculate roc curve
    fpr, tpr, _ = roc_curve(Y[:,0],y_score[:,0])
    roc_auc = auc(fpr,tpr)

    """
    # get benchmark
    benchmark_files = sorted(glob.glob(datdir + 'benchmark_prob_*'))
    pred_bench,Y_bench = construct_np_arrays(datdir,benchmark_files,np.empty((0,2),int))
    predicted_label_bench = [item.argmax() for item in pred_bench]
    valid_label_bench = [item.argmax() for item in Y_bench]

    # calculate scores
    print('Accuracy: ', accuracy_score(valid_label_bench,predicted_label_bench))
    print('Confusion matrix: ', confusion_matrix(valid_label_bench,predicted_label_bench))

    # calculate roc_curve
    fpr_bench, tpr_bench, _ = roc_curve(Y_bench[:,0],pred_bench[:,0])
    roc_auc_bench = auc(fpr_bench,tpr_bench)
    """

    outdir = '/home/silviar/Dokumente/Abschlussarbeit/training/models/'
    plt.figure()
    plt.plot(fpr,tpr, color = 'deeppink', lw = 2, label = 'class 1 (area = %0.2f)'\
            % roc_auc)
    #plt.plot(fpr_bench,tpr_bench, color = 'purple', lw = 2, \
    #        label = 'Benchmark (area = %0.2f)' % roc_auc_bench)
    plt.plot([0,1],[0,1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for region ' + str(region))
    plt.legend(loc='lower right')
    plt.savefig(outdir + 'roc_' + str(region) + '.png')

    cm = confusion_matrix(valid_label,predicted_label,labels=[1,0])
    plt.figure()
    plot_confusion_matrix(cm,['p>=10mm','p<10mm'],normalize=True,title='Normalized confusion matrix')
    plt.savefig(outdir + 'confusion_matrix_norm_' + str(region) + '.png')

    plt.figure()
    plot_confusion_matrix(cm,['p>=10mm','p<10mm'])
    plt.savefig(outdir + 'confusion_matrix_' + str(region) + '.png')

if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("region")
    args = p.parse_args()

    main(int(args.region))
