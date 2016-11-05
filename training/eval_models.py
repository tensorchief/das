#!/usr/bin/env python

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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


 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
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

    thresh = 0.85*cm.max()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calculate_scores(valid_y, pred_y, valid_Y_np, pred_prob):
    """
    Calculates Accuracy, Confusion matrix, classification report and roc/auc
    :param valid_y: true label (as a 1D-list)
    :param pred_y: predicted label (as a 1D-list)
    :param valid_Y_np: true label (as a 1-D numpy array)
    :param pred_prob: predicted probability (as a 1-D numpy array)
    :return: false positive rate, true positive rate, area under curve
    """
    print('Accuracy: ', accuracy_score(valid_y,pred_y))
    print('Confusion matrix: ', confusion_matrix(valid_y,pred_y))
    print(classification_report(valid_y,pred_y,target_names = ['p<10mm','p>=10mm']))
    
    fpr, tpr, _ = roc_curve(valid_Y_np,pred_prob)
    return fpr, tpr, auc(fpr,tpr)


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

    # do predictions
    pred = model.predict(X)
    y_score = np.array(pred)
    predicted_label = [round(item) for item in y_score[:,0]]
    valid_label = [item for item in Y[:,0]]
    
    print('***** CNN *****')
    # calculate scores
    fpr, tpr, roc_auc = calculate_scores(valid_label, predicted_label,\
                                        Y[:,0], y_score[:,0])
    
    # bootstrap to get standard deviation of auc score
    test_indices = np.random.choice(len(predicted_label),(1000,50),replace=True)
    auc_list = list()
    for index in test_indices:
        fpr_tmp, tpr_tmp, _ = roc_curve(Y[index,0],y_score[index,0])
        auc_list.append(auc(fpr_tmp,tpr_tmp))
    print("Mean AUC: ", np.nanmean(auc_list))
    print("Std. Deviation: ", np.nanstd(auc_list))
        
    # get benchmark
    benchmark_files = sorted(glob.glob(datdir + 'benchmark_prob_*_' + str(region) + '.npy'))
    pred_bench,Y_bench = construct_np_arrays(datdir,benchmark_files,np.empty((0,2),int))
    predicted_label_bench = [round(item) for item in pred_bench[:,0]]
    valid_label_bench = [item for item in Y_bench[:,0]]

    print('***** BENCHMARK *****')
    # calculate scores
    fpr_bench, tpr_bench, roc_auc_bench = calculate_scores(valid_label_bench,\
                                         predicted_label_bench, Y_bench[:,0], pred_bench[:,0])

    # plot roc curves
    outdir = '/home/silviar/Dokumente/Abschlussarbeit/training/models/'
    plt.figure()
    plt.plot(fpr,tpr, color = 'deeppink', lw = 2, label = 'CNN (area = %0.2f)'\
            % roc_auc)
    plt.plot(fpr_bench,tpr_bench, color = 'purple', lw = 2, \
            label = 'COSMO-E (area = %0.2f)' % roc_auc_bench)
    plt.plot([0,1],[0,1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve P(p $\geq$ 10mm), region ' + str(region))
    plt.legend(loc='lower right')
    plt.savefig(outdir + 'roc_' + str(region) + '.png')

    # plot confusion matrices
    cm = confusion_matrix(valid_label,predicted_label,labels=[1,0])
    plt.figure()
    plot_confusion_matrix(cm,['p $\geq$ 10mm','p < 10mm'],normalize=True,title='Normalized confusion matrix')
    plt.savefig(outdir + 'confusion_matrix_norm_' + str(region) + '.png')

    plt.figure()
    plot_confusion_matrix(cm,['p $\geq$ 10mm','p < 10mm'])
    plt.savefig(outdir + 'confusion_matrix_' + str(region) + '.png')

    cm = confusion_matrix(valid_label_bench,predicted_label_bench,labels=[1,0])
    plt.figure()
    plot_confusion_matrix(cm,['p $\geq$ 10mm','p < 10mm'],normalize=True,title='Normalized confusion matrix')
    plt.savefig(outdir + 'benchmark_confusion_matrix_norm_' + str(region) + '.png')

    plt.figure()
    plot_confusion_matrix(cm,['p $\geq$ 10mm','p < 10mm'])
    plt.savefig(outdir + 'benchmark_confusion_matrix_' + str(region) + '.png')


if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("region")
    args = p.parse_args()

    main(int(args.region))
