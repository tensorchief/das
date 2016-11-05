#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import glob
import os
from shutil import copyfile
import re
import argparse

p = argparse.ArgumentParser()
p.add_argument("region")
args = p.parse_args()
region = args.region

#with open('training_set.txt', 'r') as infile:
#    training_data = infile.readlines()
with open('test_set.txt','r') as infile:
    training_data = infile.readlines()
training_runs = [item.strip() for item in training_data]

all_runs = glob.glob('/scratch/rsb/Test/*COSMO_E')
destination = '/users/rsb/Training_set/'

for run in all_runs:
    #if run not in training_runs:
    if run in training_runs:
        files = glob.glob(run.strip() + '/training_*' + str(region) + '*.npy')
        for item in files:
            print(item)
            file_name = re.match('.*/(training_.*\.npy)',item).group(1)
            print(file_name)
            copyfile(item,destination+file_name)
        files = glob.glob(run.strip() + '/benchmark_*_*_' + str(region) + '*.npy')
        for item in files:
            print(item)
            file_name = re.match('.*/(benchmark_.*\.npy)',item).group(1)
            print(file_name)
            copyfile(item,destination+file_name)
