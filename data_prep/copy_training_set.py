#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import glob
import os
from shutil import copyfile
import re

with open('training_set.txt', 'r') as infile:
    training_runs = infile.readlines()

print(training_runs)
destination = '/users/rsb/Training_set/'

for run in training_runs:
    files = glob.glob(run.strip() + '/benchmark_labels*')
    for item in files:
        print(item)
        file_name = re.match('.*/(benchmark_labels.*\.npy)',item).group(1)
        print(file_name)
        copyfile(item,destination+file_name)
