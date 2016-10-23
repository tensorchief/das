#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import numpy as np
import glob
import os
import re

region = 144

# list all model runs
runs = glob.glob('/scratch/rsb/Test/*COSMO_E')
#runs = ['/scratch/rsb/Test/15030512_COSMO_E/']

# go through each model run
for run in runs:
    with open('log.txt','a') as logfile:
        logfile.write(run + '\n')
    # get run identifier
    identifier = re.match('.+/([0-9]{8})_COSMO_E',run).group(1)
    
    # check obs labels for missing values
    try:
        obs = np.load(glob.glob(os.path.join(run, 'labels_*_'+str(region)+'.npy'))[0])
        bench = np.load(glob.glob(os.path.join(run, 'benchmark_'+str(region)+'.npy'))[0])
        
        # remove missing time steps
        indices_delete = list()
        
        for index,item in enumerate(obs):
            if not np.any(item):
                # remove invalid timesteps from obs
                indices_delete.append(index)
        
        for index in sorted(indices_delete, reverse=True):
            obs = np.delete(obs,(index),axis=0)
            bench = np.delete(bench,(index),axis=0)
        
        # save new arrays
        np.save(os.path.join(run,'benchmark_data_' + identifier + '_' + str(region) + '.npy'), bench)
        np.save(os.path.join(run,'benchmark_labels_' + identifier + '_' + str(region) + '.npy'), obs)    
    except:
        with open('error_log.txt','a') as logfile:
            logfile.write('could not rearrange ' + run + '\n')
        pass
# split model runs into training & test set

# remove surrounding runs

# stack training & test set (-1,28,28,21) & (-1,2)

# save as npy files
