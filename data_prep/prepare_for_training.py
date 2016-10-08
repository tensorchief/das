#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import numpy as np
import glob
import os
import re

region = 232

# list all model runs
runs = glob.glob('/scratch/rsb/Test/*COSMO_E')
#runs = ['/scratch/rsb/Test/14111912_COSMO_E']

# go through each model run
for run in runs:
    with open('log.txt','a') as logfile:
        logfile.write(run + '\n')
    # get run identifier
    identifier = re.match('.+/([0-9]{8})_COSMO_E',run).group(1)
    
    # check obs labels for missing values
    obs = np.load(glob.glob(os.path.join(run, 'labels_*_'+str(region)+'.npy'))[0])
    steps = sorted(glob.glob(os.path.join(run, 'tensor*_'+str(region)+'.npy')))

    # load model data
    model_data = list()
    for step in steps:
        cur_data = np.load(step)
        model_data.append(cur_data[np.newaxis,...])

    # remove missing time steps
    indices_delete = list()
    model_data_np = np.empty((0,28,28,21),int)
    try:
        for index,item in enumerate(obs):
            if not np.any(item):
                # remove invalid timesteps from obs
                indices_delete.append(index)
            else:
                # stack valid timesteps of model data
                model_data_np = np.concatenate((model_data_np,model_data[index]),axis=0)
        
        for index in sorted(indices_delete, reverse=True):
            obs = np.delete(obs,(index),axis=0)
        
        # save new arrays
        np.save(os.path.join(run,'training_labels_' + identifier + '_' + str(region) + '.npy'), obs)
        np.save(os.path.join(run,'training_data_' + identifier + '_' + str(region) + '.npy'), model_data_np)
    
    except:
        with open('error_log.txt','a') as logfile:
            logfile.write('could not rearrange ' + run + '\n')
        pass
# split model runs into training & test set

# remove surrounding runs

# stack training & test set (-1,28,28,21) & (-1,2)

# save as npy files
