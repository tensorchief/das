#!/apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import argparse
import glob
import numpy as np
import os
import re
from datetime import datetime, timedelta

def get_region_number(region_3digit):
    """
    Returns a 4-digit region index corresponding to the input 3-digit index
    :param region_3digit: 3-digit index
    :return:    4-digit index
    """
    with open('WARN_REGIONS.txt', 'r', encoding = 'latin1') as regionsfile:
        regionlist = regionsfile.readlines()
    
    numbers = list(); indices = list()
    for item in regionlist:
        numbers.append(item.split()[0])
        indices.append(item.split()[1])

    return int(numbers[indices.index(str(region_3digit))])
    
def get_obs_data(region_number,infile='/scratch/rsb/Test/events_REGEN_24H_10.txt'):
    """
    gets observed data (timestamp + value)
    :param region_number: 4-digit region number
    :param infile:  input file
    :return: list of timestamps and observed values in region
    """
    with open(infile,'r') as obsfile:
        obslist = obsfile.readlines()
    
    reg = list(); timestamps = list(); obs = list()
    for item in obslist:
        if item.split()[0] == str(region_number):
            timestamps.append(datetime.strptime(item.split()[1],'%Y%m%d%H%M'))
            obs.append(item.split()[2])

    return timestamps,obs


def get_obs_labels(obs_data,thresh):
    """
    Assigns binary event label: 1 if threshold thresh is exceeded, 0 if not
    :param obs_data: list of observation data (float)
    :param thresh:   numerical threshold
    :return:    list of labels
    """
    labels = list()
    for item in obs_data:
        if item != 'None':
            labels.append([1,0] if float(item) > thresh else [0,1])
        else:
            labels.append([None,None])

    return labels


def main(region, threshold=10):
    """
    Main run loop.
    :param region: Warn region
    :param threshold: Warning threshold that distinguishes between event and non-event
    :return:
    """
    # get 4-digit region number
    number = get_region_number(region)
    # get corresponding observations
    obs_time,obs_num = get_obs_data(number,'/scratch/rsb/Test/events_REGEN_24H_10Zug.txt')
    # label observations
    obs_labels = get_obs_labels(obs_num,threshold)
    

    # assign labels to model runs
    #model_runs = ['/scratch/rsb/Test/15030512_COSMO_E/']
    model_runs = glob.glob('/scratch/rsb/Test/*_COSMO_E')
    for run in model_runs:
        tensors = sorted(glob.glob(run + '/tensor*'+str(region)+'.npy'))
        mod_ini = datetime.strptime(re.match('.+/([0-9]{8})_COSMO_E', run).group(1),'%y%m%d%H')
        timesteps = [int(re.match('.+/tensor_([0-9]{3})_*',tensor).group(1)) for tensor in tensors]

        # construct np.array (24,2)
        labels_run = list()
        for step in timesteps:
            cur_time = mod_ini + timedelta(hours=step)
            if cur_time in obs_time:
                cur_obs = obs_labels[obs_time.index(cur_time)]
            else:
                cur_obs = [None,None]
            print(step,cur_time,cur_obs)
            labels_run.append(cur_obs)

        cur_labels = np.array(labels_run)
        
        # write data
        np.save(os.path.join(run,'labels_' + datetime.strftime(mod_ini,'%y%m%d%H') + '_' + str(region) + '.npy'),cur_labels)
        with open('log.txt','a') as logfile:
            logfile.write('done with ' + run + '\n')
        
if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("Region")
    args = p.parse_args()

    main(int(args.Region))

