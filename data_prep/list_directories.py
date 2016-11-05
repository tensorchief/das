#!/apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import glob

datdir = '/scratch/rsb/Test/'

runs = glob.glob(datdir + '*COSMO_E')
for item in sorted(runs):
    print(item)

"""
with open('training_set.txt') as infile:
    training = infile.readlines()

training_set = [item.strip() for item in training]

with open('test_set.txt','w') as outfile:
    for run in runs:
        if run not in training_set:
            outfile.write(str(run) + '\n')
"""
