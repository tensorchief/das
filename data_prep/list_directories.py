#!/apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import glob

datdir = '/scratch/rsb/Test/'

runs = glob.glob(datdir + '*COSMO_E')

with open('datlist.txt','w') as outfile:
    for run in runs:
        outfile.write(str(run) + '\n')
