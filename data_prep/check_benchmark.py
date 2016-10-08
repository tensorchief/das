#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import re
import os
import glob

datdir = '/scratch/rsb/Test/'

# list all directories with given pattern
folders = glob.glob(datdir + '*COSMO_E')

# go through each directory & check if there are 24 files
for folder in folders:
	# check if there is a file called benchmark*
    if not glob.glob(folder + '/benchmark*'):
        print(folder)

