#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import re
import os
import argparse

datdir = '/scratch/rsb/Test/'

# get name pattern as argument
p = argparse.ArgumentParser()
p.add_argument("pattern")
args = p.parse_args()
pattern = args.pattern

# list all directories with given pattern
folders = [item for item in os.listdir() if re.match('[0-9]{8}_COSMO_E',item)]
#folders = ['16011512_COSMO_E']

# go through each directory & check if there are files matching the given pattern
for folder in folders:
	print('checking ' + folder)
	files = [file for file in os.listdir(folder) \
        	if re.match('.+' + pattern + '.*',file)]
	# check if the files are large enough
	for file in files:
		curfile = os.path.abspath(os.path.join(datdir,os.path.join(folder,file))) 
		#print(curfile)
		os.remove(curfile)
