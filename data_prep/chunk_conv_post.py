#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python
import os
import re
from subprocess import Popen
import math
#import threading
import argparse

def get_subdir(directory):
	'''
	returns list of sub-subdirectories of the input variable directory
	'''
	subdir = list()
	for dir in os.listdir(directory):
		subdir.extend([directory+dir+'/'+item+'/grib/' for item in os.listdir(directory+dir) if re.match('[0-9]{8}_[0-9]{3}',item)])
	return subdir


def print_namelist(path,template):
	'''
	writes a fieldextra namelist that converts files of pattern
	'''
	# get pattern of file names
	if re.match('.+EXP_TST.+',path):
		pattern = 'verif_<mmm>_<HHH>'
	else:
		pattern = 'ceffsurf<HHH>_<mmm>'

	# preallocate in and outfile names
	date = re.match('.+/([0-9]{8}).+',path).group(1)
	infile = path + pattern
	outfile_nc = date + '_' + pattern
	
	# write fieldextra file according to template
	filename_fx = 'fieldextra_'+date
	outfile_fx = open(filename_fx,'w')
	for line in template:
		if 'in_file' in line:
			outfile_fx.write(line.strip() + ' "' + infile + '"\n')
		elif 'out_file' in line:
			outfile_fx.write(line.strip() + ' "' + outfile_nc + '.nc", out_type = "NETCDF",\n')
		else:
			outfile_fx.write(line)
	outfile_fx.close()
	
	# return filename of fieldextra file
	return filename_fx
	

def convert(namelist):
	'''
	converts grib file to netcdf and stores it in current directory
	'''
	p = Popen(['/oprusers/owm/opr/abs/fieldextra', namelist])
	p.wait()
	os.remove(namelist)

# set number of chunks for parallel processing
num_chunks = 1

# read files that need re-doing
redo_id = open('log.txt', 'r')
redo_files = redo_id.readlines()
redo_id.close()

redo_dates = [re.match('([0-9]{8}).+',date).group(1) for date in redo_files]

# get fieldextra pattern
fx_file = 'fieldextra_nl'
file_obj = open(fx_file,'r')
fx_lines = file_obj.readlines()
file_obj.close()

# get locators
rootdir = ['/store/s83/tsm/EXP_TST/590/', '/store/s83/owm/COSMO-E/']
locator = list()

for dir in rootdir:
	locator.extend(get_subdir(dir))

# remove done ones and duplicates
locator_new = [locator[id] for id in range(1,len(locator)) if any(substring in locator[id] for substring in redo_dates)]

# conversion
for item in locator_new:
	# create namelist
	#item = locator_new[10]
	namelist_fx = print_namelist(item,fx_lines)
	# do the conversion
	convert(namelist_fx)
