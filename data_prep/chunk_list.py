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
			outfile_fx.write(line.strip() + ' "' + outfile_nc + '"\n')
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
	'''
	for file in files:
		# get output file name
		match = re.match('.+/([0-9]{8})_[0-9]{3}.+/('+pattern+')', file)
		filename = match.group(1) + '_' + match.group(2)
		tmp = filename + '_tmp'
		# extract TOT_PRec
		p1 = Popen(re.split(' ', 'fxfilter -s TOT_PREC '+ file + ' -o ' + tmp))
		p1.wait()
		# crop fields
		p2 = Popen(re.split(' ', 'fxcrop -i 23,560 -j 23,368 '+ tmp))
		p2.wait()
		# #convert to nc
		p3= Popen(re.split(' ', 'fxconvert nc '+ tmp + '_cropped -o ' + filename + '.nc'))
		p3.wait()

		# remove intermediate files
		for f in os.listdir(os.getcwd()):
			if re.search(tmp,f):
				os.remove(f)	
	'''
# set number of chunks for parallel processing
num_chunks = 24

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

# get runs that have already been calculated
done = sorted([re.match('([0-9]+)_COSMO_E',item).group(1) for item in os.listdir('/scratch/rsb/Test/') if re.match('([0-9]+)_COSMO_E',item)])

# remove done ones and duplicates
locator_new = [locator[id] for id in range(1,len(locator))\
 if not any(re.match('.+' + done_item + '.+', locator[id]) for done_item in done)\
 and re.match('.+([0-9]{8}).+',locator[id]).group(1) != re.match('.+([0-9]{8}).+',locator[id-1]).group(1)]

locator_new.append(locator[0])

# split the runs into num_chunks partitions
ids = math.ceil(len(locator_new)/num_chunks)
chunks = [locator_new[i:i+ids] for i in range(0,len(locator_new),ids)]

# get Id of chunk
p = argparse.ArgumentParser()
p.add_argument("chunk")
args = p.parse_args()
chunk = int(args.chunk)

# conversion
for item in chunks[chunk]:
	# create namelist
	#item = locator_new[10]
	#namelist_fx = print_namelist(item,fx_lines)
	# do the conversion
	#convert(namelist_fx)
	print(item+'\n')
