#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python

import os
import re

fileid = open('log.txt','r')
data = fileid.read()
fileid.close()

lines = data.split('done with')

print(len(lines))

folders = os.listdir('/scratch/rsb/Test/')
runs = [item for item in folders if re.match('.*COSMO_E.*',item)]

print(len(runs))
