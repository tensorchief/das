#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python
import argparse
import numpy as np
from scipy import spatial
import math
import glob
import os
import datetime


def get_centroid(cur_region):
    """
    Calculates centroid of a warning region
    input: 		cur_region (int): 3-digit index of region of interest
    returns: 	np.array([longitude, latitude])
    """
    # get coordinates of Region
    with open('polygon.txt', 'r') as coords_id:
        cur_coords = coords_id.readlines()

    num = [line.split(',')[0] for line in cur_coords]
    lonlat = np.array([[float(line.split(',')[1]), float(line.split(',')[2].strip())]
                       for line_id, line in enumerate(cur_coords) if num[line_id] == str(cur_region)])

    # calculate centroid of Region
    # APPROXIMATION! Assume that Earth is flat (error of approx. 2km E-W)
    return np.mean(lonlat, axis=0)


def get_field_netcdf(fields, ncfile):
    """
    Extracts the given fields from a netcdf file
    input:		fields: list of strings that denote the fieldnames
                ncfile: full file path
    returns:	list of numpy arrays
    """
    from scipy.io import netcdf

    with netcdf.netcdf_file(ncfile, 'r') as f:
        return [f.variables[item][:] for item in fields]


def get_subsample(length, width, cur_row, cur_col, field):
    """
    Gets a subsample of a 2D-gridded field around a center point
    input:	length,width:       integers defining the size of the sample
            cur_row,cur_col:    coordinates of the center point (int)
            field:              2D field from which subsample is taken (list)
    returns: length x width numpy array
    """
    row_start = cur_row - (length / 2)
    col_start = cur_col - (width / 2)

    sample = np.empty((0, width), float)
    for j in range(length):
        cur_row = field[row_start + j][col_start:col_start + width]
        sample = np.vstack([sample,cur_row])
    return sample

def classify_values(threshold,field):
    """
    goes through a field of data and marks all entries larger than threshold as 1 (else 0)
    :param threshold:  = Event threshold
    :param field: Field to be classified
    :return: classified field
    """
    field[field < threshold] = 0
    field[field >= threshold] = 1
    return field

def rearrange_coords(lat, lon):
    """
    Rearranges the coordinates and values.

    :param lat: Latitude of the coordinate
    :param lon: Longitude of the coordinate
    :param threshold: Event threshold
    :param totprec: Total precipitation
    :return:
    """
    coords = np.empty((0, 2), float)

    for i in range(len(lon)):
        for j in range(len(lon[0])):
            coords = np.append(coords, [[lon[i][j], lat[i][j]]], axis=0)

    return coords


def build_member_list(folder,time):
    """
    Get a list of all member files in folder for the given time

    :param folder: The folder
    :param time: a three-digit string giving the time since initialisation
    :return members: a list of all member files at the given time
    """
    members = sorted(glob.glob(folder + '/*_ceffsurf' + time + '_*.nc'))
    if not members:
        members = sorted(glob.glob(folder + '/*_verif_*_' + time + '.nc'))
    return members


def main(chunk, threshold=10):
    """
    Main run loop.

    :param region: The region to calculate.
    :param threshold: The threshold to differentiate between events and non-events (below = 0, above = 1).
    :return:
    """
    region = 310
    num_chunks = 15
    centroid = get_centroid(region)

    # build member paths
    datdir = '/scratch/rsb/Test/16060800_COSMO_E/'
    member_list = sorted(glob.glob(datdir + '*f024_*.nc'))

    # get coordinates from first netcdf
    [lon, lat] = get_field_netcdf(['lon_1', 'lat_1'],member_list[0])
   
    # rearrange coordinates
    coords= rearrange_coords(lat, lon)
    
    # find grid-point in netcdf closest to centroid
    distance, index = spatial.KDTree(coords).query(centroid)

    # take a 28x28 sample around centroid (14 values in each direction)
    cur_row = math.floor(index / len(lon[0]))
    cur_col = index % len(lon[0])

    # get all runs
    runs = glob.glob('/scratch/rsb/Test/*COSMO_E')
    #with open('ls.txt','r') as infile:
    #    runs = infile.read().split('\n')
    ids = math.ceil(len(runs)/num_chunks)
    chunks = [runs[i:i+ids] for i in range(0,len(runs),ids)]
    
    for run in chunks[chunk]:
        for step in range(0,24):
                    
            # build member_listi
            timestep = '0' + str(24 + step)
            member_list = build_member_list(run,timestep)
            substractor = '00' + str(step) if step <10 else '0' + str(step)
            substractor_list = build_member_list(run,substractor)
            
            # get subsample (28x28)
            prec_all = np.empty((28,28,0),int)

            for num,member in enumerate(member_list):

                # get precipitation field 
                totprec = get_field_netcdf(['TOT_PREC'],member)

                # get subsample of current member
                totprec_sample = get_subsample(28, 28, cur_row, cur_col, totprec[0][0][0])
               
                # do the same for substractor
                if step > 0:
                    subprec = get_field_netcdf(['TOT_PREC'],substractor_list[num])
                    subprec_sample = get_subsample(28, 28, cur_row, cur_col, subprec[0][0][0])

                    # get the 24 hour sum
                    prec_24h = np.subtract(totprec_sample,subprec_sample)
                else:
                    prec_24h = totprec_sample

                # classify subsample
                prec_sample = classify_values(threshold,prec_24h)

                # stack to get 3d-array ...
                prec_all = np.dstack((prec_all,prec_sample))
                
            # Write to file
            np.save(os.path.join(run,'tensor_' + timestep + '_' + str(region) + '.npy'),prec_all)
            with open('log'+str(chunk)+'.txt','a') as logfile:
                logfile.write('done with ' + timestep + '\n')
        with open('log'+str(chunk)+'.txt','a') as logfile:
            logfile.write('done with ' + run + '\n')

    # /TODO: get corresponding training labels

    # /TODO: rearrange training labels

    # /TODO: write training labels to file

    # /TODO: Sampling (Train & Test data)

    print('done')


if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("Chunk")
    args = p.parse_args()

    main(int(args.Chunk))
