#! /apps/escha/UES/RH6.7/easybuild/software/Python/3.5.0-gmvolf-15.11/bin/python
import argparse
import numpy as np
from scipy import spatial
import math
import glob
import os
import datetime
from matplotlib import path

def get_polygon(cur_region):
    """
    Returns a list of (lon,lat) tuple pairs marking the border of cur_region
    :param cur_region:  3-digit region index (int)
    :return:    list of (lon,lat) tuple pairs marking region border
    """
    with open('polygon.txt','r') as coords_id:
        cur_coords = coords_id.readlines()

    return [(float(item.strip().split(',')[1]),float(item.strip().split(',')[2]))\
        for item in cur_coords\
        if item.strip().split(',')[0] == str(cur_region)]


def in_polygon(cur_point, cur_polygon):
    """
    Returns a true or false np.array depending on whether cur_point is inside the polygon given by cur_region
    :param cur_point:   a (lon,lat) tuple pair
    :param cur_polygon:  list of (lon,lat) tuple pairs marking region border
    :return:    boolean np.array
    """
    p = path.Path(cur_polygon)
    return p.contains_points([cur_point])


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
    :return: lon,lat coordinate pairs
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

def get_subsample(index_list, field):
    """
    returns field values at index points given in index_list
    :param index_list: list of integers denoting field coordinates
    :param field: gridded field to be filtered
    :return:    numpy array of field values
    """

    field_value = list()
    for index in index_list:
        row = math.floor(index / len(field[0]))
        col = index % len(field[0])
    
        field_value.append(field[row][col])
    return np.array(field_value)


def main(chunk, threshold=10):
    """
    Main run loop.

    :param region: The region to calculate.
    :param threshold: The threshold to differentiate between events and non-events (below = 0, above = 1).
    :return:
    """
    region = 310
    prob_crit = 0.7

    # build member paths
    datdir = '/scratch/rsb/Test/16060800_COSMO_E/'
    member_list = sorted(glob.glob(datdir + '*f024_*.nc'))

    # get coordinates from first netcdf
    [lon, lat] = get_field_netcdf(['lon_1', 'lat_1'],member_list[0])
   
    # rearrange coordinates
    coords= rearrange_coords(lat, lon)
    
    # get polygon of current region
    polygon = get_polygon(region)

    # get indices of points that are in current polygon
    indices = list()
    for num,pair in enumerate(coords):
        if in_polygon(pair,polygon):
            indices.append(num)

    # get all runs
    #runs = glob.glob('/scratch/rsb/Test/*COSMO_E')
    with open('ls.txt','r') as infile:
        runs = infile.read().split('\n')
    num_chunks = 1
    ids = math.ceil(len(runs)/num_chunks)
    chunks = [runs[i:i+ids] for i in range(0,len(runs),ids)]
    
    for run in chunks[chunk]:
        prec_all = np.empty((0,2), int)
        
        for step in range(0,24):

            # build member_listi
            timestep = '0' + str(24 + step)
            member_list = build_member_list(run,timestep)
            substractor = '00' + str(step) if step <10 else '0' + str(step)
            substractor_list = build_member_list(run,substractor)
            
            # get subsample (28x28)
            class_all = list()
            for num,member in enumerate(member_list):
                
                # get precipitation field
                totprec = get_field_netcdf(['TOT_PREC'],member)
                subprec = get_field_netcdf(['TOT_PREC'],substractor_list[num])

                # get subsample of current member
                totprec_sample = get_subsample(indices, totprec[0][0][0])
                
                if step > 0:
                    subprec = get_field_netcdf(['TOT_PREC'],substractor_list[num])
                    subprec_sample = get_subsample(indices, subprec[0][0][0])

                    # get the 24 hour sum
                    prec_24h = np.subtract(totprec_sample,subprec_sample)
                else:
                    prec_24h = totprec_sample
                
                # calculate 80th percentile
                prec_value = np.percentile(prec_24h,80)
                class_all.append(1 if prec_value >= threshold else 0)

            # check if probability exceeds critical probability
            events = [item for item in class_all if item == 1]
            label = [1,0] if len(events)/float(len(class_all)) >= prob_crit else [0,1]

            # collect labels
            prec_all = np.vstack((prec_all,label))
            
            with open('log'+str(chunk)+'.txt','a') as logfile:
                logfile.write('done with ' + timestep + '\n')

        # Write to file
        np.save(os.path.join(run,'benchmark_' + str(region) + '.npy'),prec_all)
        
        with open('log'+str(chunk)+'.txt','a') as logfile:
            logfile.write('done with ' + run + '\n')

    print('done')


if __name__ == "__main__":
    # get Region
    p = argparse.ArgumentParser()
    p.add_argument("Chunk")
    args = p.parse_args()

    main(int(args.Chunk))
