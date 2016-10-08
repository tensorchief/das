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


def main():
    """
    Main run loop.
    """
    region = 232
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

    # get lon & lat for each point in subsample
    lon_sample = get_subsample(28, 28, cur_row, cur_col, lon)
    lat_sample = get_subsample(28, 28, cur_row, cur_col, lat)
    
    # rearrange
    coords_sample = rearrange_coords(lat_sample,lon_sample)

    # write coords to file
    np.savetxt('coords.txt', coords_sample,'%3.5f')


    print('done')


if __name__ == "__main__":
    main()
