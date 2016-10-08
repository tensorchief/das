#!/usr/local/bin/sh

for i in $(seq 0 14)
do
	#echo "$i"
    batchPP -t 72 -p pp-long ./crop_netcdf_24hours.py $i
	#batchPP -t 24 -p pp-long ./crop_netcdf_24hours_II.py $i
	#./chunk_list.py $i > "outfile$i.txt"
done
