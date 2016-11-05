#!/usr/local/bin/sh

for i in $(seq 0 10)
do
	#echo "$i"
    batchPP -t 16 ./crop_netcdf_24hours.py $i
    batchPP -t 16 ./crop_netcdf_24hours_tmp_144.py $i
    batchPP -t 16 ./crop_netcdf_24hours_tmp_232.py $i
    batchPP -t 16 ./crop_netcdf_24hours_benchmark.py $i
    batchPP -t 16 ./crop_netcdf_24hours_benchmark_tmp_144.py $i
    batchPP -t 16 ./crop_netcdf_24hours_benchmark_tmp_232.py $i
	#batchPP -t 24 -p pp-long ./crop_netcdf_24hours_II.py $i
	#./chunk_list.py $i > "outfile$i.txt"
done
