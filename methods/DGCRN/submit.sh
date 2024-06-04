#!/bin/bash 

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:05:00
#PBS -l filesystems=home
#PBS -q debug
#PBS -A radix-io


#PBS -o hello_world.out
#PBS -e hello_world.err


DIR=$PWD
echo $DIR

python3 /home/treewalker/Dask-Parallel-Traffic-Benchmark/methods/DGCRN/hello_world.py



