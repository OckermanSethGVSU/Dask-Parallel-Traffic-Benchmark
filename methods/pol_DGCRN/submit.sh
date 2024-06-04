#!/bin/bash 

#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:05:00
#PBS -l filesystems=home
#PBS -q debug
#PBS -A radix-io


#PBS -o hello_world.out
#PBS -e hello_world.err



num_worker=2

# -d is for cpu cores? 
mpirun -np 1  dask scheduler --scheduler-file cluster.info &
scheduler_pid=$!

while ! [ -f cluster.info ]; do
    sleep 1
    echo -n .
done


# Your commands to be executed in each iteration go here
mpirun -np $num_worker dask worker --scheduler-file cluster.info --local-directory . &
worker_pid=$!

mpirun -np 1 python3 distributed_ddp_train.py --print_every 1 --epochs 3
client_pid=$!


wait $client_pid

kill -9 $scheduler_pid
kill -9 $client_pid
kill -9 $worker_pid

