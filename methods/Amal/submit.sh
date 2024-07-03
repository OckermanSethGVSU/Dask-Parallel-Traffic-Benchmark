

module use /soft/modulefiles 
module load conda; conda activate
conda activate dask


export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=./nccl_debug_${nodes}_${num_worker}.log
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

total=$((num_worker * nodes))
NDEPTH=$((32 / num_worker))


cd /eagle/projects/radix-io/sockerman/Dask-Parallel-Traffic-Benchmark/methods/Amal/
DIR=$PWD

DATE=$(date +"%Y-%m-%d_%T")
dir="real_${nodes}_workers_${num_worker}_${DATE}"
mkdir -p $dir

# cp train.out $dir/
# cp train.err $dir/
cp *.py $dir/
cp *.pth $dir/
cp actual_submit_nodes_${nodes}_${num_worker}.sh $dir/
cd $dir



readarray -t all_nodes < "$PBS_NODEFILE"


scheduler_node=${all_nodes[0]}


tail -n +2 $PBS_NODEFILE > worker_nodefile.txt
# launch scheduler
mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node dask scheduler --scheduler-file cluster.info &
scheduler_pid=$!

while ! [ -f cluster.info ]; do
    sleep 1
    echo .
done

nvidia-smi > /dev/null
echo "Scheduler launched"

# launch workers
mpiexec -n $total --ppn $num_worker -d ${NDEPTH} --exclusive --hostfile worker_nodefile.txt dask worker --scheduler-file cluster.info  &
worker_pid=$!

echo "$total workers launched" 

# launch client and start computation

# BAY
# mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node `which python3` distributed_ddp_train.py --adj_data /home/treewalker/Dask-Parallel-Traffic-Benchmark/methods/pol_DGCRN/data/sensor_graph/adj_mx_bay.pkl --data /home/treewalker/Dask-Parallel-Traffic-Benchmark/methods/pol_DGCRN/data/PEMS-BAY --num_nodes 325 --runs 1 --epochs 110 --print_every 1 --batch_size 64 --tolerance 100 --expid DGCRN_pemsbay  --cl_decay_steps 5500 --rnn_size 96 --npar $total

# LA
# mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node `which python3` distributed_ddp_train.py --adj_data ../data/sensor_graph/adj_mx.pkl  --num_nodes 207 --runs 1  --epochs 5 --print_every 1 --batch_size 64 --tolerance 100  --cl_decay_steps 4000 --npar $total --mode dist

 
# CALI (big)
# mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node `which python3` distributed_ddp_train.py --adj_data ../data/cali_adj_mat.pkl --num_nodes 11160 --epochs 30 --print_every 1 --rnn_size 2 --batch_size 8 --npar $total 
# mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node `which python3` distributed_ddp_train.py --adj_data ../data/cali_adj_mat.pkl --num_nodes 2790 --epochs 30 --print_every 1 --rnn_size 2 --batch_size 64 --npar $total 

# CALI (2GB)
mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node `which python3` distributed_ddp_train.py --adj_data ../LA_ALL_2018/adj_mat.pkl --data ../LA_ALL_2018/speed.h5 --num_nodes 2716 --epochs 30 --print_every 1 --rnn_size 2 --batch_size 16 --npar $total --mode dist 
# --load_path model_7.pth
# mpiexec -n 1 --ppn 1 -d ${NDEPTH} --exclusive --hosts $scheduler_node `which python3` old_with_ddp.py --adj_data ../LA_ALL_2018/adj_mat.pkl --num_nodes 2716 --epochs 30 --print_every 1 --rnn_size 2 --batch_size 64 --npar $total 
client_pid=$!

wait

mv ../train_${nodes}_${num_worker}.out . 
mv ../train_${nodes}_${num_worker}.err . 


total_time=$(grep 'total_time' stats.txt | awk '{print $2}')
pre_pros_time=$(grep 'pre_processing_time' stats.txt | awk '{print $2}')
training_time=$(grep 'training_time' stats.txt | awk '{print $2}')
opt_loss=$(grep 'opt_loss' stats.txt | awk '{print $2}')
opt_rmse=$(grep 'opt_rmse' stats.txt | awk '{print $2}')
opt_mape=$(grep 'opt_mape' stats.txt | awk '{print $2}')

line="${nodes},${num_worker}, ${total_time}, ${pre_pros_time}, ${training_time}, ${opt_mape}, ${opt_rmse}, ${opt_loss}"

echo $line >> ../overall.csv
# dir -p BAY/



