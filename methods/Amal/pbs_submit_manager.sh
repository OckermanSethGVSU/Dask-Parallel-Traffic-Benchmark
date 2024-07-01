
nodes=(1 2 4 8 16 32)
gpus=(4)

for num_nodes in "${nodes[@]}"
do
    for num_gpus in "${gpus[@]}"
    do  

        total_nodes=$((num_nodes + 1))

        target_file="actual_submit_nodes_${num_nodes}_${num_gpus}.sh"
        rm $target_file
        echo "#!/bin/bash" >> $target_file
        echo "#PBS -l select=${total_nodes}:system=polaris" >> $target_file
        echo "#PBS -l place=scatter" >> $target_file
        echo "#PBS -l walltime=00:10:00" >> $target_file
        echo "#PBS -l filesystems=home" >> $target_file

        echo "#PBS -q debug" >> $target_file
        # echo "#PBS -q prod" >> $target_file
        # echo "#PBS -q preemptable" >> $target_file
        # echo  "#PBS -r y" >> $target_file

        echo "#PBS -A radix-io" >> $target_file
        echo "#PBS -o train_${num_nodes}_${num_gpus}.out" >> $target_file
        echo "#PBS -e train_${num_nodes}_${num_gpus}.err" >> $target_file


        echo "nodes=${num_nodes}" >> $target_file
        echo "num_worker=${num_gpus}" >> $target_file
        
        cat submit.sh >> $target_file

        echo "mv ../train_${num_nodes}_${num_gpus}.out ." >> $target_file
        echo "mv ../train_${num_nodes}_${num_gpus}.err ." >> $target_file


        qsub $target_file
        exit 1
        #PBS -o train.out
#PBS -e train.err
        break
    done
done