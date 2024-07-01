import torch
import numpy as np
import argparse
import time

import setproctitle
import os

import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
import pandas as pd
import sys
import torch.profiler as profiler

import seaborn as sns
import dask
# from dask_saturn import SaturnCluster
from dask.distributed import LocalCluster, wait
from dask.distributed import Client
from distributed.worker import logger

import uuid
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dask_pytorch_ddp import dispatch, results
from dask.distributed import Variable, Lock
from dask.distributed import performance_report
setproctitle.setproctitle("DGCRN@lifuxian")
from torch.utils.data import DataLoader, Dataset

import dask.array as da
import dask.dataframe as dd
from dask.array.lib.stride_tricks import sliding_window_view
import pandas as pd
import numpy as np
import time
from util import *
from trainer import Trainer
from net import DGCRN

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TrainDataset(Dataset):
    def __init__(self,x, y, ycl, device=None):
         self.x = x 
         self.y = y 
         self.ycl = ycl
         self.device = device
    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Fetch the sample at the specified index
        # x_sample = self.x[idx]
        # y_sample = self.y[idx]
        # ycl_sample = self.ycl[idx]
        
        x_sample = self.x[idx].compute()
        y_sample = self.y[idx].compute()
        ycl_sample = self.ycl[idx].compute()
        # print('_____')
        # print(type(x_sample))
        # print(x_sample.dtype)
        # print(type(x_sample[0]))
        # # print(x_sample[0])
        # print("____")
     
        # Convert to PyTorch tensors
      
        x_tensor = torch.from_numpy(x_sample)
        y_tensor = torch.from_numpy(y_sample)
        ycl_tensor = torch.from_numpy(ycl_sample)
       
        # print("converted")
        # Move to the specified device if applicable
        if self.device:
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
            ycl_tensor = ycl_tensor.to(self.device)
        
        return x_tensor, y_tensor, ycl_tensor

class ValDataset(Dataset):
    def __init__(self, x,y, device=None):
         self.x = x
         self.y = y
         self.device = device

    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Fetch the sample at the specified index
        x_sample = self.x[idx].compute()
        y_sample = self.y[idx].compute()
        
        # Convert to PyTorch tensors
        
        x_tensor = torch.from_numpy(x_sample)
        y_tensor = torch.from_numpy(y_sample)
        
        
        # Move to the specified device if applicable
        if self.device:
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
        
        return x_tensor, y_tensor




def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='number of runs')
parser.add_argument('--LOAD_INITIAL',
                    default=False,
                    type=str_to_bool,
                    help='If LOAD_INITIAL.')
parser.add_argument('--TEST_ONLY',
                    default=False,
                    type=str_to_bool,
                    help='If TEST_ONLY.')

parser.add_argument('--tolerance',
                    type=int,
                    default=100,
                    help='tolerance for earlystopping')
parser.add_argument('--OUTPUT_PREDICTION',
                    default=False,
                    type=str_to_bool,
                    help='If OUTPUT_PREDICTION.')

parser.add_argument('--cl_decay_steps',
                    default=2000,
                    type=float,
                    help='cl_decay_steps.')
parser.add_argument('--new_training_method',
                    default=False,
                    type=str_to_bool,
                    help='new_training_method.')
parser.add_argument('--rnn_size', default=64, type=int, help='rnn_size.')
parser.add_argument('--hyperGNN_dim',
                    default=16,
                    type=int,
                    help='hyperGNN_dim.')

parser.add_argument('--device', type=str, default='cuda:1', help='')

parser.add_argument('--cl',
                    type=str_to_bool,
                    default=True,
                    help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',
                    type=int,
                    default=2,
                    help='graph convolution depth')
parser.add_argument('--num_nodes',
                    type=int,
                    default=207,
                    help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')

parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--seq_in_len',
                    type=int,
                    default=12,
                    help='input sequence length')
parser.add_argument('--seq_out_len',
                    type=int,
                    default=12,
                    help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0001,
                    help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')

parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./save/', help='save path')

parser.add_argument('--expid', type=str, default='1', help='experiment id')
parser.add_argument('--scheduler_file_path', type=str, default='1',)

parser.add_argument('--data',
                    type=str,
                    default='/home/treewalker/Dask-Parallel-Traffic-Benchmark/methods/pol_DGCRN/data/PEMS-BAY',
                    help='data path')

parser.add_argument('--adj_data',
                    type=str,
                    default='sensor_graph/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--npar', type=int, default=1, help='prop alpha')



args = parser.parse_args()
# torch.set_num_threads(3)

# os.makedirs(args.save, exist_ok=True)



# device = torch.device(args.device)

def readPD():
    df = pd.read_hdf("../LA_ALL_2018/speed.h5", key="df")
    # df = pd.read_hdf("new_speed.h5", key="data")
    # df = df.astype('float32')
    df.index.freq='5min'  # Manually assign the index frequency
    df.index.freq = df.index.inferred_freq
    return df

    # cols = df.shape[1]
    # return df.iloc[:,:int(cols*0.25)]

# device=None

# scaler = dataloader['scaler']
# npar = 1
# file_path = "/home/seth/Documents/research/Argonne/DCRNN/DATA/speed.h5"
# file_path = "metr-la.h5"
def main(runid):
    
    start_time = time.time()
    dask.config.set(**{'array.slicing.split_large_chunks': True})
    print("start preprocessing...", flush=True)
    
    # prefix = "/home/treewalker/Dask-Parallel-Traffic-Benchmark/methods/pol_DGCRN/"
    
    client = Client(scheduler_file = f"cluster.info")

    # order matters - I am preserving the depencies
    
    
    # with performance_report(filename="dask-report.html"):
    

    # cluster = LocalCluster(n_workers=npar)

    # client = Client(cluster)
    print("Step 0: Reading DF")
    import h5py
    f = h5py.File("../LA_ALL_2018/speed.h5") # HDF5 file
    data1 = da.from_array(f['df']['block0_values'], chunks='auto')
    data2 = da.from_array(f['df']['axis1'], chunks='auto')
    # from dask.delayed import delayed
    # dfs = delayed(readPD)()
    # df = dd.from_delayed(dfs)
    # df = df.repartition(npartitions=10)
     
    num_samples, num_nodes = data1.shape
    data1 = da.expand_dims(data1, axis=-1) 
    
    
    x_offsets = np.sort(np.arange(-11, 1, 1))
    y_offsets = np.sort(np.arange(1, 13, 1))
    

    data2 = (np.datetime64('1970-01-01T00:00') + data2.astype('timedelta64[ns]')) 
    data2 = (data2 - data2.astype("datetime64[D]")) / np.timedelta64(1, "D")
   
    data2 = da.tile(data2,[1, num_nodes, 1]).transpose((2, 1, 0))
    
    # print(np.array_equal(data2.compute(), test2.compute()))
    # data2 = data2.rechunk((data1.chunks))
    
    
    # print("\rStep 1c Starting: Tiling", end="\n", flush=True)
    # memmap_array = da.concatenate([data1, data2], axis=-1)
    memmap_array= da.concatenate([data1, data2], axis=-1)
    
    # print(np.array_equal(memmap_array.compute(), test_array
    
    # num_samples, num_nodes = df.shape

    # num_samples = num_samples.compute()
    
    # x_offsets = np.sort(np.arange(-11, 1, 1))
    # y_offsets = np.sort(np.arange(1, 13, 1))
    
    # print("\rStep 1a Starting: df.to_dask_array", flush=True)
    # data1 =  df.to_dask_array(lengths=True)
    # # print(data1.shape)
    # data1 = da.expand_dims(data1, axis=-1)
    # data1 = data1.rechunk("auto")



    # print("\rStep 1b Starting: Tiling", flush=True)
    # data2 = da.tile((df.index.values.compute() - df.index.values.compute().astype("datetime64[D]")) / np.timedelta64(1, "D"), [1, num_nodes, 1]).transpose((2, 1, 0))
    # data2 = data2.rechunk((data1.chunks))
    
    
    # # print("\rStep 1c Starting: Tiling", end="\n", flush=True)
    # memmap_array = da.concatenate([data1, data2], axis=-1)
    
  
    # del df
    # # print("\rStep 1a Done; Step 1b Starting", flush=True)


   
   
   

    del data1 
    del data2

    
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    total = max_t - min_t

    window_size = 12
    original_shape = memmap_array.shape

    
    # Define the window shape
    window_shape = (window_size,) + original_shape[1:]  # (12, 207, 2)

    # Use sliding_window_view to create the sliding windows
    sliding_windows = sliding_window_view(memmap_array, window_shape).squeeze()
    # time.sleep(15)
    # print(sliding_windows.compute().shape)
    # print(sliding_windows.compute())
    
    x_array = sliding_windows[:total]
    y_array = sliding_windows[window_size:]
    del memmap_array
    del sliding_windows


   


    num_samples = x_array.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    

    x_train = x_array[:num_train]
    y_train = y_array[:num_train]
    ycl_train = y_array[:num_train]

    


    # print("Step 3: Computing Mean and Std-Dev", flush=True)
    mean = x_train[..., 0].mean()
    std = x_train[..., 0].std()
    

    # print("Step 4a: Standardizing Train x Dataset",end="",  flush=True)
    x_train[..., 0] = (x_train[..., 0] - mean) / std
    # x_train = x_train)
    
    
    # print("\rStep 4b: Standardizing Train ycl Dataset",  flush=True)
    ycl_train[..., 0] = (ycl_train[..., 0] - mean) / std
    # ycl_train = ycl_train)
    
   

    x_val = x_array[num_train: num_train + num_val]
    y_val = y_array[num_train: num_train + num_val]



    # print("Step 5: Standardizing Validation Dataset")
    x_val[..., 0] = (x_val[..., 0] - mean) / std
    
    # x_val = x_val)
    print("\rStep 1 Preprocessing data" , flush=True)
    mean, std, x_train, y_train, ycl_train, x_val, y_val = client.persist([mean, std, x_train, y_train, ycl_train, x_val, y_val])
    wait([mean, std, x_train, y_train, ycl_train, x_val, y_val])
    
    # time.sleep(30)
    mean = mean.compute()
    std = std.compute()

    pre_end = time.time()
    print(f"Preprocessing complete in {pre_end - start_time}; Training Starting", flush=True)
    print("X shape: ", x_train.shape )

    
    # x_train = x_train.compute()
    # y_train = y_train.compute()
    # ycl_train = ycl_train.compute()
    
    # x_val = x_val.compute()
    # y_val = y_val.compute()
    # wait([x_train, y_train, ycl_train, x_val, y_val])
    # time.sleep(60)
    del x_array
    del y_array
    

    
    
    

    # args = (x_train, y_train, ycl_train, x_val, y_val)
    for f in ['layer.py', 'net.py', 'util.py', 'net.py', 'trainer.py']:
        client.upload_file(f)
        print(f"{f} uploaded")


    futures = dispatch.run(client, my_train, x_train=x_train, mean=mean, std=std, y_train=y_train, ycl_train=ycl_train, x_val=x_val, y_val=y_val)
    key = uuid.uuid4().hex
    rh = results.DaskResultsHandler(key)
    rh.process_results(".", futures, raise_errors=False)
    end_time = time.time()
   
    client.shutdown()
    
    with open("stats.txt", "a") as file:
        file.write(f"total_time: {end_time - start_time}\n")
        file.write(f"pre_processing_time: {pre_end - start_time}\n")
        file.write(f"training_time: {end_time - pre_end}\n")
    
   

def my_train(x_train=None, y_train=None, ycl_train=None, x_val=None, y_val=None, mean=None, std=None):

    
    
    npar = int(args.npar)
    # return None
    scaler = StandardScaler(mean=mean, std=std)
    train_dataset = TrainDataset(x_train, y_train, ycl_train)
    val_dataset = ValDataset(x_val, y_val)
    
    runid = 1
    num_epochs = args.epochs
    batch_size = args.batch_size

    
    worker_rank = int(dist.get_rank())

    device = f"cuda:{worker_rank % 4}"
    # print("Device: ", device, flush=True)
    
    predefined_A = load_adj(args.adj_data)
    predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]
    # print("setup adj matrix")
    train_sampler = DistributedSampler(train_dataset, num_replicas=npar, rank=worker_rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    print("Train batches: ", len(train_loader))
    train_per_epoch = len(train_loader)
    val_sampler = DistributedSampler(val_dataset, num_replicas=npar, rank=worker_rank)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    print("Val batches: ", len(val_loader))
    val_per_epoch = len(val_loader)
    # print("Setup Samplers" ,flush=True)
    model = DGCRN(args.gcn_depth,
                args.num_nodes,
                device,
                predefined_A=predefined_A,
                dropout=args.dropout,
                subgraph_size=args.subgraph_size,
                node_dim=args.node_dim,
                middle_dim=2,
                seq_length=args.seq_in_len,
                in_dim=args.in_dim,
                out_dim=args.seq_out_len,
                layers=args.layers,
                list_weight=[0.05, 0.95, 0.95],
                tanhalpha=args.tanhalpha,
                cl_decay_steps=args.cl_decay_steps,
                rnn_size=args.rnn_size,
                hyperGNN_dim=args.hyperGNN_dim).to(device)
    model = DDP(model, device_ids=[device], gradient_as_bucket_view=True)
    # print("Moved model to device", flush=True)
    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                    args.step_size1, args.seq_out_len, scaler, device,
                    args.cl, args.new_training_method)
    worker_rank = int(dist.get_rank()) 

    his_loss = []
    his_rmse = []
    his_mape = []

    val_time = []
    train_time = []
    minl = 1e5
    epoch_best = -1
    tolerance = args.tolerance
    count_lfx = 0
    batches_seen = 0


    print("about to start epochs", flush=True)

   
    # train_start = time.time()
    for epoch in range(1, num_epochs + 1):
        if worker_rank == 0:
            print("Epoch: ", epoch, flush=True)
            print("******************************************************")
        train_sampler.set_epoch(epoch)
        
        
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
    
        
        for i, (x, y, ycl) in enumerate(train_loader):
            if worker_rank == 0:
                print(f"\r Epoch {epoch}: Train {((i / train_per_epoch) * 100):.2f}%, shape: {x.shape}", flush=True, end="")
            trainx = torch.Tensor(x.float()).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y.float()).to(device)
            trainy = trainy.transpose(1, 3)

            trainycl = torch.Tensor(ycl.float()).to(device)
            trainycl = trainycl.transpose(1, 3)
            # print("tensors moved", flush=True)
            # print(f"worker rank: {worker_rank} epoch: {epoch} batch: {batches_seen}")
            
    
            metrics = engine.train( trainx,
                                    trainy[:, 0, :, :],
                                    trainycl,
                                    idx=None,
                                    batches_seen=batches_seen)
            
            # unused_params = [name for name, param in model.named_parameters() if param.grad is None]
            # print("Unused Parameters:", unused_params)


            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            # print(batch_x.shape, batch_y.shape, batch_ycl.shape)
    
        
        
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        
    
        s1 = time.time()
        for i, (x, y) in enumerate(val_loader):
            if worker_rank == 0:
                print(f"\r Epoch {epoch}: Val {((i / val_per_epoch) * 100):.2f}%", flush=True, end="")
            testx = torch.Tensor(x.float()).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y.float()).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :], testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            
        s2 = time.time()
        
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_rmse.append(mvalid_rmse)
        his_loss.append(mvalid_loss)
        his_mape.append(mvalid_mape)
        t2 = time.time()
        train_time.append(t2 - t1)
        if (epoch - 1) % args.print_every == 0:
            if worker_rank == 0:
                # print("epoch: ", epoch, flush=True)
                print(f"\nEpoch {epoch} VAL |  time: {t2 - t1} loss: {mvalid_loss}, mape: {mvalid_mape}, rmse: {mvalid_rmse}",flush=True)
            # log = 'Rank: {:02d}, Epoch: {:03d}, Inference Time: {:.4f} secs'
            # print(log.format(worker_rank, epoch, (s2 - s1)))
            # log = 'Rank: {:02d}, Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            # print(log.format(worker_rank, epoch, mtrain_loss, mtrain_mape, mtrain_rmse,
            #                     mvalid_loss, mvalid_mape, mvalid_rmse,
            #                     (t2 - t1)),
            #         flush=True)
        if mvalid_loss < minl:
                # torch.save(
                #     engine.model.state_dict(), args.save + "exp" +
                #     str(args.expid) + "_" + str(runid) + ".pth")
                minl = mvalid_loss
                epoch_best = epoch
                count_lfx = 0
        else:
            count_lfx += 1
            if count_lfx > tolerance:
                break
    
    if worker_rank == 0:
        with open("stats.txt", "w") as file:
            file.write(f"opt_loss: {min(his_loss)}\n")
            file.write(f"opt_rmse: {min(his_rmse)}\n")
            file.write(f"opt_mape: {min(his_mape)}\n")
        
        with open("per_epoch_stats.txt", "w") as file:
            file.write(f"epoch, per_epoch_runtime, loss, rmse, mape\n")

            for i in range(len(his_loss)):
                file.write(f"{i}, {train_time[i]}, {his_loss[i]}, {his_rmse[i]}, {his_mape[i]}\n")
                
        
        # print("Total training time: ", train_end - train_start, flush=True)  

       

        
    # sampler = DistributedSampler(dataloader['train_loader'])

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    
    main(0)
    # for i in range(args.runs):
    #     if args.TEST_ONLY:
    #         vm1, vm2, vm3, m1, m2, m3 = main(i)
    #     else:
            # vm1, vm2, vm3, m1, m2, m3 = main(i)
           
    #     vmae.append(vm1)
    #     vmape.append(vm2)
    #     vrmse.append(vm3)
    #     mae.append(m1)
    #     mape.append(m2)
    #     rmse.append(m3)

    # mae = np.array(mae)
    # mape = np.array(mape)
    # rmse = np.array(rmse)

    # amae = np.mean(mae, 0)
    # amape = np.mean(mape, 0)
    # armse = np.mean(rmse, 0)

    # smae = np.std(mae, 0)
    # smape = np.std(mape, 0)
    # srmse = np.std(rmse, 0)

    # print('\n\nResults for ' + str(args.runs) + ' runs\n\n')

    # print('valid\tMAE\tRMSE\tMAPE')
    # log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    # print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    # log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    # print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    # print('\n\n')

    # print(
    #     'test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std'
    # )
    # for i in range(4):
    #     log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    #     print(
    #         log.format([3, 6, 9, 12][i], amae[i], armse[i], amape[i], smae[i],
    #                    srmse[i], smape[i]))
