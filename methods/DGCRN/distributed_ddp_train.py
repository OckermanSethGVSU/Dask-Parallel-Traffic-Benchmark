import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer

from net import DGCRN
import setproctitle
import os

import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
import pandas as pd

import seaborn as sns
import dask
# from dask_saturn import SaturnCluster
from dask.distributed import LocalCluster
from dask.distributed import Client
from distributed.worker import logger
from dask_jobqueue import PBSCluster

import uuid
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dask_pytorch_ddp import dispatch, results
from dask.distributed import Variable, Lock
setproctitle.setproctitle("DGCRN@lifuxian")
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, dataloader, device=None):
        temp_x = []
        temp_y = []
        temp_ycl = []
        for iter, (x, y, ycl) in enumerate(dataloader.get_iterator()):
            for i in range(x.shape[0]):
                temp_x.append(x[i])
                temp_y.append(y[i])
                temp_ycl.append(ycl[i])

        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        self.ycl = torch.tensor(ycl).float()  
        self.permute()

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return self.x[idx], self.y[idx], self.ycl[idx]

    def __len__(self):
        return len(self.x)

    def permute(self):
        self.permutation = torch.randperm(len(self.x))

class ValDataset(Dataset):
    def __init__(self, dataloader, device=None):
        temp_x = []
        temp_y = []
        
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            for i in range(x.shape[0]):
                temp_x.append(x[i])
                temp_y.append(y[i])
               
            # print(type(x), type(y))
            # print(x[0])
           

        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        
        self.permute()

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def permute(self):
        self.permutation = torch.randperm(len(self.x))


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
parser.add_argument('--data',
                    type=str,
                    default='data/METR-LA',
                    help='data path')

parser.add_argument('--adj_data',
                    type=str,
                    default='data/sensor_graph/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')

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

args = parser.parse_args()
torch.set_num_threads(3)

os.makedirs(args.save, exist_ok=True)

rnn_size = args.rnn_size

# device = torch.device(args.device)
dataloader = load_dataset(args.data, args.batch_size, args.batch_size,
                          args.batch_size)
scaler = dataloader['scaler']
train_dataset = TrainDataset(dataloader['train_loader'])
val_dataset = ValDataset(dataloader['val_loader'])

device=None
predefined_A = load_adj(args.adj_data)
predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]

key = uuid.uuid4().hex
rh = results.DaskResultsHandler(key)

npar=2


def main(runid):
    
    
    
    print("start distributed training...", flush=True)
    
    
    
    
    cluster = LocalCluster(n_workers=npar)
    client = Client(cluster)
    
    futures = dispatch.run(client, my_train, backend="gloo")
    rh.process_results(".", futures, raise_errors=False)
    
    client.close()
    cluster.close()
    
   

def my_train():
    runid = 1
    num_epochs = args.epochs
    batch_size = args.batch_size
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=npar)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    worker_rank = int(dist.get_rank())
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
                  rnn_size=rnn_size,
                  hyperGNN_dim=args.hyperGNN_dim)
    model = DDP(model)
    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                     args.step_size1, args.seq_out_len, scaler, device,
                     args.cl, args.new_training_method)
    worker_rank = int(dist.get_rank()) 
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    epoch_best = -1
    tolerance = args.tolerance
    count_lfx = 0
    batches_seen = 0


 
    


    for epoch in range(1, num_epochs + 1):
            
        train_sampler.set_epoch(epoch)
        train_dataset.permute()
        
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
       
        
        for i, (x, y, ycl) in enumerate(train_loader):

            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            trainycl = torch.Tensor(ycl).to(device)
            trainycl = trainycl.transpose(1, 3)
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
    
        t2 = time.time()
        train_time.append(t2 - t1)
        
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        
       
        s1 = time.time()
        for i, (x, y) in enumerate(val_loader):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
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
        his_loss.append(mvalid_loss)

        if (epoch - 1) % args.print_every == 0:
            log = 'Rank: {:02d}, Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(worker_rank, epoch, (s2 - s1)))
            log = 'Rank: {:02d}, Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(worker_rank, epoch, mtrain_loss, mtrain_mape, mtrain_rmse,
                                mvalid_loss, mvalid_mape, mvalid_rmse,
                                (t2 - t1)),
                    flush=True)
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
        
    print("Rank {:02d} Average Training Time: {:.4f} secs/epoch".format(worker_rank,
        np.mean(train_time)))
    print("Rank {:02d} Average Inference Time: {:.4f} secs".format(worker_rank, np.mean(val_time)))

        # bestid = np.argmin(his_loss)
        # engine.model.load_state_dict(
        #     torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +
        #                ".pth",
        #                map_location='cpu'))

        # print("Training finished")
        # print("The valid loss on best model is {}, epoch:{}".format(
        #     str(round(his_loss[bestid], 4)), epoch_best))
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
