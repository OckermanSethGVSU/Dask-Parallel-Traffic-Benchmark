import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer

from net import DGCRN
import setproctitle
import os
from torch.utils.data import Dataset, DataLoader

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
            # print(type(x), type(y))
            # print(x[0])
            for i in range(x.shape[0]):
                temp_x.append(x[i])
                temp_y.append(y[i])

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
# torch.set_num_threads(3)

os.makedirs(args.save, exist_ok=True)

rnn_size = args.rnn_size

device = torch.device(args.device)
device="cuda:0"
dataloader = load_dataset(args.data, args.batch_size, args.batch_size,
                          args.batch_size)
scaler = dataloader['scaler']
train_dataset = TrainDataset(dataloader['train_loader'])
val_dataset = ValDataset(dataloader['val_loader'])


predefined_A = load_adj(args.adj_data)
predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]


def main(runid):

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

    print(args)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                     args.step_size1, args.seq_out_len, scaler, device,
                     args.cl, args.new_training_method)
    
    print("start training...", flush=True)
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    start_time = time.time()
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
       
        for i, (x, y, ycl) in enumerate(train_loader):
            batches_seen += 1
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            trainycl = torch.Tensor(ycl).to(device)
            trainycl = trainycl.transpose(1, 3)

            metrics = engine.train(trainx,
                                    trainy[:, 0, :, :],
                                    trainycl,
                                    idx=None,
                                    batches_seen=batches_seen)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

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
        his_mape.append(mvalid_mape)
        his_rmse.append(mvalid_rmse)

        if (i - 1) % args.print_every == 0:
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i, (s2 - s1)))
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                                mvalid_loss, mvalid_mape, mvalid_rmse,
                                (t2 - t1)),
                    flush=True)

        if mvalid_loss < minl:
            # torch.save(
            #     engine.model.state_dict(), args.save + "exp" +
            #     str(args.expid) + "_" + str(runid) + ".pth")
            minl = mvalid_loss
            epoch_best = i
            count_lfx = 0
        else:
            count_lfx += 1
            if count_lfx > tolerance:
                break
    end_time = time.time()
    print("Total Training Time: ", end_time - start_time)
    with open("stats.txt", "w") as file:
            file.write(f"training_time: {end_time - start_time}\n")
            file.write(f"opt_loss: {min(his_loss)}\n")
            file.write(f"opt_rmse: {min(his_rmse)}\n")
            file.write(f"opt_mape: {min(his_mape)}")
    



   


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    main(1)
   