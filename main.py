import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader, ConcatDataset

from sklearn.metrics import r2_score, mean_squared_error
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

from mouse_model.data_utils_new import MouseDatasetSegNewBehav, smoothing_with_np_conv
from mouse_model.evaluation import cor_in_time, evaluate_model
import matplotlib.pyplot as plt


def load_train_val_ds(args):
    ds_list = [MouseDatasetSegNewBehav(file_id=args.file_id, segment_num=args.segment_num, seg_idx=i, data_split="train",
                               vid_type=args.vid_type, seq_len=args.seq_len, predict_offset=1,
                                       behav_mode=args.behav_mode, norm_mode="01")
               for i in range(args.segment_num)]
    train_ds, val_ds = [], []
    for ds in ds_list:
        train_ratio = 0.8
        train_ds_len = int(len(ds) * train_ratio)
        train_ds.append(Subset(ds, np.arange(0, train_ds_len, 1)))
        val_ds.append(Subset(ds, np.arange(train_ds_len, len(ds), 1)))
    train_ds = ConcatDataset(train_ds)
    val_ds = ConcatDataset(val_ds)
    print(len(train_ds), len(val_ds))
    return train_ds, val_ds

def load_test_ds(args):
    test_ds = [MouseDatasetSegNewBehav(file_id=args.file_id, segment_num=args.segment_num, seg_idx=i, data_split="test",
                               vid_type=args.vid_type, seq_len=args.seq_len, predict_offset=1,
                                       behav_mode=args.behav_mode, norm_mode="01")
               for i in range(args.segment_num)]
    test_ds = ConcatDataset(test_ds)
    return test_ds

def train_model(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_ds, val_ds = load_train_val_ds()

    train_dataloader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_train_spike_loss = np.inf
    best_val_spike_loss = np.inf
    train_loss_list = []
    val_loss_list = []

    # start training
    ct = 0

    for epoch in range(args.epochs):

        print("Start epoch", epoch)

        model.train()

        epoch_train_loss, epoch_train_spike_loss = 0, 0

        for (image, behav, spikes) in train_dataloader:

            image, behav, spikes = image.to(device), behav.to(device), spikes.to(device)

            pred = model(image, behav)

            spike_loss = nn.functional.poisson_nll_loss(pred, spikes, reduction='mean', log_input=False)

            l1_reg, l1_reg_num_param = 0.0, 0
            for name, param in model.named_parameters():
                if name == "behav_encoder.layers.1.weight":
                    l1_reg += param.abs().sum()
                    l1_reg_num_param += param.shape[0]*param.shape[1]
            l1_reg /= l1_reg_num_param

            total_loss = spike_loss + args.l1_reg_w * l1_reg

            epoch_train_loss += total_loss.item()
            epoch_train_spike_loss += spike_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        epoch_train_loss = epoch_train_loss / len(train_dataloader)
        epoch_train_spike_loss = epoch_train_spike_loss / len(train_dataloader)

        train_loss_list.append(epoch_train_loss)

        print("Epoch {} train loss: {}".format(epoch, epoch_train_loss))

        if epoch_train_spike_loss < best_train_spike_loss:

            print("save train model at epoch", epoch)
            torch.save(model.state_dict(), args.best_train_path)
            best_train_spike_loss = epoch_train_spike_loss

        model.eval()

        epoch_val_spike_loss = 0

        with torch.no_grad():

            for (image, behav, spikes) in val_dataloader:

                image, behav, spikes = image.to(device), behav.to(device), spikes.to(device)

                pred = model(image, behav)

                loss = nn.functional.poisson_nll_loss(pred, spikes, reduction='mean', log_input=False)

                epoch_val_spike_loss += loss.item()

        epoch_val_spike_loss = epoch_val_spike_loss / len(val_dataloader)

        val_loss_list.append(epoch_val_spike_loss)

        print("Epoch {} val loss: {}".format(epoch, epoch_val_spike_loss))

        if epoch_val_spike_loss < best_val_spike_loss:
            ct = 0

            print("save val model at epoch", epoch)
            torch.save(model.state_dict(), args.best_val_path)
            best_val_spike_loss = epoch_val_spike_loss
        else:
            ct += 1
            if ct >=5:
                print('stop training')
                break


        print("End epoch", epoch)

    return train_loss_list, val_loss_list

if __name__ == "__main__":

    class Args:
        seed = 0
        file_id = None
        epochs = 50
        batch_size = 256
        learning_rate = 0.0002
        l1_reg_w = 1
        seq_len = None
        num_neurons = None
        behav_mode = None
        behav_dim = None
        best_val_path = None
        best_train_path = None
        vid_type = "vid_mean"
        segment_num = 10
        hidden_size = 512
        shifter = True

    args=Args()

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.empty_cache()
    print(torch.cuda.is_available())

    # train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for file_id, num_neurons in  [("070921_J553RT", 68), ("110421_J569LT", 32), ("101521_J559NC", 49) ]:
        for behav_mode, behav_dim in [("orig_prod", 21)]:
        # for behav_mode, behav_dim in [("orig", 6), ("velo", 6), ("all", 11), ("orig_prod", 21), ("velo_prod", 21)]:
            for seq_len in range(1, 2):
                print(file_id, behav_mode, seq_len)

                args.file_id = file_id
                args.vid_type = "vid_mean"
                args.num_neurons = num_neurons
                args.shifter=True

                args.behav_mode = behav_mode
                args.behav_dim = behav_dim

                args.seq_len = seq_len

                args.best_train_path = "/hdd/yuchen/train_baseline_{}_{}_seq_{}.pth".format(
                    args.file_id, args.behav_mode, args.seq_len)
                args.best_val_path = "/hdd/yuchen/val_baseline_{}_{}_seq_{}.pth".format(
                    args.file_id, args.behav_mode, args.seq_len)

                model = LSTMPerNeuronCombiner(num_neurons=args.num_neurons,
                                            behav_dim=args.behav_dim,
                                            k1=7, k2=7, k3=7,
                                            seq_len=args.seq_len,
                                            hidden_size=args.hidden_size).to(device)

                train_loss_list, val_loss_list = train_model()

                train_ds, val_ds = load_train_val_ds()
                test_ds = load_test_ds()

                pred, label = evaluate_model(model, weights_path=args.best_val_path, dataset=test_ds, device=device)
                cor_array = cor_in_time(pred, label)
            #     print("best val model on test dataset, {:.3f}+-{:.3f}".format(np.mean(cor_array), np.std(cor_array)))
                pred = smoothing_with_np_conv(pred)
                label = smoothing_with_np_conv(label)
                # print("R2", "{:.6f}".format(r2_score(label.T, pred.T)))
                print("MSE", "{:.6f}".format(mean_squared_error(label, pred)))
                cor_array = cor_in_time(pred, label)
                print("mean corr, {:.3f}+-{:.3f}".format(np.mean(cor_array), np.std(cor_array)))
                # print("max corr", "{:.6f}".format(np.max(cor_array)))
                # print("min corr", "{:.6f}".format(np.min(cor_array)))