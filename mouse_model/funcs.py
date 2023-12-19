import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .data_utils_new import load_train_val_ds

def train_model(model, device, args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_ds, val_ds = load_train_val_ds(args)

    train_dataloader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(dataset=val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_train_spike_loss = np.inf
    best_val_spike_loss = np.inf
    train_loss_list = []
    val_loss_list = []

    # start training
    ct = 0

    for epoch in range(args.epochs):

        start_time = time.time()

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

        # print("Epoch {} train loss: {}".format(epoch, epoch_train_loss))

        if epoch_train_spike_loss < best_train_spike_loss:
            save_train_model = 'Train_model_saved'
            # print("save train model at epoch", epoch)
            torch.save(model.state_dict(), args.best_train_path)
            best_train_spike_loss = epoch_train_spike_loss
        else:
            save_train_model = ''

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

        # print("Epoch {} val loss: {}".format(epoch, epoch_val_spike_loss))

        if epoch_val_spike_loss < best_val_spike_loss:
            ct = 0
            save_valid_model = 'Valid_model_saved'
            # print("save val model at epoch", epoch)
            torch.save(model.state_dict(), args.best_val_path)
            best_val_spike_loss = epoch_val_spike_loss
        else:
            save_valid_model = ''
            ct += 1
            if ct >=5:
                print('Stop training (Early Stopped)')
                break

        minutes = (time.time()-start_time)/60
        print(f'Epoch: {epoch:4}/{args.epochs:4} | Time: {minutes:6.2f} min | Loss Train/Valid: {epoch_train_loss:.3f}/{epoch_val_spike_loss:.3f} |', save_train_model, save_valid_model)
    print('Stop training (Epoch Stopped)')

    return train_loss_list, val_loss_list

def evaluate_model(model, weights_path, dataset, device):

    dl = DataLoader(dataset=dataset, batch_size=256, shuffle=False, num_workers=4)

    model.load_state_dict(torch.load(weights_path))

    ground_truth_all = []
    pred_all = []

    model.eval()

    with torch.no_grad():

        for (image, behav, spikes) in dl:

            image = image.to(device)
            behav = behav.to(device)

            pred = model(image, behav)

            ground_truth_all.append(spikes.numpy())
            pred_all.append(pred.cpu().numpy())

    return np.concatenate(pred_all, axis=0), np.concatenate(ground_truth_all, axis=0)