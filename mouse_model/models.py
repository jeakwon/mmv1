import torch
import torch.nn as nn
import numpy as np
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

class Shifter(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_dim=256, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.bias = nn.Parameter(torch.zeros(3))
    def forward(self, x):
        x = x.reshape(-1,self.input_dim )
        x = self.layers(x)
        x0 = (x[...,0] + self.bias[0]) * 80/5.5
        x1 = (x[...,1] + self.bias[1]) * 60/5.5
        x2 = (x[...,2] + self.bias[2]) * 180/4
        x = torch.stack([x0, x1, x2], dim=-1)
        x = x.reshape(-1,self.seq_len,self.output_dim)
        return x

class PrintLayer(nn.Module):

    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

def size_helper(in_length, kernel_size, padding=0, dilation=1, stride=1):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    res = in_length + 2 * padding - dilation * (kernel_size - 1) - 1
    res /= stride
    res += 1
    return np.floor(res)

# CNN, the last fully connected layer maps to output_dim
class VisualEncoder(nn.Module):

    def __init__(self, output_dim, input_shape=(60, 80), k1=7, k2=7, k3=7):

        super().__init__()

        self.input_shape = (60, 80)
        out_shape_0 = size_helper(in_length=input_shape[0], kernel_size=k1, stride=2)
        out_shape_0 = size_helper(in_length=out_shape_0, kernel_size=k2, stride=2)
        out_shape_0 = size_helper(in_length=out_shape_0, kernel_size=k3, stride=2)
        out_shape_1 = size_helper(in_length=input_shape[1], kernel_size=k1, stride=2)
        out_shape_1 = size_helper(in_length=out_shape_1, kernel_size=k2, stride=2)
        out_shape_1 = size_helper(in_length=out_shape_1, kernel_size=k3, stride=2)
        self.output_shape = (int(out_shape_0), int(out_shape_1)) # shape of the final feature map

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=k1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=k2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=k3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(480, output_dim)
        )


    def forward(self, x):

        x = self.layers(x)

        return x


# may consider adding an activation after linear
class BehavEncoder(nn.Module):

    def __init__(self, behav_dim, output_dim):

        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(behav_dim),
            nn.Linear(behav_dim, output_dim),
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class LSTMPerNeuronCombiner(nn.Module):

    def __init__(self, num_neurons, behav_dim, k1, k2, k3, seq_len, hidden_size=512):

        super().__init__()

        self.seq_len = seq_len
        self.num_neurons = num_neurons
        self.shifter = Shifter(seq_len = seq_len)
        self.visual_encoder = VisualEncoder(output_dim=num_neurons, k1=k1, k2=k2, k3=k3)
        self.behav_encoder = BehavEncoder(behav_dim=behav_dim, output_dim=num_neurons)
        self.bn = nn.BatchNorm1d(3) # apply bn to vis_feats, beh_feats, prod
        self.lstm_net = nn.GRU(input_size=num_neurons*3, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_neurons)
        self.softplus = nn.Softplus() # we could also do relu or elu offset by 1

    def forward(self, images, behav):
        if args.shifter:
            bs = images.size()[0]
            behav_shifter = torch.concat((behav[...,4].unsqueeze(-1),   # theta
                                          behav[...,3].unsqueeze(-1),   # phi
                                          behav[...,1].unsqueeze(-1),  # pitch
                                         behav[...,2].unsqueeze(-1),  # roll
                                         ), dim=-1)
            shift_param = self.shifter(behav_shifter)
            shift_param = shift_param.reshape(-1,3)
            scale_param = torch.ones_like(shift_param[..., 0:2]).to(shift_param.device)
            affine_mat = get_affine_matrix2d(
                                            translations=shift_param[..., 0:2] ,
                                             scale = scale_param,
                                             center =torch.repeat_interleave(torch.tensor([[30,40]], dtype=torch.float),
                                                                            bs*self.seq_len, dim=0).to(shift_param.device),
                                             angle=shift_param[..., 2])
            affine_mat = affine_mat[:, :2, :]
            images = warp_affine(images.reshape(-1,1,60,80), affine_mat, dsize=(60,80)).reshape(bs, self.seq_len,1,60,80)

        # get visual behavioral features in time
        vis_beh_feats = []
        for i in range(self.seq_len):
            v = self.visual_encoder(images[:, i, :, :, :])
            b = self.behav_encoder(behav[:, i, :])
            vb = v * b
            vis_beh_feat = torch.stack([v, b, vb], axis=1)
            vis_beh_feat = self.bn(vis_beh_feat)
            vis_beh_feats.append(vis_beh_feat)
        vis_beh_feats = torch.stack(vis_beh_feats, axis=1)

        # flatten features to (batch_size, seq_len, num_neurons*3)
        vis_beh_feats = torch.flatten(vis_beh_feats, start_dim=2)

        # get LSTM output
        output, _ = self.lstm_net(vis_beh_feats)
        output = output[:, -1, :] # extract the last hidden state

        # fully connected layer and activation function
        output = self.fc(output)
        pred_spikes = self.softplus(output)

        return pred_spikes