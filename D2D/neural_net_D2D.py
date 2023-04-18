import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from setup import *
import os

class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Models_D2D")
        self.fc_1 = nn.Linear(N_LINKS**2, 6*N_LINKS**2)
        self.fc_2 = nn.Linear(6*N_LINKS**2, 6*N_LINKS**2)
        self.fc_3 = nn.Linear(6*N_LINKS**2, 6*N_LINKS**2)
        self.fc_4 = nn.Linear(6*N_LINKS**2, 6*N_LINKS**2)
        self.fc_5 = nn.Linear(6*N_LINKS**2, N_LINKS)
        self.input_mean = torch.tensor(np.load(os.path.join(self.base_dir, "pl_train_mean_{}.npy".format(SETTING_STRING))),dtype=torch.float32).to(DEVICE)
        self.input_std = torch.tensor(np.load(os.path.join(self.base_dir, "pl_train_std_{}.npy".format(SETTING_STRING))),dtype=torch.float32).to(DEVICE)

    def input_processing(self, pl):
        assert pl.ndim == 3
        x = pl.view(-1, N_LINKS*N_LINKS)
        x = (x-self.input_mean.view(1, N_LINKS*N_LINKS))/self.input_std.view(1, N_LINKS*N_LINKS)
        return x

    def computer_power_control(self, pl):
        x = self.input_processing(pl)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        pc = torch.sigmoid(self.fc_5(x))
        return pc

    def compute_sinrs(self, pc, channels):
        dl = torch.diagonal(channels, dim1=1, dim2=2)
        cl = channels * (1.0-torch.eye(N_LINKS, dtype=torch.float).to(DEVICE))
        sinrs_numerators = pc * dl
        sinrs_denominators = torch.squeeze(torch.matmul(cl, torch.unsqueeze(pc,-1)), -1) + NOISE_POWER/TX_POWER
        sinrs = sinrs_numerators / (sinrs_denominators * SINR_GAP)
        return sinrs

    # Un-normalized to ensure not in the flat region
    def compute_rates(self, sinrs):
        return torch.log(1 + sinrs)

    def load_model(self, str):
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            print("[{}] Load trained model from: {}".format(str, self.model_path))
        else:
            print("[{}] Train from scratch.".format(str))

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print("Model saved at {}".format(self.model_path))
        return

class Regular_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(self.base_dir, "regular_net_{}.ckpt".format(SETTING_STRING))
        self.load_model("Regular Net")

    def forward(self, pl):
        pc = self.computer_power_control(pl)
        sinrs = self.compute_sinrs(pc, pl)
        rates = self.compute_rates(sinrs)
        objectives, _ = torch.min(rates, dim=1)
        return torch.mean(objectives), pc

class Robust_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.n_realizations = 1000
        self.model_path = os.path.join(self.base_dir, "robust_net_{}.ckpt".format(SETTING_STRING))
        self.load_model("Robust Net")

    def add_fadings(self, pl):
        # generate shadowing coefficients
        shadow_realizations = torch.randn(size=pl.size())*8.0 # std=8.0
        shadow_realizations = torch.pow(10.0, shadow_realizations / 10.0)
        # generate fast fading factors with circular Gaussian
        h_real = torch.randn(size=pl.size())
        h_imag = torch.randn(size=pl.size())
        ff_realizations = (torch.pow(h_real,2) + torch.pow(h_imag, 2))/2
        return pl * shadow_realizations.to(DEVICE) * ff_realizations.to(DEVICE)

    def forward(self, pl):
        pc = self.computer_power_control(pl)
        csi = self.add_fadings(pl.repeat(self.n_realizations, 1, 1))
        sinrs = self.compute_sinrs(pc.repeat(self.n_realizations, 1), csi)
        sinrs = sinrs.view(self.n_realizations, -1, N_LINKS)
        rates = self.compute_rates(sinrs)
        min_rates, _ = torch.min(rates, dim=-1)
        objectives = torch.quantile(min_rates, q=0.1, dim=0)
        return torch.mean(objectives), pc

if __name__ == "__main__":
    print("Robust net weights:")
    net = Robust_Net()
    print(net.fc_1.weight)
    print(net.fc_2.weight)
    print(net.fc_3.weight)
