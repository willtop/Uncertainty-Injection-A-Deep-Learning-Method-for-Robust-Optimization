import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings_MIMO import *
import os

class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.n_hidden_layer_neurons = 200 
        self.fc_1 = nn.Linear(2*K*K, self.n_hidden_layer_neurons)
        self.fc_2 = nn.Linear(self.n_hidden_layer_neurons, self.n_hidden_layer_neurons)
        self.fc_3 = nn.Linear(self.n_hidden_layer_neurons, self.n_hidden_layer_neurons)
        self.fc_4 = nn.Linear(self.n_hidden_layer_neurons, self.n_hidden_layer_neurons)
        self.fc_5 = nn.Linear(self.n_hidden_layer_neurons, K)
        
    def compute_power_loading(self, H_est_effective):
        x = torch.view_as_real(H_est_effective).view(-1, 2*K*K)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        pc = F.softmax(self.fc_5(x), dim=-1)
        # pc = torch.sigmoid(self.fc_5(x))
        return pc

    # Same as power control, only assume estimated channels are available for computing
    # the beamformer
    def get_beamformer(self, H_est):
        # Use Zero-forcing beamformer direction
        B = torch.matmul(H_est, torch.inverse(torch.matmul(torch.transpose(torch.conj(H_est), 1, 2), H_est)+RCI_BF_ALPHA*torch.eye(K).to(DEVICE))) 
        B = B / torch.norm(B, dim=1, keepdim=True)
        return B

    def compute_sinrs(self, H, pc, B):
        pc_abs = pc * TRANSMIT_POWER_TOTAL
        pc_mat = torch.diag_embed(pc_abs)
        pc_mat = torch.view_as_complex(torch.stack((pc_mat, torch.zeros_like(pc_mat)),-1))
        sinr_numerators = torch.matmul(torch.transpose(torch.conj(H), 1, 2), B)
        sinr_numerators = torch.pow(torch.diagonal(sinr_numerators, offset=0, dim1=1, dim2=2).abs(), 2) * pc_abs
        sinr_denominators = torch.matmul(torch.transpose(torch.conj(H), 1, 2), B) * (1 - torch.eye(K)).to(DEVICE)
        sinr_denominators = torch.matmul(torch.matmul(sinr_denominators, pc_mat), torch.transpose(torch.conj(sinr_denominators), 1, 2))
        sinr_denominators = torch.diagonal(sinr_denominators, offset=0, dim1=1, dim2=2).real + NOISE_POWER
        sinrs = sinr_numerators / sinr_denominators  # turns out pytorch matmul would have such errors for a vector times its conjugate not resulting in precisely 0 complex part
        assert sinrs.size()[1:] == (K,)
        return sinrs
    
    # not scaled with bandwidth, ensure the gradients at the log function region with reasonble slope
    def compute_rates(self, sinrs):
        return torch.log(1+sinrs)

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
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Models_MIMO",
                                       "regular_net_{}.ckpt".format(SETTING_STRING))
        self.load_model("Regular Net")

    def forward(self, H_est):
        B = self.get_beamformer(H_est)
        H_est_effective = torch.matmul(torch.transpose(torch.conj(H_est), 1, 2), B)
        pc = self.compute_power_loading(H_est_effective)
        sinrs = self.compute_sinrs(H_est, pc, B)
        rates = self.compute_rates(sinrs)
        objectives, _ = torch.min(rates, dim=-1)
        return objectives.mean(), pc

class Robust_Net(Neural_Net):
    def __init__(self):
        super().__init__()
        self.n_realizations = 1000
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Models_MIMO",
                                       "robust_net_{}.ckpt".format(SETTING_STRING))
        self.load_model("Robust Net")

    def inject_uncertainties(self, H_est):
        n_layouts = H_est.size()[0]
        H_est = H_est.repeat(self.n_realizations, 1, 1) 
        # Generate channel estimation errors 
        e = torch.view_as_complex(torch.randn(size=torch.view_as_real(H_est).size())*((ESTIMATION_ERROR_VARIANCE/2)**0.5)).to(DEVICE)
        H_realize = H_est + e
        assert H_realize.size() == (n_layouts*self.n_realizations, M, K)
        return H_realize

    def forward(self, H_est):
        B = self.get_beamformer(H_est)
        H_est_effective = torch.matmul(torch.transpose(torch.conj(H_est), 1, 2), B)
        pc = self.compute_power_loading(H_est_effective)
        H_realize = self.inject_uncertainties(H_est)
        pc_multi, B_multi = pc.repeat(self.n_realizations, 1), B.repeat(self.n_realizations, 1, 1)
        sinrs = self.compute_sinrs(H_realize, pc_multi, B_multi)
        sinrs = sinrs.view(self.n_realizations, -1, K)
        rates = self.compute_rates(sinrs)
        min_rates, _ = torch.min(rates, dim=-1)
        objectives = torch.quantile(min_rates, q=0.05, dim=0)
        return objectives.mean(), pc
