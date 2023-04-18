# Training script for all the models

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
from settings_MIMO import *
from networks_generator_MIMO import generate_real_channels, estimate_channels
from neural_net_MIMO import Regular_Net, Robust_Net

N_EPOCHES = 500
N_MINIBATCHES = 50
MINIBATCH_SIZE = 1000
VALIDATION_SIZE = 2000
TRAINING_DEBUG = False

def plot_training_curves(objs, starting_point):
    print("Plotting training curves...")
    fig, axes = plt.subplots(1,2)
    fig.suptitle("MIMO over {}".format(SETTING_STRING))
    # Regular Net
    axes[0].set_xlabel("Training Epoches")
    axes[0].set_ylabel(f"Nominal Min Rate (Unscaled)")
    axes[0].set_title("Regular Neural Network")
    axes[0].plot(objs[starting_point:, 0], 'r', label="Train")
    axes[0].plot(objs[starting_point:, 2], 'b--', linewidth=0.8, label="Valid")
    axes[0].legend()
    # Robust Net
    axes[1].set_xlabel("Training Epoches")
    axes[1].set_ylabel(f"Robust Min Rate (Unscaled)")
    axes[1].set_title("Robust Neural Network")
    axes[1].plot(objs[starting_point:, 1], 'r', label="Train")
    axes[1].plot(objs[starting_point:, 3], 'b--', linewidth=0.8, label="Valid")
    axes[1].legend()
    plt.show()
    print("Finished plotting!")
    return

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    parser.add_argument("--start", type=int, default=0, help="The starting point of training updates to plot training curves")
    args = parser.parse_args()
    if args.plot:
        sinrs = np.load(f"Trained_Models_MIMO/training_curves_Min-Rate_{SETTING_STRING}.npy")
        plot_training_curves(sinrs, args.start)
        exit(0)

    regular_net = Regular_Net().to(DEVICE)
    robust_net = Robust_Net().to(DEVICE)
    regular_optimizer, robust_optimizer = optim.Adam(regular_net.parameters(), lr=1e-4), optim.Adam(robust_net.parameters(), lr=1e-4)
    regular_obj_max, robust_obj_max = -np.inf, -np.inf
    obj_eps = []
    for i in trange(1, N_EPOCHES+1):
        # [TRAIN]
        regular_obj_ep, robust_obj_ep = 0, 0
        regular_net.train()
        robust_net.train()
        for j in range(N_MINIBATCHES):
            # Generate minibatch training data online
            H_est = estimate_channels(generate_real_channels(MINIBATCH_SIZE))
            # [Regular net]
            regular_optimizer.zero_grad()
            robust_optimizer.zero_grad()
            regular_obj, _ = regular_net(torch.tensor(H_est, dtype=torch.cfloat).to(DEVICE))
            robust_obj, _ = robust_net(torch.tensor(H_est, dtype=torch.cfloat).to(DEVICE))
            (-regular_obj).backward(); regular_optimizer.step()
            (-robust_obj).backward(); robust_optimizer.step()
            regular_obj_ep += regular_obj.item() / N_MINIBATCHES
            robust_obj_ep += robust_obj.item() / N_MINIBATCHES
        # [VALIDATE]
        H_est = estimate_channels(generate_real_channels(VALIDATION_SIZE))
        with torch.no_grad():
            regular_obj, _ = regular_net(torch.tensor(H_est, dtype=torch.cfloat).to(DEVICE))
            regular_obj = regular_obj.item()
            robust_obj, _ = robust_net(torch.tensor(H_est, dtype=torch.cfloat).to(DEVICE))
            robust_obj = robust_obj.item()
        obj_eps.append([regular_obj_ep, robust_obj_ep, regular_obj, robust_obj])
        print("[Min Rate Unscaled with RCI alpha {}] [Regular] Tr:{:7.4e}; Va:{:7.4e} [Robust] Tr:{:7.4e}; Va:{:7.4e}".format(
            RCI_BF_ALPHA, regular_obj_ep, regular_obj, robust_obj_ep, robust_obj))
        if regular_obj > regular_obj_max:
            regular_net.save_model()
            regular_obj_max = regular_obj
        if robust_obj > robust_obj_max:
            robust_net.save_model()
            robust_obj_max = robust_obj
        np.save(f"Trained_Models_MIMO/training_curves_Min-Rate_{SETTING_STRING}.npy", np.array(obj_eps))

    print("Training Script finished!")
