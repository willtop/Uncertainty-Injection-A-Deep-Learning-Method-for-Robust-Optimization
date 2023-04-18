# Training script for all the models

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
from setup import *
from neural_net_D2D import Regular_Net, Robust_Net

N_EPOCHES = 150
MINIBATCH_SIZE = 1000
TRAINING_DEBUG = False

def plot_training_curves(objs):
    print("Plotting training curves...")
    fig, axes = plt.subplots(1,2)
    fig.suptitle(f"D2D Min-Rate (Unscaled) over {SETTING_STRING}")
    axes[0].set_title("Regular Neural Net")
    axes[0].set_xlabel("Training Epoches")
    axes[0].set_ylabel("Average Nominal Objective")
    axes[0].plot(objs[:,0], 'r', label="Train")
    axes[0].plot(objs[:,1], 'b', label="Valid")
    axes[0].legend()
    axes[1].set_title("Robust Neural Net")
    axes[1].set_xlabel("Training Epoches")
    axes[1].set_ylabel("Average Robust Objective")
    axes[1].plot(objs[:, 2], 'r', label="Train")
    axes[1].plot(objs[:, 3], 'b', label="Valid")
    axes[1].legend()
    plt.show()
    print("Finished plotting.")
    return

def shuffle_divide_batches(data, n_batches):
    n_layouts = np.shape(data)[0]
    perm = np.arange(n_layouts)
    np.random.shuffle(perm)
    data_batches = np.split(data[perm], n_batches, axis=0)
    return data_batches

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description="main script argument parser")
    parser.add_argument("--plot", default=False, help="Whether plot the training curves (executed after training)")
    args = parser.parse_args()
    if args.plot:
        metrics = np.load(f"Trained_Models_D2D/training_curves_Min-Rate_{SETTING_STRING}.npy")
        plot_training_curves(metrics)
        exit(0)

    # data preparation
    pl_train, pl_valid = np.load("Data_D2D/pl_train_{}.npy".format(SETTING_STRING)), np.load("Data_D2D/pl_valid_{}.npy".format(SETTING_STRING))
    n_train = np.shape(pl_train)[0]
    assert n_train % MINIBATCH_SIZE == 0
    n_minibatches = int(n_train / MINIBATCH_SIZE)
    print("[D2D Min-Rate] Data Loaded! With {} training samples ({} minibatches) and {} validation samples.".format(n_train, n_minibatches, np.shape(pl_valid)[0]))

    regular_net, robust_net = Regular_Net().to(DEVICE), Robust_Net().to(DEVICE)
    optimizer_regular, optimizer_robust = optim.Adam(regular_net.parameters(), lr=5e-3), optim.Adam(robust_net.parameters(), lr=5e-3)
    regular_obj_max, robust_obj_max = -np.inf, -np.inf
    obj_eps = []
    for i in trange(1, N_EPOCHES+1):
        regular_obj_ep = 0; robust_obj_ep = 0
        pl_batches = shuffle_divide_batches(pl_train, n_minibatches)
        for j in range(n_minibatches):
            optimizer_regular.zero_grad()
            optimizer_robust.zero_grad()
            regular_obj, _ = regular_net(torch.tensor(pl_batches[j], dtype=torch.float32).to(DEVICE))
            robust_obj, _ = robust_net(torch.tensor(pl_batches[j], dtype=torch.float32).to(DEVICE))
            (-regular_obj).backward(); optimizer_regular.step()
            (-robust_obj).backward(); optimizer_robust.step()           
            regular_obj_ep += regular_obj.item()
            robust_obj_ep += robust_obj.item()
            if (j+1) % min(50,n_minibatches) == 0:
                # Validation
                with torch.no_grad():
                    regular_obj, _ = regular_net(torch.tensor(pl_valid, dtype=torch.float32).to(DEVICE))
                    regular_obj = regular_obj.item()
                    robust_obj, _ = robust_net(torch.tensor(pl_valid, dtype=torch.float32).to(DEVICE))
                    robust_obj = robust_obj.item()
                obj_eps.append([regular_obj_ep/(j+1), regular_obj, robust_obj_ep/(j+1), robust_obj])
                print("[D2D Min-Rate][Regular] Tr:{:7.4e}; Va:{:7.4e} [Robust] Tr: {:7.4e}; Va:{:7.4e}".format(regular_obj_ep/(j+1), regular_obj, robust_obj_ep/(j+1), robust_obj))
                if (regular_obj > regular_obj_max):
                    regular_net.save_model()
                    regular_obj_max = regular_obj
                if (robust_obj > robust_obj_max):
                    robust_net.save_model()
                    robust_obj_max = robust_obj
                np.save(f"Trained_Models_D2D/training_curves_Min-Rate_{SETTING_STRING}.npy", np.array(obj_eps))

    print("Script finished!")
