# This script contains the generator code for producing wireless network layouts and path-losses

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from utils import *
from setup import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--visualize', help='whether visualize a layout or not', default=False)
    args = parser.parse_args()
    
    if args.visualize:
        n_layouts = 3
        pl, _ = generate_D2D_pathLosses(n_layouts)
        sinrs_pl = compute_SINRs(np.ones([n_layouts, N_LINKS]), pl)
        csi = add_fadings(pl)
        sinrs_csi = compute_SINRs(np.ones([n_layouts, N_LINKS]), csi)
        for i in range(n_layouts):
            plt.plot(np.arange(1, N_LINKS+1), sinrs_pl[i], 'r', label='pathloss based SINRs')
            plt.plot(np.arange(1, N_LINKS+1), sinrs_csi[i], 'b--', label='CSI based SINRs')
            plt.show()
        print("Visualization finished!")
        exit(0)

    pl, _ = generate_D2D_pathLosses(N_TRAIN+N_VALID+N_TEST)

    np.save("Data_D2D/pl_train_{}.npy".format(SETTING_STRING), pl[:N_TRAIN])
    np.save("Data_D2D/pl_valid_{}.npy".format(SETTING_STRING), pl[N_TRAIN:N_TRAIN+N_VALID])
    np.save("Data_D2D/pl_test_{}.npy".format(SETTING_STRING), pl[-N_TEST:])
    savemat("Data_D2D/pl_test_{}.mat".format(SETTING_STRING), {'effectiveChannelGains': pl[-N_TEST:]})

    # Save input normalization stats
    np.save("Trained_Models_D2D/pl_train_mean_{}.npy".format(SETTING_STRING), np.mean(pl[:N_TRAIN], axis=0))
    np.save("Trained_Models_D2D/pl_train_std_{}.npy".format(SETTING_STRING), np.std(pl[:N_TRAIN], axis=0))

    print("Script Completed!")
