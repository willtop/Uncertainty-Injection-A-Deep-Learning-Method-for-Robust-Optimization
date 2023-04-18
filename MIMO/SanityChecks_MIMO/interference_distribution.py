import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from settings_MIMO import *
from networks_generator_MIMO import generate_real_channels, estimate_channels

N_NETWORKS = 3
N_REALIZATIONS_SET = [100, 200, 500, 1000, 2000, 5000]
P_LEVEL = 0.2

h = generate_real_channels(N_NETWORKS)
h_est = estimate_channels(h)

for i in range(N_NETWORKS):
    for N_REALIZATIONS in N_REALIZATIONS_SET:
        h_est_one = h_est[i] # M X K
        h_est_one = np.expand_dims(h_est_one, axis=0) # 1 X M X K
        h_est_one = np.tile(h_est_one, (N_REALIZATIONS, 1, 1)) # N_REALIZE X M X K
        B = np.matmul(h_est_one, np.linalg.inv(np.matmul(np.transpose(np.conjugate(h_est_one), (0, 2, 1)), h_est_one)))  # N_REALIZE X M X K
        e = np.random.normal(size=np.shape(h_est_one), scale=np.sqrt(ESTIMATION_ERROR_VARIANCE/2)) + \
            1j * np.random.normal(size=np.shape(h_est_one), scale=np.sqrt(ESTIMATION_ERROR_VARIANCE/2))
        h_realize_one = h_est_one + e # N_REALIZE X M X K
        pc = np.expand_dims(np.eye(K)*P_LEVEL, axis=0)
        pc = np.tile(pc, (N_REALIZATIONS, 1, 1)) # N_REALIZE X K X K
        sinr_denominators_component = np.matmul(np.transpose(np.conjugate(h_realize_one), (0, 2, 1)), B) * (1 - np.eye(K))
        sinr_denominators = np.matmul(np.matmul(sinr_denominators_component, pc),
                                      np.transpose(np.conjugate(sinr_denominators_component), (0, 2, 1)))
        sinr_denominators = np.diagonal(sinr_denominators, offset=0, axis1=1, axis2=2).real
        assert np.shape(sinr_denominators) == (N_REALIZATIONS, K)
        str = "[Interferences with ZF on Imperfect CSI Layout #{}] Power level {} under uncertainty variance {} \n 95-percentile Interference value over {} realizations: {}".format(
            i+1, P_LEVEL, ESTIMATION_ERROR_VARIANCE, N_REALIZATIONS, np.percentile(sinr_denominators.flatten(), q=95))
        print(str)
        plt.title(str)
        plt.hist(sinr_denominators.flatten())
        plt.axvline(np.percentile(sinr_denominators.flatten(), q=95))
        plt.show()

print("Script finished successfully!")