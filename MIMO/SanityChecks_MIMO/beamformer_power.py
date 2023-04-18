import numpy as np
import sys
sys.path.append("../")
from settings_MIMO import *
from networks_generator_MIMO import generate_real_channels, estimate_channels, schedule_users
n_networks = 10000


def power_computation(h_est):
    B = np.matmul(h_est, np.linalg.inv(np.matmul(np.transpose(np.conjugate(h_est), (0, 2, 1)), h_est)))
    beamformer_powers = np.power(np.linalg.norm(B, ord=2, axis=1), 2)
    print("[Schedule: {}] M: {}; original K: {}; avg power on precoder: {:5.2f} with std: {:5.2f}".format(USER_SCHEDULING, M, K_orig, np.mean(beamformer_powers), np.std(beamformer_powers)))

# generate real channels
h = generate_real_channels(n_networks)
h_est = estimate_channels(h)
if USER_SCHEDULING:
    h_est = schedule_users(h_est)

power_computation(h_est)

