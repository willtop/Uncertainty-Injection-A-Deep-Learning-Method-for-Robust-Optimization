# This script contains the python implementation of FPLinQ scheduling algorithm (original work by K. Shen and W. Yu)

import numpy as np
from utils import compute_SINRs
from utils import get_crossLink_channels, get_directLink_channels
from setup import *
from scipy.optimize import linprog
from scipy.io import loadmat
from tqdm import trange

# Parallel computation over multiple layouts
def FP_power_control(g):
    n_layouts = np.shape(g)[0]
    assert np.shape(g)==(n_layouts, N_LINKS, N_LINKS)
    # For this project, not doing weighted sum-rate optimization
    weights = np.ones([n_layouts, N_LINKS])
    g_diag = get_directLink_channels(g)
    g_nondiag = get_crossLink_channels(g)
    # For matrix multiplication and dimension matching requirement, reshape into column vectors
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = np.ones([n_layouts, N_LINKS, 1])
    tx_powers = np.ones([n_layouts, N_LINKS, 1]) * TX_POWER  # assume same power for each transmitter
    # In the computation below, every step's output is with shape: number of samples X N X 1
    for i in range(150):
        # Compute z
        p_x_prod = x * tx_powers
        z_denominator = np.matmul(g_nondiag, p_x_prod) + NOISE_POWER
        z_numerator = g_diag * p_x_prod
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.matmul(g, p_x_prod) + NOISE_POWER
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))
        y = y_numerator / y_denominator
        # compute x
        x_denominator = np.matmul(np.transpose(g, (0,2,1)), np.power(y, 2)) * tx_powers
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag * tx_powers)
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    assert np.shape(x)==(n_layouts, N_LINKS, 1)
    x = np.squeeze(x, axis=-1)
    return x

# Load the results from MATLAB optimizer
def GP_power_control(channels):
    res = loadmat('Data_D2D/GP_{}.mat'.format(SETTING_STRING))
    pc = res['power_controls_all']
    assert np.shape(pc) == (N_TEST, N_LINKS)
    sinrs = compute_SINRs(pc, channels)
    # check the SINR error (theoretically should all be the same)
    print("<<<<<<<<<<<<<<<<<<<GP SINR ERRORS>>>>>>>>>>>>>>>>>>>")
    print((np.max(sinrs, axis=1)-np.min(sinrs, axis=1))/np.min(sinrs, axis=1))
    print("<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>")
    return pc
