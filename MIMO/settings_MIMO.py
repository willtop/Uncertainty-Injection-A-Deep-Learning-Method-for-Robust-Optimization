import os
import random
import numpy as np
import torch
# for windows specific error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Environment Setting
BANDWIDTH = 10e6
_NOISE_dBm_Hz = -75
NOISE_POWER = np.power(10, ((_NOISE_dBm_Hz-30)/10))*BANDWIDTH
# Channel estimation setting
ESTIMATION_PILOT_LENGTH = 1
ESTIMATION_ERROR_VARIANCE = 0.075
ESTIMATION_PILOT_POWER = NOISE_POWER*(1-ESTIMATION_ERROR_VARIANCE)/(ESTIMATION_PILOT_LENGTH*ESTIMATION_ERROR_VARIANCE)
# Data Transmission Setting
TRANSMIT_POWER_TOTAL = 1
M = 4  # Number of antennas at base-station
K = 4
assert M >= K
RCI_BF_ALPHA = 0.2 # Based on the numerical 50-percentile min-rate result from uniform power allocation
SETTING_STRING = "M{}_K{}_alpha_{}".format(M, K, RCI_BF_ALPHA)
# Testing
N_TEST = 2000

# set random seed for reproducible results
RANDOM_SEED = 123
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
