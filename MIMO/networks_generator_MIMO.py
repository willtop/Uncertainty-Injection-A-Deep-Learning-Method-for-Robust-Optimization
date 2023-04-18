# This script contains the generator code for producing wireless network layouts and path-losses

import numpy as np
from settings_MIMO import *
from scipy.io import savemat

# Generate i.i.d Rayleigh fading channels
def generate_real_channels(n_networks):
    H_reals = np.random.normal(loc=0.0, scale=np.sqrt(2)/2, size=(n_networks, M, K))
    H_imags = np.random.normal(loc=0.0, scale=np.sqrt(2)/2, size=(n_networks, M, K))
    return H_reals + 1j*H_imags

def estimate_channels(H):
    n_networks = np.shape(H)[0]
    n_BS_reals = np.random.normal(loc=0.0, scale=np.sqrt(NOISE_POWER/2), size=(n_networks, M, K))
    n_BS_imags = np.random.normal(loc=0.0, scale=np.sqrt(NOISE_POWER/2), size=(n_networks, M, K))
    n_BS = n_BS_reals + 1j*n_BS_imags
    pilots_received_BS = np.sqrt(ESTIMATION_PILOT_POWER)*np.conjugate(H) + n_BS
    H_conj_est = np.sqrt(ESTIMATION_PILOT_POWER)/(ESTIMATION_PILOT_POWER+NOISE_POWER) * pilots_received_BS
    H_est = np.conjugate(H_conj_est)
    return H_est

# Utilize only channel estimates
def get_beamformers(H_est):
    # zero-forcing beamforming
    B = np.matmul(H_est, np.linalg.inv(np.matmul(np.transpose(np.conjugate(H_est), (0,2,1)), H_est)+RCI_BF_ALPHA*np.eye(K)))
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B

if(__name__=="__main__"):
    # Due to training and validation data being generated online
    # only generate testing data here (to be read by matlab code for computing geometric programming solutions)
    print(f'[MIMO {SETTING_STRING}] Generating testing data...')
    H_est = estimate_channels(generate_real_channels(N_TEST))
    B = get_beamformers(H_est)

    # compute effective channels to be read by geometric programming
    effectiveChannels = np.matmul(np.transpose(np.conjugate(H_est), (0, 2, 1)), B)
    # compute channel gains, with (i,j)th component being jth beamformer to ith user
    effectiveChannelGains = np.power(np.absolute(effectiveChannels),2)
    
    # save files
    np.save(f'Data_MIMO/channelEstimates_test_{SETTING_STRING}.npy', H_est)
    np.save(f'Data_MIMO/RCI_beamformers_test_{SETTING_STRING}.npy', B)
    savemat(f'Data_MIMO/effectiveChannelGains_test_{SETTING_STRING}.mat', {'effectiveChannelGains':effectiveChannelGains})
    print('Script finished successfully!')
