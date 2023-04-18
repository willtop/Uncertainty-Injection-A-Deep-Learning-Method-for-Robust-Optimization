import numpy as np
from setup import *
from tqdm import trange

# Compute the angle based on three points
def compute_angle(center_loc, loc1, loc2):
    vec1, vec2 = loc1-center_loc, loc2-center_loc
    assert np.linalg.norm(vec1)>0 and np.linalg.norm(vec2)>0
    vec1, vec2 = vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2)
    dotprod = np.dot(vec1, vec2)
    assert dotprod > -1-1e-3 and dotprod < 1+1e-3
    angle = np.arccos(np.clip(dotprod, -1.0, 1.0))
    return angle

# Generate layout one at a time
def generate_one_D2D_layout():
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=FIELD_LENGTH, size=[N_LINKS,1])
    tx_ys = np.random.uniform(low=0, high=FIELD_LENGTH, size=[N_LINKS,1])
    while(True): # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []; rx_ys = []
        for i in range(N_LINKS):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=SHORTEST_DIRECTLINK, high=LONGEST_DIRECTLINK)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=FIELD_LENGTH and 0<=rx_y<=FIELD_LENGTH):
                    got_valid_rx = True
            rx_xs.append(rx_x); rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N_LINKS, N_LINKS])
        angles_rx = np.zeros([N_LINKS, N_LINKS])
        angles_tx = np.zeros([N_LINKS, N_LINKS])
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N_LINKS):
            for tx_index in range(N_LINKS):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
                # compute the angle from rx perspective
                angles_rx[rx_index][tx_index] = compute_angle(center_loc=layout[rx_index][2:4], loc1=layout[rx_index][0:2], loc2=layout[tx_index][0:2])
                angles_tx[rx_index][tx_index] = compute_angle(center_loc=layout[tx_index][0:2], loc1=layout[tx_index][2:4], loc2=layout[rx_index][2:4])
        if(np.min(distances+np.eye(N_LINKS)*SHORTEST_CROSSLINK)<SHORTEST_CROSSLINK):
            pass
        else:
            break # go ahead and return the layout
    return layout, distances, angles_rx, angles_tx

def generate_D2D_layouts(n_layouts):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(n_layouts, SETTING_STRING))
    layouts_all, distances_all, angles_rx_all, angles_tx_all = [], [], [], []
    for i in trange(n_layouts):
        layouts, distances, angles_rx, angles_tx = generate_one_D2D_layout()
        layouts_all.append(layouts)
        distances_all.append(distances)
        angles_rx_all.append(angles_rx)
        angles_tx_all.append(angles_tx)
    layouts_all, distances_all, angles_rx_all, angles_tx_all = \
           np.array(layouts_all), np.array(distances_all), np.array(angles_rx_all), np.array(angles_tx_all)
    assert np.shape(layouts_all)==(n_layouts, N_LINKS, 4)
    assert np.shape(distances_all)==np.shape(angles_rx_all)==np.shape(angles_tx_all)==(n_layouts, N_LINKS, N_LINKS)
    return layouts_all, distances_all, angles_rx_all, angles_tx_all

def generate_D2D_pathLosses(n_layouts):
    layouts, distances, angles_rx, angles_tx = generate_D2D_layouts(n_layouts)
    assert np.shape(distances) == (n_layouts, N_LINKS, N_LINKS)
    h1, h2 = TX_HEIGHT, RX_HEIGHT
    signal_lambda = 2.998e8 / CARRIER_FREQUENCY
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss
    # add beamforming gain based on directions
    Tx_gains = (np.ones([N_LINKS, N_LINKS])*MAIN_LOBE_HALF_WIDTH >= angles_tx) * MAIN_LOBE_GAIN_dB + \
               (np.ones([N_LINKS, N_LINKS])*MAIN_LOBE_HALF_WIDTH <= angles_tx) * SIDE_LOBE_GAIN_dB
    Rx_gains = (np.ones([N_LINKS, N_LINKS])*MAIN_LOBE_HALF_WIDTH >= angles_rx) * MAIN_LOBE_GAIN_dB + \
               (np.ones([N_LINKS, N_LINKS])*MAIN_LOBE_HALF_WIDTH <= angles_rx) * SIDE_LOBE_GAIN_dB
    # add beamforming gains
    pathLosses = -Tx_over_Rx + Tx_gains + Rx_gains 
    # add the best beamforming gains only for direct channels
    pathLosses = pathLosses + np.eye(N_LINKS) * (DIRECTLINK_GAIN_dB-MAIN_LOBE_GAIN_dB)
    pathLosses = np.power(10, (pathLosses / 10))  # convert from decibel to absolute
    return pathLosses, layouts # Shape: n_layouts X N X N

def get_directLink_channels(channels):
    assert channels.ndim==3
    return np.diagonal(channels, axis1=1, axis2=2)

def get_crossLink_channels(channels):
    return channels*(1.0-np.identity(N_LINKS, dtype=float))

def compute_SINRs(pc, channels):
    dl = get_directLink_channels(channels)
    cl = get_crossLink_channels(channels)
    signals = pc * dl * TX_POWER # layouts X N
    interferences = np.squeeze(np.matmul(cl, np.expand_dims(pc, axis=-1))) * TX_POWER # layouts X N
    sinrs = signals / ((interferences + NOISE_POWER)*SINR_GAP)   # layouts X N
    return sinrs

def compute_rates(sinrs):
    return BANDWIDTH * np.log2(1 + sinrs) 

def add_fadings(pl):
    # generate shadowing coefficients
    shadow_realizations = np.random.normal(size=np.shape(pl), loc=0, scale=8.0) 
    shadow_realizations = np.power(10.0, shadow_realizations / 10.0)
    # generate fast fading factors with circular Gaussian
    h_real = np.random.normal(size=np.shape(pl))
    h_imag = np.random.normal(size=np.shape(pl))
    ff_realizations = (np.power(h_real, 2) + np.power(h_imag, 2)) / 2
    return pl * shadow_realizations * ff_realizations