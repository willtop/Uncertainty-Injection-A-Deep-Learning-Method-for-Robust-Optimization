# Script for evaluating MIMO network objectives: percentile threshold and power consumptions

import numpy as np
import matplotlib.pyplot as plt
import torch
from neural_net_MIMO import Regular_Net, Robust_Net
from settings_MIMO import *
from networks_generator_MIMO import generate_real_channels, estimate_channels
from benchmarks_MIMO import *
from scipy.io import loadmat

N_REALIZATIONS = 1000
DEBUG_VISUALIZE = False # visualize allocations and rates of each method


def compute_sinrs(H, pc, B):
    pc_abs = pc*TRANSMIT_POWER_TOTAL
    pc_mat = np.array([np.diag(pc_vector) for pc_vector in pc_abs])
    pc_mat = pc_mat + 1j*np.zeros_like(pc_mat)
    sinr_numerators = np.matmul(np.transpose(np.conjugate(H), (0,2,1)), B)
    signal_powers = np.power(np.absolute(np.diagonal(sinr_numerators, offset=0, axis1=1, axis2=2)), 2) * pc_abs
    sinr_denominators = np.matmul(np.transpose(np.conjugate(H), (0,2,1)), B) * (1 - np.eye(K))
    sinr_denominators = np.matmul(np.matmul(sinr_denominators, pc_mat), np.transpose(np.conjugate(sinr_denominators), (0,2,1)))
    interference_powers = np.diagonal(sinr_denominators, offset=0, axis1=1, axis2=2).real 
    sinrs = signal_powers / (interference_powers + NOISE_POWER)
    return sinrs, signal_powers, interference_powers

def compute_rates(sinrs):
    return BANDWIDTH * np.log2(1+sinrs)

def inject_uncertainties(H_est):
    assert np.shape(H_est) == (N_TEST, M, K)
    H_est = np.tile(H_est, (N_REALIZATIONS, 1, 1))
    # Generate channel estimation errors 
    e_reals = np.random.normal(loc=0.0, scale=np.sqrt(ESTIMATION_ERROR_VARIANCE/2), size=(N_TEST*N_REALIZATIONS, M, K))
    e_imags = np.random.normal(loc=0.0, scale=np.sqrt(ESTIMATION_ERROR_VARIANCE/2), size=(N_TEST*N_REALIZATIONS, M, K))
    e = e_reals + 1j*e_imags
    H_realize = H_est + e
    return H_realize

def GP_power_loading(H_est, B):
    source_file = f'Data_MIMO/GP_{SETTING_STRING}.mat'
    print(f'Loading GP solutions from {source_file}...')
    res = loadmat(source_file)
    pc = res['power_controls_all'] # power loading solutions are named as power control in matlab script
    sinrs,_,_ = compute_sinrs(H_est, pc, B)
    # check the SINR error (theoretically should be all the same)
    print("<<<<<<<<<<<<<<<<<GP SINR ERRORS>>>>>>>>>>>>>>>")
    print((np.max(sinrs, axis=1)-np.min(sinrs, axis=1))/np.min(sinrs,axis=1))
    print("<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
    return pc

if(__name__ =='__main__'):
    print(f"[MIMO Downlink] Evaluate {SETTING_STRING} over {N_TEST} layouts.")
    # load the channel estimates and common beamformers used for all methods
    H_est = np.load(f"Data_MIMO/channelEstimates_test_{SETTING_STRING}.npy")
    B = np.load(f"Data_MIMO/RCI_beamformers_test_{SETTING_STRING}.npy")
    # ensure normalized beamformers
    assert np.all(np.linalg.norm(B, axis=1) > 1-1e-4) and np.all(np.linalg.norm(B, axis=1) < 1+1e-4)

    all_power_loadings, plot_colors, plot_linestyles = {}, {}, {}
    regular_net = Regular_Net().to(DEVICE)
    robust_net = Robust_Net().to(DEVICE)

    pc = GP_power_loading(H_est, B)
    all_power_loadings["Geometric Programming"] = pc
    plot_colors["Geometric Programming"] = 'b'
    plot_linestyles["Geometric Programming"] = '-.'

    _, pc = regular_net(torch.tensor(H_est, dtype=torch.cfloat).to(DEVICE))
    all_power_loadings["Deep Learning without Uncertainty Injection"] = pc.detach().cpu().numpy()
    plot_colors["Deep Learning without Uncertainty Injection"] = 'm'
    plot_linestyles["Deep Learning without Uncertainty Injection"] = '--'

    _, pc = robust_net(torch.tensor(H_est, dtype=torch.cfloat).to(DEVICE))
    all_power_loadings["Deep Learning with Uncertainty Injection"] = pc.detach().cpu().numpy()
    plot_colors["Deep Learning with Uncertainty Injection"] = 'r'
    plot_linestyles["Deep Learning with Uncertainty Injection"] = '-'

    all_power_loadings["Uniform Power"] = np.ones([N_TEST, K], dtype=float)/K
    plot_colors["Uniform Power"] = 'k'
    plot_linestyles["Uniform Power"] = ':'


    
    print("<<<<<<<<<<<<<<<<<<<<<<<Evaluating>>>>>>>>>>>>>>>>>>>")
    print("Average std of power loading results: ")
    for (method_key, pc) in all_power_loadings.items():
        assert np.shape(pc) == (N_TEST, K), "{}".format(np.shape(pc))
        # Ensure it's a legit power loading solution
        assert np.all(np.sum(pc, axis=-1) > 1-1e-4) and np.all(np.sum(pc, axis=-1) < 1+1e-4)
        print("[{}] {:.3f}".format(method_key, np.mean(np.std(pc,axis=-1))))

    # Nominal Evaluation
    nominal_objectives = {}
    print("Nominal Objectives: ")
    for (method_key, pc) in all_power_loadings.items():
        sinrs, signals, interferences = compute_sinrs(H_est, pc, B)
        rates = compute_rates(sinrs)
        assert np.shape(sinrs) == np.shape(rates) == (N_TEST, K)
        min_sinrs, min_rates = np.min(sinrs, axis=-1), np.min(rates, axis=-1)
        nominal_objectives[method_key] = min_rates
        print("[{}] Min-SINR: {:.3f}; Min-Rate: {:.3f}Mbps".format(method_key, np.mean(min_sinrs), np.mean(min_rates)/1e6))
        
    # Robust Evaluation
    robust_objectives = {}
    H_realize = inject_uncertainties(H_est)
    print("Robust Objectives: ")
    for (method_key, pc) in all_power_loadings.items():
        sinrs, signals, interferences = compute_sinrs(H_realize, np.tile(pc, (N_REALIZATIONS,1)), np.tile(B, (N_REALIZATIONS,1,1)))
        sinrs = np.reshape(sinrs, [N_REALIZATIONS, N_TEST, K])
        rates = compute_rates(sinrs)
        min_sinrs = np.percentile(np.min(sinrs, axis=-1), q=5, axis=0, interpolation="lower") 
        min_rates = np.percentile(np.min(rates, axis=-1), q=5, axis=0, interpolation="lower")   
        min_rates_medium = np.percentile(np.min(rates, axis=-1), q=50, axis=0, interpolation="lower")   
        robust_objectives[method_key] = min_rates
        print("[{}] Min-SINR: {:.3f}; Min-Rate: {:.3f}Mbps".format(method_key, np.mean(min_sinrs), np.mean(min_rates)/1e6))

    # Plot SINR CDF
    # get the plot bound
    lowerbound_plot, upperbound_plot = np.inf, -np.inf
    for val in robust_objectives.values():
        lowerbound_plot = min(lowerbound_plot, np.percentile(val,q=10,interpolation='lower'))
        upperbound_plot = max(upperbound_plot, np.percentile(val,q=90,interpolation='lower'))
    plt.xlabel("Min-Rates (Mbps)", fontsize=20)
    plt.ylabel("Cumulative Distribution of Robust Min-Rates", fontsize=20)
    plt.xticks(fontsize=21)
    plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0,stop=1,num=5)], fontsize=21)
    plt.grid(linestyle="dotted")
    plt.ylim(bottom=0)
    for method_key in all_power_loadings.keys():
        legend_label = method_key
        if legend_label == "Deep Learning without Uncertainty Injection":
             legend_label = "Deep Learning without \n Uncertainty Injection" 
        if legend_label == "Deep Learning with Uncertainty Injection":
             legend_label = "Deep Learning with \n Uncertainty Injection"
        plt.plot(np.sort(robust_objectives[method_key])/1e6, np.arange(1, N_TEST+1)/N_TEST, color=plot_colors[method_key], linestyle=plot_linestyles[method_key], linewidth=2.0, label=legend_label)
    plt.xlim(left=lowerbound_plot/1e6, right=upperbound_plot/1e6)
    plt.legend(prop={'size':20},loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.show()

    print("Script Completed Successfully!")
