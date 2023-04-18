# Script for evaluating D2D network objectives: sum rate or min rate

import numpy as np
import torch
import matplotlib.pyplot as plt
from neural_net_D2D import Regular_Net, Robust_Net
from utils import *
from setup import *
from networks_generator_D2D import add_fadings
from benchmarks_D2D import *

N_REALIZATIONS = 1000
BENCHMARKS = ["Geometric Programming", "Full Power"]

if(__name__ =='__main__'):
    pl = np.load("Data_D2D/pl_test_{}.npy".format(SETTING_STRING))
    assert np.shape(pl)[0] == N_TEST
    print(f"[D2D] Evaluate {SETTING_STRING} over {N_TEST} layouts.")

    power_controls, plot_colors, plot_linestyles = {}, {}, {}
    # Geometric Programming
    if "Geometric Programming" in BENCHMARKS:
        power_controls["Geometric Programming"] = GP_power_control(pl)
        plot_colors["Geometric Programming"] = 'b'
        plot_linestyles["Geometric Programming"] = '-.'
    # Regular Deep Learning
    regular_net = Regular_Net().to(DEVICE)
    _, pc = regular_net(torch.tensor(pl, dtype=torch.float32).to(DEVICE))
    power_controls["Deep Learning without Uncertainty Injection"] = pc.detach().cpu().numpy()
    plot_colors["Deep Learning without Uncertainty Injection"] = 'm'
    plot_linestyles["Deep Learning without Uncertainty Injection"] = '--'
    # Robust Deep Learning
    robust_net = Robust_Net().to(DEVICE)
    _, pc = robust_net(torch.tensor(pl, dtype=torch.float32).to(DEVICE))
    power_controls["Deep Learning with Uncertainty Injection"] = pc.detach().cpu().numpy()
    plot_colors["Deep Learning with Uncertainty Injection"] = 'r'
    plot_linestyles["Deep Learning with Uncertainty Injection"] = '-'
    # Full Power
    if "Full Power" in BENCHMARKS:
        power_controls["Full Power"] = np.ones([N_TEST, N_LINKS])
        plot_colors["Full Power"] = 'k'
        plot_linestyles["Full Power"] = ':'

    # EVALUATION
    print("Power Allocation Percentages: ")
    for method_key, power_percentages in power_controls.items():
        print("[{}]: {:.3f}%".format(method_key, np.mean(power_percentages)*100))

    print("Nominal Objectives: ")
    nominal_objectives = {}
    for method_key in power_controls.keys():
        sinrs = compute_SINRs(power_controls[method_key], pl)
        rates = compute_rates(sinrs)
        assert np.shape(sinrs) == np.shape(rates) == (N_TEST, N_LINKS)
        min_sinrs, min_rates = np.min(sinrs, axis=1), np.min(rates, axis=1) 
        nominal_objectives[method_key] = min_rates
        print("[{}] Min-SINR: {:.3f}dB / Min-Rate: {:.3f}Mbps".format(method_key, 10*np.log10(np.mean(min_sinrs)), np.mean(min_rates)/1e6))

    print("Robust Objectives:")
    robust_objectives, robust_rates = {}, {}
    csi = add_fadings(np.tile(pl, reps=[N_REALIZATIONS, 1, 1]))
    for method_key in power_controls.keys():
        pc = np.tile(power_controls[method_key], reps=[N_REALIZATIONS, 1])
        sinrs = compute_SINRs(pc, csi)
        sinrs = np.reshape(sinrs, [N_REALIZATIONS, N_TEST, N_LINKS])
        rates = compute_rates(sinrs)
        min_sinrs = np.percentile(np.min(sinrs, axis=-1), q=10, axis=0, interpolation="lower")
        min_rates = np.percentile(np.min(rates, axis=-1), q=10, axis=0, interpolation="lower")
        robust_objectives[method_key] = min_rates
        robust_rates[method_key] = rates
        print("[{}] Min-SINR: {:.3f}dB / Min-Rate: {:.3f}Mbps".format(method_key, 10*np.log10(np.mean(min_sinrs)), np.mean(min_rates)/1e6))


    # Plot the CDF curve
    # get the lower bound plot
    lowerbound_plot, upperbound_plot = np.inf, -np.inf
    for val in robust_objectives.values():
        lowerbound_plot = min(lowerbound_plot, np.percentile(val,q=10, interpolation="lower"))
        upperbound_plot = max(upperbound_plot, np.percentile(val,q=90, interpolation="lower"))

    fig = plt.figure()
    plt.xlabel("Min-Rate (Mbps)", fontsize=20)
    plt.ylabel("Cumulative Distribution of Robust Min-Rates", fontsize=20)
    plt.xticks(fontsize=21)
    plt.yticks(np.linspace(start=0, stop=1, num=5), ["{}%".format(int(i*100)) for i in np.linspace(start=0, stop=1, num=5)], fontsize=21)
    plt.grid(linestyle="dotted")
    plt.ylim(bottom=0)
    for method_key in power_controls.keys():
        legend_label = method_key
        if legend_label == "Deep Learning without Uncertainty Injection":
            legend_label = "Deep Learning without \n Uncertainty Injection"
        if legend_label == "Deep Learning with Uncertainty Injection":
            legend_label = "Deep Learning with \n Uncertainty Injection"
        plt.plot(np.sort(robust_objectives[method_key])/1e6, np.arange(1,N_TEST+1)/N_TEST, color=plot_colors[method_key], linestyle=plot_linestyles[method_key], linewidth=2.0, label=legend_label)
    plt.xlim(left=lowerbound_plot/1e6, right=upperbound_plot/1e6)
    plt.legend(prop={'size':20}, loc='lower right')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.show()

    # Inspect the allocation difference: for the worst link under 5-percentile rate
    #methods_plot = ["Geometric Programming", "Regular Deep Learning", "Robust Deep Learning"]
    #layout_idx = np.random.randint(N_TEST)
    #fig, axes = plt.subplots(1+len(methods_plot), 1)
    #bar_width = 0.1
    #for method_idx, method_key in enumerate(methods_plot):
    #    rates = robust_rates[method_key]
    #    # get the index of uncertainty realization where the 5-percentile rate occurs
    #    realization_idx = np.argsort(np.min(rates,axis=-1),axis=0)[int(N_TEST*0.1),:][layout_idx]
    #    # firstly, plot link rate (where 5-percentile rate occurs for each method)
    #    axes[0].bar(np.arange(N_LINKS)-2.0*(1.0-method_idx)*bar_width, rates[realization_idx, layout_idx], bar_width, label=method_key)
    #    # then, plot power allocation by each method
    #    axes[1].bar(np.arange(N_LINKS)-2.0*(1.0-method_idx)*bar_width, 100*power_controls[method_key][layout_idx], bar_width, label=method_key)
    #for ax in axes:
    #    ax.set_xlim(left=-0.5, right=N_LINKS-0.5)
    #    ax.set_xticks(np.arange(N_LINKS))
    #    for x_separate in np.linspace(start=-0.5, stop=N_LINKS-0.5, num=N_LINKS+1):
    #        ax.axvline(x=x_separate, linewidth=0.5, linestyle="--")
    #    ax.legend()
    #plt.show()        

    print("Script Completed Successfully!")
