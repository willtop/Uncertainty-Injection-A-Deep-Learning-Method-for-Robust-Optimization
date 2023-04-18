import numpy as np
import matplotlib.pyplot as plt

alphas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5]
robust_min_rates = [3.783, 3.788, 3.815, 3.853, 4.254, 4.814, 5.718, 6.364, 6.827, 7.170, 8.066, 8.595, 8.785]
medium_min_rates = [11.756, 11.784, 11.915, 12.090, 13.130, 13.724, 14.092, 14.122, 14.045, 13.938, 13.421, 12.781, 11.920]
nominal_min_rates = [74.706, 74.695, 72.085, 67.833, 47.115, 36.814, 28.332, 24.511, 22.303, 20.848, 17.517, 15.454, 13.713]
first_n_vals = 10
plt.xlabel('alpha in Regularized Zero-Forcing')
plt.ylabel('rates in Mbps')
plt.xticks(ticks=alphas, labels=alphas)
plt.plot(alphas[:first_n_vals], robust_min_rates[:first_n_vals], c='r', marker='*', label='robust min rate')
plt.plot(alphas[:first_n_vals], medium_min_rates[:first_n_vals], c='b', marker='*', label='medium min rate')
#plt.semilogy(alphas[:first_n_vals], nominal_min_rates[:first_n_vals], c='g', label='nominal min rate')
plt.legend()
plt.show()
