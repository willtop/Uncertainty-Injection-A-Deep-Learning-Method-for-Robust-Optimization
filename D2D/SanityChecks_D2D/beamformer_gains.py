import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from setup import *

n_points = 1001
midpoint = int((n_points - 1)/2) # zero indexing

thetas = np.linspace(start=0,stop=np.pi,num=n_points,endpoint=True)
assert thetas[midpoint] > np.pi/2-1e-3 and thetas[midpoint] < np.pi/2+1e-3
beamformer = np.ones(N_ANTENNAS)/np.sqrt(N_ANTENNAS)
assert np.linalg.norm(beamformer) > 1-1e-3 and np.linalg.norm(beamformer) < 1+1e-3
gains = []

delta = ANTENNA_SEPARATION / WAVELENGTH

for theta in thetas:
    response_vector = np.exp(-np.arange(start=0,stop=-N_ANTENNAS,step=-1)*2*np.pi*delta*np.cos(theta)*1j)
    gain = np.power(np.abs(np.dot(np.conjugate(beamformer), response_vector)),2)
    gains.append(10*np.log10(gain))

thetas = thetas/np.pi*180

# plt.title(f"Linear antenna array beamforming pattern with {N_ANTENNAS} antennas", fontdict={'fontsize':17})
plt.xlabel("Angle of Incidence/Arrival", fontdict={'fontsize':20})
plt.ylabel("Gain (dB)", fontdict={'fontsize':20})
plt.xlim(left=0, right=180)
plt.xticks(np.arange(start=0,stop=181,step=10), fontsize=21)
plt.ylim(bottom=-35,top=20)
plt.grid()
plt.plot(thetas, gains, linewidth=1.5)
# label the max point
plt.annotate('Max Gain: {:.3f}dB'.format(gains[midpoint]), (thetas[midpoint], gains[midpoint]+3), ha='center', fontsize=20)
# label the main-lobe level
plt.hlines(y=MAIN_LOBE_GAIN_dB,xmin=80,xmax=100,linestyle='--',color='r',linewidth=2.3)
# label the direct-link gain
# plt.vlines(x=90,ymin=MAIN_LOBE_GAIN_dB,ymax=DIRECTLINK_GAIN_dB,linestyle='-',color='r',linewidth=2.5)
# label the side-lobe level
plt.hlines(y=SIDE_LOBE_GAIN_dB,xmin=0,xmax=80,linestyle='--',color='r',linewidth=2.3)
plt.hlines(y=SIDE_LOBE_GAIN_dB,xmin=100,xmax=180,linestyle='--',color='r',linewidth=2.3)
plt.vlines(x=80,ymin=SIDE_LOBE_GAIN_dB,ymax=MAIN_LOBE_GAIN_dB,linestyle='--',color='r',linewidth=2.3)
plt.vlines(x=100,ymin=SIDE_LOBE_GAIN_dB,ymax=MAIN_LOBE_GAIN_dB,linestyle='--',color='r',linewidth=2.3)
yticks_list = list(plt.yticks()[0])+[MAIN_LOBE_GAIN_dB,SIDE_LOBE_GAIN_dB]
yticks_list.remove(-10)
plt.yticks(yticks_list, fontsize=21)
plt.tick_params(axis='both', labelsize=21)
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.09, top=0.97)
plt.show()

