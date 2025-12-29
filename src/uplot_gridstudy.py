import numpy as np
from matplotlib import pyplot as plt


n_fine = np.load('simulation_data/n_avg.npy')
T_fine = np.load('simulation_data/T_avg.npy')
n_coarse = np.load('simulation_data/n_avg_coarse.npy')
T_coarse = np.load('simulation_data/T_avg_coarse.npy')

x_fine = np.linspace(-25, 25, 201)
x_coarse = np.linspace(-25, 25, 101)

def scaler(x):
    return (x - x.min()) / (x.max() - x.min())

x_scale_fine = scaler(x_fine)
x_scale_coarse = scaler(x_coarse)

plt.rcParams['font.family'] = "serif"

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)

ax1.plot(x_scale_fine, n_fine, color='green')
ax1.plot(x_scale_fine, T_fine, color='red')

ax1.plot(x_scale_coarse, n_coarse, '--', color='green')
ax1.plot(x_scale_coarse, T_coarse, '--', color='red')

ax1.set_xlabel(r'Scaled Location', fontsize=20)
ax1.set_ylabel(r'Normalized Property', fontsize=20)
ax1.tick_params(axis='both',labelsize=16)
ax1.set_xlim(0.0, 1.0)
ax1.legend(['n - fine', 'T - fine', r'n - coarse', r'T - coarse'], fontsize=14)
plt.tight_layout()
plt.show()