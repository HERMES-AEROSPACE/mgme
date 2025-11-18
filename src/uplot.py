import numpy as np
from matplotlib import pyplot as plt


# data = np.load('simulation_data/U20.npy')
data1 = np.load('simulation_data/U0.npy')
data2 = np.load('simulation_data/U12.npy')
dsmc = np.loadtxt('src/dsmc.txt')

# plt.plot([1, 2, 4, 8, 14], [65.3/65.3, 65.3/37.7, 65.3/24.8, 65.3/22.1, 65.3/19.9], '^-', color='black')
# plt.xlabel('Threads', fontsize=16)
# plt.ylabel('Relative speedup', fontsize=16)
# plt.show()

x = np.linspace(-14, 10, 201)
x_scale = (x - x.min()) / (x.max() - x.min())

# y = np.sum(data, axis=1)[:, 0]
n1 = np.sum(data1, axis=1)[:, 0]
n2 = np.sum(data2, axis=1)[:, 0]
T2 = np.sum(data2, axis=1)[:, 4]
n1_scale = (n1 - n1.min()) / (n1[-1] - n1.min())
n2_scale = (n2 - n2.min()) / (n2[-1] - n2.min())
T2_scale = (T2 - T2.min()) / (T2[-1] - T2.min())

shock_thick = (np.max(n2_scale) - np.min(n2_scale)) / np.max(np.gradient(n2_scale, x))
shock_thick2 = (np.max(dsmc[:, 1]) - np.min(dsmc[:, 1])) / np.max(np.gradient(dsmc[:, 1], dsmc[:, 0]))
lambda_inf = 1.098
print(lambda_inf/shock_thick2, lambda_inf/shock_thick)

plt.rcParams['font.family'] = "serif"
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
ax1.plot(1 - dsmc[:, 0], dsmc[:, 1], '--', color='black')
ax1.plot(x_scale, n2_scale, color='red')
ax1.plot(x_scale, n1_scale, color='black')
ax1.set_xlabel(r'$X_j$', fontsize=20)
ax1.set_ylabel(r'Density', fontsize=20)
ax1.tick_params(axis='both',labelsize=16)
ax1.legend(['DSMC'], fontsize=14)
plt.tight_layout()

fig3 = plt.figure(figsize=(6, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(x_scale, n2_scale, color='black')
ax3.plot(x_scale, T2_scale, color='red')
ax3.set_xlabel(r'$X_j$', fontsize=20)
ax3.set_ylabel(r'Normalized property', fontsize=20)
ax3.tick_params(axis='both',labelsize=16)
ax3.legend(['Density', 'Temperature'], fontsize=14)
plt.tight_layout()

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(x_scale, (np.sum(data2, axis=1)[:, 0] - np.sum(data1, axis=1)[:, 0]), color='black')
ax2.set_xlabel(r'$X_j$', fontsize=20)
ax2.set_ylabel(r'Density difference', fontsize=20)
ax2.tick_params(axis='both',labelsize=16)
plt.tight_layout()
plt.show()