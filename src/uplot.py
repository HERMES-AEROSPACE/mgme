import numpy as np
from matplotlib import pyplot as plt


# data = np.load('simulation_data/U20.npy')
data1 = np.load('simulation_data/U108.npy')
data2 = np.load('simulation_data/U109.npy')
dsmc = np.loadtxt('src/dsmc.txt')

# plt.plot([1, 2, 4, 8, 14], [65.3/65.3, 65.3/37.7, 65.3/24.8, 65.3/22.1, 65.3/19.9], '^-', color='black')
# plt.xlabel('Threads', fontsize=16)
# plt.ylabel('Relative speedup', fontsize=16)
# plt.show()

x = np.linspace(-14, 10, 201)
x_scale = (x - x.min()) / (x.max() - x.min())

# y = np.sum(data, axis=1)[:, 0]
y1 = np.sum(data2, axis=1)[:, 0]
y2 = np.sum(data2, axis=1)[:, 4]
y1_scale = (y1 - y1.min()) / (y1.max() - y1.min())
y2_scale = (y2 - y2.min()) / (y2.max() - y2.min())

shock_thick = (np.max(y2_scale) - np.min(y2_scale)) / np.max(np.gradient(y2_scale, x))
lamba_inf = 1.098
print(lamba_inf, lamba_inf/shock_thick)

plt.rcParams['font.family'] = "serif"
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)

# ax1.plot(x, y, color='black')
ax1.plot(x_scale, y2_scale, color='red')
ax1.plot(x_scale, y1_scale, color='black')
# ax1.plot(1 - dsmc[:, 0], dsmc[:, 1], '--', color='black')
ax1.set_xlabel(r'$X_j$', fontsize=20)
ax1.set_ylabel(r'Density', fontsize=20)
ax1.tick_params(axis='both',labelsize=16)
plt.tight_layout()
# plt.plot(x_scale, np.sum(data0, axis=1)[:, 0], color='red')

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111)
ax2.plot((np.sum(data2, axis=1)[:, 0] - np.sum(data1, axis=1)[:, 0]), color='black')
ax2.set_xlabel(r'$X_j$', fontsize=20)
ax2.set_ylabel(r'Density difference', fontsize=20)
ax2.tick_params(axis='both',labelsize=16)
plt.tight_layout()
plt.show()