import numpy as np
from matplotlib import pyplot as plt
from .config import CONSTANTS, FREESTREAM_PARAMS
from .moment_utils import invert
from scipy.interpolate import interp1d


# data = np.load('simulation_data/U20.npy')
data1 = np.load('simulation_data/U0.npy')
data2 = np.load('simulation_data/U25.npy')
dsmc = np.loadtxt('src/dsmc.txt')
dsmcT = np.loadtxt('src/dsmcT.txt')

# plt.plot([1, 2, 4, 8, 14], [65.3/65.3, 65.3/37.7, 65.3/24.8, 65.3/22.1, 65.3/19.9], '^-', color='black')
# plt.xlabel('Threads', fontsize=16)
# plt.ylabel('Relative speedup', fontsize=16)
# plt.show()

x = np.linspace(-25, 25, 201)
x2 = np.linspace(-30, 20, 251)
x_scale = (x - x.min()) / (x.max() - x.min())
x2_scale = (x2 - x2.min()) / (x2.max() - x2.min())

# y = np.sum(data, axis=1)[:, 0]
n1 = np.sum(data1, axis=1)[:, 0]
n2 = np.sum(data2, axis=1)[:, 0]
u2 = np.sum(data2, axis=1)[:, 1]
T2 = np.sum(data2, axis=1)[:, 4]
temperature = 2/3 * ((T2 / n2) - (u2 / n2)**2)
n1_scale = (n1 - n1[0]) / (n1[-1] - n1[0])
n2_scale = (n2 - n2[0]) / (n2[-1] - n2[0])
u2_scale = (u2 - u2[0]) / (u2[-1] - u2[0])
T2_scale = (T2 - T2[0]) / (T2[-1] - T2[0])
temperature_scale = (temperature - temperature[0]) / (temperature[-1] - temperature[0])

# shock_thick = (np.max(n2_scale) - np.min(n2_scale)) / np.max(np.gradient(n2_scale, x))

gamma = CONSTANTS['gamma']
R = CONSTANTS['R']
T1 = FREESTREAM_PARAMS['T1']
P1 = FREESTREAM_PARAMS['P1']

a1 = np.sqrt(gamma * R * T1)
rho1 = P1/(R * T1)
mu1 = 2.26e-5  # [Pa * s]

# shock_thick2 = (np.max(dsmc[:, 1]) - np.min(dsmc[:, 1])) / np.max(np.gradient(dsmc[:, 1], dsmc[:, 0]))
lambda_inf = 16/5 * (gamma / (2 * np.pi))**0.5 * mu1 / (rho1 * a1) * 1000
# lambda_inf = 1.098
# print(lambda_inf, lambda_inf/shock_thick)

# Calculate some distributions and plot them.
p = 100
negx_n = np.sum(data2[:, 0:4], axis=1)[p, 0]
posx_n = np.sum(data2[:, 4:], axis=1)[p, 0]
negx_u = np.sum(data2[:, 0:4], axis=1)[p, 1]
posx_u = np.sum(data2[:, 4:], axis=1)[p, 1]
negx_e = np.sum(data2[:, 0:4], axis=1)[p, 4]
posx_e = np.sum(data2[:, 4:], axis=1)[p, 4]

# point = (x[p] - x.min()) / (x.max() - x.min())
# A, b, wx, _, _ = invert([negx_n, negx_u, 0.0, 0.0, negx_e], [1.0, 0.0, 0.0, 0.0], {'ci_cx': -7, 'cf_cx': 0.4666666666666668, 'ci_cy': -7, 'cf_cy': 7, 'ci_cz': -7, 'cf_cz': 7})
# cx_vec, cy_vec, cz_vec = np.linspace(-7, 7, 121), np.linspace(-7, 7, 121), np.linspace(-7, 7, 121)
# cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
# negx_f = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

# A, b, wx, _, _ = invert([posx_n, posx_u, 0.0, 0.0, posx_e], [1.0, 0.0, 0.0, 0.0], {'ci_cx': 0.4666666666666668, 'cf_cx': 7, 'ci_cy': -7, 'cf_cy': 7, 'ci_cz': -7, 'cf_cz': 7})
# cx_vec, cy_vec, cz_vec = np.linspace(-7, 7, 121), np.linspace(-7, 7, 121), np.linspace(-7, 7, 121)
# cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
# posx_f = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

f = interp1d(1 - dsmc[:, 0], dsmc[:, 1], kind='cubic')
ft = interp1d(1 - dsmcT[:, 0], dsmcT[:, 1], kind='cubic')
x_new = np.linspace(0.03, 0.99, 20)

plt.rcParams['font.family'] = "serif"

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
ax1.plot(1 - dsmc[:, 0], dsmc[:, 1], '--', color='green')
# ax1.scatter(x_new, f(x_new), color='green', marker='s', facecolors='none')
ax1.plot(1 - dsmcT[:, 0], dsmcT[:, 1], '--', color='red')
# ax1.scatter(x_new, ft(x_new), color='red', marker='s', facecolors='none')
ax1.plot(x_scale, temperature_scale, color='red')
# ax1.plot(x2_scale - 0.01814, n1_scale, color='purple')
ax1.plot(x_scale, n2_scale, color='green')
ax1.set_xlabel(r'Scaled Location', fontsize=20)
ax1.set_ylabel(r'Normalized Property', fontsize=20)
ax1.tick_params(axis='both',labelsize=16)
ax1.set_xlim(0.0, 1.0)
ax1.legend(['DSMC - n', 'DSMC - T', r'T', r'n'], fontsize=14)
plt.savefig('simulation_data/density_shock.jpg', bbox_inches='tight')
plt.tight_layout()

# fig3 = plt.figure(figsize=(6, 6))
# ax3 = fig3.add_subplot(111)
# ax3.plot(cx_vec[0:65], negx_f[0:65], color='green')
# ax3.plot(cx_vec[64:], posx_f[64:], color='red')
# ax3.set_xlabel(r'$C_x$', fontsize=20)
# ax3.set_ylabel(r'f', fontsize=20)
# ax3.tick_params(axis='both',labelsize=16)
# ax3.legend([r'Group $x_0$', r'Group $x_1$'], fontsize=14, loc='upper left')
# ax3.set_title(r'$X_j$ = {}'.format(point), fontsize=18)
# plt.savefig('simulation_data/dist1.jpg', bbox_inches='tight')
# plt.tight_layout()

# fig2 = plt.figure(figsize=(6, 6))
# ax2 = fig2.add_subplot(111)
# ax2.plot(x_scale, (np.sum(data2, axis=1)[:, 0] - np.sum(data1, axis=1)[:, 0]), color='black')
# ax2.set_xlabel(r'$X_j$', fontsize=20)
# ax2.set_ylabel(r'Density difference', fontsize=20)
# ax2.tick_params(axis='both',labelsize=16)
# plt.tight_layout() 
plt.show()