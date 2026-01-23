import numpy as np
from matplotlib import pyplot as plt
from .config import CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE
from .moment_utils import invert
from scipy.interpolate import interp1d


# data = np.load('simulation_data/U20.npy')
data1 = np.load('simulation_data/U2000.npy')
data2 = np.load('simulation_data/U1180.npy')
dsmc = np.loadtxt('src/dsmc.txt')
dsmcT = np.loadtxt('src/dsmcT.txt')

# plt.plot([1, 2, 4, 8, 14], [65.3/65.3, 65.3/37.7, 65.3/24.8, 65.3/22.1, 65.3/19.9], '^-', color='black')
# plt.xlabel('Threads', fontsize=16)
# plt.ylabel('Relative speedup', fontsize=16)
# plt.show()

x = np.linspace(*PHYS_SPACE['xj_range'], PHYS_SPACE['num_xj'])
x2 = np.linspace(-30, 20, 126)
x_scale = (x - x.min()) / (x.max() - x.min())
x2_scale = (x2 - x2.min()) / (x2.max() - x2.min())

# y = np.sum(data, axis=1)[:, 0]
n1 = np.sum(data1, axis=1)[:, 0]
n2 = np.sum(data2, axis=1)[:, 0]
u1 = np.sum(data1, axis=1)[:, 1]
u2 = np.sum(data2, axis=1)[:, 1]
T1 = np.sum(data1, axis=1)[:, 4]
T2 = np.sum(data2, axis=1)[:, 4]
temperature1 = 2/3 * ((T1 / n1) - (u1 / n1)**2)
temperature2 = 2/3 * ((T2 / n2) - (u2 / n2)**2)
vel2 = u2 / n2
n1_scale = (n1 - n1[0]) / (n1[-1] - n1[0])
n2_scale = (n2 - n2[0]) / (n2[-1] - n2[0])
u2_scale = (u2 - u2[0]) / (u2[-1] - u2[0])
T1_scale = (T1 - T1[0]) / (T1[-1] - T1[0])
T2_scale = (T2 - T2[0]) / (T2[-1] - T2[0])
temperature1_scale = (temperature1 - temperature1[0]) / (temperature1[-1] - temperature1[0])
temperature2_scale = (temperature2 - temperature2[0]) / (temperature2[-1] - temperature2[0])
vel2_scale = (vel2 - vel2[-1]) / (vel2[0] - vel2[-1])

shock_thick = (np.max(n2_scale) - np.min(n2_scale)) / np.max(np.abs(np.gradient(n2_scale, x_scale)))

gamma = CONSTANTS['gamma']
R = CONSTANTS['R']
T1 = FREESTREAM_PARAMS['T1']
P1 = FREESTREAM_PARAMS['P1']

a1 = np.sqrt(gamma * R * T1)
rho1 = P1/(R * T1)
mu1 = 2.26e-5  # [Pa * s]

shock_thick2 = (np.max(dsmc[:, 1]) - np.min(dsmc[:, 1])) / np.max(np.abs(np.gradient(dsmc[:, 1], dsmc[:, 0])))
# lambda_inf = 16/5 * (gamma / (2 * np.pi))**0.5 * mu1 / (rho1 * a1) * 1000
lambda_inf = 1.098
print('DSMC:', lambda_inf/shock_thick2, 'Current:', lambda_inf/shock_thick)

# Calculate some distributions and plot them.
p = 70
nx1 = np.sum(data2[p, 0:4], axis=0)[0]
nx2 = np.sum(data2[p, 4:8], axis=0)[0]
# nx3 = np.sum(data2[p, 8:], axis=0)[0]
ux1 = np.sum(data2[p, 0:4], axis=0)[1]
ux2 = np.sum(data2[p, 4:8], axis=0)[1]
# ux3 = np.sum(data2[p, 8:], axis=0)[1]
ex1 = np.sum(data2[p, 0:4], axis=0)[4]
ex2 = np.sum(data2[p, 4:8], axis=0)[4]
# ex3 = np.sum(data2[p, 8:], axis=0)[4]

point = (x[p] - x.min()) / (x.max() - x.min())
cx_vec, cy_vec, cz_vec = np.linspace(-5, 5.5, 106), np.linspace(-5, 5.5, 106), np.linspace(-5, 5.5, 106)
cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

A, b, wx, _, _ = invert([nx1, ux1, 0.0, 0.0, ex1], [1.0, 0.0, 0.0, 0.0], {'ci_cx': -5, 'cf_cx': 0.6, 'ci_cy': -5, 'cf_cy': 5.5, 'ci_cz': -5, 'cf_cz': 5.5})
fx1 = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

# A, b, wx, _, _ = invert([nx2, ux2, 0.0, 0.0, ex2], [1.0, 0.0, 0.0, 0.0], {'ci_cx': -0.5, 'cf_cx': 1.2, 'ci_cy': -5, 'cf_cy': 5.5, 'ci_cz': -5, 'cf_cz': 5.5})
# fx2 = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

A, b, wx, _, _ = invert([nx2, ux2, 0.0, 0.0, ex2], [1.0, 0.0, 0.0, 0.0], {'ci_cx': 0.6, 'cf_cx': 5.5, 'ci_cy': -5, 'cf_cy': 5.5, 'ci_cz': -5, 'cf_cz': 5.5})
fx2 = np.trapezoid(np.trapezoid(A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)), cz_vec, axis=2), cy_vec, axis=1)

# Interpolate to get smooth curves of the DSMC data.
f = interp1d(1 - dsmc[:, 0], dsmc[:, 1], kind='cubic')
ft = interp1d(1 - dsmcT[:, 0], dsmcT[:, 1], kind='cubic')
x_new = np.linspace(0.03, 0.99, 20)

# Time average the data for anything noisy due to low collisions.
# n_avg = np.zeros(PHYS_SPACE['num_xj'])
# T_avg = np.zeros(PHYS_SPACE['num_xj'])
# avg_t = 40
# for i in range(630, 630 + avg_t):
#     data = np.load('simulation_data/U{}.npy'.format(i))
#     n = np.sum(data, axis=1)[:, 0]
#     u = np.sum(data, axis=1)[:, 1]
#     e = np.sum(data, axis=1)[:, 4]
#     T = 2/3 * ((e / n) - (u / n)**2)
#     n_scale = (n - n[0]) / (n[-1] - n[0])
#     T_scale = (T - T[0]) / (T[-1] - T[0])
#     n_avg += n_scale
#     T_avg += T_scale

# n_avg /= avg_t
# T_avg /= avg_t
# np.save('simulation_data/n_avg_coarse.npy', n_avg)
# np.save('simulation_data/T_avg_coarse.npy', T_avg)

plt.rcParams['font.family'] = "serif"
x_scale_shifted = x_scale + 0.08176

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
# ax1.plot(x_scale, temperature1_scale, color='indigo')
ax1.plot(x_scale, temperature2_scale, color='red')
# ax1.plot(x_scale, n1_scale, color='purple')
ax1.plot(x_scale, n2_scale, color='green')
# ax1.plot(x_scale, vel2_scale, color='blue')
# ax1.plot(x_scale, n_avg, '-.', color='green')
# ax1.plot(x_scale, T_avg, '-.', color='red')
# ax1.plot(1 - dsmc[:, 0], dsmc[:, 1], '--', color='green')
ax1.scatter(x_new, f(x_new), color='green', marker='s', facecolors='none')
# ax1.plot(1 - dsmcT[:, 0], dsmcT[:, 1], '--', color='red')
ax1.scatter(x_new, ft(x_new), color='red', marker='s', facecolors='none')

ax1.set_xlabel(r'Scaled Location', fontsize=20)
ax1.set_ylabel(r'Normalized Property', fontsize=20)
ax1.tick_params(axis='both',labelsize=16)
# ax1.set_xlim(-30.0, 20.0)
ax1.legend(['T', 'n', 'DSMC - n', 'DSMC - T'], fontsize=14)
plt.savefig('simulation_data/density_shock.jpg', bbox_inches='tight')
ax1.grid()
plt.tight_layout()

fig3 = plt.figure(figsize=(6, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(cx_vec[0:57], fx1[0:57], color='green')
ax3.plot(cx_vec[56:106], fx2[56:106], color='red')
# ax3.plot(cx_vec[62:], fx3[62:], color='blue')
ax3.set_xlabel(r'$C_x$', fontsize=20)
ax3.set_ylabel(r'f', fontsize=20)
ax3.tick_params(axis='both',labelsize=16)
ax3.legend([r'Group $x_0$', r'Group $x_1$', r'Group $x_2$'], fontsize=14, loc='upper left')
ax3.set_title(r'$X_j$ = {}'.format(point), fontsize=18)
plt.savefig('simulation_data/dist1.jpg', bbox_inches='tight')
plt.tight_layout()

plt.show()