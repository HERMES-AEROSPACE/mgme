import numpy as np
from matplotlib import pyplot as plt
from .config_1d import CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE, GROUP_PARAMS, VELOCITY_SPACE
from .shock_helper import calculate_velocity_grid
from .moment_utils import invert
from scipy.interpolate import interp1d
from scipy import special


# data = np.load('simulation_data/U20.npy')
data1 = np.load('simulation_data/U2400.npy')
data2 = np.load('simulation_data/U1090.npy')
dsmc = np.loadtxt('src/dsmc.txt')
dsmcT = np.loadtxt('src/dsmcT.txt')
dsmc_hard = np.loadtxt('src/dsmc_hard.txt')
dsmcT_hard = np.loadtxt('src/dsmcT_hard.txt')
dsmc_vhs = np.loadtxt('src/dsmc_vhs.txt')
dsmcT_vhs = np.loadtxt('src/dsmcT_vhs.txt')

alsmeyer_205 = np.loadtxt('src/alsmeyer_205.txt')
# plt.plot([1, 2, 4, 8, 14], [65.3/65.3, 65.3/37.7, 65.3/24.8, 65.3/22.1, 65.3/19.9], '^-', color='black')
# plt.xlabel('Threads', fontsize=16)
# plt.ylabel('Relative speedup', fontsize=16)
# plt.show()

x = np.linspace(*PHYS_SPACE['xj_range'], PHYS_SPACE['num_xj'])
# x = np.linspace(-20, 20, 201)
x2 = np.linspace(-15, 15, 101)

R = CONSTANTS['R']
m = CONSTANTS['m']
T1 = FREESTREAM_PARAMS['T1']
P1 = FREESTREAM_PARAMS['P1']
d_ref = CONSTANTS['d']
gamma = CONSTANTS['gamma']
rho1    = P1 / (R * T1)
n_ref = P1/(R * T1) * 1/m
sigma_ref = np.pi * d_ref**2
lam_ref = 1/(n_ref * sigma_ref)

omega = 0.811
# this is lam_vhs / lam_ref
mu_argon_300K = 2.2948e-5   # Pa.s
lam_alsmeyer  = 16/5 * mu_argon_300K / (rho1 * np.sqrt(2 * np.pi * R * T1))
print(f'lam_ref      = {lam_ref*1000:.4f} mm')
print(f'lam_alsmeyer = {lam_alsmeyer*1000:.4f} mm  (target: 1.098 mm)')

x_scale = x #* lam_ref/0.001098
x2_scale = x2 * (lam_ref / lam_alsmeyer)

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
temperature2_scale = (temperature2 - temperature2.min()) / (temperature2.max() - temperature2.min())
vel2_scale = (vel2 - vel2[-1]) / (vel2[0] - vel2[-1])

shock_thick  = np.max(np.abs(np.gradient(n2_scale, x2_scale)))
shock_thick_als  = np.max(np.abs(np.gradient(alsmeyer_205[:, 1], alsmeyer_205[:, 0])))

shock_thick_dsmc = np.max(np.abs(np.gradient(dsmc[:, 1], dsmc[:, 0]))) / 50.0
shock_thick_vhs = np.max(np.abs(np.gradient(dsmc_vhs[:, 1], dsmc_vhs[:, 0]))) / 50.0
print('Alsmeyer:', shock_thick_als, 'Current:', shock_thick)

# Calculate some distributions and plot them.
p = 50
point = x2[p]
fx_groups = []
cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)

for i in range(0, 5):
    ci = GROUP_PARAMS['ci_cx'][i]
    cf = GROUP_PARAMS['cf_cx'][i]
    group_slice = slice(i*1, (i+1)*1)

    nx = np.sum(data2[p, group_slice], axis=0)[0]
    ux = np.sum(data2[p, group_slice], axis=0)[1]
    ex = np.sum(data2[p, group_slice], axis=0)[4]

    # Use a tighter initial guess for non-first groups
    initial_guess = [1.0, 0.0, 0.0, 0.0]

    A, b, wx, _, _ = invert(
        [nx, ux, 0.0, 0.0, ex],
        initial_guess,
        {'ci_cx': ci, 'cf_cx': cf,
         'ci_cy': -5, 'cf_cy': 5.5,
         'ci_cz': -5, 'cf_cz': 5.5}
    )

    fx = np.trapezoid(
             np.trapezoid(
                 A * np.exp(-b * ((cx - wx)**2 + cy**2 + cz**2)),
             cz_vec, axis=2),
         cy_vec, axis=1)

    fx_groups.append(fx)
    print(f'Group {i}: cx=[{ci}, {cf}], A={A:.4f}, b={b:.4f}, wx={wx:.4f}')


# Interpolate to get smooth curves of the DSMC data.
f = interp1d(x2_scale, n2_scale, kind='cubic')
ft = interp1d(x2_scale, temperature2_scale, kind='cubic')
x_new = np.linspace(0.00, 1.0, 48)

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

plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'mathtext.fontset': 'cm',   # Computer Modern — same font as LaTeX
})
interp    = interp1d(n2_scale, x2_scale)
x_center  = interp(0.5)

x_centered = x2_scale - x_center
mask = (x_centered >= -7.8) & (x_centered <= 9.1)
idx  = np.where(mask)[0]
f_als       = interp1d(alsmeyer_205[:, 0], alsmeyer_205[:, 1],
                        kind='cubic', bounds_error=False, fill_value=(0.0, 1.0))
x_als_fine  = np.linspace(alsmeyer_205[0, 0], alsmeyer_205[-1, 0], 300)
y_als_fine  = f_als(x_als_fine)

fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.plot(x_centered[idx[::3]], n2_scale[idx[::3]], color='black', linewidth=1.6)
ax1.plot(x_centered[idx[::3]], temperature2_scale[idx[::3]], color='red', linewidth=1.6)
ax1.scatter(x_als_fine[0::10], y_als_fine[0::10],  color='black', marker='o', s=40, linewidths=1.3, facecolors='none')
# ax1.plot(x2_scale + 0.07, n1_scale, color='purple')
# ax1.plot(x2_scale + 0.055, temperature1_scale, color='indigo')
# ax1.plot(x_scale, vel2_scale, color='blue')
# ax1.scatter(x_plot, y_plot, color='black', marker='o', facecolors='none', s=60, linewidths=1.3, zorder=5)
# ax1.plot(1 - dsmc[:, 0], dsmc[:, 1], color='black', linewidth=1.3)
# ax1.plot(1 - dsmc_hard[:, 0], dsmc_hard[:, 1], '--', color='green')
# ax1.plot(1 - dsmc_vhs[:, 0], dsmc_vhs[:, 1], '--', color='green')
# ax1.scatter(x_plot, y_plot, color='red', marker='o', facecolors='none', s=60, linewidths=1.3, zorder=5)
# ax1.plot(1 - dsmcT[:, 0], dsmcT[:, 1], color='red', linewidth=1.3)
# ax1.plot(1 - dsmcT_hard[:, 0], dsmcT_hard[:, 1], '--', color='red')
# ax1.plot(1 - dsmcT_vhs[:, 0], dsmcT_vhs[:, 1], '--', color='red')


ax1.set_xlabel(r'$x/\lambda_{1}$ ', fontsize=18)
ax1.set_ylabel(r'Normalized $n$, $T$', fontsize=18)
ax1.tick_params(axis='both',labelsize=16)
ax1.legend([r'$n$', r'$T$', r'Alsmeyer - $n$', r'DSMC - $T$'], fontsize=16)
ax1.tick_params(axis='both', direction='out', length=6, width=1.2)
ax1.minorticks_on()
# ax1.grid()
# ax1.set_xlim(-10, 10)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_linewidth(1.2)
plt.tight_layout()
plt.savefig('plots/profile.pdf')

fig3 = plt.figure(figsize=(6, 6))
ax3 = fig3.add_subplot(111)
for i in range(0, 5):
    bounds = GROUP_PARAMS['group_bounds_cx'][i]
    group_slice = slice(bounds[0], bounds[1])
    ax3.plot(cx_vec[group_slice], fx_groups[i][group_slice], linewidth=1.4)
ax3.set_xlabel(r'$C_x$', fontsize=20)
ax3.set_ylabel(r'f', fontsize=20)
ax3.tick_params(axis='both',labelsize=16)
# ax3.legend([r'Group $x_0$', r'Group $x_1$', r'Group $x_2$'], fontsize=14, loc='upper left')
ax3.set_title(r'$X_j$ = {}'.format(point), fontsize=18)
plt.tight_layout()
plt.savefig('plots/dist.pdf')


plt.show()