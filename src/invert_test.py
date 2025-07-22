import numpy as np
from matplotlib import pyplot as plt

from .moment_utils import invert, moments
from .config import calculate_velocity_grid


mu = np.array([ 0.00808594, -0.00223938, -0.00223938, -0.00223938,  0.00236188])
# mu2 = np.array([ 0.01756732, -0.00423398, -0.00419785, -0.00416251,  0.00409948])
initial_guess = [1.0, 0.0, 0.0, 0.0]
group_bounds = {'ci_cx': -0.5, 'cf_cx': 0.0, 'group_bounds_cx': np.array([100, 121]), \
                'ci_cy': -0.5, 'cf_cy': 0.0, 'group_bounds_cy': np.array([100, 121]), \
                    'ci_cz': -0.5, 'cf_cz': 0.0, 'group_bounds_cz': np.array([100, 121])}
# group_bounds2 = {'ci_cx': -0.5, 'cf_cx': 0.0, 'group_bounds_cx': np.array([100, 121]), \
                #  'ci_cy': -0.5, 'cf_cy': 0.0, 'group_bounds_cy': np.array([100, 121]), \
                # 'ci_cz': -0.5, 'cf_cz': 0.0, 'group_bounds_cz': np.array([100, 121])}

A, b, wx, wy, wz = invert(mu, initial_guess, group_bounds)
print(A, b, wx, wy, wz)
# A, b, wx, wy, wz = invert(mu2, initial_guess, group_bounds2)
# print(A, b, wx, wy, wz)

# bgrid, wxgrid, wygrid, wzgrid = np.meshgrid(np.linspace(1e-5, 10.0, 20), np.linspace(-3, 3, 20), np.linspace(-3, 3, 20), \
#                         np.linspace(-3, 3, 20), indexing='ij')
# ux, uy, uz, e = moments(bgrid, wxgrid, wygrid, wzgrid, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0)

cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
f = 9905.559065736357 * np.exp(-0.10866370008187215 * ((cx - 6.313558874611846)**2 + (cy - 6.313839727517143)**2 + (cz - 6.3137320210358565)**2))
K = 1 - 0.4 * np.exp(-0/6)
f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
ax1.plot(cx_vec[100:121], np.trapz(np.trapz(f[100:121, 100:121, 100:121], cz_vec[100:121], axis=2), cy_vec[100:121], axis=1))
ax1.plot(cx_vec, np.trapz(np.trapz(f0, cz_vec, axis=2), cy_vec, axis=1))
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot_surface(bgrid[:, :, 10, 10], wxgrid[:, :, 10, 10], ux[:, :, 10, 10])
# # ax1.plot_surface(wygrid[10, 10, :, :], wzgrid[10, 10, :, :], ux[10, 10, : :])
# ax1.set_xlabel('b')
# ax1.set_ylabel('wx')
# ax1.set_zlabel('ux')
plt.show()
