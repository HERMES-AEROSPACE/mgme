import numpy as np
from matplotlib import pyplot as plt

from .moment_utils import invert, moments, calc_moment, moment_eq
from .sampling import calculate_velocity_grid
from scipy import optimize, special


sol = optimize.least_squares(moment_eq, [0.27589075599575336, 3.598356344275178, -1.7102274992290973, 1.7054816195116538], args=(0.0070341124913311836, 0.6402750574510462, -0.6410232590766655, 2.090761964678449, \
     -7.0, 0.4666666666666668, 0.0, 7.0, -7.0, 0.0), \
                                                bounds=([0.0, -20, -20, -20], [100.0, 20, 20, 20]), method='trf', loss='soft_l1')
print(sol.x, np.linalg.norm(sol.fun))
b = sol.x[0]
wx = sol.x[1]
wy = sol.x[2]
wz = sol.x[3]
I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (7.0 - wx)) - special.erf(np.sqrt(b) * (2.0999999999999996 - wx)))
I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (0.0 - wy)) - special.erf(np.sqrt(b) * (-7.0 - wy)))
I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (7.0 - wz)) - special.erf(np.sqrt(b) * (0.0 - wz)))
A = 0.020356526229492456 / (I0x * I0y * I0z)
print(A, b, wx, wy, wz)

cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()

testf = A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2))

K = 1 - 0.4 * np.exp(-0/6)
f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)

mu = calc_moment(f0[108:121], cx[108:121], cy[108:121], cz[108:121], cx_vec[108:121], cy_vec, cz_vec)

initial_guess = [0.2, 0.0, 0.0, 0.0]
group_bounds = {'ci_cx': -0.5, 'cf_cx': 0.0, 'group_bounds_cx': np.array([108, 121]), \
                'ci_cy': -5.0, 'cf_cy': 5.0, 'group_bounds_cy': np.array([0, 241]), \
                    'ci_cz': -5.0, 'cf_cz': 5.0, 'group_bounds_cz': np.array([0, 241])}

A, b, wx, wy, wz = invert(mu, initial_guess, group_bounds)
# print(A, b, wx, wy, wz)
f = A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2))
# print(mu)
# mu = calc_moment(f[108:121], cx[108:121], cy[108:121], cz[108:121], cx_vec[108:121], cy_vec, cz_vec)
fI = np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1)
f0I = np.trapezoid(np.trapezoid(f0, cz_vec, axis=2), cy_vec, axis=1)
testfI = np.trapezoid(np.trapezoid(testf, cz_vec, axis=2), cy_vec, axis=1)

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
ax1.plot(cx_vec, testfI, color='red')
# ax1.plot(cx_vec, f0I, color='black')
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot_surface(bgrid[:, :, 10, 10], wxgrid[:, :, 10, 10], ux[:, :, 10, 10])
# # ax1.plot_surface(wygrid[10, 10, :, :], wzgrid[10, 10, :, :], ux[10, 10, : :])
# ax1.set_xlabel('b')
# ax1.set_ylabel('wx')
# ax1.set_zlabel('ux')
plt.show()
