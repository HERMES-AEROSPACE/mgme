import numpy as np
from matplotlib import pyplot as plt

from .moment_utils import invert, moments, calc_moment, moment_eq
from .sampling import calculate_velocity_grid
from scipy import optimize, special


sol = optimize.least_squares(moment_eq, [0.01, -1.0, -1.0, 0.0], args=(-0.07781224965889887, -0.6134569726632962, 0.6158573426598, 1.7276316918440353, \
     -7.0, 0.4666666666666668, -7.0, 0.0, 0.0, 7.0), \
                                                bounds=([0.0, -7, -7, 0], [np.inf, 0.5, 0, 7]), method='trf', loss='soft_l1')
print(sol.x, np.linalg.norm(sol.fun))
b = sol.x[0]
wx = sol.x[1]
wy = sol.x[2]
wz = sol.x[3]
I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (0.4666666666666668 - wx)) - special.erf(np.sqrt(b) * (-7.0 - wx)))
I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (0.0 - wy)) - special.erf(np.sqrt(b) * (-7.0 - wy)))
I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (7.0 - wz)) - special.erf(np.sqrt(b) * (0.0 - wz)))
A = 0.01688446754551784 / (I0x * I0y * I0z)
# print(A, b, wx, wy, wz)

ci_cx = -6
cf_cx = 0
ci_cy = -6
cf_cy = 0
ci_cz = -6
cf_cz = 0

cx_vec = np.linspace(-6, 6, 241)
cy_vec = np.linspace(-6, 6, 241)
cz_vec = np.linspace(-6, 6, 241)
cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

# K = 1 - 0.4 * np.exp(-0/6)
# f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
# f0 = 1 / (np.pi**1.5) * np.exp(-1 * ((cx - .75  )**2 + cy**2 + cz**2))
f0 = 1 * (1.0 / (np.pi * 1))**1.5 * np.exp(-(1/1) * ((cx - 1.8256910592827011)**2 + cy**2 + cz**2))

mu = calc_moment(f0[121:241, 121:241, 121:241], cx[121:241, 121:241, 121:241], cy[121:241, 121:241, 121:241], cz[121:241, 121:241, 121:241], cx_vec[121:241], cy_vec[121:241], cz_vec[121:241])
print('moments:', mu[0], mu[1], mu[2], mu[3], mu[4])

initial_guess = [0.2, 0.0, 0.0, 0.0]
group_bounds = {'ci_cx': ci_cx, 'cf_cx': cf_cx, 'group_bounds_cx': np.array([0, 121]), \
                'ci_cy': ci_cy, 'cf_cy': cf_cy, 'group_bounds_cy': np.array([0, 121]), \
                    'ci_cz': ci_cz, 'cf_cz': cf_cz, 'group_bounds_cz': np.array([0, 121])}

A, b, wx, wy, wz = invert(mu, initial_guess, group_bounds)
# print(A, b, wx, wy, wz)

# Calculate integrals for testing flux calculation.
I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cx - wx)) - special.erf(np.sqrt(b) * (ci_cx - wx)))
I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cy - wy)) - special.erf(np.sqrt(b) * (ci_cy - wy)))
I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cz - wz)) - special.erf(np.sqrt(b) * (ci_cz - wz)))

I1x = (np.exp(-b * (ci_cx - wx)**2) - np.exp(-b * (cf_cx - wx)**2)) / (2 * b)
I1y = (np.exp(-b * (ci_cy - wy)**2) - np.exp(-b * (cf_cy - wy)**2)) / (2 * b)
I1z = (np.exp(-b * (ci_cz - wz)**2) - np.exp(-b * (cf_cz - wz)**2)) / (2 * b)

I2x = -np.sqrt(np.pi) / (2 * np.sqrt(b)) * \
    ((np.exp(-b * (cf_cx - wx)**2) * (cf_cx - wx))/np.sqrt(np.pi * b) - (np.exp(-b * (ci_cx - wx)**2) * (ci_cx - wx))/np.sqrt(np.pi * b)) + \
        np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (special.erf(np.sqrt(b) * (cf_cx - wx)) - special.erf(np.sqrt(b) * (ci_cx - wx)))
I2y = -np.sqrt(np.pi) / (2 * np.sqrt(b)) * \
    ((np.exp(-b * (cf_cy - wy)**2) * (cf_cy - wy))/np.sqrt(np.pi * b) - (np.exp(-b * (ci_cy - wy)**2) * (ci_cy - wy))/np.sqrt(np.pi * b)) + \
        np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (special.erf(np.sqrt(b) * (cf_cy - wy)) - special.erf(np.sqrt(b) * (ci_cy - wy)))
I2z = -np.sqrt(np.pi) / (2 * np.sqrt(b)) * \
    ((np.exp(-b * (cf_cz - wz)**2) * (cf_cz - wz))/np.sqrt(np.pi * b) - (np.exp(-b * (ci_cz - wz)**2) * (ci_cz - wz))/np.sqrt(np.pi * b)) + \
        np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (special.erf(np.sqrt(b) * (cf_cz - wz)) - special.erf(np.sqrt(b) * (ci_cz - wz)))

I3x = 1/b * I1x + 1 / (2 * b) * ((ci_cx - wx)**2 * np.exp(-b * (ci_cx - wx)**2) - (cf_cx - wx)**2 * np.exp(-b * (cf_cx - wx)**2))
I3y = 1/b * I1y + 1 / (2 * b) * ((ci_cy - wy)**2 * np.exp(-b * (ci_cy - wy)**2) - (cf_cy - wy)**2 * np.exp(-b * (cf_cy - wy)**2))
I3z = 1/b * I1z + 1 / (2 * b) * ((ci_cz - wz)**2 * np.exp(-b * (ci_cz - wz)**2) - (cf_cz - wz)**2 * np.exp(-b * (cf_cz - wz)**2))

F1 = A * (I1x + wx * I0x) * I0y * I0z
F2x = A * (I2x + wx**2 * I0x + 2 * wx * I1x) * I0y * I0z
F3 = A * ((I3x + I0x * wx**3 + 3 * I1x * wx**2 + 3 * I2x * wx) * I0y * I0z \
        + (I2y + 2 * wy * I1y + wy**2 * I0y) * (I1x + wx * I0x) * I0z + \
        (I2z + 2 * wz * I1z + wz**2 * I0z) * (I1x + wx * I0x) * I0y)

# Flux the integral way.
intF1 = np.trapezoid(np.trapezoid(np.trapezoid(cx[121:241, 121:241, 121:241] * f0[121:241, 121:241, 121:241], cz_vec[121:241], axis=2), cy_vec[121:241], axis=1), cx_vec[121:241], axis=0)
intF2 = np.trapezoid(np.trapezoid(np.trapezoid((cx[121:241, 121:241, 121:241]**2) * f0[121:241, 121:241, 121:241], cz_vec[121:241], axis=2), cy_vec[121:241], axis=1), cx_vec[121:241], axis=0)
ccTc = cx**3 + cy**2 * cx + cz**2 * cx
intF3 = np.trapezoid(np.trapezoid(np.trapezoid(ccTc[121:241, 121:241, 121:241] * f0[121:241, 121:241, 121:241], cz_vec[121:241], axis=2), cy_vec[121:241], axis=1), cx_vec[121:241], axis=0)
print('fluxes:', F1, F2x, F3)
print('flux the int way:', intF1, intF2, intF3)

f = A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2))
fI = np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1)
f0I = np.trapezoid(np.trapezoid(f0, cz_vec, axis=2), cy_vec, axis=1)

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
ax1.plot(cx_vec[0:241], f0I[0:241], color='red')
ax1.plot(cx_vec[0:241], fI[0:241], color='black')

plt.show()
