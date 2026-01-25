import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from .moment_utils import invert
from scipy import special
from scipy.stats import qmc


def calc_flux(A, b, wx, wy, wz, bounds):
    F = np.zeros(5)

    ci_cx = bounds['ci_cx']
    cf_cx = bounds['cf_cx']
    ci_cy = bounds['ci_cy']
    cf_cy = bounds['cf_cy']
    ci_cz = bounds['ci_cz']
    cf_cz = bounds['cf_cz']

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
    # I3y = 1/b * I1y + 1 / (2 * b) * ((ci_cy - wy)**2 * np.exp(-b * (ci_cy - wy)**2) - (cf_cy - wy)**2 * np.exp(-b * (cf_cy - wy)**2))
    # I3z = 1/b * I1z + 1 / (2 * b) * ((ci_cz - wz)**2 * np.exp(-b * (ci_cz - wz)**2) - (cf_cz - wz)**2 * np.exp(-b * (cf_cz - wz)**2))

    F1 = A * (I1x + wx * I0x) * I0y * I0z
    F2x = A * (I2x + wx**2 * I0x + 2 * wx * I1x) * I0y * I0z
    F3 = A * ((I3x + I0x * wx**3 + 3 * I1x * wx**2 + 3 * I2x * wx) * I0y * I0z + \
        (I2y + 2 * wy * I1y + wy**2 * I0y) * (I1x + wx * I0x) * I0z + \
        (I2z + 2 * wz * I1z + wz**2 * I0z) * (I1x + wx * I0x) * I0y)
    F = np.array([F1, F2x, 0.0, 0.0, F3])

    return F

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    # Density moment
    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Momentum moment
    uk = cx * f
    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cy * f
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cz * f
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Energy moment
    c2 = cx**2 + cy**2 + cz**2
    ek = c2 * f
    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid(ek, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return mu 


# np.random.seed(34957293)

cx_vec_smooth = np.linspace(-5, 5.5, 106)
cy_vec_smooth = np.linspace(-5, 5.5, 106)
cz_vec_smooth = np.linspace(-5, 5.5, 106)

cx_s, cy_s, cz_s = np.meshgrid(cx_vec_smooth, cy_vec_smooth, cz_vec_smooth, indexing='ij')
f0 = 1 / (np.pi**1.5) * np.exp(-1 * ((cx_s - 1.8256910592827011)**2 + cy_s**2 + cz_s**2))
# K = 1 - 0.4 * np.exp(-0/6)
# f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx_s**2 + cy_s**2 + cz_s**2)) * np.exp(-(cx_s**2 + cy_s**2 + cz_s**2) / K)
# f0_init = np.trapezoid(np.trapezoid(f0[100:121, 0:121, 0:121], cz_vec_smooth[0:121], axis=2), cy_vec_smooth[0:121], axis=1)

ici = 0
icf = 121
mu1 = calc_moment(f0[ici:icf, 0:121, 0:121], cx_s[ici:icf, 0:121, 0:121], cy_s[ici:icf, 0:121, 0:121], cz_s[ici:icf, 0:121, 0:121], \
    cx_vec_smooth[ici:icf], cy_vec_smooth[0:121], cz_vec_smooth[0:121])

mu1 = np.array([0.0007763650355286552, -0.0016730375615871755, 0.00019104354358810463, 0.00018702198653392654, 0.007557816246825353])
bounds = {'ci_cx': -5.0, 'cf_cx': -0.5, \
        'ci_cy': 0.0, 'cf_cy': 5.5,\
        'ci_cz': 0.0, 'cf_cz': 5.5}

# cx_vec = np.linspace(bounds['ci_cx'], bounds['cf_cx'], num_cx, endpoint=False) + (bounds['cf_cx'] - bounds['ci_cx']) / (2 * num_cx)
# cy_vec = np.linspace(bounds['ci_cy'], bounds['cf_cy'], num_cy, endpoint=False) + (bounds['cf_cy'] - bounds['ci_cy']) / (2 * num_cy)
# cz_vec = np.linspace(bounds['ci_cz'], bounds['cf_cz'], num_cz, endpoint=False) + (bounds['cf_cz'] - bounds['ci_cz']) / (2 * num_cz)
# cx_vec = np.linspace(bounds['ci_cx'], bounds['cf_cx'], num_cx)
# cy_vec = np.linspace(bounds['ci_cy'], bounds['cf_cy'], num_cy)
# cz_vec = np.linspace(bounds['ci_cz'], bounds['cf_cz'], num_cz)
# dx = cx_vec[1] - cx_vec[0]
# dy = cy_vec[1] - cy_vec[0]
# dz = cz_vec[1] - cz_vec[0]

# cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

# x_sample = cx.flatten()
# y_sample = cy.flatten()
# z_sample = cz.flatten()
num_sample = 2500
l_bounds = [-5.0, 0.0, 0.0]
u_bounds = [-0.5, 5.5, 5.5]
sampler = qmc.LatinHypercube(d=3)
sample = qmc.scale(sampler.random(n=num_sample), l_bounds, u_bounds)

x_sample = sample[:, 0]
y_sample = sample[:, 1]
z_sample = sample[:, 2]

A = np.zeros((5, num_sample))
b = np.zeros(5)
A[0, :] = 1
A[1, :] = x_sample
A[2, :] = y_sample
A[3, :] = z_sample
A[4, :] = x_sample**2 + y_sample**2 + z_sample**2
b[0] = mu1[0]
b[1] = mu1[1]
b[2] = mu1[2]
b[3] = mu1[3]
b[4] = mu1[4]

x = cp.Variable(shape=num_sample, nonneg=True)

# Least squares
# cost = cp.sum_squares(A @ x - b)
# prob2 = cp.Problem(cp.Minimize(cost), [x >= 0])
# prob2.solve(verbose=True)

# print("\nThe optimal value is:", prob2.value)
# print('\nThe optimal solution is:')
# print('density:', np.sum(x.value))
# print('x-momentum:', np.sum(x.value * x_sample))
# print('y-momentum:', np.sum(x.value * y_sample))
# print('z-momentum:', np.sum(x.value * z_sample))
# print('energy:', np.sum((x_sample**2 + y_sample**2 + z_sample**2) * x.value))

# plt.rc('font', family='serif')
# plt.hist(x_sample, weights=x.value / (dx), bins=num_cx_group, rwidth=0.2)
# plt.plot(cx_vec_smooth[0:56], f0_init[0:56], color='black')
# plt.xlabel('Cx', fontsize=18)
# plt.ylabel('Density', fontsize=18)
# plt.tight_layout()
# plt.show()

# Entropy maximization
obj = cp.Maximize(cp.sum(cp.entr(x)))
constraints = [cp.sum(x) == mu1[0], cp.sum(cp.multiply(x_sample, x)) == mu1[1], \
               cp.sum(cp.multiply(y_sample, x)) == mu1[2], cp.sum(cp.multiply(z_sample, x)) == mu1[3], \
               cp.sum(cp.multiply(x_sample**2 + y_sample**2 + z_sample**2, x)) == mu1[4]]

prob = cp.Problem(obj, constraints)
prob.solve(verbose=True, tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-6)

real_weight = x.value

# Inversion.
A, b, wx, wy, wz = invert(mu1, [0.001, 0.0, 0.0, 0.0], bounds)
f_invert = A * np.exp(-b * ((cx_s - wx)**2 + (cy_s - wy)**2 + (cz_s - wz)**2))
fI_invert = np.trapezoid(np.trapezoid(f_invert[0:46, 50:, 50:], cz_vec_smooth[50:], axis=2), cy_vec_smooth[50:], axis=1)

# Compute flux using inversion vs. integrating weights.
flux = calc_flux(A, b, wx, wy, wz, bounds)

# Group temperature investigation.
ux = mu1[1] / mu1[0]
uy = mu1[2] / mu1[0]
uz = mu1[3] / mu1[0]
T = 2/3 * ((mu1[4] / mu1[0]) - (ux**2 + uy**2 + uz**2))
print(ux - 3*np.sqrt(T), uy - 3 * np.sqrt(T), uz - 3 * np.sqrt(T))
print(ux + 3*np.sqrt(T), uy + 3 * np.sqrt(T), uz + 3 * np.sqrt(T))

# Print result.
print("\nThe optimal value is:", prob.value)
print('\nThe optimal solution is:')
print('density:', np.sum(real_weight))
print('x-momentum:', np.sum(real_weight * x_sample))
print('y-momentum:', np.sum(real_weight * y_sample))
print('z-momentum:', np.sum(real_weight * z_sample))
print('energy:', np.sum((x_sample**2 + y_sample**2 + z_sample**2) * real_weight))
print(mu1)

# width = (bounds['cf_cx'] - bounds['ci_cx'])/num_cx
# half_width = width /2
# bin_edges = np.array([cx_vec[0] - half_width] + 
#                       [cx_vec[i] + half_width for i in range(num_cx)])

# counts, ed = np.histogram(x_sample, weights=real_weight / dx, bins=bin_edges)
# shape_weights = np.reshape(real_weight / (dx*dy*dz), (10, 8, 8))
# f1 = np.trapezoid(np.trapezoid(np.trapezoid(cx * shape_weights, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
# f2 = np.trapezoid(np.trapezoid(np.trapezoid(cx**2 * shape_weights, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
ccTc = x_sample**3 + y_sample**2 * x_sample + z_sample**2 * x_sample
# ccTc_s = cx_s**3 + cy_s**2 * cx_s + cz_s**2 * cx_s
# f3 = np.trapezoid(np.trapezoid(np.trapezoid(ccTc * shape_weights, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
f1 = np.sum(real_weight * x_sample)
f2 = np.sum(real_weight * x_sample**2)
f3 = np.sum(real_weight * ccTc)

# f1_invert = np.trapezoid(np.trapezoid(np.trapezoid(cx_s[100:121, 0:121, 0:121] * f_invert[100:121, 0:121, 0:121], \
#     cz_vec_smooth[0:121], axis=2), cy_vec_smooth[0:121], axis=1), cx_vec_smooth[ici:icf], axis=0)
# f2_invert = np.trapezoid(np.trapezoid(np.trapezoid(cx_s[100:121, 0:121, 0:121]**2 * f_invert[100:121, 0:121, 0:121], \
#     cz_vec_smooth[0:121], axis=2), cy_vec_smooth[0:121], axis=1), cx_vec_smooth[ici:icf], axis=0)
# f3_invert = np.trapezoid(np.trapezoid(np.trapezoid(ccTc_s[100:121, 0:121, 0:121] * f_invert[100:121, 0:121, 0:121], \
#     cz_vec_smooth[0:121], axis=2), cy_vec_smooth[0:121], axis=1), cx_vec_smooth[ici:icf], axis=0)
print(flux)
print(f1, f2, f3)
# print(f1_invert, f2_invert, f3_invert)

plt.rc('font', family='serif')
# fig = plt.figure(figsize=(6, 6))
# ax1 = fig.add_subplot(111)
# plt.hist(x_sample, weights=real_weight / dx, bins=bin_edges, rwidth=0.85)
# ax1.plot(cx_vec, counts, '-o', color='red', linewidth=2)
# ax1.plot(cx_vec_smooth[ici:icf], fI_invert, '--', color='black')
# ax1.plot(cx_vec_smooth[ici:icf], f0_init, color='black')
# ax1.set_xlabel('Cx', fontsize=18)
# ax1.set_ylabel('Density', fontsize=18)
# ax1.tick_params(axis='both', labelsize=14)
# ax1.legend(['CVX samples', 'Inverted distribution', 'True distribution'], fontsize=14)
# dx = cx_vec[1] - cx_vec[0]
# density = (x.value.reshape(-1, num_cy * num_cz).sum(axis=1)) / dx

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
# ax1.bar(cx_vec, density, width=dx * 0.1, color='green')
ax1.plot(cy_vec_smooth[0:46], fI_invert, color='black')
ax1.set_xlabel('Cy', fontsize=18)
ax1.set_ylabel('f', fontsize=18)
ax1.set_yscale('log')
plt.tight_layout()
plt.savefig('plots/cvx_fI_invert.pdf')
plt.show()
