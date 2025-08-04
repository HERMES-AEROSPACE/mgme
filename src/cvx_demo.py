import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt


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


np.random.seed(34957293)

cx_vec_smooth = np.linspace(-5, 5, 241)
cy_vec_smooth = np.linspace(-5, 5, 241)
cz_vec_smooth = np.linspace(-5, 5, 241)

cx_s, cy_s, cz_s = np.meshgrid(cx_vec_smooth, cy_vec_smooth, cz_vec_smooth, indexing='ij')
# f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx_s**2 + cy_s**2 + cz_s**2))
K = 1 - 0.4 * np.exp(-0/6)
f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx_s**2 + cy_s**2 + cz_s**2)) * np.exp(-(cx_s**2 + cy_s**2 + cz_s**2) / K)
f0_init = np.trapezoid(np.trapezoid(f0, cy_vec_smooth, axis=1), cx_vec_smooth, axis=0)

mu1 = calc_moment(f0[0:56], cx_s[0:56], cy_s[0:56], cz_s[0:56], cx_vec_smooth[0:56], cy_vec_smooth, cz_vec_smooth)

num_cx = 48
num_cy = 48
num_cz = 48

cx_vec = np.linspace(-5, 5, num_cx)
cy_vec = np.linspace(-5, 5, num_cy)
cz_vec = np.linspace(-5, 5, num_cz)
dx = cx_vec[1] - cx_vec[0]

cx_vec_group = cx_vec[(cx_vec > -5.0) & (cx_vec < -2.5)]
num_cx_group = len(cx_vec_group)

cx, cy, cz = np.meshgrid(cx_vec_group, cy_vec, cz_vec, indexing='ij')

x_sample = cx.flatten()
y_sample = cy.flatten()
z_sample = cz.flatten()

n = num_cx_group * num_cy * num_cz  # number of samples

A = np.zeros((5, n))
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

x = cp.Variable(shape=n)

# Least squares
cost = cp.sum_squares(A @ x - b)
prob2 = cp.Problem(cp.Minimize(cost), [x >= 0])
prob2.solve(verbose=True)

print("\nThe optimal value is:", prob2.value)
print('\nThe optimal solution is:')
print('density:', np.sum(x.value))
print('x-momentum:', np.sum(x.value * x_sample))
print('y-momentum:', np.sum(x.value * y_sample))
print('z-momentum:', np.sum(x.value * z_sample))
print('energy:', np.sum((x_sample**2 + y_sample**2 + z_sample**2) * x.value))

plt.rc('font', family='serif')
plt.hist(x_sample, weights=x.value / (dx), bins=num_cx_group, rwidth=0.2)
plt.plot(cx_vec_smooth[0:56], f0_init[0:56], color='black')
plt.xlabel('Cx', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.tight_layout()
plt.show()

# Entropy maximization
obj = cp.Maximize(cp.sum(cp.entr(x)))
constraints = [cp.sum(x) == mu1[0], x >= 0, cp.sum(cp.multiply(x_sample, x)) == mu1[1], \
               cp.sum(cp.multiply(y_sample, x)) == mu1[2], cp.sum(cp.multiply(z_sample, x)) == mu1[3], \
               cp.sum(cp.multiply(x_sample**2 + y_sample**2 + z_sample**2, x)) == mu1[4]]
prob = cp.Problem(obj, constraints)
prob.solve(verbose=True)

# Print result.
print("\nThe optimal value is:", prob.value)
print('\nThe optimal solution is:')
print('density:', np.sum(x.value))
print('x-momentum:', np.sum(x.value * x_sample))
print('y-momentum:', np.sum(x.value * y_sample))
print('z-momentum:', np.sum(x.value * z_sample))
print('energy:', np.sum((x_sample**2 + y_sample**2 + z_sample**2) * x.value))

plt.rc('font', family='serif')

unique_x = np.unique(x_sample)
integrated_weights = np.array([np.sum(x.value[x_sample == x_val]) for x_val in unique_x])

plt.hist(x_sample, weights=x.value / (dx), bins=num_cx_group, rwidth=0.2)
plt.plot(cx_vec_smooth[0:56], f0_init[0:56], color='black')
plt.xlabel('Cx', fontsize=18)
plt.ylabel('Density', fontsize=18)
# dx = cx_vec[1] - cx_vec[0]
# density = (x.value.reshape(-1, num_cy * num_cz).sum(axis=1)) / dx

# fig = plt.figure(figsize=(6, 6))
# ax1 = fig.add_subplot(111)
# ax1.bar(cx_vec, density, width=dx * 0.1, color='green')
# ax1.plot(cx_vec_smooth, f0_init, color='black')
# ax1.set_xlabel('Cx', fontsize=18)
# ax1.set_ylabel('Weight', fontsize=18)
plt.tight_layout()
plt.show()
