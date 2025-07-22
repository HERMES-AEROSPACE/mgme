import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt


def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    # Density moment
    mu[0] = np.trapz(np.trapz(np.trapz(f, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Momentum moment
    uk = cx * f
    mu[1] = np.trapz(np.trapz(np.trapz(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cy * f
    mu[2] = np.trapz(np.trapz(np.trapz(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cz * f
    mu[3] = np.trapz(np.trapz(np.trapz(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Energy moment
    c2 = cx**2 + cy**2 + cz**2
    ek = c2 * f
    mu[4] = np.trapz(np.trapz(np.trapz(ek, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return mu 


np.random.seed(34957293)

cx_vec_smooth = np.linspace(-3, 3, 241)
cy_vec_smooth = np.linspace(-3, 3, 241)
cz_vec_smooth = np.linspace(-3, 3, 241)

cx_s, cy_s, cz_s = np.meshgrid(cx_vec_smooth, cy_vec_smooth, cz_vec_smooth, indexing='ij')
# f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx_s**2 + cy_s**2 + cz_s**2))
K = 1 - 0.4 * np.exp(-0/6)
f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx_s**2 + cy_s**2 + cz_s**2)) * np.exp(-(cx_s**2 + cy_s**2 + cz_s**2) / K)
f0_init = np.trapz(np.trapz(f0, cy_vec_smooth, axis=1), cx_vec_smooth, axis=0)

mu1 = calc_moment(f0[100:121], cx_s[100:121], cy_s[100:121], cz_s[100:121], cx_vec_smooth[100:121], cy_vec_smooth, cz_vec_smooth)
print(mu1)

num_cx = 48
num_cy = 48
num_cz = 48

cx_vec = np.linspace(-3, 3, num_cx)
cy_vec = np.linspace(-3, 3, num_cy)
cz_vec = np.linspace(-3, 3, num_cz)
dx = cx_vec[1] - cx_vec[0]

cx_vec_group = cx_vec[(cx_vec > -0.5) & (cx_vec < 0.0)]
num_cx_group = len(cx_vec_group)

cx, cy, cz = np.meshgrid(cx_vec_group, cy_vec, cz_vec, indexing='ij')

x_sample = cx.flatten()
y_sample = cy.flatten()
z_sample = cz.flatten()

n = num_cx_group * num_cy * num_cz  # number of samples

x = cp.Variable(shape=n)
obj = cp.Maximize(cp.sum(cp.entr(x)))

ux = cp.Parameter()
ux.value = mu1[1]

constraints = [cp.sum(x) == mu1[0], x >= 0, cp.sum(cp.multiply(x_sample, x)) == ux, \
               cp.sum(cp.multiply(y_sample, x)) == mu1[2], cp.sum(cp.multiply(z_sample, x)) == mu1[3], \
               cp.sum(cp.multiply(x_sample**2 + y_sample**2 + z_sample**2, x)) == mu1[4]]
prob = cp.Problem(obj, constraints)
# prob.solve(verbose=True)

test = np.array([prob])
test[0].solve(verbose=True)

ux.value = mu1[1] + 1e-2
test[0].solve(verbose=True)

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
plt.plot(cx_vec_smooth[100:121], f0_init[100:121], color='black')
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
