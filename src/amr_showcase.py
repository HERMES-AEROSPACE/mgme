import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from .moment_utils import invert, calc_moment
from .amr import calculate_hellinger_distance


class GroupNode:
    def __init__(self, data: dict):
        self.group_bounds = data
        self.children = []

    def set_mu(self, mu):
        self.mu = mu

    def set_dist_param(self, A, b, wx, wy, wz):
        self.A = A
        self.b = b
        self.wx = wx
        self.wy = wy
        self.wz = wz

    def set_hellinger_distance(self, dist):
        self.hellinger_dist = dist

    def add_child(self, child):
        self.children.append(child)

    def update_parameters(self, dt, dn, dpx, dpy, dpz, de):
        self.mu[0] += dt * dn
        self.mu[1] += dt * dpx
        self.mu[2] += dt * dpy
        self.mu[3] += dt * dpz
        self.mu[4] += dt * de

        if self.mu[0] > 1e-4:
            self._update_group_dist_params([self.b, self.wx, self.wy, self.wz])
    
    def _update_group_dist_params(self, initial_guess):
        self.A, self.b, self.wx, self.wy, self.wz = invert(self.mu, initial_guess, self.group_bounds)


cx_vec = np.linspace(-7, 7, 121)
cy_vec = np.linspace(-7, 7, 121)
cz_vec = np.linspace(-7, 7, 121)
cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

f0 = 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 3)**2 + cy**2 + cz**2))
f02 = 0.04 / (np.pi**1.5) * np.exp(-0.2 * ((cx + 1)**2 + cy**2 + cz**2))
second_fraction = 0.5

plt.rcParams['font.family'] = "serif"
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlabel(r'C_x', fontsize=18)
ax1.set_ylabel('f', fontsize=18)
ax1.tick_params(axis='both', labelsize=14)

# Set up tree structure to keep track of the distributions.
root = GroupNode({'ci_cx': -7, 'cf_cx': 7, 'group_bounds_cx': np.array([0, 121]),
                  'ci_cy': -7, 'cf_cy': 7, 'group_bounds_cy': np.array([0, 121]),
                  'ci_cz': -7, 'cf_cz': 7, 'group_bounds_cz': np.array([0, 121])})

# Calculate moments in the big group.
mu = calc_moment(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)

# Invert to get the initial distribution and calculate the Hellinger distance.
A, b, wx, wy, wz = invert(mu, [1.0, mu[1], mu[2], mu[3]], root.group_bounds)
f = A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2))
h_dist = calculate_hellinger_distance(f0, f, cx_vec, cy_vec, cz_vec)
root.set_dist_param(A, b, wx, wy, wz)
root.set_hellinger_distance(h_dist)

# Add the two initial child nodes.
child1 = GroupNode({'ci_cx': -7, 'cf_cx': 0, 'group_bounds_cx': np.array([0, 61]),
                  'ci_cy': -7, 'cf_cy': 7, 'group_bounds_cy': np.array([0, 121]),
                  'ci_cz': -7, 'cf_cz': 7, 'group_bounds_cz': np.array([0, 121])})
child2 = GroupNode({'ci_cx': 0, 'cf_cx': 7, 'group_bounds_cx': np.array([60, 121]),
                  'ci_cy': -7, 'cf_cy': 7, 'group_bounds_cy': np.array([0, 121]),
                  'ci_cz': -7, 'cf_cz': 7, 'group_bounds_cz': np.array([0, 121])})
root.add_child(child1)
root.add_child(child2)

# n_step = 50
# for t in range(0, n_step):
def animate(t):
    ax1.clear()
    # Calculate input moments to the simulation.
    alpha = t / 99

    f01 = 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 3 + 0.05 * t)**2 + cy**2 + cz**2))
    weight2 = second_fraction * alpha
    weight1 = 1 - weight2

    ft = weight1 * f01 + weight2 * f02

    # Update the root distribution.
    mu = calc_moment(ft, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    A, b, wx, wy, wz = invert(mu, [1.0, mu[1], mu[2], mu[3]], root.group_bounds)
    f = A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2))

    # Calculate the moments in the two child groups.
    mu1 = calc_moment(ft[0:61], cx[0:61], cy[0:61], cz[0:61], cx_vec[0:61], cy_vec, cz_vec)
    mu2 = calc_moment(ft[60:121], cx[60:121], cy[60:121], cz[60:121], cx_vec[60:121], cy_vec, cz_vec)

    # Invert and compare to the root distribution.
    A1, b1, wx1, wy1, wz1 = invert(mu1, [1.0, mu1[1], mu1[2], mu1[3]], child1.group_bounds)
    A2, b2, wx2, wy2, wz2 = invert(mu2, [1.0, mu2[1], mu2[2], mu2[3]], child2.group_bounds)
    f1 = A1 * np.exp(-b1 * ((cx - wx1)**2 + (cy - wy1)**2 + (cz - wz1)**2))
    f2 = A2 * np.exp(-b2 * ((cx - wx2)**2 + (cy - wy2)**2 + (cz - wz2)**2))
    h_dist1 = calculate_hellinger_distance(f[0:61], f1[0:61], cx_vec[0:61], cy_vec, cz_vec)
    h_dist2 = calculate_hellinger_distance(f[60:121], f2[60:121], cx_vec[60:121], cy_vec, cz_vec)
    print(t, h_dist1, h_dist2)

    ax1.plot(cx_vec, np.trapezoid(np.trapezoid(ft, cz_vec, axis=2), cy_vec, axis=1), color='black')
    # In actuality, instead of these if statements, we should refine another level and run the whole thing again.
    # Also the values here would be replaced with the frontier of nodes instead of hard-coded.
    if h_dist1 > 0.05 or h_dist2 > 0.05:
        ax1.plot(cx_vec[0:61], np.trapezoid(np.trapezoid(f1[0:61], cz_vec, axis=2), cy_vec, axis=1), color='indigo')
        ax1.plot(cx_vec[60:121], np.trapezoid(np.trapezoid(f2[60:121], cz_vec, axis=2), cy_vec, axis=1), color='firebrick')
        ax1.legend(['test f', 'group 1', 'group 2'], fontsize=14, loc='upper right')
    else:
        ax1.plot(cx_vec, np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1), color='firebrick')
        ax1.legend(['test f', 'group 1'], fontsize=14, loc='upper right')

    ax1.set_ylim(-0.05, 0.6)


anim = animation.FuncAnimation(fig, animate, frames=91, interval=100)
anim.save('amr_showcase.mp4')
