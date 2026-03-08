import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy import optimize, special
import h5py


def moment_eq(x, ux, uy, uz, e, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    """
    Moment equation for solving distribution parameters.
    
    Args:
        x: Array containing [beta, wx, wy, wz]
        u: Target velocity
        e: Target energy
        ci: Lower bound
        cf: Upper bound
    
    Returns:
        Array of moment equations
    """
    I0x = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cx - x[1])) - special.erf(np.sqrt(x[0]) * (ci_cx - x[1])))
    I0y = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cy - x[2])) - special.erf(np.sqrt(x[0]) * (ci_cy - x[2])))
    I0z = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cz - x[3])) - special.erf(np.sqrt(x[0]) * (ci_cz - x[3])))

    I1x = (np.exp(-x[0] * (ci_cx - x[1])**2) - np.exp(-x[0] * (cf_cx - x[1])**2)) / (2 * x[0])
    I1y = (np.exp(-x[0] * (ci_cy - x[2])**2) - np.exp(-x[0] * (cf_cy - x[2])**2)) / (2 * x[0])
    I1z = (np.exp(-x[0] * (ci_cz - x[3])**2) - np.exp(-x[0] * (cf_cz - x[3])**2)) / (2 * x[0])

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cx - x[1])**2) * (cf_cx - x[1]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cx - x[1])**2) * (ci_cx - x[1])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cx - x[1])) - special.erf(np.sqrt(x[0]) * (ci_cx - x[1])))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cy - x[2])**2) * (cf_cy - x[2]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cy - x[2])**2) * (ci_cy - x[2])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cy - x[2])) - special.erf(np.sqrt(x[0]) * (ci_cy - x[2])))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cz - x[3])**2) * (cf_cz - x[3]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cz - x[3])**2) * (ci_cz - x[3])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cz - x[3])) - special.erf(np.sqrt(x[0]) * (ci_cz - x[3])))

    return [(I1x + x[1] * I0x) / I0x - ux, (I1y + x[2] * I0y) / I0y - uy, (I1z + x[3] * I0z) / I0z - uz, \
            (I2x + 2 * x[1] * I1x + x[1]**2 * I0x) / (I0x) + (I2y + 2 * x[2] * I1y + x[2]**2 * I0y) / (I0y) + (I2z + 2 * x[3] * I1z + x[3]**2 * I0z) / (I0z) - e]

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    """
    Calculate moments (density, momentum, energy) for a given distribution function.
    
    Args:
        f: Distribution function
        cx, cy, cz: Velocity components
        cx_vec, cy_vec, cz_vec: Velocity grid vectors
    
    Returns:
        Array of moments [density, x-momentum, y-momentum, z-momentum, energy]
    """
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

def invert(mu, initial_guess, group_bounds):
    A, b, wx, wy, wz = 0.0, 0.0, 0.0, 0.0, 0.0
    ci_cx = group_bounds['ci_cx']
    cf_cx = group_bounds['cf_cx']
    ci_cy = group_bounds['ci_cy']
    cf_cy = group_bounds['cf_cy']
    ci_cz = group_bounds['ci_cz']
    cf_cz = group_bounds['cf_cz']

    sol = optimize.least_squares(moment_eq, initial_guess, args=(mu[1] / mu[0], mu[2] / mu[0], \
                                    mu[3] / mu[0], mu[4] / mu[0], \
                                    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), \
                                        bounds=([0.0, -10, -10, -10], [np.inf, 10, 10, 10]), method='trf', loss='soft_l1')
    # print(sol)
    # print('residual:', np.linalg.norm(sol.fun))
    
    # sol = optimize.root(moment_eq, initial_guess, args=(mu[1] / mu[0], mu[2] / mu[0], \
    #                                 mu[3] / mu[0], mu[4] / mu[0], \
    #                                     ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), method='lm')
    if sol.success:
        b = sol.x[0]
        wx = sol.x[1]
        wy = sol.x[2]
        wz = sol.x[3]
    
    I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cx'] - wx)) - special.erf(np.sqrt(b) * (group_bounds['ci_cx'] - wx)))
    I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cy'] - wy)) - special.erf(np.sqrt(b) * (group_bounds['ci_cy'] - wy)))
    I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cz'] - wz)) - special.erf(np.sqrt(b) * (group_bounds['ci_cz'] - wz)))
    A = mu[0] / (I0x * I0y * I0z)

    return A, b, wx, wy, wz

def calculate_hellinger_distance(f1, f2, cx_vec, cy_vec, cz_vec):
    """
    Calculate the Hellinger distance between two distributions in a specific group.
    Make sure distributions are normalized to 1!!!!!!
    
    Args:
        f1, f2: Group distribution functions
        cx_vec, cy_vec, cz_vec: Velocity space vectors
        
    Returns:
        Hellinger distance between f1 and f2
    """
    # Calculate Hellinger distance
    # H(P,Q) = √(1/2) * √(∫(√P(x) - √Q(x))² dx)
    diff = np.sqrt(f1) - np.sqrt(f2)
    squared_diff = diff**2
    
    # Integrate over the group volume
    integral = np.trapezoid(np.trapezoid(np.trapezoid(squared_diff, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return np.sqrt(0.5 * integral)

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

output_file = "amr_showcase_data.h5"
frame_data = {}

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

    # --- Compute marginals (1D projections over cy, cz) ---
    ft_marginal = np.trapezoid(np.trapezoid(ft,  cz_vec, axis=2), cy_vec, axis=1)
    f_marginal  = np.trapezoid(np.trapezoid(f,   cz_vec, axis=2), cy_vec, axis=1)
    f1_marginal = np.trapezoid(np.trapezoid(f1[0:61],  cz_vec, axis=2), cy_vec, axis=1)
    f2_marginal = np.trapezoid(np.trapezoid(f2[60:121], cz_vec, axis=2), cy_vec, axis=1)

    frame_data[t] = {
        "t":          t,
        "ft":         ft_marginal,
        "f":          f_marginal,
        "f1":         f1_marginal,   # defined on cx_vec[0:61]
        "f2":         f2_marginal,   # defined on cx_vec[60:121]
        "h_dist1":    h_dist1,
        "h_dist2":    h_dist2,
        "refined":    int(h_dist1 > 0.05 or h_dist2 > 0.05),
    }

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

def save_frame_data(path=output_file):
    with h5py.File(path, "w") as hf:
        # Shared axis — stored once
        hf.create_dataset("cx_vec", data=cx_vec)

        for t, fd in sorted(frame_data.items()):
            grp = hf.create_group(f"frame_{fd['t']:04d}")
            grp.attrs["t"]       = fd["t"]
            grp.attrs["h_dist1"] = fd["h_dist1"]
            grp.attrs["h_dist2"] = fd["h_dist2"]
            grp.attrs["refined"] = fd["refined"]
            grp.create_dataset("ft", data=fd["ft"])
            grp.create_dataset("f",  data=fd["f"])
            grp.create_dataset("f1", data=fd["f1"])
            grp.create_dataset("f2", data=fd["f2"])

    print(f"Saved {len(frame_data)} frames to {path}")

anim = animation.FuncAnimation(fig, animate, frames=91, interval=100)
anim.save('amr_showcase.gif')
save_frame_data()
