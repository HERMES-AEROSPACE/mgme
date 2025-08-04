import numpy as np
from .config import VELOCITY_SPACE, GROUP_PARAMS
from numba import jit
from scipy import optimize
import sys
from matplotlib import pyplot as plt
import cvxpy as cp


@jit(nopython=True)
def f(x, y, z, A, b, wx, wy, wz):
    return A * np.exp(-b * ((x - wx)**2 + (y - wy)**2 + (z - wz)**2))

def func(x, M, Q):
    return np.matmul(M, x) - Q

def calculate_volume_elements(x_centers, y_centers, z_centers):
    """
    Calculate volume elements for discretization with given center points.
    Boundary points get full spacing to neighbors (extending past centers).
    """
    def get_spacings(centers):
        # Calculate spacings between adjacent centers
        center_spacings = centers[1:] - centers[:-1]
        
        spacings = np.zeros_like(centers)
        
        # Interior points: average of adjacent spacings
        spacings[1:-1] = (center_spacings[:-1] + center_spacings[1:]) / 2
        
        # Boundary points: full spacing to neighbor
        spacings[0] = center_spacings[0]   # Full distance to next center
        spacings[-1] = center_spacings[-1]  # Full distance to previous center
        
        return spacings
    
    dx = get_spacings(x_centers)
    dy = get_spacings(y_centers) 
    dz = get_spacings(z_centers)
    
    # Create 3D mesh of volume elements
    DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing='ij')
    volume_elements = DX * DY * DZ
    
    return volume_elements

@jit(nopython=True)
def calculate_entropy(weights, volume_elements, sample_loc_x, sample_loc_y, sample_loc_z):
    f = np.reshape(weights, (48, 48, 48)) / volume_elements
    return np.trapezoid(np.trapezoid(np.trapezoid(-f * np.log(f), sample_loc_z), sample_loc_y), sample_loc_x)

def generate_grid(n_samples_x, n_samples_y, n_samples_z):
    sample_loc_x_neg = np.append(np.linspace(-4., -2.6, 4), np.linspace(-2.4, -0.01, 20))
    sample_loc_x_pos = -1 * np.append(np.linspace(-2.4, -0.01, 20)[::-1], np.linspace(-4, -2.6, 4)[::-1])

    sample_loc_x = np.append(sample_loc_x_neg, sample_loc_x_pos)
    sample_loc_y = sample_loc_x
    sample_loc_z = sample_loc_x
    print(sample_loc_x)
    
    [xgrid, ygrid, zgrid] = np.meshgrid(sample_loc_x, sample_loc_y, sample_loc_z, indexing='ij')

    x_sample = xgrid.flatten()
    y_sample = ygrid.flatten()
    z_sample = zgrid.flatten()

    return x_sample, y_sample, z_sample, sample_loc_x, sample_loc_y, sample_loc_z

@jit(nopython=True)
def generate_regular_samples_helper(mu, x_sample, y_sample, z_sample, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, Ak, bk, wxk, wyk, wzk):
    mask = (x_sample >= ci_cx) & (x_sample <= cf_cx) & \
    (y_sample >= ci_cy) & (y_sample <= cf_cy) & \
    (z_sample >= ci_cz) & (z_sample <= cf_cz)

    x_sample_slice = x_sample[mask]
    y_sample_slice = y_sample[mask]
    z_sample_slice = z_sample[mask]
    dx = 3 - 2.93814433
    test = f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk) * (dx)**3

    sum_f_group = np.sum(f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk))
    num_sample_group = len(x_sample_slice)
    if Ak == 0.0 and bk == 0.0  and wxk == 0.0 and wyk == 0.0 and wzk == 0.0:
        weights = np.zeros(len(x_sample_slice))
    else:
        weights = mu[0] * f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk) / sum_f_group

    return num_sample_group, test, mask, test

def generate_convex_helper(mu, x_sample, y_sample, z_sample, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    mask = (x_sample >= ci_cx) & (x_sample <= cf_cx) & \
    (y_sample >= ci_cy) & (y_sample <= cf_cy) & \
    (z_sample >= ci_cz) & (z_sample <= cf_cz)

    x_sample_slice = x_sample[mask]
    y_sample_slice = y_sample[mask]
    z_sample_slice = z_sample[mask]

    num_sample_group = len(x_sample_slice)

    if mu[0] < 1e-4: # change this to a different scheme for better AMR stuff.
        A = np.zeros((5, num_sample_group))
        b = np.zeros(5)
        A[0, :] = 1
        A[1, :] = x_sample_slice
        A[2, :] = y_sample_slice
        A[3, :] = z_sample_slice
        A[4, :] = x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2
        b[0] = mu[0]
        b[1] = mu[1]
        b[2] = mu[2]
        b[3] = mu[3]
        b[4] = mu[4]

        x = cp.Variable(shape=num_sample_group, nonneg=True)
        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()

        weights = x.value
    else:
        x = cp.Variable(shape=num_sample_group, nonneg=True)
        obj = cp.Maximize(cp.sum(cp.entr(x)))

        constraints = [cp.sum(x) == mu[0], cp.sum(cp.multiply(x_sample_slice, x)) == mu[1], \
                    cp.sum(cp.multiply(y_sample_slice, x)) == mu[2], cp.sum(cp.multiply(z_sample_slice, x)) == mu[3], \
                    cp.sum(cp.multiply(x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2, x)) == mu[4]]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        weights = x.value

    return num_sample_group, weights, mask

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups):
    weights = np.zeros(n_samples)
    num_sample_group = np.zeros(len(curr_groups))

    for i, group in enumerate(curr_groups):
        ci_cx, cf_cx = group.group_bounds['ci_cx'], group.group_bounds['cf_cx']
        ci_cy, cf_cy = group.group_bounds['ci_cy'], group.group_bounds['cf_cy']
        ci_cz, cf_cz = group.group_bounds['ci_cz'], group.group_bounds['cf_cz']

        Ak, bk, wxk, wyk, wzk = group.A, group.b, group.wx, group.wy, group.wz
        mu = group.mu
        # print(ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, mu[0])
        # n_group_sample, group_weights, mask, test = generate_regular_samples_helper(mu, x_sample, y_sample, z_sample, \
                                                                                    #    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, Ak, bk, wxk, wyk, wzk)
        n_group_sample, group_weights, mask = generate_convex_helper(mu, x_sample, y_sample, z_sample, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz)
        # print(group_weights, test)
        # print()
        num_sample_group[i] = n_group_sample
        weights[mask] = group_weights

    return weights, num_sample_group
