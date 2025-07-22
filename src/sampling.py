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

def generate_grid(n_samples_x, n_samples_y, n_samples_z):
    # sample_loc_x = np.linspace(*VELOCITY_SPACE['cx_range'], n_samples_x)
    # sample_loc_y = np.linspace(*VELOCITY_SPACE['cy_range'], n_samples_y)
    # sample_loc_z = np.linspace(*VELOCITY_SPACE['cz_range'], n_samples_z)

    sample_loc_x_neg = np.append(np.linspace(-2.5, -0.51, 12), np.linspace(-0.49, 0.0, 8, endpoint=False))
    sample_loc_x_pos = -1 * np.append(np.linspace(-0.49, 0.0, 8, endpoint=False)[::-1], np.linspace(-2.5, -0.51, 12)[::-1])

    sample_loc_x = np.append(sample_loc_x_neg, sample_loc_x_pos)
    sample_loc_y = sample_loc_x
    sample_loc_z = sample_loc_x
    print(sample_loc_x)
    
    [xgrid, ygrid, zgrid] = np.meshgrid(sample_loc_x, sample_loc_y, sample_loc_z, indexing='ij')

    x_sample = xgrid.flatten()
    y_sample = ygrid.flatten()
    z_sample = zgrid.flatten()

    return x_sample, y_sample, z_sample

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
    x = cp.Variable(shape=num_sample_group)
    obj = cp.Maximize(cp.sum(cp.entr(x)))

    constraints = [cp.sum(x) == mu[0], x >= 0, cp.sum(cp.multiply(x_sample_slice, x)) == mu[1], \
                cp.sum(cp.multiply(y_sample_slice, x)) == mu[2], cp.sum(cp.multiply(z_sample_slice, x)) == mu[3], \
                cp.sum(cp.multiply(x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2, x)) == mu[4]]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # print('density:', np.sum(x.value))
    # print('x-momentum:', np.sum(x.value * x_sample_slice))
    # print('y-momentum:', np.sum(x.value * y_sample_slice))
    # print('z-momentum:', np.sum(x.value * z_sample_slice))
    # print('energy:', np.sum((x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2) * x.value))

    return num_sample_group, x.value, mask

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups):
    weights = np.zeros(n_samples)
    num_sample_group = np.zeros(len(curr_groups))

    for i, group in enumerate(curr_groups):
        ci_cx, cf_cx = group.group_bounds['ci_cx'], group.group_bounds['cf_cx']
        ci_cy, cf_cy = group.group_bounds['ci_cy'], group.group_bounds['cf_cy']
        ci_cz, cf_cz = group.group_bounds['ci_cz'], group.group_bounds['cf_cz']

        Ak, bk, wxk, wyk, wzk = group.A, group.b, group.wx, group.wy, group.wz
        mu = group.mu
        # n_group_sample, group_weights, mask, test = generate_regular_samples_helper(mu, x_sample, y_sample, z_sample, \
                                                                                    #    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, Ak, bk, wxk, wyk, wzk)
        n_group_sample, group_weights, mask = generate_convex_helper(mu, x_sample, y_sample, z_sample, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz)
        # print(group_weights, test)
        # print()
        num_sample_group[i] = n_group_sample
        weights[mask] = group_weights

    return weights, num_sample_group
