import numpy as np
from .config import VELOCITY_SPACE, GROUP_PARAMS
from numba import jit
from scipy import optimize
import sys
from matplotlib import pyplot as plt


@jit(nopython=True)
def f(x, y, z, A, b, wx, wy, wz):
    return A * np.exp(-b * ((x - wx)**2 + (y - wy)**2 + (z - wz)**2))

def func(x, M, Q):
    return np.matmul(M, x) - Q

def generate_grid(n_samples_x, n_samples_y, n_samples_z):
    sample_loc_x = np.linspace(*VELOCITY_SPACE['cx_range'], n_samples_x)
    sample_loc_y = np.linspace(*VELOCITY_SPACE['cy_range'], n_samples_y)
    sample_loc_z = np.linspace(*VELOCITY_SPACE['cz_range'], n_samples_z)
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

    sum_f_group = np.sum(f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk))
    num_group_sample = len(x_sample_slice)
    if Ak == 0.0 and bk == 0.0  and wxk == 0.0 and wyk == 0.0 and wzk == 0.0:
        weights = np.zeros(len(x_sample_slice))
    else:
        weights = mu[0] * f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk) / sum_f_group

    return num_group_sample, weights, mask

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups):
    weights = np.zeros(n_samples)
    num_group_sample = np.zeros(len(curr_groups))

    for i, group in enumerate(curr_groups):
        ci_cx, cf_cx = group.group_bounds['ci_cx'], group.group_bounds['cf_cx']
        ci_cy, cf_cy = group.group_bounds['ci_cy'], group.group_bounds['cf_cy']
        ci_cz, cf_cz = group.group_bounds['ci_cz'], group.group_bounds['cf_cz']

        Ak, bk, wxk, wyk, wzk = group.A, group.b, group.wx, group.wy, group.wz
        mu = group.mu
        n_group_sample, group_weights, mask = generate_regular_samples_helper(mu, x_sample, y_sample, z_sample, \
                                                                                       ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, Ak, bk, wxk, wyk, wzk)
        
        num_group_sample[i] = n_group_sample
        weights[mask] = group_weights

    return weights, num_group_sample

# def reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu):
#     new_weights = np.zeros(int(np.sum(num_group_sample)))

#     for i in range(0, NUM_GROUPS_CX):
#         for j in range(0, NUM_GROUPS_CY):
#             for k in range(0, NUM_GROUPS_CZ):
#                 M = np.zeros((5, int(num_group_sample[i, j, k])))
#                 Q = np.zeros((5,))

#                 x_mask = np.asarray(np.logical_and(x_sample >= CI_CX[i], x_sample <= CF_CX[i])).nonzero()
#                 y_mask = np.asarray(np.logical_and(y_sample >= CI_CY[j], y_sample <= CF_CY[j])).nonzero()
#                 z_mask = np.asarray(np.logical_and(z_sample >= CI_CZ[k], z_sample <= CF_CZ[k])).nonzero()
#                 mask = np.array(list(set(x_mask[0].flatten()) & set(y_mask[0].flatten()) & set(z_mask[0].flatten())))

#                 M[0, :] = 1
#                 M[1, :] = x_sample[mask]
#                 M[2, :] = y_sample[mask]
#                 M[3, :] = z_sample[mask]
#                 M[4, :] = (x_sample[mask]**2 + y_sample[mask]**2 + z_sample[mask]**2)

#                 Q[0] = mu[i, j, k, 0]
#                 Q[1] = mu[i, j, k, 1]
#                 Q[2] = mu[i, j, k, 2]
#                 Q[3] = mu[i, j, k, 3]
#                 Q[4] = mu[i, j, k, 4]
                
#                 sol = optimize.least_squares(func, weights[mask], args=(M, Q), bounds=(0.0, 1.0), loss='soft_l1')
#                 new_weights[mask] = sol.x

#     return new_weights
