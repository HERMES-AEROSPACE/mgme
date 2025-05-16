import numpy as np
from .config import VELOCITY_SPACE, GROUP_PARAMS
from numba import jit


NUM_GROUPS_CX = GROUP_PARAMS['num_groups_cx']
NUM_GROUPS_CY = GROUP_PARAMS['num_groups_cy']
NUM_GROUPS_CZ = GROUP_PARAMS['num_groups_cz']

CI_CX = GROUP_PARAMS['ci_cx']
CF_CX = GROUP_PARAMS['cf_cx']
CI_CY = GROUP_PARAMS['ci_cy']
CF_CY = GROUP_PARAMS['cf_cy']
CI_CZ = GROUP_PARAMS['ci_cz']
CF_CZ = GROUP_PARAMS['cf_cz']

@jit(nopython=True)
def f(x, y, z, A, b, wx, wy, wz):
    return A * np.exp(-b * ((x - wx)**2 + (y - wy)**2 + (z - wz)**2))

def generate_grid(n_samples_x, n_samples_y, n_samples_z):
    sample_loc_x = np.linspace(*VELOCITY_SPACE['cx_range'], n_samples_x)
    sample_loc_y = np.linspace(*VELOCITY_SPACE['cy_range'], n_samples_y)
    sample_loc_z = np.linspace(*VELOCITY_SPACE['cz_range'], n_samples_z)

    [xgrid, ygrid, zgrid] = np.meshgrid(sample_loc_x, sample_loc_y, sample_loc_z, indexing='ij')

    x_sample = xgrid.flatten()
    y_sample = ygrid.flatten()
    z_sample = zgrid.flatten()

    return x_sample, y_sample, z_sample

@jit(nopython=True)
def calc_sum_f_group(n_samples, x_sample, y_sample, z_sample, Ak, bk, wxk, wyk, wzk):
    sum_f_group = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    num_group_sample = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))

    for i in range(0, n_samples):
        group_idx_x = np.argmax(np.logical_and(x_sample[i] >= CI_CX, x_sample[i] <= CF_CX))
        group_idx_y = np.argmax(np.logical_and(y_sample[i] >= CI_CY, y_sample[i] <= CF_CY))
        group_idx_z = np.argmax(np.logical_and(z_sample[i] >= CI_CZ, z_sample[i] <= CF_CZ))

        sum_f_group[group_idx_x, group_idx_y, group_idx_z] += f(x_sample[i], y_sample[i], z_sample[i], \
                                                                Ak[group_idx_x, group_idx_y, group_idx_z], bk[group_idx_x, group_idx_y, group_idx_z], \
                                                                    wxk[group_idx_x, group_idx_y, group_idx_z], wyk[group_idx_x, group_idx_y, group_idx_z], \
                                                                        wzk[group_idx_x, group_idx_y, group_idx_z])
        
        num_group_sample[group_idx_x, group_idx_y, group_idx_z] += 1

    return sum_f_group, num_group_sample

@jit(nopython=True)
def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, Ak, bk, wxk, wyk, wzk, mu):
    weights = np.zeros(n_samples)

    sum_f_group, num_group_sample = calc_sum_f_group(n_samples, x_sample, y_sample, z_sample, Ak, bk, wxk, wyk, wzk)

    for i in range(0, n_samples):
        group_idx_x = np.argmax(np.logical_and(x_sample[i] >= CI_CX, x_sample[i] <= CF_CX))
        group_idx_y = np.argmax(np.logical_and(y_sample[i] >= CI_CY, y_sample[i] <= CF_CY))
        group_idx_z = np.argmax(np.logical_and(z_sample[i] >= CI_CZ, z_sample[i] <= CF_CZ))
        
        weights[i] = mu[group_idx_x, group_idx_y, group_idx_z, 0] * f(x_sample[i], y_sample[i], z_sample[i], \
                                                                      Ak[group_idx_x, group_idx_y, group_idx_z], bk[group_idx_x, group_idx_y, group_idx_z], \
                                                                          wxk[group_idx_x, group_idx_y, group_idx_z], wyk[group_idx_x, group_idx_y, group_idx_z], \
                                                                              wzk[group_idx_x, group_idx_y, group_idx_z]) / sum_f_group[group_idx_x, group_idx_y, group_idx_z]

    return weights, num_group_sample

def calculate_moments_from_weights(x_sample, y_sample, z_sample, weights, n_samples):
    n = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    ux = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    uy = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    uz = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    e = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))

    for i in range(0, n_samples):
        group_idx_x = np.argmax(np.logical_and(x_sample[i] >= GROUP_PARAMS['ci_cx'], x_sample[i] <= GROUP_PARAMS['cf_cx']))
        group_idx_y = np.argmax(np.logical_and(y_sample[i] >= GROUP_PARAMS['ci_cy'], y_sample[i] <= GROUP_PARAMS['cf_cy']))
        group_idx_z = np.argmax(np.logical_and(z_sample[i] >= GROUP_PARAMS['ci_cz'], z_sample[i] <= GROUP_PARAMS['cf_cz']))

        n[group_idx_x, group_idx_y, group_idx_z] += weights[i] 
        ux[group_idx_x, group_idx_y, group_idx_z] += x_sample[i] * weights[i]
        uy[group_idx_x, group_idx_y, group_idx_z] += y_sample[i] * weights[i]
        uz[group_idx_x, group_idx_y, group_idx_z] += z_sample[i] * weights[i]
        e[group_idx_x, group_idx_y, group_idx_z] += (x_sample[i]**2 + y_sample[i]**2 + z_sample[i]**2) * weights[i]

    return n, ux, uy, uz, e
