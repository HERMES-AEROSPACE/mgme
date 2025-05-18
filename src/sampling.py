import numpy as np
from .config import VELOCITY_SPACE, GROUP_PARAMS
from numba import jit
from scipy import optimize


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

def func(x, M, Q):
    return np.matmul(M, x) - Q

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

@jit(nopython=True)
def calculate_moments_from_weights(x_sample, y_sample, z_sample, weights, n_samples):
    n = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    ux = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    uy = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    uz = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    e = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))

    for i in range(0, n_samples):
        group_idx_x = np.argmax(np.logical_and(x_sample[i] >= CI_CX, x_sample[i] <= CF_CX))
        group_idx_y = np.argmax(np.logical_and(y_sample[i] >= CI_CY, y_sample[i] <= CF_CY))
        group_idx_z = np.argmax(np.logical_and(z_sample[i] >= CI_CZ, z_sample[i] <= CF_CZ))

        n[group_idx_x, group_idx_y, group_idx_z] += weights[i] 
        ux[group_idx_x, group_idx_y, group_idx_z] += x_sample[i] * weights[i]
        uy[group_idx_x, group_idx_y, group_idx_z] += y_sample[i] * weights[i]
        uz[group_idx_x, group_idx_y, group_idx_z] += z_sample[i] * weights[i]
        e[group_idx_x, group_idx_y, group_idx_z] += (x_sample[i]**2 + y_sample[i]**2 + z_sample[i]**2) * weights[i]

    return n, ux, uy, uz, e

def reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu):
    new_weights = np.zeros(int(np.sum(num_group_sample)))
    l = 0
    u_old = 0

    for i in range(0, NUM_GROUPS_CX):
        for j in range(0, NUM_GROUPS_CY):
            for k in range(0, NUM_GROUPS_CZ):
                u = u_old + int(num_group_sample[i, j, k])

                M = np.zeros((5, u - l))
                Q = np.zeros((5,))

                M[0, :] = 1
                M[1, :] = x_sample[l:u]
                M[2, :] = y_sample[l:u]
                M[3, :] = z_sample[l:u]
                M[4, :] = (x_sample[l:u]**2 + y_sample[l:u]**2 + z_sample[l:u]**2)

                Q[0] = mu[i, j, k, 0]
                Q[1] = mu[i, j, k, 1]
                Q[2] = mu[i, j, k, 2]
                Q[3] = mu[i, j, k, 3]
                Q[4] = mu[i, j, k, 4]
                
                sol = optimize.least_squares(func, weights[l:u], args=(M, Q), bounds=(0.0, 1.0), loss='soft_l1')
                new_weights[l:u] = sol.x

                l = int(num_group_sample[i, j, k])
                u_old = u

    return new_weights
