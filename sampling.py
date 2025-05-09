import numpy as np
from config import VELOCITY_SPACE, GROUP_PARAMS


def f(x, y, z, A, b, w):
    return A * np.exp(-b * ((x - w)**2 + y**2 + z**2))

def generate_grid(n_samples_dir):
    sample_loc = np.linspace(VELOCITY_SPACE['cx_range'][0], VELOCITY_SPACE['cx_range'][1], n_samples_dir)
    [xgrid, ygrid, zgrid] = np.meshgrid(sample_loc, sample_loc, sample_loc, indexing='ij')

    x_sample = xgrid.flatten()
    y_sample = ygrid.flatten()
    z_sample = zgrid.flatten()

    return x_sample, y_sample, z_sample

def calc_sum_f_group(n_samples, x_sample, y_sample, z_sample, Ak, bk, wk):
    sum_f_group = np.zeros(GROUP_PARAMS['num_groups'])
    num_group_sample = np.zeros(GROUP_PARAMS['num_groups'])
    for i in range(0, n_samples):
        group_idx = np.argmax(np.logical_and(x_sample[i] >= GROUP_PARAMS['ci'], x_sample[i] <= GROUP_PARAMS['cf']))
        sum_f_group[group_idx] += f(x_sample[i], y_sample[i], z_sample[i], Ak[group_idx], bk[group_idx], wk[group_idx])
        num_group_sample[group_idx] += 1

    return sum_f_group, num_group_sample

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, Ak, bk, wk, mu):
    weights = np.zeros(n_samples)

    sum_f_group, num_group_sample = calc_sum_f_group(n_samples, x_sample, y_sample, z_sample, Ak, bk, wk)
    for i in range(0, n_samples):
        group_idx = np.argmax(np.logical_and(x_sample[i] >= GROUP_PARAMS['ci'], x_sample[i] <= GROUP_PARAMS['cf']))
        weights[i] = mu[group_idx, 0] * f(x_sample[i], y_sample[i], z_sample[i], Ak[group_idx], bk[group_idx], wk[group_idx]) / sum_f_group[group_idx]

    return weights, num_group_sample

def calculate_moments_from_weights(x_sample, y_sample, z_sample, weights, n_samples):
    n = np.zeros(GROUP_PARAMS['num_groups'])
    ux = np.zeros(GROUP_PARAMS['num_groups'])
    e = np.zeros(GROUP_PARAMS['num_groups'])

    for i in range(0, n_samples):
        group_idx = np.argmax(np.logical_and(x_sample[i] >= GROUP_PARAMS['ci'], x_sample[i] <= GROUP_PARAMS['cf']))
        n[group_idx] += weights[i] 
        ux[group_idx] += x_sample[i] * weights[i]
        e[group_idx] += (x_sample[i]**2 + y_sample[i]**2 + z_sample[i]**2) * weights[i]

    return n, ux, e
