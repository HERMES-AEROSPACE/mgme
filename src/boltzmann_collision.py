import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy import optimize
from scipy.stats import qmc, norm
from .config import (
    VELOCITY_SPACE,
    GROUP_PARAMS, 
    AMR,
    COLLISION_PARAMS, 
    SAMPLING_PARAMS,
    calculate_velocity_grid
)
from .moment_utils import moment_eq, calculate_group_moments, invert, calc_moment
from .sampling import generate_regular_samples, calculate_moments_from_weights, generate_grid, reweight_samples
from .virtual_collisions import collide
from .data_utils import save_simulation_data
from .banner import print_banner
from .amr import calculate_hellinger_distance, GroupNode, refine_init, print_tree_structure, get_current_groups
import sys


def run_simulation():
    # Print banner at startup
    print_banner()
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()

    # Initial distribution function.
    # K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    # f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    f0 = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))

    # Use AMR to calculate initial groups.
    root = GroupNode({'ci_cx': VELOCITY_SPACE['cx_range'][0], 'cf_cx': VELOCITY_SPACE['cx_range'][1], 'group_bounds_cx': np.array([0, VELOCITY_SPACE['num_cx']]),
                      'ci_cy': VELOCITY_SPACE['cy_range'][0], 'cf_cy': VELOCITY_SPACE['cy_range'][1], 'group_bounds_cy': np.array([0, VELOCITY_SPACE['num_cy']]), 
                      'ci_cz': VELOCITY_SPACE['cz_range'][0], 'cf_cz': VELOCITY_SPACE['cz_range'][1], 'group_bounds_cz': np.array([0, VELOCITY_SPACE['num_cz']])})
    mu = calc_moment(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    root.set_mu(mu)
    A, b, wx, wy, wz = invert(root.mu, root.group_bounds)
    root_f = A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2))
    dist = calculate_hellinger_distance(f0, root_f, cx_vec, cy_vec, cz_vec, root.group_bounds)
    root.set_hellinger_distance(dist)

    print('Running AMR to get initial groups...\n')
    refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root)

    print_tree_structure(root)

    print('Initial group generation complete. Generating samples...\n')

    curr_groups = get_current_groups(root)

    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups)

    print('Reweighting samples...\n')

    # reweighted_weights = reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu)

    print('Weights generated. Starting simulation...\n')

    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
            # save_simulation_data(t, Ak_list, bk_list, wk_list)

        group_n, group_px, group_py, group_pz, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, n_samples)

        for i in range(GROUP_PARAMS['num_groups_cx']):
            for j in range(GROUP_PARAMS['num_groups_cy']):
                for k in range(GROUP_PARAMS['num_groups_cz']):
                    mu[i, j, k, 0] += COLLISION_PARAMS['dt'] * group_n[i, j, k]
                    mu[i, j, k, 1] += COLLISION_PARAMS['dt'] * group_px[i, j, k]
                    mu[i, j, k, 2] += COLLISION_PARAMS['dt'] * group_py[i, j, k]
                    mu[i, j, k, 3] += COLLISION_PARAMS['dt'] * group_pz[i, j, k]
                    mu[i, j, k, 4] += COLLISION_PARAMS['dt'] * group_e[i, j, k]

        A, b, wx, wy, wz = invert(mu, bk_list[t - 1], wxk_list[t - 1], wyk_list[t - 1], wzk_list[t - 1])
        Ak_list[t] = A
        bk_list[t] = b
        wxk_list[t] = wx
        wyk_list[t] = wy
        wzk_list[t] = wz

        weights, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, Ak_list[t], bk_list[t], wxk_list[t], wyk_list[t], wzk_list[t], mu)
        # reweighted_weights = reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu)

    # Save final state
    save_simulation_data(COLLISION_PARAMS['n_t'], Ak_list, bk_list, wxk_list, wyk_list, wzk_list)

if __name__ == '__main__':
    run_simulation()
