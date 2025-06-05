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
    root._update_group_dist_params()
    root_f = root.A * np.exp(-root.b * ((cx - root.wx)**2 + (cy - root.wy)**2 + (cz - root.wz)**2))
    dist = calculate_hellinger_distance(f0, root_f, cx_vec, cy_vec, cz_vec, root.group_bounds)
    root.set_hellinger_distance(dist)

    print('Running AMR to get initial groups...\n')
    refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root)
    curr_groups = get_current_groups(root)
    n_groups = len(curr_groups)

    bounds_list = np.zeros((n_groups, 6))
    for i, group in enumerate(curr_groups):
        bounds_list[i] = np.array([group.group_bounds['ci_cx'], group.group_bounds['cf_cx'], group.group_bounds['ci_cy'], \
                                   group.group_bounds['cf_cy'], group.group_bounds['ci_cz'], group.group_bounds['cf_cz']])

    print(bounds_list)
    print_tree_structure(root)

    print('Initial group generation complete. Generating samples...\n')

    Ak_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, n_groups))
    bk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, n_groups))
    wxk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, n_groups))
    wyk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, n_groups))
    wzk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, n_groups))

    for i, group in enumerate(curr_groups):
        Ak_list[0, i] = group.A
        bk_list[0, i] = group.b
        wxk_list[0, i] = group.wx
        wyk_list[0, i] = group.wy
        wzk_list[0, i] = group.wz

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

        group_n, group_px, group_py, group_pz, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_samples, n_groups)

        for i, group in enumerate(curr_groups):
            # Update group parameters after collisions.
            group.update_parameters(COLLISION_PARAMS['dt'], group_n[i], group_px[i], group_py[i], group_pz[i], group_e[i])

            # Save results for plotting and such.
            Ak_list[t, i] = group.A
            bk_list[t, i] = group.b
            wxk_list[t, i] = group.wx
            wyk_list[t, i] = group.wy
            wzk_list[t, i] = group.wz

        # Update weights for next simulation step.
        weights, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups)
        # reweighted_weights = reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu)

    # Save final state
    save_simulation_data(COLLISION_PARAMS['n_t'], Ak_list, bk_list, wxk_list, wyk_list, wzk_list)

if __name__ == '__main__':
    run_simulation()
