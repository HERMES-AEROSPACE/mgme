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
from .sampling import generate_regular_samples, generate_grid
from .virtual_collisions import collide
from .data_utils import save_simulation_data
from .banner import print_banner
from .amr import calculate_hellinger_distance, GroupNode, refine_init, print_tree_structure, get_current_groups, custom_groups
import copy


def run_simulation():
    # Print banner at startup
    print_banner()
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()

    # Initial distribution function.
    # K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    # f0 = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))

    # Use AMR to calculate initial groups.
    root = GroupNode({'ci_cx': VELOCITY_SPACE['cx_range'][0], 'cf_cx': VELOCITY_SPACE['cx_range'][1], 'group_bounds_cx': np.array([0, VELOCITY_SPACE['num_cx']]),
                      'ci_cy': VELOCITY_SPACE['cy_range'][0], 'cf_cy': VELOCITY_SPACE['cy_range'][1], 'group_bounds_cy': np.array([0, VELOCITY_SPACE['num_cy']]), 
                      'ci_cz': VELOCITY_SPACE['cz_range'][0], 'cf_cz': VELOCITY_SPACE['cz_range'][1], 'group_bounds_cz': np.array([0, VELOCITY_SPACE['num_cz']])})
    mu = calc_moment(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    root.set_mu(mu)
    root._update_group_dist_params([1.0, 0.0, 0.0, 0.0])
    root_f = root.A * np.exp(-root.b * ((cx - root.wx)**2 + (cy - root.wy)**2 + (cz - root.wz)**2))
    dist = calculate_hellinger_distance(f0, root_f, cx_vec, cy_vec, cz_vec, root.group_bounds)
    root.set_hellinger_distance(dist)

    print('Running AMR to get initial groups...\n')
    custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    # refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, 2)
    curr_groups = get_current_groups(root)
    n_groups = len(curr_groups)

    bounds_list = np.zeros((n_groups, 6))
    for i, group in enumerate(curr_groups):
        bounds_list[i] = np.array([group.group_bounds['ci_cx'], group.group_bounds['cf_cx'], group.group_bounds['ci_cy'], \
                                   group.group_bounds['cf_cy'], group.group_bounds['ci_cz'], group.group_bounds['cf_cz']])
        # print(group.A, group.b, group.wx, group.wy, group.wz, group.mu, group.group_bounds)

    # print_tree_structure(root)
    print(bounds_list)

    print('Initial group generation complete. Generating samples...\n')

    curr_groups_list = [0 for x in range(COLLISION_PARAMS['n_t'] + 1)]
    curr_groups_list[0] = copy.deepcopy(curr_groups)

    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups)
    print('Reweighting samples...\n')

    # print(np.sum(weights[0:n_samples//2]), np.sum(weights[n_samples//2:]))
    # print(np.sum(weights[0:n_samples//2] * x_sample[0:n_samples//2]), np.sum(weights[n_samples//2:] * x_sample[n_samples//2:]))
    # print(np.sum(weights[0:n_samples//2] * y_sample[0:n_samples//2]), np.sum(weights[n_samples//2:] * y_sample[n_samples//2:]))
    # print(np.sum(weights[0:n_samples//2] * z_sample[0:n_samples//2]), np.sum(weights[n_samples//2:] * z_sample[n_samples//2:]))
    # print(np.sum((x_sample[0:n_samples//2]**2 + y_sample[0:n_samples//2]**2 + z_sample[0:n_samples//2]**2) * weights[0:n_samples//2]))

    # reweighted_weights = reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu)

    print('Weights generated. Starting simulation...\n')

    group_collector = np.zeros((n_groups, 5))
    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

        group_n_d, group_n_r, group_px, group_py, group_pz, group_e, d_group, r_group, vx_collector, vxp_collector\
              = collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_samples, n_groups)
        # ax1.bar(np.linspace(0, n_groups, n_groups), group_n_d)
        # ax2.bar(np.linspace(0, n_groups, n_groups), group_n_r)
        # ax3.bar(np.linspace(0, n_groups, n_groups), group_n_r + group_n_d)
        # ax4.bar(np.linspace(0, n_groups, n_groups), d_group)
        # ax5.bar(np.linspace(0, n_groups, n_groups), r_group)
        # ax6.bar(np.linspace(0, n_groups, n_groups), group_px)
        # plt.tight_layout()
        # plt.show()

        # fig2 = plt.figure(figsize=(6, 6))
        # ax1 = fig2.add_subplot(111)
        # ax1.hist(vxp_collector, bins=200)
        # ax1.hist(vx_collector, bins=200)
        # plt.tight_layout()
        # plt.show()
        group_n = group_n_d + group_n_r

        for i, group in enumerate(curr_groups):
            # Update group parameters after collisions.
            group.update_parameters(COLLISION_PARAMS['dt'], group_n[i], group_px[i], group_py[i], group_pz[i], group_e[i])
            # print(group.A, group.b, group.wx, group.wy, group.wz, group.group_bounds)

        # Save data for plotting.
        curr_groups_list[t] = copy.deepcopy(curr_groups)

        # Update weights for next simulation step.
        weights, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups)
        # reweighted_weights = reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu)

    # Save final state
    save_simulation_data(COLLISION_PARAMS['n_t'], curr_groups_list)

    # np.save('notebooks/group_collector.npy', group_collector)
if __name__ == '__main__':
    run_simulation()
