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
    SAMPLING_PARAMS
)
from .moment_utils import calc_moment
from .sampling import generate_regular_samples, generate_grid, calculate_volume_elements, calculate_entropy, calculate_velocity_grid
from .virtual_collisions import collide
from .data_utils import save_simulation_data
from .banner import print_banner
from .amr import calculate_hellinger_distance, GroupNode, refine_init, get_current_groups, custom_groups
import copy
import sys


def run_simulation():
    print_banner()
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()

    # Initial distribution function. Still have to uncomment the correct one.
    # K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    # f0 = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))

    # Create the root node of the AMR tree.
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

    # Choose between using custom groups or AMR to get initial groups.
    custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    # refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, 4)
    curr_groups = get_current_groups(root)
    n_groups = len(curr_groups)
    print('# of groups:', n_groups)

    # Set up group bounds for use in collisions.
    bounds_list = np.zeros((n_groups, 6))
    for i, group in enumerate(curr_groups):
        bounds_list[i] = np.array([group.group_bounds['ci_cx'], group.group_bounds['cf_cx'], group.group_bounds['ci_cy'], \
                                   group.group_bounds['cf_cy'], group.group_bounds['ci_cz'], group.group_bounds['cf_cz']])

    print('Initial group generation complete. Generating samples...\n')

    # Sets up list of current groups in refinement level. Will need other arrays for finer/coarser grids in AMR.
    curr_groups_list = [0 for x in range(COLLISION_PARAMS['n_t'] + 1)]
    curr_groups_list[0] = copy.deepcopy(curr_groups)

    # Generate regular samples on a grid. Generate initial weights on grid.
    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample, sample_loc_x, sample_loc_y, sample_loc_z = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups)
    volume_elements = calculate_volume_elements(sample_loc_x, sample_loc_y, sample_loc_z)

    print('Weights generated. Starting simulation...\n')

    # Set up array for entropy calculation and outputting.
    entropy_list = np.zeros(COLLISION_PARAMS['n_t'] + 1)
    entropy_list[0] = calculate_entropy(weights, volume_elements, sample_loc_x, sample_loc_y, sample_loc_z, SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    print(entropy_list[0])

    # MAIN SIMULATION LOOP.
    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
            print('Entropy: ', entropy_list[t - 1])

        # Random values necessary for collision routine.
        Rf1 = np.random.uniform(0.0, 1.0, COLLISION_PARAMS['n_coll'])
        Rf2 = np.random.uniform(0.0, 1.0, COLLISION_PARAMS['n_coll'])
        depl_idx1 = np.random.randint(0, n_samples, COLLISION_PARAMS['n_coll'])
        depl_idx2 = np.random.randint(0, n_samples, COLLISION_PARAMS['n_coll'])

        # BINARY ELASTIC COLLISIONS.
        group_n, group_px, group_py, group_pz, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_groups, Rf1, Rf2, depl_idx1, depl_idx2)
        
        # Update group parameters after collisions. Do not invert the distribution.
        for i, group in enumerate(curr_groups):
            group.update_parameters(COLLISION_PARAMS['dt'], group_n[i], group_px[i], group_py[i], group_pz[i], group_e[i])

        # Save data for plotting.
        curr_groups_list[t] = copy.deepcopy(curr_groups)

        # Update weights for next simulation step. Update entropy.
        weights, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups)
        entropy_list[t] = calculate_entropy(weights, volume_elements, sample_loc_x, sample_loc_y, sample_loc_z, SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])

    # Save final state.
    save_simulation_data(COLLISION_PARAMS['n_t'], curr_groups_list, entropy_list)

if __name__ == '__main__':
    run_simulation()
