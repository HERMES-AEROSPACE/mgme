import numpy as np
from scipy import special
from scipy.stats import qmc, norm
from .config_0d import (
    VELOCITY_SPACE,
    GROUP_PARAMS, 
    AMR,
    COLLISION_PARAMS
)
from .collision_helper import calculate_velocity_grid, calc_moment, GroupNode, refine_init
from .banner import print_banner
import copy
import sys


def run_simulation():
    print_banner()
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    dx = np.abs(cx_vec[1] - cx_vec[0])
    dy = np.abs(cy_vec[1] - cy_vec[0])
    dz = np.abs(cz_vec[1] - cz_vec[0])

    # Initial distribution function. Still have to uncomment the correct one.
    # Ak = 0.00384934
    # bk = 0.03110795
    # wxk = 2.36981369
    # wyk = 0.0
    # wzk = 0.0
    # K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    # f0 = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))
    # f0 = Ak * np.exp(-bk * ((cx - wxk)**2 + (cy - wyk)**2 + (cz - wzk)**2))

    # Latin Hypercube sampler for generating group samples.
    sampler = qmc.LatinHypercube(d=3)

    # Create the root node of the AMR tree.
    root = GroupNode(np.array([-3, 3, -3, 3, -3, 3]))
    mu = calc_moment(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    root.set_mu(mu)
    root.generate_samples(sampler)
    root.optimize_weights()

    print('Running AMR to get initial groups...\n')

    # Choose between using custom groups or AMR to get initial groups.
    # custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, 4)
    curr_groups = get_current_groups(root)
    n_groups = len(curr_groups)
    print('# of groups:', n_groups)

    # Set up group bounds for use in collisions.
    bounds_list = np.zeros((n_groups, 6))
    for i, group in enumerate(curr_groups):
        bounds_list[i] = np.array([group.group_bounds['ci_cx'], group.group_bounds['cf_cx'], group.group_bounds['ci_cy'], \
                                   group.group_bounds['cf_cy'], group.group_bounds['ci_cz'], group.group_bounds['cf_cz']])
    # print(bounds_list)
    print('Initial group generation complete. Generating samples...\n')

    # Sets up list of current groups in refinement level. Will need other arrays for finer/coarser grids in AMR.
    curr_groups_list = [0 for x in range(COLLISION_PARAMS['n_t'] + 1)]
    curr_groups_list[0] = copy.deepcopy(curr_groups)

    # Generate regular samples on a grid. Generate initial weights on grid.
    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample, sample_loc_x, sample_loc_y, sample_loc_z = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    vol_elem = calculate_volume_elements(sample_loc_x, sample_loc_y, sample_loc_z)
    weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups, vol_elem)
    print('Reweighting samples...\n')
    
    # plt.rcParams['font.family'] = "serif"
    # fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    # ax[0].plot(cx_vec, np.trapz(np.trapz(f0, cz_vec, axis=2), cy_vec, axis=1))
    # ax[0].plot(sample_loc_x, np.sum(np.reshape(weights, (24, 20, 20)) / 2, axis=(1, 2)), '--o', color='black')
    # ax[0].hist(x_sample, weights=weights / dx, bins=24)
    # ax[1].plot(cy_vec, np.trapz(np.trapz(f0, cz_vec, axis=2), cx_vec, axis=0))
    # ax[1].plot(sample_loc_y, np.sum(np.reshape(weights, (24, 20, 20)), axis=(0, 2)), '--o', color='black')
    # ax[1].hist(y_sample, weights=weights, bins=12)
    # ax[2].plot(cz_vec, np.trapz(np.trapz(f0, cy_vec, axis=1), cx_vec, axis=0))
    # ax[2].hist(z_sample, weights=weights, bins=12)
    # ax[2].plot(sample_loc_z, np.sum(np.reshape(weights, (24, 20, 20)), axis=(0, 1)), '--o', color='black')
    # ax[0].set_xlabel('Cx', fontsize=20)
    # ax[1].set_xlabel('Cy', fontsize=20)
    # ax[2].set_xlabel('Cz', fontsize=20)
    # plt.tight_layout()
    # plt.show()

    # reweighted_weights = reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu)

    print('Weights generated. Starting simulation...\n')

    # Set up array for entropy calculation and outputting.
    entropy_list = np.zeros(COLLISION_PARAMS['n_t'] + 1)
    # entropy_list[0] = calculate_entropy(weights, volume_elements, sample_loc_x, sample_loc_y, sample_loc_z, SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    # print(entropy_list[0])

    # MAIN SIMULATION LOOP.
    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
            # print('Entropy: ', entropy_list[t - 1])

        # Random values necessary for collision routine.
        Rf1 = np.random.uniform(0.0, 1.0, COLLISION_PARAMS['n_coll'])
        Rf2 = np.random.uniform(0.0, 1.0, COLLISION_PARAMS['n_coll'])
        depl_idx1 = np.random.randint(0, n_samples, COLLISION_PARAMS['n_coll'])
        depl_idx2 = np.random.randint(0, n_samples, COLLISION_PARAMS['n_coll'])

        # BINARY ELASTIC COLLISIONS.
        group_n, group_px, group_py, group_pz, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_groups, Rf1, Rf2, depl_idx1, depl_idx2)
        # print(np.sum(group_n.reshape(-1, 1), axis=1))
        # Update group parameters after collisions. Do not invert the distribution.
        for i, group in enumerate(curr_groups):
            group.update_parameters(COLLISION_PARAMS['dt'], group_n[i], group_px[i], group_py[i], group_pz[i], group_e[i])
            # print(group.A, group.b, group.wx, group.wy, group.wz)

        # Save data for plotting.
        curr_groups_list[t] = copy.deepcopy(curr_groups)

        # Update weights for next simulation step. Update entropy.
        weights, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups, vol_elem)
        # entropy_list[t] = calculate_entropy(weights, volume_elements, sample_loc_x, sample_loc_y, sample_loc_z, SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])

    # Save final state.
    save_simulation_data(COLLISION_PARAMS['n_t'], curr_groups_list, entropy_list)

if __name__ == '__main__':
    run_simulation()
