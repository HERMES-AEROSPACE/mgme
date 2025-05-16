import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy import optimize
from scipy.stats import qmc, norm
from .config import (
    GROUP_PARAMS, 
    COLLISION_PARAMS, 
    SAMPLING_PARAMS,
    calculate_beta_w_lists,
    calculate_velocity_grid
)
from .moment_utils import moment_eq, solve_equation, calculate_group_moments, create_table, invert
from .sampling import generate_regular_samples, calculate_moments_from_weights, generate_grid
from .virtual_collisions import collide
from .data_utils import save_simulation_data
from .banner import print_banner


def run_simulation():
    # Print banner at startup
    print_banner()
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()

    # Get beta and w lists.
    beta_list, w_list = calculate_beta_w_lists()

    # Initial distribution function.
    # K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))

    # Calculate group moments.
    mu = calculate_group_moments(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    print(mu)

    # Initialize parameter lists
    Ak_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    bk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    wxk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    wyk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    wzk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))

    # Create table for moments
    table = create_table(beta_list, w_list)

    print('Table created.\n')

    b_guess = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    wx_guess = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    wy_guess = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    wz_guess = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))

    for i in range(0, GROUP_PARAMS['num_groups_cx']):
        for j in range(0, GROUP_PARAMS['num_groups_cy']):
            for k in range(0, GROUP_PARAMS['num_groups_cz']):
                if np.abs(mu[i, j, k, 1]) < 1e-8:
                    b_guess[i, j, k] = 1.0
                    wx_guess[i, j, k] = 0.0
                else:
                    b_guess[i, j, k], wx_guess[i, j, k] = solve_equation(mu[i, j, k, 1] / mu[i, j, k, 0], mu[i, j, k, 4] / mu[i, j, k, 0], beta_list, w_list, table[i, j, k, 0], table[i, j, k, 3])
                if np.abs(mu[i, j, k, 2]) < 1e-8:
                    b_guess[i, j, k] = 1.0
                    wy_guess[i, j, k] = 0.0
                else:
                    b_guess[i, j, k], wy_guess[i, j, k] = solve_equation(mu[i, j, k, 2] / mu[i, j, k, 0], mu[i, j, k, 4] / mu[i, j, k, 0], beta_list, w_list, table[i, j, k, 1], table[i, j, k, 3])
                if np.abs(mu[i, j, k, 3]) < 1e-8:
                    b_guess[i, j, k] = 1.0
                    wz_guess[i, j, k] = 0.0
                else:
                    b_guess[i, j, k], wz_guess[i, j, k] = solve_equation(mu[i, j, k, 3] / mu[i, j, k, 0], mu[i, j, k, 4] / mu[i, j, k, 0], beta_list, w_list, table[i, j, k, 2], table[i, j, k, 3])

    A, b, wx, wy, wz = invert(mu, b_guess, wx_guess, wy_guess, wz_guess)
    Ak_list[0] = A
    bk_list[0] = b
    wxk_list[0] = wx
    wyk_list[0] = wy
    wzk_list[0] = wz

    # Save initial state
    # save_simulation_data(0, Ak_list, bk_list, wxk_list, wyk_list, wzk_list)
    print('Inversion complete.\n')

    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, Ak_list[0], bk_list[0], wxk_list[0], wyk_list[0], wzk_list[0], mu)

    print('Weights generated. Starting simulation...\n')

    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
            # save_simulation_data(t, Ak_list, bk_list, wk_list)

        group_n, group_px, group_py, group_pz, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, n_samples, COLLISION_PARAMS['n_coll'])

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

    # Save final state
    save_simulation_data(COLLISION_PARAMS['n_t'], Ak_list, bk_list, wxk_list, wyk_list, wzk_list)

if __name__ == '__main__':
    run_simulation()
