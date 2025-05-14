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

    # Initialize parameter lists
    Ak_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups']))
    bk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups']))
    wk_list = np.zeros((COLLISION_PARAMS['n_t'] + 1, GROUP_PARAMS['num_groups']))

    # Create table for moments
    table = create_table(beta_list, w_list)

    b_guess = np.zeros(GROUP_PARAMS['num_groups'])
    w_guess = np.zeros(GROUP_PARAMS['num_groups'])
    for i in range(0, GROUP_PARAMS['num_groups']):
        if np.abs(mu[i][1]) < 1e-8:
            b_guess[i] = 1.0
            w_guess[i] = 0.0
        else:
            b_guess[i], w_guess[i] = solve_equation(mu[i, 1] / mu[i, 0], mu[i, 2] / mu[i, 0], beta_list, w_list, table[i, 0], table[i, 1])

    A, b, w = invert(mu, b_guess, w_guess)
    Ak_list[0] = A
    bk_list[0] = b
    wk_list[0] = w

    # Save initial state
    # save_simulation_data(0, Ak_list, bk_list, wk_list)

    n_samples = SAMPLING_PARAMS['n_samples_dir']**3
    x_sample, y_sample, z_sample = generate_grid(SAMPLING_PARAMS['n_samples_dir'])
    weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, Ak_list[0], bk_list[0], wk_list[0], mu)

    print('Setup complete. Starting simulation...\n')

    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
            # save_simulation_data(t, Ak_list, bk_list, wk_list)

        group_n, group_p, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, n_samples, COLLISION_PARAMS['n_coll'])

        for i in range(GROUP_PARAMS['num_groups']):
            mu[i][0] += COLLISION_PARAMS['dt'] * group_n[i]
            mu[i][1] += COLLISION_PARAMS['dt'] * group_p[i]
            mu[i][2] += COLLISION_PARAMS['dt'] * group_e[i]

        A, b, w = invert(mu, bk_list[t - 1], wk_list[t - 1])
        Ak_list[t] = A
        bk_list[t] = b
        wk_list[t] = w

        weights, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, Ak_list[t], bk_list[t], wk_list[t], mu)

    # Save final state
    save_simulation_data(COLLISION_PARAMS['n_t'], Ak_list, bk_list, wk_list)

if __name__ == '__main__':
    run_simulation()
