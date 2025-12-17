import numpy as np
from .banner import print_banner
from .shock_helper import calc_flux_int, ic, invert, calc_integral, calc_flux, LF_central1, KT_central2, generate_regular_samples, lookup_table, collide, calculate_velocity_grid, generate_grid
from .config import CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE, GROUP_PARAMS, VELOCITY_SPACE, SAMPLING_PARAMS, COLLISION_PARAMS
from matplotlib import pyplot as plt
import itertools
import cProfile, pstats
from scipy import interpolate
from joblib import Parallel, delayed
import time, sys
from numba import types


def run_simulation():
    print_banner()
    print("---------------------------CALCULATING SHOCK QUANTITIES---------------------------")

    gamma = CONSTANTS['gamma']
    R = CONSTANTS['R']
    T1 = FREESTREAM_PARAMS['T1']
    P1 = FREESTREAM_PARAMS['P1']
    Ma1 = FREESTREAM_PARAMS['Ma1']
    m = CONSTANTS['m']
    k = CONSTANTS['k']

    a1 = np.sqrt(gamma * R * T1)
    u1 = Ma1 * a1
    rho1 = P1/(R * T1)
    n1 = P1/(R * T1) * 1/m

    # Post shock quantities.
    T2 = T1 * (((gamma - 1) * Ma1**2 + 2) * (2 * gamma * Ma1**2 - (gamma - 1)))/((gamma + 1)**2 * Ma1**2)
    P2 = P1 * ((2 * gamma * Ma1**2) - (gamma - 1))/(gamma + 1)
    rho2 = P2/(R * T2)  
    a2 = np.sqrt(gamma * R * T2)
    u2 = u1 * rho1/rho2
    Ma2 = Ma1 * u2/u1 * (T1/T2)**0.5
    n2 = P2/(R * T2) * 1/m
    print(n1, n2, u1, u2, T1, T2)

    m_ref = m
    n_ref = n1
    T_ref = T2
    c_ref = np.sqrt((2 * k * T_ref)/m_ref)
    # c_ref = np.sqrt((2 * k * T1)/m_ref)
    # T_ref = m * c_ref**2 / (2 * k)

    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    xj_vec = np.linspace(PHYS_SPACE['xj_range'][0], PHYS_SPACE['xj_range'][1], PHYS_SPACE['num_xj'])
    dx = np.abs(xj_vec[1] - xj_vec[0])
    dcx = np.abs(cx_vec[1] - cx_vec[0])
    dcy = np.abs(cy_vec[1] - cy_vec[0])
    dcz = np.abs(cz_vec[1] - cz_vec[0])
    Xj_l = PHYS_SPACE['xj_range'][0]
    Xj_u = PHYS_SPACE['xj_range'][1]
    numXj = PHYS_SPACE['num_xj']

    transition_start = -0.8
    transition_end = 0.8
    ramp_length = transition_end - transition_start

    t = (xj_vec - transition_start) / ramp_length
    t = np.clip(t, 0, 1)
    cosine_factor = 0.5 * (1 - np.cos(np.pi * t))

    T_val = (T1 + cosine_factor * (T2 - T1)) / T_ref
    u_val = (u1 + cosine_factor * (u2 - u1)) / c_ref
    n_val = (n1 + cosine_factor * (n2 - n1)) / n_ref

    print("---------------------------SETTING UP INITIAL CONDITION---------------------------")
    group_bounds_cx = GROUP_PARAMS['group_bounds_cx']
    group_bounds_cy = GROUP_PARAMS['group_bounds_cy']
    group_bounds_cz = GROUP_PARAMS['group_bounds_cz']
    ci_cx, cf_cx = GROUP_PARAMS['ci_cx'], GROUP_PARAMS['cf_cx']
    ci_cy, cf_cy = GROUP_PARAMS['ci_cy'], GROUP_PARAMS['cf_cy']
    ci_cz, cf_cz = GROUP_PARAMS['ci_cz'], GROUP_PARAMS['cf_cz']
    combinations = np.array(list(itertools.product(group_bounds_cx, group_bounds_cy, group_bounds_cz)))
    ci_combo = np.array(list(itertools.product(ci_cx, ci_cy, ci_cz)))
    cf_combo = np.array(list(itertools.product(cf_cx, cf_cy, cf_cz)))
    num_groups = combinations.shape[0]

    restart = 0

    U0, f = ic(cx, cy, cz, cx_vec, cy_vec, cz_vec, n_val, u_val, T_val, VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'], VELOCITY_SPACE['num_cz'], \
        numXj, num_groups, combinations)
    if restart:
        data = np.load('simulation_data/U1500.npy')
        print('Restarting from...')
        U = data
    else:
        U = U0.copy()

    # plt.plot(xj_vec, 2/3 * ((np.sum(U[:, :, 4], axis=1) / np.sum(U[:, :, 0], axis=1)) - (np.sum(U[:, :, 1], axis=1) / np.sum(U[:, :, 0], axis=1))**2))
    # plt.plot(xj_vec, T_val, '--')
    # plt.xlabel('xj', fontsize=16)
    # plt.ylabel('T', fontsize=16)
    # plt.show()
    # f[f < 1e-12] = 0.0
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(n_val[-1] * f[-1], cz_vec, axis=2), cy_vec, axis=1))
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(n_val[-1] * f[-1], cy_vec, axis=1), cx_vec, axis=0))
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f[0], cz_vec, axis=2), cy_vec, axis=1))
    # plt.show()

    # b_range = np.logspace(-8, 1, 20, endpoint=True)
    # interpolater_list = []
    bounds_list = np.zeros((num_groups, 6))
    
    for i in range(num_groups):
    #     inputs, outputs = lookup_table(b_range, np.linspace(ci_combo[i, 0], cf_combo[i, 0], 20), \
    #         np.linspace(ci_combo[i, 1], cf_combo[i, 1], 20), np.linspace(ci_combo[i, 2], cf_combo[i, 2], 20), \
    #             ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2])

    #     interpolater_list.append(interpolate.NearestNDInterpolator(outputs, inputs))
        bounds_list[i] = np.array([ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2]])

    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample, cx_loc, cy_loc, cz_loc = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])

    print("--------------------------------BEGIN SIMULATION----------------------------------")
    t_end = 15.0
    dt = 0.025
    n_coll = COLLISION_PARAMS['n_coll']
    CX_LB, CX_UB = VELOCITY_SPACE['cx_range']
    CY_LB, CY_UB = VELOCITY_SPACE['cy_range']
    CZ_LB, CZ_UB = VELOCITY_SPACE['cz_range']
    key_type = types.UniTuple(types.int64, 2)

    def coll_step(i, n_samples, x_sample, y_sample, z_sample, U_i_c, bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type):
        # Generate random numbers for collisions.
        Rf1 = np.random.uniform(0.0, 1.0, n_coll)
        Rf2 = np.random.uniform(0.0, 1.0, n_coll)
        depl_idx1 = np.random.randint(0, n_samples, n_coll)
        depl_idx2 = np.random.randint(0, n_samples, n_coll)

        # Advance the collision forward.
        weights, num_group_sample, _ = generate_regular_samples(
            i, n_samples, x_sample, y_sample, z_sample, U_i_c, bounds_list, num_groups
        )
        coll = collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, num_groups, \
                        Rf1, Rf2, depl_idx1, depl_idx2, n_coll, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type)

        return i, coll

    def flux_step(i, n_samples, x_sample, y_sample, z_sample, U_i_f, bounds_list, num_groups, cx_loc, cy_loc, cz_loc):
        # Advance the flux term forward.
        weights_f, _, masks_f = generate_regular_samples(
            i, n_samples, x_sample, y_sample, z_sample, U_i_f, bounds_list, num_groups
        )
        flux = calc_flux_int(num_groups, weights_f, masks_f, bounds_list, cx_loc, cy_loc, cz_loc)
        
        return i, flux

    profiler = cProfile.Profile()
    for t in range(0, int(np.ceil(int(t_end / dt) / 100) * 100) + 1):
        # Inversion and calculate flux.
        # Ak, bk, wxk, wyk, wzk = invert(U, numXj, num_groups, bounds_list, interpolater_list)
        # I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z = calc_integral(bk, wxk, wyk, wzk, bounds_list, numXj, num_groups)
        # F = calc_flux(Ak, bk, wxk, wyk, wzk, I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z, numXj, num_groups)
        
        # Boundary conditions.
        U[0, :] = U0[0, :]
        U[-1, :] = U[-2, :]

        # RK2 integration.
        k1_c = np.zeros((numXj, num_groups, 5))
        k2_c = np.zeros((numXj, num_groups, 5))

        F1 = np.zeros((numXj, num_groups, 5))
        F2 = np.zeros((numXj, num_groups, 5))
        F3 = np.zeros((numXj, num_groups, 5))
        
        # Use Strang splitting to first advance the collision term by dt/2. Then apply the flux term. Finally do the collision term again.
        # Apply RK1 (Euler) to advance collision term to t^{n + 1/2}.
        coll_dt_half = Parallel(n_jobs=12)(
            delayed(coll_step)(i, n_samples, x_sample, y_sample, z_sample, U[i], bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type)
            for i in range(0, numXj)
        )
        for i, coll in coll_dt_half:
            k1_c[i, :, 0] = coll[0]
            k1_c[i, :, 1] = coll[1]
            k1_c[i, :, 2] = coll[2]
            k1_c[i, :, 3] = coll[3]
            k1_c[i, :, 4] = coll[4]
        U_half = U + dt/2 * k1_c

        # Apply RK2 to step the flux term to t^{n + 1}.
        flux_dt_1 = Parallel(n_jobs=12)(
            delayed(flux_step)(i, n_samples, x_sample, y_sample, z_sample, U_half[i], bounds_list, num_groups, cx_loc, cy_loc, cz_loc) 
            for i in range(0, numXj)
        )
        for i, flux in flux_dt_1:
            F1[i] = flux
        k1_f = KT_central2(U_half, F1, numXj, num_groups, dt, dx, CX_LB, CX_UB)
        U1_flux = U_half + dt * k1_f

        flux_dt_2 = Parallel(n_jobs=12)(
            delayed(flux_step)(i, n_samples, x_sample, y_sample, z_sample, U1_flux[i], bounds_list, num_groups, cx_loc, cy_loc, cz_loc) 
            for i in range(0, numXj)
        )
        for i, flux in flux_dt_2:
            F2[i] = flux
        k2_f = KT_central2(U1_flux, F2, numXj, num_groups, dt, dx, CX_LB, CX_UB)
        U2_flux = 1/2 * U_half + 1/2 * (U1_flux + dt * k2_f)

        # flux_dt_3 = Parallel(n_jobs=12)(
        #     delayed(flux_step)(i, n_samples, x_sample, y_sample, z_sample, U2_flux[i], bounds_list, num_groups, cx_loc, cy_loc, cz_loc) 
        #     for i in range(0, numXj)
        # )
        # for i, flux in flux_dt_3:
        #     F3[i] = flux
        # k3_f = KT_central2(U2_flux, F3, numXj, num_groups, dt, dx, CX_LB, CX_UB)
        # U3_flux = 1/3 * U_half + 2/3 * (U2_flux + dt * k3_f)
        
        # Apply Euler to advance collision term to t^{n + 1}.
        coll_dt = Parallel(n_jobs=12)(
            delayed(coll_step)(i, n_samples, x_sample, y_sample, z_sample, U2_flux[i], bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type)
            for i in range(0, numXj)
        )
        for i, coll in coll_dt:
            k2_c[i, :, 0] = coll[0]
            k2_c[i, :, 1] = coll[1]
            k2_c[i, :, 2] = coll[2]
            k2_c[i, :, 3] = coll[3]
            k2_c[i, :, 4] = coll[4]

        # Update solution.
        U = U2_flux + dt/2 * k2_c

        # Save solution.
        f1 = 'simulation_data/U{}.npy'.format(t + 0)
        with open(f1, 'wb') as file:
            np.save(file, U)

        print(t * dt,  t + 0)

if __name__ == '__main__':
    run_simulation()