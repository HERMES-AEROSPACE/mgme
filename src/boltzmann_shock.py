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
    d = CONSTANTS['d']
    T1 = FREESTREAM_PARAMS['T1']
    P1 = FREESTREAM_PARAMS['P1']
    Ma1 = FREESTREAM_PARAMS['Ma1']
    m = CONSTANTS['m']
    k = CONSTANTS['k']

    n_coll = COLLISION_PARAMS['n_coll']
    CX_LB, CX_UB = VELOCITY_SPACE['cx_range']
    CY_LB, CY_UB = VELOCITY_SPACE['cy_range']
    CZ_LB, CZ_UB = VELOCITY_SPACE['cz_range']
    key_type = types.UniTuple(types.int64, 2)

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
    # print(n1, n2, u1, u2, T1, T2)

    m_ref = m
    n_ref = n1
    T_ref = T1
    d_ref = d
    c_ref = np.sqrt((2 * k * T_ref)/m_ref)
    sigma_ref = np.pi * d_ref**2
    lam_ref = 1/(n_ref * sigma_ref)
    L_ref = lam_ref
    Kn = lam_ref / L_ref
    t_ref = L_ref / c_ref
    print('Reference mean free path:', lam_ref, '[m]')
    print('Knudsen number:', Kn)
    # c_ref = np.sqrt((2 * k * T1)/m_ref)
    # T_ref = m * c_ref**2 / (2 * k)

    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    print(cx_vec[68])
    xj_vec = np.linspace(PHYS_SPACE['xj_range'][0], PHYS_SPACE['xj_range'][1], PHYS_SPACE['num_xj'])
    dx = np.abs(xj_vec[1] - xj_vec[0])
    dcx = np.abs(cx_vec[1] - cx_vec[0])
    dcy = np.abs(cy_vec[1] - cy_vec[0])
    dcz = np.abs(cz_vec[1] - cz_vec[0])
    Xj_l = PHYS_SPACE['xj_range'][0]
    Xj_u = PHYS_SPACE['xj_range'][1]
    numXj = PHYS_SPACE['num_xj']

    cfl = 0.7
    t_end = 80.0
    tc = 1/(n2/n_ref * (d/d_ref)**2 * np.sqrt(2) * 1)
    dt = np.round(cfl/(1/tc + CX_UB/dx), 3)
    print('CFL number:', cfl)
    print('Collision time scale:', tc * t_ref, '[s]')
    print('Time step:', dt)
    print('dx:', dx)

    transition_start = -30
    transition_end = 30
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

    restart = 1

    U0, f = ic(cx, cy, cz, cx_vec, cy_vec, cz_vec, n_val, u_val, T_val, VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'], VELOCITY_SPACE['num_cz'], \
        numXj, num_groups, combinations)
    if restart:
        data = np.load('simulation_data/U2200.npy')
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
    # plt.plot(cz_vec, np.trapezoid(np.trapezoid(n_val[-1] * f[-1], cy_vec, axis=1), cx_vec, axis=0))
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f[0], cz_vec, axis=2), cy_vec, axis=1))
    # plt.show()

    bounds_list = np.zeros((num_groups, 6))
    for i in range(num_groups):
        bounds_list[i] = np.array([ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2]])
    
    x_sample, y_sample, z_sample, offsets, num_samples = generate_grid(bounds_list, num_groups)
    print(bounds_list)
    print("--------------------------------BEGIN SIMULATION----------------------------------")

    def step(i, U_i, bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, x_sample, y_sample, z_sample, offsets, num_samples):
        # Calculate weights through convex optimization.
        weights, num_valid_samples, x_sample_mod, y_sample_mod, z_sample_mod = generate_regular_samples(
            i, offsets, num_samples, x_sample, y_sample, z_sample, U_i, num_groups, bounds_list,
            max_retries=10
        )

        # Advance the collision and flux forward.
        coll = collide(x_sample_mod, y_sample_mod, z_sample_mod, weights, num_valid_samples, bounds_list, num_groups, \
                        n_coll, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type)
        flux = calc_flux_int(num_groups, weights, offsets, x_sample_mod, y_sample_mod, z_sample_mod)

        return i, coll, flux

    profiler = cProfile.Profile()
    for t in range(0, int(np.ceil(int(t_end / dt) / 100) * 100) + 1):
        # Boundary conditions.
        U[0, :] = U0[0, :]
        U[-1, :] = U[-2, :]

        # RK1 integration.
        k1_c = np.zeros((numXj, num_groups, 5))
        F1 = np.zeros((numXj, num_groups, 5))    
        
        # Integrate collision term and flux term separately. Integrate in time using explicit Euler.
        step_dt = Parallel(n_jobs=16)(
            delayed(step)(i, U[i], bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, x_sample, y_sample, z_sample, offsets, num_samples)
            for i in range(0, numXj)
        )
        for i, coll, flux in step_dt:
            k1_c[i, :, 0] = coll[0]
            k1_c[i, :, 1] = coll[1]
            k1_c[i, :, 2] = coll[2]
            k1_c[i, :, 3] = coll[3]
            k1_c[i, :, 4] = coll[4]
            F1[i] = flux

        # 2nd order central difference using MUSCL reconstruction and slope limiters.
        if t == 0: F0 = F1
        k1_f = KT_central2(U, F1, numXj, num_groups, dt, dx, CX_LB, CX_UB)
        # k1_f = LF_central1(U, F1, numXj, num_groups, dt/dx)
        U += (k1_f + k1_c) * dt

        # Save solution.
        f1 = 'simulation_data/U{}.npy'.format(t + 2201)
        with open(f1, 'wb') as file:
            np.save(file, U)

        print(t * dt,  t + 2201)

if __name__ == '__main__':
    run_simulation()