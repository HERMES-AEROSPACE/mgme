import numpy as np
from ..banner import print_banner
from ..shock_helper import ic, KT_central2, generate_regular_samples
from ..shock_helper import calc_flux_analytical, calc_flux_int
from ..physics.grid import calculate_velocity_grid
from ..physics.collide import collide
from ..config_1d import CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE, GROUP_PARAMS, VELOCITY_SPACE, SIMULATION_PARAMS
import itertools
from scipy import special
from joblib import Parallel, delayed, parallel_backend
import time, sys, cProfile, io, pstats
from numba import types
from scipy.stats import qmc
from matplotlib import pyplot as plt


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
    omega = FREESTREAM_PARAMS['omega']

    alpha = SIMULATION_PARAMS['alpha']
    n_coll = SIMULATION_PARAMS['n_coll']
    CX_LB, CX_UB = VELOCITY_SPACE['cx_range']
    CY_LB, CY_UB = VELOCITY_SPACE['cy_range']
    CZ_LB, CZ_UB = VELOCITY_SPACE['cz_range']
    key_type = types.UniTuple(types.int64, 2)

    # Some pre-shock quantities.
    a1 = np.sqrt(gamma * R * T1)
    u1 = Ma1 * a1
    rho1 = P1/(R * T1)
    n1 = P1/(R * T1) * 1/m
    m_r = 0.5  # reduced mass

    # Post shock quantities.
    T2 = T1 * (((gamma - 1) * Ma1**2 + 2) * (2 * gamma * Ma1**2 - (gamma - 1)))/((gamma + 1)**2 * Ma1**2)
    P2 = P1 * ((2 * gamma * Ma1**2) - (gamma - 1))/(gamma + 1)
    rho2 = P2/(R * T2)  
    a2 = np.sqrt(gamma * R * T2)
    u2 = u1 * rho1/rho2
    Ma2 = Ma1 * u2/u1 * (T1/T2)**0.5
    n2 = P2/(R * T2) * 1/m
    print(n1)

    # Set up reference variables.
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
    gamma_omega = special.gamma(5/2 - omega)
    sigma_coeff_hat = 1/gamma_omega * (1 / m_r)**(0.5 - omega)
    print('Reference mean free path:', lam_ref, '[m]')
    print('Characteristic length:', L_ref, '[m]')
    print('Knudsen number:', Kn)
    print('Collision cross section omega:', omega)
    
    print("---------------------------SETTING UP INITIAL CONDITION---------------------------")
    # Set up velocity and physical space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    xj_vec = np.linspace(PHYS_SPACE['xj_range'][0], PHYS_SPACE['xj_range'][1], PHYS_SPACE['num_xj'])
    dx = np.abs(xj_vec[1] - xj_vec[0])
    dcx = np.abs(cx_vec[1] - cx_vec[0])
    dcy = np.abs(cy_vec[1] - cy_vec[0])
    dcz = np.abs(cz_vec[1] - cz_vec[0])
    Xj_l = PHYS_SPACE['xj_range'][0]
    Xj_u = PHYS_SPACE['xj_range'][1]
    numXj = PHYS_SPACE['num_xj']

    # Set up time step and simulation parameters.
    cfl = SIMULATION_PARAMS['cfl']
    t_end = SIMULATION_PARAMS['t_end']
    T_local = T2 / T_ref   # downstream temperature, nondimensional
    T_d = 1.0          # reference temperature used to calculate species diameter
    v_max = np.max([np.abs(CX_UB), np.abs(CX_LB)])

    tc_vhs = 1 / (2 * (d/d_ref)**2 * (n2/n_ref) * np.sqrt(2 * T_d / np.pi) * (T_local/T_d)**(1-omega))
    tc_conv = dx / v_max

    # dt = np.round(cfl/(1/tc + CX_UB/dx), 3)
    dt = np.round(cfl * np.min([tc_vhs, tc_conv]), 3)
    
    print('CFL number:', cfl)
    print('Collision time scale:', tc_vhs * t_ref, '[s]')
    print('Time step:', dt)
    print('dx:', dx)

    # Calculate intial condition on physical grid.
    transition_start = -25
    transition_end = 15
    ramp_length = transition_end - transition_start

    t = (xj_vec - transition_start) / ramp_length
    t = np.clip(t, 0, 1)
    cosine_factor = 0.5 * (1 - np.cos(np.pi * t))

    T_val = (T1 + cosine_factor * (T2 - T1)) / T_ref
    u_val = (u1 + cosine_factor * (u2 - u1)) / c_ref
    n_val = (n1 + cosine_factor * (n2 - n1)) / n_ref

    # Create arrays for group bounds.
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

    bounds_list = np.zeros((num_groups, 6))
    for i in range(num_groups):
        bounds_list[i] = np.array([ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2]])
    print(bounds_list)

    # Option to restart simulation from a saved moment list. Calculate initial moments in space.
    restart = 0
    U0, f = ic(cx, cy, cz, cx_vec, cy_vec, cz_vec, n_val, u_val, T_val, VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'], VELOCITY_SPACE['num_cz'], \
        numXj, num_groups, combinations)
    if restart:
        data = np.load('simulation_data/U2000.npy')
        print('Restarting from...')
        U = data
    else:
        U = U0.copy()

    initial_f = 'simulation_data/U{}.npy'.format(0)
    with open(initial_f, 'wb') as file:
        np.save(file, U)

    # plt.plot(xj_vec, 2/3 * ((np.sum(U[:, :, 4], axis=1) / np.sum(U[:, :, 0], axis=1)) - (np.sum(U[:, :, 1], axis=1) / np.sum(U[:, :, 0], axis=1))**2))
    # plt.plot(T_val, '--')
    # plt.xlabel('xj', fontsize=16)
    # plt.ylabel('T', fontsize=16)
    # plt.savefig('plots/ic.pdf')
    # f[f < 1e-12] = 0.0
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(n_val[-1] * f[-1], cz_vec, axis=2), cy_vec, axis=1))
    # plt.plot(cz_vec, np.trapezoid(np.trapezoid(n_val[0] * f[0], cy_vec, axis=1), cx_vec, axis=0))
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f[0], cz_vec, axis=2), cy_vec, axis=1))
    # plt.savefig('plots/icdist.pdf')
    # plt.show()
    
    print("--------------------------------BEGIN SIMULATION----------------------------------")
    # Cache distribution parameters for Newton solver.
    lam_cache = np.zeros((numXj, num_groups, 5))
    pr = cProfile.Profile()
    
    def step(i, U_i, bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, sigma_coeff_hat, omega, alpha, lam_cache_i):
        # Calculate weights through optimization.
        weights, num_valid_samples, lam_out, x_s, y_s, z_s, offsets = generate_regular_samples(i, U_i, num_groups, bounds_list, lam_cache_i)
        
        # Advance the collision and flux forward.
        coll = collide(x_s, y_s, z_s, weights, num_valid_samples, bounds_list, num_groups, \
                        n_coll, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, sigma_coeff_hat, omega, alpha)
        # flux = calc_flux_analytical(lam_out, bounds_list, num_groups, U_i)
        flux = calc_flux_int(num_groups, weights, offsets, x_s, y_s, z_s)
        
        second_moment_x = 0.0
        rho  = np.sum(U_i[:, 0])          # cell total density
        rhou = np.sum(U_i[:, 1])          # cell total x-momentum
        ux   = rhou / rho 
        for j in range(num_groups):
            start = int(offsets[j])
            end   = int(offsets[j+1])
            w_i   = weights[start:end]     # f * dv for this group
            vx_i  = x_s[start:end]
            second_moment_x += np.sum(vx_i**2 * w_i)
        Tx = second_moment_x / rho - ux**2

        return i, coll, flux, lam_out, Tx

    for t in range(1, int(np.ceil(int(t_end / dt) / 100) * 100) + 1):
        # Boundary conditions.
        U[0, :] = U0[0, :]
        U[-1, :] = U[-2, :]

        # RK1 integration.
        k1_c = np.zeros((numXj, num_groups, 5))
        F1 = np.zeros((numXj, num_groups, 5))
        
        # Integrate collision term and flux term separately. Integrate in time using explicit Euler.
        with parallel_backend('loky', inner_max_num_threads=1):
            step_dt = Parallel(n_jobs=32)(
                delayed(step)(i, U[i], bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, \
                            key_type, sigma_coeff_hat, omega, alpha, lam_cache[i])
                for i in range(0, numXj)
            )

        Tx_hist = np.zeros(numXj)

        for i, coll, flux, lam_out, Tx in step_dt:
            k1_c[i, :, 0] = coll[0]
            k1_c[i, :, 1] = coll[1]
            k1_c[i, :, 2] = coll[2]
            k1_c[i, :, 3] = coll[3]
            k1_c[i, :, 4] = coll[4]
            F1[i] = flux
            lam_cache[i] = lam_out
            Tx_hist[i] = Tx

        # 2nd order central difference using MUSCL reconstruction and slope limiters.
        k1_f = KT_central2(U, F1, numXj, num_groups, dt, dx, CX_LB, CX_UB)
        U += (k1_f + k1_c) * dt

        if t % 50 == 0:
            # Save solution.
            f1 = 'simulation_data/U{}.npy'.format(t + 0)
            with open(f1, 'wb') as file:
                np.save(file, U)

            f2 = 'simulation_data/Tx{}.npy'.format(t + 0)
            with open(f2, 'wb') as file:
                np.save(file, Tx_hist)
            print(t * dt,  t + 0)

if __name__ == '__main__':
    run_simulation()