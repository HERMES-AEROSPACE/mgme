import numpy as np
from .banner import print_banner
from .sampling import calculate_velocity_grid, generate_grid
from .shock_helper import calc_flux_int, ic, invert, calc_integral, calc_flux, RK_LF, generate_regular_samples, coll_source, lookup_table
from .config import CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE, GROUP_PARAMS, VELOCITY_SPACE, SAMPLING_PARAMS, COLLISION_PARAMS
from matplotlib import pyplot as plt
import itertools
import cProfile, pstats
from scipy import interpolate
from joblib import Parallel, delayed
import time


def run_simulation():
    print_banner()

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
    c_ref = np.sqrt((2 * k * T1)/m_ref)
    T_ref = m * c_ref**2 / (2 * k)

    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid()
    xj_vec = np.linspace(PHYS_SPACE['xj_range'][0], PHYS_SPACE['xj_range'][1], PHYS_SPACE['num_xj'])
    dx = np.abs(xj_vec[1] - xj_vec[0])
    dcx = np.abs(cx_vec[1] - cx_vec[0])
    dcy = np.abs(cy_vec[1] - cy_vec[0])
    dcz = np.abs(cz_vec[1] - cz_vec[0])
    Xj_l = PHYS_SPACE['xj_range'][0]
    Xj_u = PHYS_SPACE['xj_range'][1]
    numXj = PHYS_SPACE['num_xj']

    cos_ramp = 0.2
    T_val = (0.5 * (np.cos(np.pi/2.0/cos_ramp * (xj_vec + np.abs(Xj_l) * cos_ramp) / np.abs(Xj_l)) + 1) * (T1 - T2) + T2)/T_ref
    u_val = (0.5 * (np.cos(np.pi/2.0/cos_ramp * (xj_vec + np.abs(Xj_l) * cos_ramp) / np.abs(Xj_l)) + 1) * (u1 - u2) + u2)/c_ref
    n_val = (0.5 * (np.cos(np.pi/2.0/cos_ramp * (xj_vec + np.abs(Xj_l) * cos_ramp) / np.abs(Xj_l)) + 1) * (n1 - n2) + n2)/n_ref

    T_val[0:int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) - cos_ramp/2))] = T1/T_ref
    T_val[int((numXj) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) + cos_ramp/2)):] = T2/T_ref

    u_val[0:int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) - cos_ramp/2))] = u1/c_ref
    u_val[int((numXj) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) + cos_ramp/2)):] = u2/c_ref

    n_val[0:int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) - cos_ramp/2))] = n1/n_ref
    n_val[int((numXj) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) + cos_ramp/2)):] = n2/n_ref

    print("-------------------SETTING UP INITIAL CONDITION-------------------")
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
    U0, _ = ic(cx, cy, cz, dcx, dcy, dcz, n_val, u_val, T_val, VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'], VELOCITY_SPACE['num_cz'], \
        numXj, num_groups, combinations)
    if restart:
        data = np.load('simulation_data/U666.npy')
        print('Restarting from...')
        U_list = data
    else:
        U_list = U0.copy()

    # f[f < 1e-12] = 0.0
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(n_val[-1] * f[-1], cz_vec, axis=2), cy_vec, axis=1))
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f[0], cz_vec, axis=2), cy_vec, axis=1))
    # plt.plot(xj_vec, np.sum(U0, axis=1)[:, 4])
    # plt.plot(xj_vec, T_val, '--')
    # plt.xlabel('xj', fontsize=16)
    # plt.ylabel('u', fontsize=16)
    # plt.show()
    
    F_list = np.zeros((numXj, num_groups, 5))

    b_range = np.logspace(-8, 1, 20, endpoint=True)
    interpolater_list = []
    bounds_list = np.zeros((num_groups, 6))
    
    for i in range(num_groups):
        inputs, outputs = lookup_table(b_range, np.linspace(ci_combo[i, 0], cf_combo[i, 0], 20), \
            np.linspace(ci_combo[i, 1], cf_combo[i, 1], 20), np.linspace(ci_combo[i, 2], cf_combo[i, 2], 20), \
                ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2])

        interpolater_list.append(interpolate.NearestNDInterpolator(outputs, inputs))
        bounds_list[i] = np.array([ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2]])

    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample, cx_loc, cy_loc, cz_loc = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])

    print("-------------------------BEGIN SIMULATION-------------------------")
    t_end = 10.0
    dt = 0.005
    profiler = cProfile.Profile()
    for t in range(0, int(np.ceil(int(t_end / dt) / 100) * 100) + 1):
        # Inversion and calculate flux. 
        # profiler.enable()
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats('cumulative')
        # stats.print_stats(20)
        # Ak, bk, wxk, wyk, wzk = invert(U_list, numXj, num_groups, bounds_list, interpolater_list)
        # I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z = calc_integral(bk, wxk, wyk, wzk, bounds_list, numXj, num_groups)
        # F_list = calc_flux(Ak, bk, wxk, wyk, wzk, I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z, numXj, num_groups)
        
        # Boundary conditions.
        U_list[0, :] = U0[0, :]
        U_list[-1, :] = U_list[-2, :]

        # RK2 integration.
        k1 = np.zeros((numXj, num_groups, 5))
        k2 = np.zeros((numXj, num_groups, 5))
        k1_c = np.zeros((numXj, num_groups, 5))
        k2_c = np.zeros((numXj, num_groups, 5))
 
        def process_iter(i, n_samples, x_sample, y_sample, z_sample, U_i, bounds_list, num_groups, COLLISION_PARAMS, VELOCITY_SPACE, cx_loc, cy_loc, cz_loc, dt):
            weights, num_group_sample, masks = generate_regular_samples(
                i, n_samples, x_sample, y_sample, z_sample, U_i, bounds_list, num_groups
            )

            flux = calc_flux_int(num_groups, weights, masks, bounds_list, cx_loc, cy_loc, cz_loc)
            
            group_n, group_px, group_py, group_pz, group_e = coll_source(
                x_sample, y_sample, z_sample, weights, num_group_sample, 
                num_groups, n_samples, bounds_list, COLLISION_PARAMS, VELOCITY_SPACE
            )
            return i, group_n * dt, group_px * dt, group_py * dt, group_pz * dt, group_e * dt, flux

        res = Parallel(n_jobs=10)(
            delayed(process_iter)(i, n_samples, x_sample, y_sample, z_sample, U_list[i], bounds_list, num_groups, COLLISION_PARAMS, VELOCITY_SPACE, cx_loc, cy_loc, cz_loc, dt) 
            for i in range(0, numXj)
        )

        for i, n, px, py, pz, e, flux in res:
            F_list[i] = flux
            k1_c[i, :, 0] = n
            k1_c[i, :, 1] = px
            k1_c[i, :, 2] = py
            k1_c[i, :, 3] = pz
            k1_c[i, :, 4] = e

        k1 = RK_LF(U_list, F_list, numXj, num_groups, dx, dt)

        # res2 = Parallel(n_jobs=10)(
        #     delayed(process_iter)(i, n_samples, x_sample, y_sample, z_sample, U_list[i] + k1, bounds_list, num_groups, COLLISION_PARAMS, VELOCITY_SPACE, cx_loc, cy_loc, cz_loc, dt) 
        #     for i in range(0, numXj)
        # )

        # for i, n, px, py, pz, e, flux in res:
        #     F_list[i] = flux
        #     k1_c[i, :, 0] = n
        #     k1_c[i, :, 1] = px
        #     k1_c[i, :, 2] = py
        #     k1_c[i, :, 3] = pz
        #     k1_c[i, :, 4] = e

        # k2 = RK_LF(U_list, F_list, numXj, num_groups, dx, dt)

        dU = 0.5 * (k1 + k2 + k1_c + k2_c)
        U_list += dU

        # if t % 10 == 0:
        f1 = 'simulation_data/U{}.npy'.format(t)

        with open(f1, 'wb') as file:
            np.save(file, U_list)

        print(t * dt,  t)

if __name__ == '__main__':
    run_simulation()