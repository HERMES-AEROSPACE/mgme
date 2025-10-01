import numpy as np
from .banner import print_banner
from .sampling import calculate_velocity_grid, generate_grid
from .shock_helper import ic, invert, calc_integral, calc_flux, RK_LF, generate_regular_samples, coll_source
from .moment_utils import moments
from .config import CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE, GROUP_PARAMS, VELOCITY_SPACE, SAMPLING_PARAMS, COLLISION_PARAMS
from matplotlib import pyplot as plt


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
    T_val[int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) + cos_ramp/2)):] = T2/T_ref

    u_val[0:int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) - cos_ramp/2))] = u1/c_ref
    u_val[int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) + cos_ramp/2)):] = u2/c_ref

    n_val[0:int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) - cos_ramp/2))] = n1/n_ref
    n_val[int((numXj - 1) * (np.abs(Xj_l) / (np.abs(Xj_l) + Xj_u) + cos_ramp/2)):] = n2/n_ref
    
    print("-------------------SETTING UP INITIAL CONDITION-------------------")
    num_groups = GROUP_PARAMS['num_groups_cx']
    U0, f = ic(cx, cy, cz, dcx, dcy, dcz, n_val, u_val, T_val, VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'], VELOCITY_SPACE['num_cz'], \
               numXj, num_groups, GROUP_PARAMS['group_bounds_cx'])
    print(U0[-1, :])
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(n_val[-1] * f[-1], cz_vec, axis=2), cy_vec, axis=1))
    # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f[-1], cz_vec, axis=2), cy_vec, axis=1))
    # print(np.trapezoid(np.trapezoid(np.trapezoid(cx * f[-1], cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0) * n_val[-1], np.sum(U0[-1], axis=0)[1])
    # plt.show()
    f[f < 1e-12] = 0.0

    U_list = U0.copy()
    F_list = np.zeros((numXj, num_groups, 5))
    Ak_list = np.zeros((numXj, num_groups))
    bk_list = np.zeros((numXj, num_groups))
    wxk_list = np.zeros((numXj, num_groups))
    wyk_list = np.zeros((numXj, num_groups))
    wzk_list = np.zeros((numXj, num_groups))

    b_range = np.logspace(-8, 1, 20, endpoint=True)
    wy_range, wz_range = np.linspace(-20, 20, 20), np.linspace(-20, 20, 20)
    inputs_list = []
    outputs_list = []
    
    def lookup_table(b_range, wx_range, wy_range, wz_range, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
        for b in b_range:
            for wx in wx_range:
                for wy in wy_range:
                    for wz in wz_range:
                        ux, uy, uz, e = moments(b, wx, wy, wz, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz)

                        if np.all(np.isfinite([ux, uy, uz, e])):
                            inputs_list.append([b, wx, wy, wz])
                            outputs_list.append([ux, uy, uz, e])

        return np.array(inputs_list), np.array(outputs_list)
    
    input_list1, output_list1 = lookup_table(b_range, np.linspace(-20, 0.0, 20), wy_range, wz_range, -20.0, 0.0, -20, 20, -20, 20)
    input_list, output_list = lookup_table(b_range, np.linspace(0.0, 6.66666666666, 20), wy_range, wz_range, 0.0, 6.6666666666666, -20, 20, -20, 20)
    input_list2, output_list2 = lookup_table(b_range, np.linspace(6.666666666666, 20, 20), wy_range, wz_range, 6.6666666666666, 20.0, -20, 20, -20, 20)

    n_samples = SAMPLING_PARAMS['n_samples_x'] * SAMPLING_PARAMS['n_samples_y'] * SAMPLING_PARAMS['n_samples_z']
    x_sample, y_sample, z_sample, sample_loc_x, sample_loc_y, sample_loc_z = generate_grid(SAMPLING_PARAMS['n_samples_x'], SAMPLING_PARAMS['n_samples_y'], SAMPLING_PARAMS['n_samples_z'])
    
    bounds_list = np.zeros((3, 6))
    bounds_list[0] = np.array([-20.0, 0.0, -20.0, 20.0, -20.0, 20.0])
    bounds_list[1] = np.array([0.0, 6.6666666666666, -20.0, 20.0, -20.0, 20.0])
    bounds_list[2] = np.array([6.66666666666, 20.0, -20.0, 20.0, -20.0, 20.0])

    print("-------------------------BEGIN SIMULATION-------------------------")
    t_end = 30.0
    dt = 0.01
    for t in range(0, int(np.ceil(int(t_end / dt) / 100) * 100) + 1):
        # plt.plot(xj_vec, np.sum(U_list, axis=1)[:, 0])
        # plt.show()

        # Inversion and calculate flux.
        Ak, bk, wxk, wyk, wzk = invert(U_list, numXj, num_groups, GROUP_PARAMS, input_list, output_list, input_list2, output_list2, input_list1, output_list1)
        print(Ak[-1], bk[-1], wxk[-1], wyk[-1], wzk[-1])
        I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z = calc_integral(bk, wxk, wyk, wzk, GROUP_PARAMS, numXj, num_groups)
        F_list = calc_flux(Ak, bk, wxk, wyk, wzk, I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z, numXj, num_groups)

        # Boundary conditions.
        U_list[0, :] = U0[0, :]
        U_list[-1, :] = U_list[-2, :]

        # RK2 integration.
        k1 = np.zeros((numXj, num_groups, 5))
        k2 = np.zeros((numXj, num_groups, 5))
        k1_c = np.zeros((numXj, num_groups, 5))
        k2_c = np.zeros((numXj, num_groups, 5))

        k1 = RK_LF(U_list, F_list, numXj, num_groups, dx, dt)
        for i in range(numXj):
            weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, \
                U_list[i] / np.sum(U_list[i], axis=0)[0], GROUP_PARAMS, num_groups)
            
            weights *= np.sum(U_list[i], axis=0)[0]
            
            # print(np.sum(x_sample * weights), np.sum(U_list[i], axis=0)[1])
            # print(np.sum(y_sample * weights), np.sum(U_list[i], axis=0)[2])
            # print(np.sum(z_sample * weights), np.sum(U_list[i], axis=0)[3])
            # print(np.sum((x_sample**2 + y_sample**2 + z_sample**2) * weights), np.sum(U_list[i], axis=0)[4])
            # weights2, _ = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, \
                # U_list[-1] / np.sum(U_list[-1], axis=0)[0], GROUP_PARAMS, num_groups)

            # print(np.sum(weights), np.sum(U_list[i], axis=0)[0])
            # print(np.sum(weights2), np.sum(U_list[-1], axis=0)[0])

            # plt.rcParams['font.family'] = "serif"
            # fig, ax = plt.subplots(figsize=(6, 6))
            
            # ax.plot(cx_vec, np.trapz(np.trapz(f[i], cz_vec, axis=2), cy_vec, axis=1), color='black')
            # ax.plot(cx_vec, np.trapz(np.trapz(f[-1], cz_vec, axis=2), cy_vec, axis=1), color='purple')

            # ax.plot(sample_loc_x, np.sum(np.reshape(weights, (26, 11, 11)), axis=(1, 2)), '--o', color='black')
            # ax.plot(sample_loc_x, np.sum(np.reshape(weights2, (26, 11, 11)), axis=(1, 2)), '--o', color='purple')
            
            # ax.set_xlabel('Cz', fontsize=18)
            # ax.set_ylabel('f', fontsize=18)
            # ax.tick_params(axis='both', labelsize=14)
            # ax.legend(['Pre-shock', 'Post-shock'], fontsize=14)
            # plt.tight_layout()
            # plt.show()

            group_n, group_px, group_py, group_pz, group_e = \
                coll_source(x_sample, y_sample, z_sample, weights, num_group_sample, num_groups, n_samples, bounds_list, COLLISION_PARAMS)
            k1_c[i, :, 0] = group_n * dt
            k1_c[i, :, 1] = group_px * dt
            k1_c[i, :, 2] = group_py * dt
            k1_c[i, :, 3] = group_pz * dt
            k1_c[i, :, 4] = group_e * dt
            print(i, group_n)
        
        F_list_step2 = np.zeros((numXj, num_groups, 5))
        Ak, bk, wxk, wyk, wzk = invert(U_list + k1, numXj, num_groups, GROUP_PARAMS, input_list, output_list, input_list2, output_list2, input_list1, output_list1)
        I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z = calc_integral(bk, wxk, wyk, wzk, GROUP_PARAMS, numXj, num_groups)
        F_list_step2 = calc_flux(Ak, bk, wxk, wyk, wzk, I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z, numXj, num_groups)
        
        k2 = RK_LF(U_list + k1, F_list_step2, numXj, num_groups, dx, dt) 
        for i in range(numXj):
            weights, num_group_sample = generate_regular_samples(n_samples, x_sample, y_sample, z_sample, \
                (U_list + k1)[i], GROUP_PARAMS, num_groups)

            group_n, group_px, group_py, group_pz, group_e = \
                coll_source(x_sample, y_sample, z_sample, weights, num_group_sample, num_groups, n_samples, bounds_list, COLLISION_PARAMS)
            k2_c[i, :, 0] = group_n * dt
            k2_c[i, :, 1] = group_px * dt
            k2_c[i, :, 2] = group_py * dt
            k2_c[i, :, 3] = group_pz * dt
            k2_c[i, :, 4] = group_e * dt

        dU = 0.5 * (k1_c + k2_c)

        U_list += dU

        if t % 10 == 0:
            f1 = 'simulation_data/U{}.npy'.format(t)

            with open(f1, 'wb') as file:
                np.save(file, U_list)

        print(t * dt, t)
        plt.plot(xj_vec, np.sum(U_list, axis=1)[:, 4])
        plt.show()

if __name__ == '__main__':
    run_simulation()