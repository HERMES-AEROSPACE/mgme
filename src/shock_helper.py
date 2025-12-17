from numba import jit, types
from numba.typed import Dict
import numpy as np
from scipy import optimize, special, interpolate
import math
import cvxpy as cp
import sys


def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 

def generate_grid(n_samples_x, n_samples_y, n_samples_z):
    def gen_group_sample(a, b, n):
        return np.linspace(a, b, n, endpoint=False) + (b - a) / (2 * n)
    
    g1 = gen_group_sample(-2.5, 0.4666666666666668, 10)
    # g2 = gen_group_sample(0, 2.0999999999999996, 6)
    g3 = gen_group_sample(0.4666666666666668, 4.0, 10)
    g4 = gen_group_sample(-3.5, 3.5, 10)

    # sample_loc_x = np.append(sample_loc_x_neg, sample_loc_x_pos)
    # np.append(np.linspace(-20, -1e-5, 8), np.append(np.linspace(1e-5, 6.65, 8), np.linspace(6.67, 20, 8))) 
    sample_loc_x = np.concatenate([g1, g3])
    sample_loc_y = np.concatenate([g4])
    sample_loc_z = sample_loc_y
    print(sample_loc_x)
    
    [xgrid, ygrid, zgrid] = np.meshgrid(sample_loc_x, sample_loc_y, sample_loc_z, indexing='ij')

    x_sample = xgrid.flatten()
    y_sample = ygrid.flatten()
    z_sample = zgrid.flatten()

    return x_sample, y_sample, z_sample, sample_loc_x, sample_loc_y, sample_loc_z

@jit(nopython=True)
def moments(beta, wx, wy, wz, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    I0x = np.sqrt(np.pi / (4 * beta)) * (math.erf(np.sqrt(beta) * (cf_cx - wx)) - math.erf(np.sqrt(beta) * (ci_cx - wx)))
    I0y = np.sqrt(np.pi / (4 * beta)) * (math.erf(np.sqrt(beta) * (cf_cy - wy)) - math.erf(np.sqrt(beta) * (ci_cy - wy)))
    I0z = np.sqrt(np.pi / (4 * beta)) * (math.erf(np.sqrt(beta) * (cf_cz - wz)) - math.erf(np.sqrt(beta) * (ci_cz - wz)))

    I1x = (np.exp(-beta * (ci_cx - wx)**2) - np.exp(-beta * (cf_cx - wx)**2)) / (2 * beta)
    I1y = (np.exp(-beta * (ci_cy - wy)**2) - np.exp(-beta * (cf_cy - wy)**2)) / (2 * beta)
    I1z = (np.exp(-beta * (ci_cz - wz)**2) - np.exp(-beta * (cf_cz - wz)**2)) / (2 * beta)

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cx - wx)**2) * (cf_cx - wx))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cx - wx)**2) * (ci_cx - wx))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (math.erf(np.sqrt(beta) * (cf_cx - wx)) - math.erf(np.sqrt(beta) * (ci_cx - wx)))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cy - wy)**2) * (cf_cy - wy))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cy - wy)**2) * (ci_cy - wy))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (math.erf(np.sqrt(beta) * (cf_cy - wy)) - math.erf(np.sqrt(beta) * (ci_cy - wy)))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cz - wz)**2) * (cf_cz - wz))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cz - wz)**2) * (ci_cz - wz))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (math.erf(np.sqrt(beta) * (cf_cz - wz)) - math.erf(np.sqrt(beta) * (ci_cz - wz)))

    return [(I1x + wx*I0x) / I0x, (I1y + wy*I0y) / I0y, (I1z + wz*I0z) / I0z, \
            (I2x + 2 * wx * I1x + wx**2 * I0x) / I0x + (I2y + 2 * wy * I1y + wy**2 * I0y) / I0y + (I2z + 2 * wz * I1z + wz**2 * I0z) / I0z]

def moment_eq(x, ux, uy, uz, e, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    """
    Moment equation for solving distribution parameters.
    
    Args:
        x: Array containing [beta, wx, wy, wz]
        u: Target velocity
        e: Target energy
        ci: Lower bound
        cf: Upper bound
    
    Returns:
        Array of moment equations
    """
    I0x = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cx - x[1])) - special.erf(np.sqrt(x[0]) * (ci_cx - x[1])))
    I0y = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cy - x[2])) - special.erf(np.sqrt(x[0]) * (ci_cy - x[2])))
    I0z = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cz - x[3])) - special.erf(np.sqrt(x[0]) * (ci_cz - x[3])))

    I1x = (np.exp(-x[0] * (ci_cx - x[1])**2) - np.exp(-x[0] * (cf_cx - x[1])**2)) / (2 * x[0])
    I1y = (np.exp(-x[0] * (ci_cy - x[2])**2) - np.exp(-x[0] * (cf_cy - x[2])**2)) / (2 * x[0])
    I1z = (np.exp(-x[0] * (ci_cz - x[3])**2) - np.exp(-x[0] * (cf_cz - x[3])**2)) / (2 * x[0])

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cx - x[1])**2) * (cf_cx - x[1]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cx - x[1])**2) * (ci_cx - x[1])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cx - x[1])) - special.erf(np.sqrt(x[0]) * (ci_cx - x[1])))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cy - x[2])**2) * (cf_cy - x[2]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cy - x[2])**2) * (ci_cy - x[2])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cy - x[2])) - special.erf(np.sqrt(x[0]) * (ci_cy - x[2])))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cz - x[3])**2) * (cf_cz - x[3]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cz - x[3])**2) * (ci_cz - x[3])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cz - x[3])) - special.erf(np.sqrt(x[0]) * (ci_cz - x[3])))

    return [(I1x + x[1] * I0x) / I0x - ux, (I1y + x[2] * I0y) / I0y - uy, (I1z + x[3] * I0z) / I0z - uz, \
            (I2x + 2 * x[1] * I1x + x[1]**2 * I0x) / (I0x) + (I2y + 2 * x[2] * I1y + x[2]**2 * I0y) / (I0y) + (I2z + 2 * x[3] * I1z + x[3]**2 * I0z) / (I0z) - e]

def initialize_maxwellian(m_hat, n_hat, T_hat, v_hat, cx, cy, cz):
    A = n_hat * (m_hat / (np.pi * T_hat))**1.5
    beta = m_hat / T_hat
    dist = A * np.exp(-beta * ((cx - v_hat)**2 + cy**2 + cz**2))

    return dist

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)

    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)

    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu

def ic(cx, cy, cz, cx_vec, cy_vec, cz_vec, n_val, u_val, T_val, numCx, numCy, numCz, numXj, num_groups, group_bounds):
    f = np.zeros((numXj, numCx, numCy, numCz))
    U0 = np.zeros((numXj, num_groups, 5))

    for point in range(0, numXj):
        m_hat = 1.0
        f[point] = initialize_maxwellian(m_hat, n_val[point], T_val[point], u_val[point], cx, cy, cz)

        for i in range(0, num_groups):
            lb_cx = group_bounds[i, 0, 0]
            ub_cx = group_bounds[i, 0, 1]
            lb_cy = group_bounds[i, 1, 0]
            ub_cy = group_bounds[i, 1, 1]
            lb_cz = group_bounds[i, 2, 0]
            ub_cz = group_bounds[i, 2, 1]

            U0[point, i] = calc_moment(f[point, lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], \
                cx[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cy[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cz[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], \
                cx_vec[lb_cx:ub_cx], cy_vec[lb_cy:ub_cy], cz_vec[lb_cz:ub_cz])

    return (U0, f)

@jit(nopython=True)
def lookup_table(b_range, wx_range, wy_range, wz_range, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    list1 = []
    list2 = []

    for b in b_range:
        for wx in wx_range:
            for wy in wy_range:
                for wz in wz_range:
                    ux, uy, uz, e = moments(b, wx, wy, wz, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz)

                    if np.all(np.isfinite(np.array([ux, uy, uz, e]))):
                        list1.append([b, wx, wy, wz])
                        list2.append([ux, uy, uz, e])

    return np.array(list1), np.array(list2)

def invert(U_list, numXj, numGroups, bounds_list, interpolater_list):
    Ak = np.zeros((numXj, numGroups))
    bk = np.zeros((numXj, numGroups))
    wxk = np.zeros((numXj, numGroups))
    wyk = np.zeros((numXj, numGroups))
    wzk = np.zeros((numXj, numGroups))

    for point in range(0, numXj):
        for i in range(0, numGroups):
            if U_list[point, i, 0] > 1e-3:
                n = U_list[point, i, 0]
                ux = U_list[point, i, 1]
                uy = U_list[point, i, 2]
                uz = U_list[point, i, 3]
                e = U_list[point, i, 4]

                ci_cx = bounds_list[i, 0]
                cf_cx = bounds_list[i, 1]
                ci_cy = bounds_list[i, 2]
                cf_cy = bounds_list[i, 3]
                ci_cz = bounds_list[i, 4]
                cf_cz = bounds_list[i, 5]

                target_point = np.array([ux / n, uy / n, uz / n, e / n])
                initial_guess = interpolater_list[i](target_point)[0]
                sol = optimize.least_squares(moment_eq, initial_guess, args=(ux / n, uy / n, \
                                            uz / n, e / n, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), \
                                                bounds=([0.0, -20, -20, -20], [100.0, 20, 20, 20]), method='trf', loss='soft_l1')
                if np.linalg.norm(sol.fun) > 1e-6:
                    print('residual:', np.linalg.norm(sol.fun), point, i, sol.x)
                    print(n, ux / n, uy / n, uz / n, e / n, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, initial_guess)

                if sol.success:
                    b = sol.x[0]
                    wx = sol.x[1]
                    wy = sol.x[2]
                    wz = sol.x[3]

                I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (bounds_list[i, 1] - wx)) - special.erf(np.sqrt(b) * (bounds_list[i, 0] - wx)))
                I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (bounds_list[i, 3] - wy)) - special.erf(np.sqrt(b) * (bounds_list[i, 2] - wy)))
                I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (bounds_list[i, 5] - wz)) - special.erf(np.sqrt(b) * (bounds_list[i, 4] - wz)))
                A = n / (I0x * I0y * I0z)

                Ak[point, i] = A
                bk[point, i] = b
                wxk[point, i] = wx
                wyk[point, i] = wy
                wzk[point, i] = wz

    return Ak, bk, wxk, wyk, wzk

def calc_integral(bk, wxk, wyk, wzk, bounds_list, numXj, numGroups):
    I0x, I0y, I0z = np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups))
    I1x, I1y, I1z = np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups))
    I2x, I2y, I2z = np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups))
    I3x, I3y, I3z = np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups)), np.zeros((numXj, numGroups))

    for point in range(0, numXj):
        for i in range(0, numGroups):
            if bk[point, i] != 0.0:
                b = bk[point, i]
                wx = wxk[point, i]
                wy = wyk[point, i]
                wz = wzk[point, i]

                ci_cx = bounds_list[i, 0]
                cf_cx = bounds_list[i, 1]
                ci_cy = bounds_list[i, 2]
                cf_cy = bounds_list[i, 3]
                ci_cz = bounds_list[i, 4]
                cf_cz = bounds_list[i, 5]

                I0x[point, i] = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cx - wx)) - special.erf(np.sqrt(b) * (ci_cx - wx)))
                I0y[point, i] = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cy - wy)) - special.erf(np.sqrt(b) * (ci_cy - wy)))
                I0z[point, i] = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cz - wz)) - special.erf(np.sqrt(b) * (ci_cz - wz)))

                I1x[point, i] = (np.exp(-b * (ci_cx - wx)**2) - np.exp(-b * (cf_cx - wx)**2)) / (2 * b)
                I1y[point, i] = (np.exp(-b * (ci_cy - wy)**2) - np.exp(-b * (cf_cy - wy)**2)) / (2 * b)
                I1z[point, i] = (np.exp(-b * (ci_cz - wz)**2) - np.exp(-b * (cf_cz - wz)**2)) / (2 * b)

                I2x[point, i] = -np.sqrt(np.pi) / (2 * np.sqrt(b)) * \
                    ((np.exp(-b * (cf_cx - wx)**2) * (cf_cx - wx))/np.sqrt(np.pi * b) - (np.exp(-b * (ci_cx - wx)**2) * (ci_cx - wx))/np.sqrt(np.pi * b)) + \
                        np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (special.erf(np.sqrt(b) * (cf_cx - wx)) - special.erf(np.sqrt(b) * (ci_cx - wx)))
                I2y[point, i] = -np.sqrt(np.pi) / (2 * np.sqrt(b)) * \
                    ((np.exp(-b * (cf_cy - wy)**2) * (cf_cy - wy))/np.sqrt(np.pi * b) - (np.exp(-b * (ci_cy - wy)**2) * (ci_cy - wy))/np.sqrt(np.pi * b)) + \
                        np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (special.erf(np.sqrt(b) * (cf_cy - wy)) - special.erf(np.sqrt(b) * (ci_cy - wy)))
                I2z[point, i] = -np.sqrt(np.pi) / (2 * np.sqrt(b)) * \
                    ((np.exp(-b * (cf_cz - wz)**2) * (cf_cz - wz))/np.sqrt(np.pi * b) - (np.exp(-b * (ci_cz - wz)**2) * (ci_cz - wz))/np.sqrt(np.pi * b)) + \
                        np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (special.erf(np.sqrt(b) * (cf_cz - wz)) - special.erf(np.sqrt(b) * (ci_cz - wz)))

                I3x[point, i] = 1/b * I1x[point, i] + 1 / (2 * b) * ((ci_cx - wx)**2 * np.exp(-b * (ci_cx - wx)**2) - (cf_cx - wx)**2 * np.exp(-b * (cf_cx - wx)**2))
                I3y[point, i] = 1/b * I1y[point, i] + 1 / (2 * b) * ((ci_cy - wy)**2 * np.exp(-b * (ci_cy - wy)**2) - (cf_cy - wy)**2 * np.exp(-b * (cf_cy - wy)**2))
                I3z[point, i] = 1/b * I1z[point, i] + 1 / (2 * b) * ((ci_cz - wz)**2 * np.exp(-b * (ci_cz - wz)**2) - (cf_cz - wz)**2 * np.exp(-b * (cf_cz - wz)**2))

                print(I1x * I1x)
                # I0x[point, i] = np.sqrt(np.pi/(4 * b)) * (math.erf(np.sqrt(b) * (cf - w)) - math.erf(np.sqrt(b) * (ci - w)))
                # I1x[point, i] = (np.exp(-b * (ci - w)**2) - np.exp(-b * (cf - w)**2))/(2 * b)
                # I2x[point, i] = -np.sqrt(np.pi)/(2 * np.sqrt(b)) * \
                #     ((np.exp(-b * (cf - w)**2) * (cf - w))/np.sqrt(np.pi * b) - (np.exp(-b * (ci - w)**2) * (ci - w))/np.sqrt(np.pi * b)) + \
                #         np.sqrt(np.pi)/(4 * np.sqrt(b**3)) * (math.erf(np.sqrt(b) * (cf - w)) - math.erf(np.sqrt(b) * (ci - w)))
                # I3x[point, i] = -((np.exp(-b * (cf - w)**2) * (cf - w)**2 - np.exp(-b * (ci  - w)**2) * (ci - w)**2)/(2*b)) - \
                #     ((np.exp(-b * (cf - w)**2) - np.exp(-b * (ci - w)**2))/(2 * b**2))
            else:
                I0x[point, i] = 0.0
                I0y[point, i] = 0.0
                I0z[point, i] = 0.0
                I1x[point, i] = 0.0
                I1y[point, i] = 0.0
                I1z[point, i] = 0.0
                I2x[point, i] = 0.0
                I2y[point, i] = 0.0
                I2z[point, i] = 0.0
                I3x[point, i] = 0.0
                I3y[point, i] = 0.0
                I3z[point, i] = 0.0
        
    return [I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z]

def calc_flux(Ak, bk, wxk, wyk, wzk, I0x, I0y, I0z, I1x, I1y, I1z, I2x, I2y, I2z, I3x, I3y, I3z, numXj, numGroups):
    F = np.zeros((numXj, numGroups, 5))

    for point in range(0, numXj):
        for i in range(0, numGroups):
            if bk[point, i] != 0.0:
                A = Ak[point, i]
                b = bk[point, i]
                wx = wxk[point, i]
                wy = wyk[point, i]
                wz = wzk[point, i]

                F1 = A * (I1x[point, i] + wx * I0x[point, i]) * I0y[point, i] * I0z[point, i]
                F2x = A * (I2x[point, i] + wx**2 * I0x[point, i] + 2 * wx * I1x[point, i]) * I0y[point, i] * I0z[point, i]
                # F2y = A * (I1x[point, i] + wx * I0x[point, is]) * (I1y[point, i] + wy * I0y[point, i]) * I0z[point, i]
                F3 = A * ((I3x[point, i] + I0x[point, i] * wx**3 + 3 * I1x[point, i] * wx**2 + 3 * I2x[point, i] * wx) * I0y[point, i] * I0z[point, i] \
                     + (I2y[point, i] + 2 * wy * I1y[point, i] + wy**2 * I0y[point, i]) * (I1x[point, i] + wx * I0x[point, i]) * I0z[point, i] + \
                        (I2z[point, i] + 2 * wz * I1z[point, i] + wz**2 * I0z[point, i]) * (I1x[point, i] + wx * I0x[point, i]) * I0y[point, i])

                F[point, i] = np.array([F1, F2x, 0.0, 0.0, F3])
            else:
                F[point, i] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    return F

@jit(nopython=True)
def calc_flux_int(num_groups, weights, masks, bounds_list, cx_loc, cy_loc, cz_loc):
    F = np.zeros((num_groups, 5))

    for i in range(0, num_groups):
        mask = masks[i]
        ci_cx = bounds_list[i, 0]
        cf_cx = bounds_list[i, 1]
        ci_cy = bounds_list[i, 2]
        cf_cy = bounds_list[i, 3]
        ci_cz = bounds_list[i, 4]
        cf_cz = bounds_list[i, 5]

        cx_vec = cx_loc[(cx_loc > ci_cx) & (cx_loc < cf_cx)]
        cy_vec = cy_loc[(cy_loc > ci_cy) & (cy_loc < cf_cy)]
        cz_vec = cz_loc[(cz_loc > ci_cz) & (cz_loc < cf_cz)]
        dx = np.abs(cx_vec[1] - cx_vec[0])
        dy = np.abs(cy_vec[1] - cy_vec[0])
        dz = np.abs(cz_vec[1] - cz_vec[0])

        shape_weights = np.reshape(weights[mask] / (dx*dy*dz), (cx_vec.size, cy_vec.size, cz_vec.size))

        # Extend ci_vec and shape_weights to include the extrapolated points.
        cx_vec_ext = np.empty(len(cx_vec) + 2)
        cy_vec_ext = np.empty(len(cy_vec) + 2)
        cz_vec_ext = np.empty(len(cz_vec) + 2)

        cx_vec_ext[0] = cx_vec[0] - dx / 2
        cx_vec_ext[1:-1] = cx_vec
        cx_vec_ext[-1] = cx_vec[-1] + dx / 2
        cy_vec_ext[0] = cy_vec[0] - dy / 2
        cy_vec_ext[1:-1] = cy_vec
        cy_vec_ext[-1] = cy_vec[-1] + dy / 2
        cz_vec_ext[0] = cz_vec[0] - dz / 2
        cz_vec_ext[1:-1] = cz_vec
        cz_vec_ext[-1] = cz_vec[-1] + dz / 2

        nx_ext, ny_ext, nz_ext = len(cx_vec_ext), len(cy_vec_ext), len(cz_vec_ext)
        weights_ext = np.zeros((nx_ext, ny_ext, nz_ext))

        weights_ext[1:-1, 1:-1, 1:-1] = shape_weights

        weights_ext[0, 1:-1, 1:-1] = 2 * shape_weights[0, :, :] - shape_weights[1, :, :]      # x_min face
        weights_ext[-1, 1:-1, 1:-1] = 2 * shape_weights[-1, :, :] - shape_weights[-2, :, :]   # x_max face
        
        weights_ext[:, 0, 1:-1] = 2 * weights_ext[:, 1, 1:-1] - weights_ext[:, 2, 1:-1]    # y_min face
        weights_ext[:, -1, 1:-1] = 2 * weights_ext[:, -2, 1:-1] - weights_ext[:, -3, 1:-1]  # y_max face
        
        weights_ext[:, :, 0] = 2 * weights_ext[:, :, 1] - weights_ext[:, :, 2]      # z_min face
        weights_ext[:, :, -1] = 2 * weights_ext[:, :, -2] - weights_ext[:, :, -3]   # z_max face

        # Create meshgrid. (Friendly with Numba).
        cx = np.empty((nx_ext, ny_ext, nz_ext))
        cy = np.empty((nx_ext, ny_ext, nz_ext))
        cz = np.empty((nx_ext, ny_ext, nz_ext))
        for x in range(nx_ext):
            cx[x, :, :] = cx_vec_ext[x]

        for y in range(ny_ext):
            cy[:, y, :] = cy_vec_ext[y]
        
        for z in range(nz_ext):
            cz[:, :, z] = cz_vec_ext[z]

        # Integrate to get flux. 
        F[i, 0] = np.trapezoid(np.trapezoid(np.trapezoid(cx * weights_ext, cz_vec_ext), cy_vec_ext), cx_vec_ext)
        F[i, 1] = np.trapezoid(np.trapezoid(np.trapezoid(cx**2 * weights_ext, cz_vec_ext), cy_vec_ext), cx_vec_ext)
        ccTc = cx**3 + cy**2 * cx + cz**2 * cx
        F[i, 4] = np.trapezoid(np.trapezoid(np.trapezoid(ccTc * weights_ext, cz_vec_ext), cy_vec_ext), cx_vec_ext)

    return F

# Flux limiter function.
def minmod3(a, b, c):  
    all_positive = (a > 0) & (b > 0) & (c > 0)
    all_negative = (a < 0) & (b < 0) & (c < 0)
    
    result = np.where(all_positive, np.minimum(np.minimum(a, b), c),
            np.where(all_negative, np.maximum(np.maximum(a, b), c), 0))

    return result

def minmod2(a, b):
    all_positive = (a > 0) & (b > 0)
    all_negative = (a < 0) & (b < 0)
    
    result = np.where(all_positive, np.minimum(a, b),
            np.where(all_negative, np.maximum(a, b), 0))

    return result

def LF_central1(U_list, F_list, numXj, n_groups, lam):
    k = np.zeros((numXj, n_groups, 5))

    p = np.arange(1, numXj - 1, 1)
    for i in range(0, n_groups):
        # Lax-Freidrichs flux.
        k[1:numXj - 1, i] = 0.5 * (U_list[p + 1, i] + U_list[p - 1, i]) - 0.5 * lam * (F_list[p + 1, i] - F_list[p - 1, i]) - U_list[p, i]

    return k

def LF_central1_conservative(U_list, F_list, numXj, n_groups, lam):
    k = np.zeros((numXj, n_groups, 5))

    p = np.arange(1, numXj - 1, 1)
    for i in range(0, n_groups):
        f_left = 0.5 * (F_list[p - 1, i] + F_list[p, i]) + 1 / (2 * lam) * (U_list[p - 1, i] - U_list[p, i])
        f_right = 0.5 * (F_list[p, i] + F_list[p + 1, i]) + 1 / (2 * lam) * (U_list[p, i] - U_list[p + 1, i])
        k[1:numXj - 1, i] = lam * (f_left - f_right)

    return k

def KT_central2(U_list, F_list, numXj, n_groups, dt, dx, CX_LB, CX_UB):
    # Returns dU/dt and not U^{n+1}. Requires a little different treatment than LF_central1 method.
    k = np.zeros((numXj, n_groups, 5))

    p = np.arange(2, numXj - 2, 1)
    theta = 2

    a_plus = CX_UB
    a_minus = CX_UB

    for i in range(0, n_groups):
        uR_plus = U_list[p + 1, i] - dx/2 * minmod3(theta * (U_list[p + 1, i] - U_list[p, i])/dx, (U_list[p + 2, i] - U_list[p, i])/(2*dx), theta * (U_list[p + 2, i] - U_list[p + 1, i])/dx)
        uL_plus = U_list[p, i] + dx/2 * minmod3(theta * (U_list[p, i] - U_list[p - 1, i])/dx, (U_list[p + 1, i] - U_list[p - 1, i])/(2*dx), theta * (U_list[p + 1, i] - U_list[p, i])/dx)
        uR_minus = U_list[p, i] - dx/2 * minmod3(theta * (U_list[p, i] - U_list[p - 1, i])/dx, (U_list[p + 1, i] - U_list[p - 1, i])/(2*dx), theta * (U_list[p + 1, i] - U_list[p, i])/dx)
        uL_minus = U_list[p - 1, i] + dx/2 * minmod3(theta * (U_list[p - 1, i] - U_list[p - 2, i])/dx, (U_list[p, i] - U_list[p - 2, i])/(2*dx), theta * (U_list[p, i] - U_list[p - 1, i])/dx)

        # Need to evaluate the flux at the recontructed values of U...
        fR_plus = F_list[p + 1, i] - dx/2 * minmod2((F_list[p + 1, i] - F_list[p, i])/dx, (F_list[p + 2, i] - F_list[p + 1, i])/dx)
        fL_plus = F_list[p, i] + dx/2 * minmod2((F_list[p, i] - F_list[p - 1, i])/dx, (F_list[p + 1, i] - F_list[p, i])/dx)
        fR_minus = F_list[p, i] - dx/2 * minmod2((F_list[p, i] - F_list[p - 1, i])/dx, (F_list[p + 1, i] - F_list[p, i])/dx)
        fL_minus = F_list[p - 1, i] + dx/2 * minmod2((F_list[p - 1, i] - F_list[p - 2, i])/dx, (F_list[p, i] - F_list[p - 1, i])/dx)

        H_plus = (fR_plus + fL_plus)/2 - (a_plus/2) * (uR_plus - uL_plus)
        H_minus = (fR_minus + fL_minus)/2 - (a_minus/2) * (uR_minus - uL_minus)

        k[2:numXj - 2, i] = -1/dx * (H_plus - H_minus)

        k[1] = 0.5 * (U_list[2, i] + U_list[0, i]) - 0.5 * dt/dx * (F_list[2, i] - F_list[0, i]) - U_list[1, i]
        k[-2] = 0.5 * (U_list[-1, i] + U_list[-3, i]) - 0.5 * dt/dx * (F_list[-1, i] - F_list[-3, i]) - U_list[-2, i]

    return k

def generate_regular_samples(p, n_samples, x_sample, y_sample, z_sample, mu, bounds_list, num_groups):
    weights = np.zeros(n_samples)
    n_sample_group = np.zeros(num_groups)
    masks = []

    for i in range(num_groups):
        ci_cx = bounds_list[i, 0]
        cf_cx = bounds_list[i, 1]
        ci_cy = bounds_list[i, 2]
        cf_cy = bounds_list[i, 3]
        ci_cz = bounds_list[i, 4]
        cf_cz = bounds_list[i, 5]

        mask = (x_sample >= ci_cx) & (x_sample <= cf_cx) & \
        (y_sample >= ci_cy) & (y_sample <= cf_cy) & \
        (z_sample >= ci_cz) & (z_sample <= cf_cz)

        x_sample_slice = x_sample[mask]
        y_sample_slice = y_sample[mask]
        z_sample_slice = z_sample[mask]
        masks.append(mask)

        num_sample_group = len(x_sample_slice)

        if mu[i, 0] < 1e-4: # wonder what a good threshold is.
            try:
                A = np.zeros((5, num_sample_group))
                b = np.zeros(5)
                A[0, :] = 1
                A[1, :] = x_sample_slice
                A[2, :] = y_sample_slice
                A[3, :] = z_sample_slice
                A[4, :] = x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2
                b[0] = mu[i, 0]
                b[1] = mu[i, 1]
                b[2] = mu[i, 2]
                b[3] = mu[i, 3]
                b[4] = mu[i, 4]

                x = cp.Variable(shape=num_sample_group, nonneg=True)
                cost = cp.sum_squares(A @ x - b)
                prob = cp.Problem(cp.Minimize(cost))
                prob.solve()

                weights = x.value
            except:
                print('i knew it')
        else:
            try:
                x = cp.Variable(shape=num_sample_group, nonneg=True)
                obj = cp.Maximize(cp.sum(cp.entr(x)))

                constraints = [cp.sum(x) == mu[i, 0], cp.sum(cp.multiply(x_sample_slice, x)) == mu[i, 1], \
                            cp.sum(cp.multiply(y_sample_slice, x)) == mu[i, 2], cp.sum(cp.multiply(z_sample_slice, x)) == mu[i, 3], \
                            cp.sum(cp.multiply(x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2, x)) == mu[i, 4]]
                prob = cp.Problem(obj, constraints)
                prob.solve()
                # if prob.status == cp.OPTIMAL_INACCURATE:
                    # print(prob.value, p, i)
                    # print(ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, mu[i, 0], mu[i, 1], mu[i, 2], mu[i, 3], mu[i, 4])

                n_sample_group[i] = num_sample_group
                weights[mask] = x.value
            except:
                print('Hi', p, i)

    return weights, n_sample_group, masks

@jit(nopython=True)
def collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_groups, Rf1, Rf2, depl_idx1, depl_idx2, n_coll, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type):
    group_n = np.zeros(n_groups)
    group_px = np.zeros(n_groups)
    group_py = np.zeros(n_groups)
    group_pz = np.zeros(n_groups)
    group_e = np.zeros(n_groups)

    mask = depl_idx1 != depl_idx2

    depl_idx1 = depl_idx1[mask]
    depl_idx2 = depl_idx2[mask]

    # Group the prospective collisions into which group they end up in.
    depl_tracker = Dict.empty(key_type=key_type, \
                              value_type=types.int64)
    
    ci_cx = bounds_list[:, 0]
    cf_cx = bounds_list[:, 1]
    ci_cy = bounds_list[:, 2]
    cf_cy = bounds_list[:, 3]
    ci_cz = bounds_list[:, 4]
    cf_cz = bounds_list[:, 5]
    
    for i in range(0, depl_idx1.size):
        x_valid = (x_sample[depl_idx1[i]] >= ci_cx) & (x_sample[depl_idx1[i]] <= cf_cx)
        y_valid = (y_sample[depl_idx1[i]] >= ci_cy) & (y_sample[depl_idx1[i]] <= cf_cy)
        z_valid = (z_sample[depl_idx1[i]] >= ci_cz) & (z_sample[depl_idx1[i]] <= cf_cz)
        depl_group1 = np.argmax(x_valid & y_valid & z_valid)

        x_valid = (x_sample[depl_idx2[i]] >= ci_cx) & (x_sample[depl_idx2[i]] <= cf_cx)
        y_valid = (y_sample[depl_idx2[i]] >= ci_cy) & (y_sample[depl_idx2[i]] <= cf_cy)
        z_valid = (z_sample[depl_idx2[i]] >= ci_cz) & (z_sample[depl_idx2[i]] <= cf_cz)
        depl_group2 = np.argmax(x_valid & y_valid & z_valid)

        if depl_group1 < depl_group2: key = (depl_group1, depl_group2)
        else: key = (depl_group2, depl_group1)

        if key in depl_tracker: depl_tracker[key] += 1
        else: depl_tracker[key] = 1

    for i in range(0, depl_idx1.size):
        d_idx1 = depl_idx1[i]
        d_idx2 = depl_idx2[i]

        vx1 = x_sample[d_idx1]
        vy1 = y_sample[d_idx1]
        vz1 = z_sample[d_idx1]
        vx2 = x_sample[d_idx2]
        vy2 = y_sample[d_idx2]
        vz2 = z_sample[d_idx2]

        # Simulate virtual collisions.
        gx = np.abs(vx2 - vx1)
        gy = np.abs(vy2 - vy1)
        gz = np.abs(vz2 - vz1)
        g = np.sqrt(gx**2 + gy**2 + gz**2)

        phi = 2 * np.pi * Rf1[i]
        cos_theta = 2 * Rf2[i] - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        gx_p = 0.5 * g * sin_theta * np.cos(phi)
        gy_p = 0.5 * g * sin_theta * np.sin(phi)
        gz_p = 0.5 * g * cos_theta

        V_x = 0.5 * (vx1 + vx2)
        V_y = 0.5 * (vy1 + vy2)
        V_z = 0.5 * (vz1 + vz2)

        vx1p = V_x - gx_p
        vy1p = V_y - gy_p
        vz1p = V_z - gz_p
        vx2p = V_x + gx_p
        vy2p = V_y + gy_p
        vz2p = V_z + gz_p

        # Calculate loss rate for mass, momentum, and energy.
        x_valid = (vx1 >= ci_cx) & (vx1 <= cf_cx)
        y_valid = (vy1 >= ci_cy) & (vy1 <= cf_cy)
        z_valid = (vz1 >= ci_cz) & (vz1 <= cf_cz)
        group_idx1 = np.argmax(x_valid & y_valid & z_valid)

        x_valid = (vx2 >= ci_cx) & (vx2 <= cf_cx)
        y_valid = (vy2 >= ci_cy) & (vy2 <= cf_cy)
        z_valid = (vz2 >= ci_cz) & (vz2 <= cf_cz)
        group_idx2 = np.argmax(x_valid & y_valid & z_valid)

        # Calculate gain rate for mass, momentum, and energy.
        vx1p_clamped = np.minimum(np.maximum(vx1p, CX_LB), CX_UB)
        vy1p_clamped = np.minimum(np.maximum(vy1p, CY_LB), CY_UB)
        vz1p_clamped = np.minimum(np.maximum(vz1p, CZ_LB), CZ_UB)
        x_valid = (vx1p_clamped >= ci_cx) & (vx1p_clamped <= cf_cx)
        y_valid = (vy1p_clamped >= ci_cy) & (vy1p_clamped <= cf_cy)
        z_valid = (vz1p_clamped >= ci_cz) & (vz1p_clamped <= cf_cz)
        if np.count_nonzero(x_valid & y_valid & z_valid) != 1: print("uho h")
        group_idx1r = np.argmax(x_valid & y_valid & z_valid)

        vx2p_clamped = np.minimum(np.maximum(vx2p, CX_LB), CX_UB)
        vy2p_clamped = np.minimum(np.maximum(vy2p, CY_LB), CY_UB)
        vz2p_clamped = np.minimum(np.maximum(vz2p, CZ_LB), CZ_UB)
        x_valid = (vx2p_clamped >= ci_cx) & (vx2p_clamped <= cf_cx)
        y_valid = (vy2p_clamped >= ci_cy) & (vy2p_clamped <= cf_cy)
        z_valid = (vz2p_clamped >= ci_cz) & (vz2p_clamped <= cf_cz)
        if np.count_nonzero(x_valid & y_valid & z_valid) != 1: print("uho h")
        group_idx2r = np.argmax(x_valid & y_valid & z_valid)

        if group_idx1 < group_idx2: key = (group_idx1, group_idx2)
        else: key = (group_idx2, group_idx1)
        n_coll_group = depl_tracker[key]

        Li = weights[d_idx1] * weights[d_idx2] * num_group_sample[group_idx1] * num_group_sample[group_idx2] / n_coll_group
        group_n[group_idx1] -= Li
        group_px[group_idx1] -= Li * vx1
        group_py[group_idx1] -= Li * vy1
        group_pz[group_idx1] -= Li * vz1
        group_e[group_idx1] -= Li * (vx1**2 + vy1**2 + vz1**2)
        
        group_n[group_idx2] -= Li
        group_px[group_idx2] -= Li * vx2
        group_py[group_idx2] -= Li * vy2
        group_pz[group_idx2] -= Li * vz2
        group_e[group_idx2] -= Li * (vx2**2 + vy2**2 + vz2**2)

        Gi = Li
        group_n[group_idx1r] += Gi
        group_px[group_idx1r] += Gi * vx1p
        group_py[group_idx1r] += Gi * vy1p
        group_pz[group_idx1r] += Gi * vz1p
        group_e[group_idx1r] += Gi * (vx1p**2 + vy1p**2 + vz1p**2)

        group_n[group_idx2r] += Gi
        group_px[group_idx2r] += Gi * vx2p
        group_py[group_idx2r] += Gi * vy2p
        group_pz[group_idx2r] += Gi * vz2p
        group_e[group_idx2r] += Gi * (vx2p**2 + vy2p**2 + vz2p**2)

    return [group_n, group_px, group_py, group_pz, group_e]