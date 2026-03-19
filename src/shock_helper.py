from numba import jit, types
from numba.typed import Dict
import numpy as np
from scipy import optimize, special, interpolate
from matplotlib import pyplot as plt
import math
import cvxpy as cp
import sys
from scipy.special import erf
from scipy.stats import norm, qmc


def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 

def generate_grid(bounds_list, num_groups):
    num_samples = np.zeros(num_groups)
    for i in range(0, num_groups):
        volume = (bounds_list[i, 1] - bounds_list[i, 0]) * \
            (bounds_list[i, 3] - bounds_list[i, 2]) * \
            (bounds_list[i, 5] - bounds_list[i, 4])
        num_samples[i] = np.max((300, int(np.ceil(10 * volume))))
    
    x_sample = np.zeros(int(np.sum(num_samples)))
    y_sample = np.zeros(int(np.sum(num_samples)))
    z_sample = np.zeros(int(np.sum(num_samples)))
    offsets = np.concatenate([[0], np.cumsum(num_samples)])

    for i in range(0, num_groups):
        l_bounds = np.array([bounds_list[i, 0], bounds_list[i, 2], bounds_list[i, 4]])
        u_bounds = np.array([bounds_list[i, 1], bounds_list[i, 3], bounds_list[i, 5]])

        if np.any(l_bounds > u_bounds): continue

        start_idx = int(offsets[i])
        end_idx = int(offsets[i+1])

        sampler = qmc.LatinHypercube(d=3)
        sample = qmc.scale(sampler.random(n=int(num_samples[i])), l_bounds, u_bounds)
    
        x_sample[start_idx:end_idx] = sample[:, 0]
        y_sample[start_idx:end_idx] = sample[:, 1]
        z_sample[start_idx:end_idx] = sample[:, 2]

    return x_sample, y_sample, z_sample, offsets, num_samples

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
def calc_flux_int(num_groups, weights, offsets, x_sample, y_sample, z_sample):
    # Calculate the flux from samples generated. Requires samples to be placed at locations covering the whole domain for best results.
    F = np.zeros((num_groups, 5))

    for i in range(0, num_groups): 
        start_idx = int(offsets[i])
        end_idx = int(offsets[i+1])

        x_group = x_sample[start_idx:end_idx]
        y_group = y_sample[start_idx:end_idx]
        z_group = z_sample[start_idx:end_idx]

        F[i, 0] = np.sum(x_group * weights[start_idx:end_idx])
        F[i, 1] = np.sum(x_group**2 * weights[start_idx:end_idx])
        F[i, 2] = np.sum(x_group * y_group * weights[start_idx:end_idx])
        F[i, 3] = np.sum(x_group * z_group * weights[start_idx:end_idx])
        ccTc = x_group**3 + y_group**2 * x_group + z_group**2 * x_group
        F[i, 4] = np.sum(ccTc * weights[start_idx:end_idx])
        
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
    theta = 1.0

    a_plus = np.abs(CX_UB)
    a_minus = np.abs(CX_UB)

    for i in range(0, n_groups):
        uR_plus = U_list[p + 1, i] - dx/2 * minmod3(theta * (U_list[p + 1, i] - U_list[p, i])/dx, (U_list[p + 2, i] - U_list[p, i])/(2*dx), theta * (U_list[p + 2, i] - U_list[p + 1, i])/dx)
        uL_plus = U_list[p, i] + dx/2 * minmod3(theta * (U_list[p, i] - U_list[p - 1, i])/dx, (U_list[p + 1, i] - U_list[p - 1, i])/(2*dx), theta * (U_list[p + 1, i] - U_list[p, i])/dx)
        uR_minus = U_list[p, i] - dx/2 * minmod3(theta * (U_list[p, i] - U_list[p - 1, i])/dx, (U_list[p + 1, i] - U_list[p - 1, i])/(2*dx), theta * (U_list[p + 1, i] - U_list[p, i])/dx)
        uL_minus = U_list[p - 1, i] + dx/2 * minmod3(theta * (U_list[p - 1, i] - U_list[p - 2, i])/dx, (U_list[p, i] - U_list[p - 2, i])/(2*dx), theta * (U_list[p, i] - U_list[p - 1, i])/dx)

        # Need to evaluate the flux at the recontructed values of U...
        fR_plus = F_list[p + 1, i] - dx/2 * minmod3(theta * (F_list[p + 1, i] - F_list[p, i])/dx, (F_list[p + 2, i] - F_list[p, i])/(2*dx), theta * (F_list[p + 2, i] - F_list[p + 1, i])/dx)
        fL_plus = F_list[p, i] + dx/2 * minmod3(theta * (F_list[p, i] - F_list[p - 1, i])/dx, (F_list[p + 1, i] - F_list[p - 1, i])/(2*dx), theta * (F_list[p + 1, i] - F_list[p, i])/dx)
        fR_minus = F_list[p, i] - dx/2 * minmod3(theta * (F_list[p, i] - F_list[p - 1, i])/dx, (F_list[p + 1, i] - F_list[p - 1, i])/(2*dx), theta * (F_list[p + 1, i] - F_list[p, i])/dx)
        fL_minus = F_list[p - 1, i] + dx/2 * minmod3(theta * (F_list[p - 1, i] - F_list[p - 2, i])/dx, (F_list[p, i] - F_list[p - 2, i])/(2*dx), theta * (F_list[p, i] - F_list[p - 1, i])/dx)

        H_plus = (fR_plus + fL_plus)/2 - (a_plus/2) * (uR_plus - uL_plus)
        H_minus = (fR_minus + fL_minus)/2 - (a_minus/2) * (uR_minus - uL_minus)

        k[2:numXj-2, i] = -1/dx * (H_plus - H_minus)

        k[1, i] = -(F_list[2, i] - F_list[0, i])/(2 * dx) + 1/(2 * dx) * (a_plus * (U_list[2, i] - U_list[1, i]) - a_minus * (U_list[1, i] - U_list[0, i]))
        k[-2, i] = -(F_list[-1, i] - F_list[-3, i])/(2 * dx) + 1/(2 * dx) * (a_plus * (U_list[-1, i] - U_list[-2, i]) - a_minus * (U_list[-2, i] - U_list[-3, i]))

    return k

@jit(nopython=True)
def max_entropy_newton(x_s, y_s, z_s, moments, lam0=None, max_iter=50, tol=1e-8):
    """
    Solve max-entropy weights directly via Newton iterations on dual.
    
    Dual problem: find lambda s.t. sum_i phi_i * exp(lam . phi_i) = moments
    """
    n = x_s.shape[0]
    r2 = x_s**2 + y_s**2 + z_s**2

    # Initial lambda guess
    if lam0 is None:
        lam = np.zeros(5)
    else:
        lam = lam0.copy()

    for iteration in range(max_iter):
        # Compute weights from current lambda
        log_w = (lam[0] + lam[1]*x_s + lam[2]*y_s + lam[3]*z_s + lam[4]*r2)
        w = np.exp(log_w)

        # Compute moment residual: g = sum(phi * w) - moments
        g = np.zeros(5)
        g[0] = np.sum(w)           - moments[0]
        g[1] = np.sum(x_s * w)    - moments[1]
        g[2] = np.sum(y_s * w)    - moments[2]
        g[3] = np.sum(z_s * w)    - moments[3]
        g[4] = np.sum(r2  * w)    - moments[4]

        if np.linalg.norm(g) < tol:
            break

        # Hessian: H_ij = sum(phi_i * phi_j * w)
        phi = np.zeros((5, n))
        phi[0] = np.ones(n)
        phi[1] = x_s
        phi[2] = y_s
        phi[3] = z_s
        phi[4] = r2

        H = np.zeros((5, 5))
        for a in range(5):
            for b in range(5):
                H[a, b] = np.sum(phi[a] * phi[b] * w)

        # Newton step: lam -= H^{-1} g
        dlam = np.linalg.solve(H, g)
        lam -= dlam

    return w, lam

def solve_group_newton(x_sample, y_sample, z_sample, U_i, lam0):
    """Attempt to solve for one group"""
    try:
        w, lam = max_entropy_newton(x_sample, y_sample, z_sample, U_i, lam0)
        residual = np.abs(np.sum(w) - U_i[0]) / U_i[0]
        if residual < 1e-7:
            return True, w, lam
        else:
            return False, w, lam
    except:
        print('Newton failed')
        return False, np.zeros_like(x_sample), np.zeros(5)

def solve_group_cvxpy(x_sample, y_sample, z_sample, U_i, flux_limit=10.0):
    # Fallback to CVXPY for ill-conditioned cases
    try:
        x = cp.Variable(shape=int(x_sample.size), nonneg=True)
        obj = cp.Maximize(cp.sum(cp.entr(x)))

        constraints = [
            cp.sum(x) == U_i[0],
            cp.sum(cp.multiply(x_sample, x)) == U_i[1],
            cp.sum(cp.multiply(y_sample, x)) == U_i[2],
            cp.sum(cp.multiply(z_sample, x)) == U_i[3],
            cp.sum(cp.multiply(x_sample**2 + y_sample**2 + z_sample**2, x)) == U_i[4]
        ]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if x.value is not None and not np.any(np.isnan(x.value)):
            predicted_flux = np.sum(x_sample * x.value)
            
            if np.abs(predicted_flux) < flux_limit:
                return True, x.value, predicted_flux
            else:
                return False, None, f"flux_too_large_{predicted_flux:.3f}"
        else:
            return False, None, f"status_{prob.status}"
    except Exception as e:
        return False, None, str(e)

def regenerate_group_samples(i, n_samples, bounds_list):
    """Generate new samples for a group"""
    l_bounds = [bounds_list[i, 0], bounds_list[i, 2], bounds_list[i, 4]]
    u_bounds = [bounds_list[i, 1], bounds_list[i, 3], bounds_list[i, 5]]
    
    sampler = qmc.LatinHypercube(d=3)
    if all(u > l for l, u in zip(l_bounds, u_bounds)):
        sample = qmc.scale(sampler.random(n=n_samples), l_bounds, u_bounds)
        return sample[:, 0], sample[:, 1], sample[:, 2]
    else:
        return None, None, None

def generate_regular_samples(p, U_i, num_groups, bounds_list, lam_cache, max_retries=6):
    num_valid_samples = np.zeros(num_groups, dtype=np.int64)

    ux = U_i[:, 1] / U_i[:, 0]
    uy = U_i[:, 2] / U_i[:, 0]
    uz = U_i[:, 3] / U_i[:, 0]
    
    thermal = (U_i[:, 4] / U_i[:, 0]) - (ux**2 + uy**2 + uz**2)
    sigma = np.sqrt(np.maximum(2 * thermal / 3, 1e-10))
    
    x_boundsl = np.maximum(bounds_list[:, 0], ux - 3*sigma)
    x_boundsu = np.minimum(bounds_list[:, 1], ux + 3*sigma)
    
    y_boundsl = np.maximum(bounds_list[:, 2], uy - 3*sigma)
    y_boundsu = np.minimum(bounds_list[:, 3], uy + 3*sigma)
    
    z_boundsl = np.maximum(bounds_list[:, 4], uz - 3*sigma)
    z_boundsu = np.minimum(bounds_list[:, 5], uz + 3*sigma)

    adaptive_bounds = np.vstack((x_boundsl, x_boundsu, y_boundsl, y_boundsu, z_boundsl, z_boundsu)).T
    x_sample, y_sample, z_sample, offsets, num_samples = generate_grid(adaptive_bounds, num_groups)

    weights = np.zeros(int(np.sum(num_samples)))
    x_sample_working = x_sample.copy()
    y_sample_working = y_sample.copy()
    z_sample_working = z_sample.copy()

    # Initialize output lambda array for all groups
    lam_out = np.zeros((num_groups, 5))

    for i in range(num_groups):
        start_idx = int(offsets[i])
        end_idx = int(offsets[i+1])
        n_samples_group = end_idx - start_idx

        # Skip if density too small
        if U_i[i, 0] <= 1e-6:
            num_valid_samples[i] = 0
            weights[start_idx:end_idx] = 0.0
            continue

        # Try multiple times
        success = False
        for attempt in range(max_retries):
            # For retry attempts, regenerate samples
            if attempt > 0:
                x_new, y_new, z_new = regenerate_group_samples(i, n_samples_group, adaptive_bounds)

                # Replace samples in modified arrays
                x_sample_working[start_idx:end_idx] = x_new
                y_sample_working[start_idx:end_idx] = y_new
                z_sample_working[start_idx:end_idx] = z_new
            
            # Get samples (from original on first attempt, modified on retries)
            x_slice = x_sample_working[start_idx:end_idx]
            y_slice = y_sample_working[start_idx:end_idx]
            z_slice = z_sample_working[start_idx:end_idx]

            # Try to solve using Newton first (much faster).
            lam0 = lam_cache[i] if (lam_cache is not None and
                                     np.any(lam_cache[i] != 0.0)) else None
            success, solution, lam = solve_group_newton(x_slice, y_slice, z_slice, U_i[i], lam0)

            if success:
                # Accept solution
                weights[start_idx:end_idx] = solution
                num_valid_samples[i] = np.sum(solution > 1e-12)
                lam_out[i] = lam
                break  # Exit retry loop
            elif not success and attempt > 3:
                success, solution, status = solve_group_cvxpy(x_slice, y_slice, z_slice, U_i[i])
                if success:
                    weights[start_idx:end_idx] = solution
                    num_valid_samples[i] = np.sum(solution > 1e-12)
                    lam_out[i] = np.zeros(5)
                    break

        if not success:
            # if attempt == max_retries - 1:
            print(f'Cell {p}, Group {i}: Failed after {max_retries} attempts ')
            print(f'  moments:  {U_i[i]}')
            print(f'  num_samples:   {n_samples_group}')
            print(f'  sigma:    {sigma[i]:.4f}, thermal: {2/3*thermal[i]:.4f}')
            print(f'  adaptive: x=[{adaptive_bounds[i,0]:.4f}, {adaptive_bounds[i,1]:.4f}] '
                  f'y=[{adaptive_bounds[i,2]:.4f}, {adaptive_bounds[i,3]:.4f}] '
                  f'z=[{adaptive_bounds[i,4]:.4f}, {adaptive_bounds[i,5]:.4f}]')
            num_valid_samples[i] = 0
            weights[start_idx:end_idx] = 0.0
            lam_out = np.zeros(5)

    return (weights, num_valid_samples, offsets,
            x_sample_working, y_sample_working, z_sample_working, lam_out)

@jit(nopython=True)
def collide(x_sample, y_sample, z_sample, weights, num_valid_samples, bounds_list, n_groups, n_coll,
            CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, sigma_coeff_hat, omega):

    group_n  = np.zeros(n_groups)
    group_px = np.zeros(n_groups)
    group_py = np.zeros(n_groups)
    group_pz = np.zeros(n_groups)
    group_e  = np.zeros(n_groups)

    ci_cx = bounds_list[:, 0]
    cf_cx = bounds_list[:, 1]
    ci_cy = bounds_list[:, 2]
    cf_cy = bounds_list[:, 3]
    ci_cz = bounds_list[:, 4]
    cf_cz = bounds_list[:, 5]

    def find_group(vx, vy, vz):
        x_valid = (vx >= ci_cx) & (vx <= cf_cx)
        y_valid = (vy >= ci_cy) & (vy <= cf_cy)
        z_valid = (vz >= ci_cz) & (vz <= cf_cz)
        return np.argmax(x_valid & y_valid & z_valid)

    def clamp_and_find_group(vx, vy, vz):
        vx_c = np.minimum(np.maximum(vx, CX_LB), CX_UB)
        vy_c = np.minimum(np.maximum(vy, CY_LB), CY_UB)
        vz_c = np.minimum(np.maximum(vz, CZ_LB), CZ_UB)
        return find_group(vx_c, vy_c, vz_c)

    # --- Generate collision pairs ---
    nonzero_indices = np.where(weights > 1e-12)[0]
    w_nonzero   = weights[nonzero_indices]
    W           = w_nonzero.sum()
    w_cdf       = np.cumsum(w_nonzero) / W   # normalized CDF, goes from 0 to 1

    # Sample using searchsorted with cdf. Clip to prevent out of bounds errors.
    u1 = np.random.uniform(0.0, 1.0, n_coll)
    u2 = np.random.uniform(0.0, 1.0, n_coll)
    i1 = np.clip(np.searchsorted(w_cdf, u1), 0, len(nonzero_indices) - 1)
    i2 = np.clip(np.searchsorted(w_cdf, u2), 0, len(nonzero_indices) - 1)

    # For each uniform sample, find which bin it falls into
    depl_idx1 = nonzero_indices[i1]
    depl_idx2 = nonzero_indices[i2]

    Rf1      = np.random.uniform(0.0, 1.0, n_coll)
    Rf2      = np.random.uniform(0.0, 1.0, n_coll)

    mask      = depl_idx1 != depl_idx2
    depl_idx1 = depl_idx1[mask]
    depl_idx2 = depl_idx2[mask]
    Rf1       = Rf1[mask]
    Rf2       = Rf2[mask]
    n_actual  = depl_idx1.size

    # --- Precompute all pre and post collision groups in one pass ---
    pre_group1  = np.zeros(n_actual, dtype=np.int64)
    pre_group2  = np.zeros(n_actual, dtype=np.int64)
    post_group1 = np.zeros(n_actual, dtype=np.int64)
    post_group2 = np.zeros(n_actual, dtype=np.int64)

    # Store post-collision velocities for second pass
    vx1_arr  = np.zeros(n_actual);  vy1_arr  = np.zeros(n_actual);  vz1_arr  = np.zeros(n_actual)
    vx2_arr  = np.zeros(n_actual);  vy2_arr  = np.zeros(n_actual);  vz2_arr  = np.zeros(n_actual)
    vx1p_arr = np.zeros(n_actual);  vy1p_arr = np.zeros(n_actual);  vz1p_arr = np.zeros(n_actual)
    vx2p_arr = np.zeros(n_actual);  vy2p_arr = np.zeros(n_actual);  vz2p_arr = np.zeros(n_actual)
    g_arr = np.zeros(n_actual)

    for i in range(n_actual):
        vx1 = x_sample[depl_idx1[i]]
        vy1 = y_sample[depl_idx1[i]]
        vz1 = z_sample[depl_idx1[i]]
        vx2 = x_sample[depl_idx2[i]]
        vy2 = y_sample[depl_idx2[i]]
        vz2 = z_sample[depl_idx2[i]]

        vx1_arr[i] = vx1;  vy1_arr[i] = vy1;  vz1_arr[i] = vz1
        vx2_arr[i] = vx2;  vy2_arr[i] = vy2;  vz2_arr[i] = vz2

        # Pre-collision groups
        pre_group1[i] = find_group(vx1, vy1, vz1)
        pre_group2[i] = find_group(vx2, vy2, vz2)

        # Post-collision velocities
        gx = vx2 - vx1
        gy = vy2 - vy1
        gz = vz2 - vz1
        g  = np.sqrt(gx**2 + gy**2 + gz**2)
        g_arr[i] = g

        phi       = 2 * np.pi * Rf1[i]
        cos_theta = 2 * Rf2[i] - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        V_x = 0.5 * (vx1 + vx2)
        V_y = 0.5 * (vy1 + vy2)
        V_z = 0.5 * (vz1 + vz2)

        vx1p = V_x - 0.5 * g * sin_theta * np.cos(phi)
        vy1p = V_y - 0.5 * g * sin_theta * np.sin(phi)
        vz1p = V_z - 0.5 * g * cos_theta
        vx2p = V_x + 0.5 * g * sin_theta * np.cos(phi)
        vy2p = V_y + 0.5 * g * sin_theta * np.sin(phi)
        vz2p = V_z + 0.5 * g * cos_theta

        vx1p_arr[i] = vx1p
        vy1p_arr[i] = vy1p
        vz1p_arr[i] = vz1p
        vx2p_arr[i] = vx2p
        vy2p_arr[i] = vy2p
        vz2p_arr[i] = vz2p

        # Post-collision groups
        post_group1[i] = clamp_and_find_group(vx1p, vy1p, vz1p)
        post_group2[i] = clamp_and_find_group(vx2p, vy2p, vz2p)

    # --- Apply collision rates ---
    for i in range(n_actual):
        g1,  g2  = pre_group1[i],  pre_group2[i]
        g1r, g2r = post_group1[i], post_group2[i]
        g = g_arr[i]

        # Single collision weight — used for BOTH loss and gain
        C = 0.5 * W**2 / n_actual * g**(2 - 2*omega) * sigma_coeff_hat

        # Loss from pre-collision groups
        group_n[g1]  -= C;       group_n[g2]  -= C
        group_px[g1] -= C * vx1_arr[i]
        group_py[g1] -= C * vy1_arr[i]
        group_pz[g1] -= C * vz1_arr[i]
        group_e[g1]  -= C * (vx1_arr[i]**2 + vy1_arr[i]**2 + vz1_arr[i]**2)
        group_px[g2] -= C * vx2_arr[i]
        group_py[g2] -= C * vy2_arr[i]
        group_pz[g2] -= C * vz2_arr[i]
        group_e[g2]  -= C * (vx2_arr[i]**2 + vy2_arr[i]**2 + vz2_arr[i]**2)

        # Gain into post-collision groups
        group_n[g1r]  += C;       group_n[g2r]  += C
        group_px[g1r] += C * vx1p_arr[i]
        group_py[g1r] += C * vy1p_arr[i]
        group_pz[g1r] += C * vz1p_arr[i]
        group_e[g1r]  += C * (vx1p_arr[i]**2 + vy1p_arr[i]**2 + vz1p_arr[i]**2)
        group_px[g2r] += C * vx2p_arr[i]
        group_py[g2r] += C * vy2p_arr[i]
        group_pz[g2r] += C * vz2p_arr[i]
        group_e[g2r]  += C * (vx2p_arr[i]**2 + vy2p_arr[i]**2 + vz2p_arr[i]**2)

    return [group_n, group_px, group_py, group_pz, group_e]