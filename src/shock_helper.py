from numba import jit
import numpy as np
from scipy import optimize, special, interpolate
from .virtual_collisions import collide
import math
import cvxpy as cp
import sys


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

def initialize_maxwellian(m_hat, T_hat, v_hat, cx, cy, cz):
    A = (m_hat / (np.pi * T_hat))**1.5
    beta = m_hat / T_hat
    dist = A * np.exp(-beta * ((cx - v_hat)**2 + cy**2 + cz**2))

    return dist

def calc_moment(f, n, cx, cy, cz, dcx, dcy, dcz):
    mu = np.zeros(5)

    mu[0] = np.trapz(np.trapz(np.trapz(n * f, dx=dcz), dx=dcy), dx=dcx)

    mu[1] = np.trapz(np.trapz(np.trapz(cx * n * f, dx=dcz), dx=dcy), dx=dcx)
    mu[2] = np.trapz(np.trapz(np.trapz(cy * n * f, dx=dcz), dx=dcy), dx=dcx)
    mu[3] = np.trapz(np.trapz(np.trapz(cz * n * f, dx=dcz), dx=dcy), dx=dcx)

    mu[4] = np.trapz(np.trapz(np.trapz((cx**2 + cy**2 + cz**2) * n * f, dx=dcz), dx=dcy), dx=dcx)

    return mu

def ic(cx, cy, cz, dcx, dcy, dcz, n_val, u_val, T_val, numCx, numCy, numCz, numXj, num_groups, group_bounds):
    f = np.zeros((numXj, numCx, numCy, numCz))
    U0 = np.zeros((numXj, num_groups, 5))

    for point in range(0, numXj):
        m_hat = 1.0
        f[point] = initialize_maxwellian(m_hat, T_val[point], u_val[point], cx, cy, cz)

        for i in range(0, num_groups):
            lb_cx = group_bounds[i, 0, 0]
            ub_cx = group_bounds[i, 0, 1]
            lb_cy = group_bounds[i, 1, 0]
            ub_cy = group_bounds[i, 1, 1]
            lb_cz = group_bounds[i, 2, 0]
            ub_cz = group_bounds[i, 2, 1]

            U0[point, i] = calc_moment(f[point, lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], n_val[point], \
                cx[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cy[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cz[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], dcx, dcy, dcz)

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
                    print('residual:', np.linalg.norm(sol.fun), point, i)
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

def calc_flux_int(weights, masks, dx, dy, dz, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    F = np.zeros(5)

    for i in range(0, len(masks)):
        shape_weights = np.reshape(weights[masks[i]] / (dx*dy*dz), (8, 8, 8))
        F[0] = np.trapezoid(np.trapezoid(np.trapezoid(cx * shape_weights, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
        F[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx**2 * shape_weights, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
        ccTc = cx**3 + cy**2 * cx + cz**2 * cx
        F[2] = np.trapezoid(np.trapezoid(np.trapezoid(ccTc * shape_weights, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return F

def RK_LF(U_list, F_list, numXj, n_groups, dx, delta_t):
    k = np.zeros((numXj, n_groups, 5))

    p = np.arange(1, numXj - 1, 1)

    for i in range(0, n_groups):
        # Lax-Freidrichs intercell flux.
        f_left = 0.5 * (F_list[p - 1, i] + F_list[p, i]) + dx / (2 * delta_t) * (U_list[p - 1, i] - U_list[p, i])
        f_right = 0.5 * (F_list[p, i] + F_list[p + 1, i]) + dx / (2 * delta_t) * (U_list[p, i] - U_list[p + 1, i])
        k[1:numXj - 1, i] = delta_t / dx * (f_left - f_right)

    return k

def generate_convex_helper(mu, x_sample, y_sample, z_sample, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    mask = (x_sample >= ci_cx) & (x_sample <= cf_cx) & \
    (y_sample >= ci_cy) & (y_sample <= cf_cy) & \
    (z_sample >= ci_cz) & (z_sample <= cf_cz)

    x_sample_slice = x_sample[mask]
    y_sample_slice = y_sample[mask]
    z_sample_slice = z_sample[mask]

    num_sample_group = len(x_sample_slice)

    if mu[0] < 1e-4: # wonder what a good threshold is.
        A = np.zeros((5, num_sample_group))
        b = np.zeros(5)
        A[0, :] = 1
        A[1, :] = x_sample_slice
        A[2, :] = y_sample_slice
        A[3, :] = z_sample_slice
        A[4, :] = x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2
        b[0] = mu[0]
        b[1] = mu[1]
        b[2] = mu[2]
        b[3] = mu[3]
        b[4] = mu[4]

        x = cp.Variable(shape=num_sample_group, nonneg=True)
        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()

        weights = x.value
    else:
        x = cp.Variable(shape=num_sample_group, nonneg=True)
        obj = cp.Maximize(cp.sum(cp.entr(x)))

        constraints = [cp.sum(x) == mu[0], cp.sum(cp.multiply(x_sample_slice, x)) == mu[1], \
                    cp.sum(cp.multiply(y_sample_slice, x)) == mu[2], cp.sum(cp.multiply(z_sample_slice, x)) == mu[3], \
                    cp.sum(cp.multiply(x_sample_slice**2 + y_sample_slice**2 + z_sample_slice**2, x)) == mu[4]]
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=True)
        # if prob.status == cp.OPTIMAL_INACCURATE:
            # prob.solve(verbose=True)  
            # sys.exit()
        
        weights = x.value

    return num_sample_group, weights, mask

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, mu, bounds_list, num_groups):
    weights = np.zeros(n_samples)
    num_sample_group = np.zeros(num_groups)

    for i in range(num_groups):
        ci_cx = bounds_list[i, 0]
        cf_cx = bounds_list[i, 1]
        ci_cy = bounds_list[i, 2]
        cf_cy = bounds_list[i, 3]
        ci_cz = bounds_list[i, 4]
        cf_cz = bounds_list[i, 5]

        n_group_sample, group_weights, mask = generate_convex_helper(mu[i], x_sample, y_sample, z_sample, \
                                                                                       ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz)

        num_sample_group[i] = n_group_sample
        weights[mask] = group_weights

    return weights, num_sample_group

def coll_source(x_sample, y_sample, z_sample, weights, num_group_sample, n_groups, n_samples, bounds_list, COLLISION_PARAMS):
    # Input: distribution parameters, bounds
    # Calculate: samples, group moment change
    # Output: moment change over the time step
    Rf1 = np.random.uniform(0.0, 1.0, COLLISION_PARAMS['n_coll'])
    Rf2 = np.random.uniform(0.0, 1.0, COLLISION_PARAMS['n_coll'])
    depl_idx1 = np.random.randint(0, n_samples, COLLISION_PARAMS['n_coll'])
    depl_idx2 = np.random.randint(0, n_samples, COLLISION_PARAMS['n_coll'])

    # BINARY ELASTIC COLLISIONS.
    group_n, group_px, group_py, group_pz, group_e = collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_groups, Rf1, Rf2, depl_idx1, depl_idx2)
    
    return group_n, group_px, group_py, group_pz, group_e