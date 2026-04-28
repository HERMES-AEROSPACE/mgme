from numba import jit, types, njit
from numba.typed import Dict
import numpy as np
from scipy import optimize, special, interpolate
from matplotlib import pyplot as plt
import math
import cvxpy as cp
import sys
from scipy.special import erf
from scipy.stats import norm, qmc
import time

from .physics.moments import moments, moment_eq, calc_moment
from .physics.grid import calculate_velocity_grid
from .physics.maxent import solve_group_newton


def initialize_maxwellian(m_hat, n_hat, T_hat, v_hat, cx, cy, cz):
    A = n_hat * (m_hat / (np.pi * T_hat))**1.5
    beta = m_hat / T_hat
    dist = A * np.exp(-beta * ((cx - v_hat)**2 + cy**2 + cz**2))

    return dist

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

@jit(nopython=True)
def calc_flux_analytical(lam_cache, bounds_list, num_groups, U_i):
    flux = np.zeros((num_groups, 5))

    for g in range(num_groups):
        lam = lam_cache[g]
        if lam[4] >= 0.0 or U_i[g, 0] < 1e-5:
            continue

        beta = -lam[4]
        wx   = -lam[1] / (2.0 * lam[4])
        wy   = -lam[2] / (2.0 * lam[4])
        wz   = -lam[3] / (2.0 * lam[4])

        xlo = bounds_list[g, 0];  xhi = bounds_list[g, 1]
        ylo = bounds_list[g, 2];  yhi = bounds_list[g, 3]
        zlo = bounds_list[g, 4];  zhi = bounds_list[g, 5]

        # --- x ---
        ax = xlo - wx;  bx = xhi - wx
        eax = np.exp(-beta * ax**2)
        ebx = np.exp(-beta * bx**2)
        erf_x = math.erf(np.sqrt(beta) * bx) - math.erf(np.sqrt(beta) * ax)
        exp_x = eax - ebx

        I0x = 0.5 * np.sqrt(np.pi / beta) * erf_x
        I1x = wx * I0x + 1.0 / (2.0 * beta) * exp_x
        I2x = (wx**2 * I0x
               + (wx / beta) * exp_x
               + 1.0 / (2.0 * beta) * (ax * eax - bx * ebx)
               + 0.5 * np.sqrt(np.pi / beta) * 1.0 / (2.0 * beta) * erf_x)
        I3x = (wx**3 * I0x
               + (3.0*wx**2/(2.0*beta) + 1.0/(2.0*beta**2)) * exp_x
               + 3.0*wx * 0.5*np.sqrt(np.pi / beta) * 1.0 / (2.0 * beta) * erf_x
               + 1.0 / (2.0 * beta) * ((ax**2 + 3.0*wx*ax) * eax - (bx**2 + 3.0*wx*bx) * ebx))

        # --- y ---
        ay = ylo - wy;  by = yhi - wy
        eay = np.exp(-beta * ay**2)
        eby = np.exp(-beta * by**2)
        erf_y = math.erf(np.sqrt(beta) * by) - math.erf(np.sqrt(beta) * ay)
        exp_y = eay - eby

        I0y = 0.5 * np.sqrt(np.pi / beta) * erf_y
        I1y = wy * I0y + 1.0 / (2.0 * beta) * exp_y
        I2y = (wy**2 * I0y
               + (wy / beta) * exp_y
               + 1.0 / (2.0 * beta) * (ay * eay - by * eby)
               + 0.5 * np.sqrt(np.pi / beta) * 1.0 / (2.0 * beta) * erf_y)

        # --- z ---
        az = zlo - wz;  bz = zhi - wz
        eaz = np.exp(-beta * az**2)
        ebz = np.exp(-beta * bz**2)
        erf_z = math.erf(np.sqrt(beta) * bz) - math.erf(np.sqrt(beta) * az)
        exp_z = eaz - ebz

        I0z = 0.5 * np.sqrt(np.pi / beta) * erf_z
        I1z = wz * I0z + 1.0 / (2.0 * beta) * exp_z
        I2z = (wz**2 * I0z
               + (wz / beta) * exp_z
               + 1.0 / (2.0 * beta) * (az * eaz - bz * ebz)
               + 0.5 * np.sqrt(np.pi / beta) * 1.0 / (2.0 * beta) * erf_z)

        # Skip if mean velocity is outside group bounds (erf_x/y/z ≈ 0 → I0 = 0)
        if I0x <= 0.0 or I0y <= 0.0 or I0z <= 0.0:
            continue

        # A from known density
        A = U_i[g, 0] / (I0x * I0y * I0z)

        flux[g, 0] = A * I1x * I0y * I0z
        flux[g, 1] = A * I2x * I0y * I0z
        flux[g, 2] = A * I1x * I1y * I0z
        flux[g, 3] = A * I1x * I0y * I1z
        flux[g, 4] = A * (I3x * I0y * I0z + I1x * I2y * I0z + I1x * I0y * I2z)

    return flux

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

        dual_vals = np.zeros(5)
        for i in range(5):
            dual_vals[i] = constraints[i].dual_value

        if x.value is not None and not np.any(np.isnan(x.value)):
            predicted_flux = np.sum(x_sample * x.value)
            
            if np.abs(predicted_flux) < flux_limit:
                return True, x.value, dual_vals
            else:
                return False, None, np.zeros(5)
        else:
            return False, None, np.zeros(5)
    except Exception as e:
        return False, None, np.zeros(5)

def generate_regular_samples(p, U_i, num_groups, bounds_list, lam_cache_i, n_sigma=3.0):
    num_valid_samples = np.zeros(num_groups, dtype=np.int64)
    n_x = 5
    n_y = 5
    n_z = 5
    total = n_x * n_y * n_z
    lam_out = np.zeros((num_groups, 5))

    n_fine = 31
    weights = np.zeros(n_fine**3 * num_groups)
    offsets = np.arange(num_groups + 1) * n_fine**3
    x_sample = np.zeros(n_fine**3 * num_groups)
    y_sample = np.zeros(n_fine**3 * num_groups)
    z_sample = np.zeros(n_fine**3 * num_groups)

    for i in range(num_groups):
        start_idx = int(offsets[i])
        end_idx = int(offsets[i+1])

        # Skip if density too small
        if U_i[i, 0] <= 1e-5:
            num_valid_samples[i] = 0
            weights[start_idx:end_idx] = 0.0
            continue

        success = False

        ux  = U_i[i, 1] / U_i[i, 0]
        uy  = U_i[i, 2] / U_i[i, 0]
        uz  = U_i[i, 3] / U_i[i, 0]
        T   = max(2 * (U_i[i, 4] / U_i[i, 0] - ux**2 - uy**2 - uz**2) / 3.0, 1e-10)
        v_th = np.sqrt(T)

        xlo = np.max([ux - n_sigma * v_th, bounds_list[i, 0]]) + (1e-10 if i > 0 else 0)
        xhi = np.min([ux + n_sigma * v_th, bounds_list[i, 1]]) - (1e-10 if i < num_groups-1 else 0)
        ylo = np.max([uy - n_sigma * v_th, bounds_list[i, 2]])
        yhi = np.min([uy + n_sigma * v_th, bounds_list[i, 3]])
        zlo = np.max([uz - n_sigma * v_th, bounds_list[i, 4]])
        zhi = np.min([uz + n_sigma * v_th, bounds_list[i, 5]])
        
        if np.all(lam_cache_i[i] == 0):
            # ──────────── CVXPY ──────────────
            gx = np.linspace(xlo, xhi, n_x)
            gy = np.linspace(ylo, yhi, n_y)
            gz = np.linspace(zlo, zhi, n_z)

            GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
            x_sub = GX.ravel()
            y_sub = GY.ravel()
            z_sub = GZ.ravel()

            
            success, solution, dual_vals = solve_group_cvxpy(x_sub, y_sub, z_sub, U_i[i])
            dual_vals = -dual_vals
            lam = dual_vals.copy()
        else:
            lam = lam_cache_i[i]

        # ──────────── Newton (project onto grid) ──────────────
        gx_fine = np.linspace(xlo, xhi, n_fine)
        gy_fine = np.linspace(ylo, yhi, n_fine)
        gz_fine = np.linspace(zlo, zhi, n_fine)
        GX, GY, GZ = np.meshgrid(gx_fine, gy_fine, gz_fine, indexing='ij')
        x_slice = GX.ravel()
        y_slice = GY.ravel()
        z_slice = GZ.ravel()
        
        w_init = np.exp(lam[1]*x_slice + lam[2]*y_slice + lam[3]*z_slice + 
                lam[4]*(x_slice**2 + y_slice**2 + z_slice**2))
        lam[0] = np.log(U_i[i, 0]) - np.log(np.sum(w_init))
        
        solution, lam, success, rel_err = solve_group_newton(x_slice, y_slice, z_slice, U_i[i], lam)

        if success:
            weights[start_idx:end_idx] = solution
            num_valid_samples[i] = np.sum(solution > 1e-12)
            lam_out[i] = lam

            x_sample[start_idx:end_idx] = x_slice
            y_sample[start_idx:end_idx] = y_slice
            z_sample[start_idx:end_idx] = z_slice

        if not success:
            print(f'Cell {p}, Group {i}: Failed')
            print(f'  moments: {U_i[i]}')
            print(f'  rel error: {rel_err}')
            num_valid_samples[i] = 0
            weights[start_idx:end_idx] = 0.0
            lam_out[i] = np.zeros(5)

    return (weights, num_valid_samples, lam_out, x_sample, y_sample, z_sample, offsets)

@njit
def collide(x_sample, y_sample, z_sample, weights, num_valid_samples, bounds_list, n_groups, n_coll,
            CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, sigma_coeff_hat, omega, alpha):

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
        for g in range(len(ci_cx)):
            if ci_cx[g] <= vx <= cf_cx[g] and ci_cy[g] <= vy <= cf_cy[g] and ci_cz[g] <= vz <= cf_cz[g]:
                return g
        return 0

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

    for i in range(n_actual):
        vx1 = x_sample[depl_idx1[i]]
        vy1 = y_sample[depl_idx1[i]]
        vz1 = z_sample[depl_idx1[i]]
        vx2 = x_sample[depl_idx2[i]]
        vy2 = y_sample[depl_idx2[i]]
        vz2 = z_sample[depl_idx2[i]]

        # Pre-collision groups
        g1 = find_group(vx1, vy1, vz1)
        g2 = find_group(vx2, vy2, vz2)

        # Post-collision velocities
        gx = vx2 - vx1
        gy = vy2 - vy1
        gz = vz2 - vz1
        g  = np.sqrt(gx**2 + gy**2 + gz**2)

        # VHS isotropic scattering. If alpha != 1, then VSS anisotropic scattering.
        phi       = 2 * np.pi * Rf1[i]
        cos_theta = 2 * Rf2[i]**(1 / alpha) - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        V_x = 0.5 * (vx1 + vx2)
        V_y = 0.5 * (vy1 + vy2)
        V_z = 0.5 * (vz1 + vz2)

        if alpha == 1.0:
            gxp = 0.5 * g * sin_theta * np.cos(phi)
            gyp = 0.5 * g * sin_theta * np.sin(phi)
            gzp = 0.5 * g * cos_theta
        else:
            gxp = 0.5 * (gx * cos_theta + (sin_theta * (g * gy * np.cos(phi) - gz * gx * np.sin(phi))) / (np.sqrt(gx**2 + gy**2)))
            gyp = 0.5 * (gy * cos_theta - (sin_theta * (g * gx * np.cos(phi) + gz * gy * np.sin(phi))) / (np.sqrt(gx**2 + gy**2)))
            gzp = 0.5 * (gz * cos_theta + np.sin(phi) * sin_theta * np.sqrt(gx**2 + gy**2))

        vx1p = V_x - gxp
        vy1p = V_y - gyp
        vz1p = V_z - gzp
        vx2p = V_x + gxp
        vy2p = V_y + gyp
        vz2p = V_z + gzp

        # Post-collision groups
        g1r = clamp_and_find_group(vx1p, vy1p, vz1p)
        g2r = clamp_and_find_group(vx2p, vy2p, vz2p)

        # Single collision weight — used for BOTH loss and gain
        C = 0.5 * W**2 / n_actual * g**(2 - 2*omega) * sigma_coeff_hat

        # Loss from pre-collision groups
        group_n[g1]  -= C;       group_n[g2]  -= C
        group_px[g1] -= C * vx1
        group_py[g1] -= C * vy1
        group_pz[g1] -= C * vz1
        group_e[g1]  -= C * (vx1**2 + vy1**2 + vz1**2)
        group_px[g2] -= C * vx2
        group_py[g2] -= C * vy2
        group_pz[g2] -= C * vz2
        group_e[g2]  -= C * (vx2**2 + vy2**2 + vz2**2)

        # Gain into post-collision groups
        group_n[g1r]  += C;       group_n[g2r]  += C
        group_px[g1r] += C * vx1p
        group_py[g1r] += C * vy1p
        group_pz[g1r] += C * vz1p
        group_e[g1r]  += C * (vx1p**2 + vy1p**2 + vz1p**2)
        group_px[g2r] += C * vx2p
        group_py[g2r] += C * vy2p
        group_pz[g2r] += C * vz2p
        group_e[g2r]  += C * (vx2p**2 + vy2p**2 + vz2p**2)

    return [group_n, group_px, group_py, group_pz, group_e]