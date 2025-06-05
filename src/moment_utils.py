import numpy as np
from scipy import special
from scipy import optimize
from .config import GROUP_PARAMS, LOOKUP_TABLE, VELOCITY_SPACE
from matplotlib import pyplot as plt

import sys

def moments(beta, w, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    """
    Calculate moments for given beta and w parameters.
    
    Args:
        beta: Beta parameter
        w: W parameter
        ci: Lower bound
        cf: Upper bound
    
    Returns:
        Tuple of (I1x + w*I0x)/I0x and (I2x + I0x*w**2 + 2*w*I1x + I0x/beta)/I0x
    """
    I0x = np.sqrt(np.pi / (4 * beta)) * (special.erf(np.sqrt(beta) * (cf_cx - w)) - special.erf(np.sqrt(beta) * (ci_cx - w)))
    I0y = np.sqrt(np.pi / (4 * beta)) * (special.erf(np.sqrt(beta) * (cf_cy - w)) - special.erf(np.sqrt(beta) * (ci_cy - w)))
    I0z = np.sqrt(np.pi / (4 * beta)) * (special.erf(np.sqrt(beta) * (cf_cz - w)) - special.erf(np.sqrt(beta) * (ci_cz - w)))

    I1x = (np.exp(-beta * (ci_cx - w)**2) - np.exp(-beta * (cf_cx - w)**2)) / (2 * beta)
    I1y = (np.exp(-beta * (ci_cy - w)**2) - np.exp(-beta * (cf_cy - w)**2)) / (2 * beta)
    I1z = (np.exp(-beta * (ci_cz - w)**2) - np.exp(-beta * (cf_cz - w)**2)) / (2 * beta)

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cx - w)**2) * (cf_cx - w))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cx - w)**2) * (ci_cx - w))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (special.erf(np.sqrt(beta) * (cf_cx - w)) - special.erf(np.sqrt(beta) * (ci_cx - w)))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cy - w)**2) * (cf_cy - w))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cy - w)**2) * (ci_cy - w))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (special.erf(np.sqrt(beta) * (cf_cy - w)) - special.erf(np.sqrt(beta) * (ci_cy - w)))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cz - w)**2) * (cf_cz - w))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cz - w)**2) * (ci_cz - w))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (special.erf(np.sqrt(beta) * (cf_cz - w)) - special.erf(np.sqrt(beta) * (ci_cz - w)))
    
    if I0x.size != 1:
        for i, row in enumerate(I0x):
            row = np.where(row > 1e-12, row, np.nan)
            I0x[i, :] = row
    
    if I0y.size != 1:
        for i, row in enumerate(I0y):
            row = np.where(row > 1e-12, row, np.nan)
            I0y[i, :] = row

    if I0z.size != 1:
        for i, row in enumerate(I0z):
            row = np.where(row > 1e-12, row, np.nan)
            I0z[i, :] = row

    return [(I1x + w*I0x) / I0x, (I1y + w*I0y) / I0y, (I1z + w*I0z) / I0z, (I2x + 2 * w * I1x + w**2 * I0x) / I0x + (I2y + 2 * w * I1y + w**2 * I0y) / I0y + (I2z + 2 * w * I1z + w**2 * I0z) / I0z]

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

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    """
    Calculate moments (density, momentum, energy) for a given distribution function.
    
    Args:
        f: Distribution function
        cx, cy, cz: Velocity components
        cx_vec, cy_vec, cz_vec: Velocity grid vectors
    
    Returns:
        Array of moments [density, x-momentum, y-momentum, z-momentum, energy]
    """
    mu = np.zeros(5)

    # Density moment
    mu[0] = np.trapz(np.trapz(np.trapz(f, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Momentum moment
    uk = cx * f
    mu[1] = np.trapz(np.trapz(np.trapz(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cy * f
    mu[2] = np.trapz(np.trapz(np.trapz(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cz * f
    mu[3] = np.trapz(np.trapz(np.trapz(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Energy moment
    c2 = cx**2 + cy**2 + cz**2
    ek = c2 * f
    mu[4] = np.trapz(np.trapz(np.trapz(ek, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return mu 

def calculate_group_moments(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, n_groups=GROUP_PARAMS, group_bounds=GROUP_PARAMS):
    mu = np.zeros((n_groups['num_groups_cx'], n_groups['num_groups_cy'], n_groups['num_groups_cz'], 5))

    for i in range(n_groups['num_groups_cx']):
        for j in range(n_groups['num_groups_cy']):
            for k in range(n_groups['num_groups_cz']):
                lb_cx = group_bounds['group_bounds_cx'][i, 0]
                ub_cx = group_bounds['group_bounds_cx'][i, 1]
                lb_cy = group_bounds['group_bounds_cy'][j, 0]
                ub_cy = group_bounds['group_bounds_cy'][j, 1]
                lb_cz = group_bounds['group_bounds_cz'][k, 0]
                ub_cz = group_bounds['group_bounds_cz'][k, 1]

                mu[i, j, k] = calc_moment(f0[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cx[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], \
                                cy[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cz[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], \
                                        cx_vec[lb_cx:ub_cx], cy_vec[lb_cy:ub_cy], cz_vec[lb_cz:ub_cz])

    return mu

def invert(mu, group_bounds=GROUP_PARAMS, max_attempts=10):
    guess_arr=[1.0, 0.0, 0.0, 0.0]

    for attempt in range(max_attempts):
        try:
            sol, _, ier, _ = optimize.fsolve(moment_eq, guess_arr, \
                                            args=(mu[1] / mu[0], mu[2] / mu[0], \
                                                    mu[3] / mu[0], mu[4] / mu[0], \
                                                    group_bounds['ci_cx'], group_bounds['cf_cx'], \
                                                        group_bounds['ci_cy'], group_bounds['cf_cy'], \
                                                            group_bounds['ci_cz'], group_bounds['cf_cz']), full_output=True)
            b = sol[0]
            wx = sol[1]
            wy = sol[2]
            wz = sol[3]
            
            if ier != 1: raise RuntimeError
            if VELOCITY_SPACE['cx_range'][0] > wx or VELOCITY_SPACE['cx_range'][1] < wx: raise RuntimeError
            if VELOCITY_SPACE['cy_range'][0] > wy or VELOCITY_SPACE['cy_range'][1] < wy: raise RuntimeError
            if VELOCITY_SPACE['cz_range'][0] > wz or VELOCITY_SPACE['cz_range'][1] < wz: raise RuntimeError
        except RuntimeError:
            guess_arr[0] /= 10
            if mu[1] > 0: guess_arr[1] += 0.1
            else: guess_arr[1] -= 0.1
            if mu[2] > 0: guess_arr[2] += 0.1
            else: guess_arr[2] -= 0.1
            if mu[3] > 0: guess_arr[3] += 0.1
            else: guess_arr[3] -= 0.1

    I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cx'] - wx)) - special.erf(np.sqrt(b) * (group_bounds['ci_cx'] - wx)))
    I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cy'] - wy)) - special.erf(np.sqrt(b) * (group_bounds['ci_cy'] - wy)))
    I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cz'] - wz)) - special.erf(np.sqrt(b) * (group_bounds['ci_cz'] - wz)))
    A = mu[0] / (I0x * I0y * I0z)

    return A, b, wx, wy, wz
