import numpy as np
from scipy import special
from scipy import optimize
from .config import GROUP_PARAMS, LOOKUP_TABLE, VELOCITY_SPACE
from matplotlib import pyplot as plt

        
def moments(beta, wx, wy, wz, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, ux, uy, uz, e):
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
    I0x = np.sqrt(np.pi / (4 * beta)) * (special.erf(np.sqrt(beta) * (cf_cx - wx)) - special.erf(np.sqrt(beta) * (ci_cx - wx)))
    I0y = np.sqrt(np.pi / (4 * beta)) * (special.erf(np.sqrt(beta) * (cf_cy - wy)) - special.erf(np.sqrt(beta) * (ci_cy - wy)))
    I0z = np.sqrt(np.pi / (4 * beta)) * (special.erf(np.sqrt(beta) * (cf_cz - wz)) - special.erf(np.sqrt(beta) * (ci_cz - wz)))

    I1x = (np.exp(-beta * (ci_cx - wx)**2) - np.exp(-beta * (cf_cx - wx)**2)) / (2 * beta)
    I1y = (np.exp(-beta * (ci_cy - wy)**2) - np.exp(-beta * (cf_cy - wy)**2)) / (2 * beta)
    I1z = (np.exp(-beta * (ci_cz - wz)**2) - np.exp(-beta * (cf_cz - wz)**2)) / (2 * beta)

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cx - wx)**2) * (cf_cx - wx))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cx - wx)**2) * (ci_cx - wx))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (special.erf(np.sqrt(beta) * (cf_cx - wx)) - special.erf(np.sqrt(beta) * (ci_cx - wx)))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cy - wy)**2) * (cf_cy - wy))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cy - wy)**2) * (ci_cy - wy))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (special.erf(np.sqrt(beta) * (cf_cy - wy)) - special.erf(np.sqrt(beta) * (ci_cy - wy)))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cz - wz)**2) * (cf_cz - wz))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cz - wz)**2) * (ci_cz - wz))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (special.erf(np.sqrt(beta) * (cf_cz - wz)) - special.erf(np.sqrt(beta) * (ci_cz - wz)))
    
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

    return [(I1x + wx*I0x) / I0x - ux, (I1y + wy*I0y) / I0y - uy, (I1z + wz*I0z) / I0z - uz, \
            (I2x + 2 * wx * I1x + wx**2 * I0x) / I0x + (I2y + 2 * wy * I1y + wy**2 * I0y) / I0y + (I2z + 2 * wz * I1z + wz**2 * I0z) / I0z - e]

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

def grid_search(group_bounds, ux, uy, uz, e):
    ci_cx = group_bounds['ci_cx']
    cf_cx = group_bounds['cf_cx']
    ci_cy = group_bounds['ci_cy']
    cf_cy = group_bounds['cf_cy']
    ci_cz = group_bounds['ci_cz']
    cf_cz = group_bounds['cf_cz']

    b_range = np.logspace(1e-8, 20.0, 20, endpoint=True)
    wx_range, wy_range, wz_range = np.linspace(ci_cx, cf_cx, 20), np.linspace(ci_cy, cf_cy, 20), np.linspace(ci_cz, cf_cz, 20)

    B, WX, WY, WZ = np.meshgrid(b_range, wx_range, wy_range, wz_range, indexing='ij')    

    f_values = moments(B, WX, WY, WZ, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, ux, uy, uz, e)

    abs_sum = np.sum(np.abs(f_values), axis=-1)

    n_guess = 3
    flat_idx = np.argpartition(abs_sum.flatten(), n_guess)[:n_guess]
    best_idx = np.unravel_index(flat_idx, abs_sum.shape)

    initial_guess = []
    for idx in range(n_guess):
        i, j, k, l = best_idx[0][idx], best_idx[1][idx], best_idx[2][idx], best_idx[3][idx]
        guess = [B[i, j, k, l], WX[i, j, k, l], WY[i, j, k, l], WZ[i, j, k, l]]
        residual_sum = abs_sum[i, j, k, l]
        initial_guess.append((guess, residual_sum))

    return initial_guess


def invert(mu, initial_guess, group_bounds=GROUP_PARAMS):
    A, b, wx, wy, wz = 0.0, 0.0, 0.0, 0.0, 0.0
    ci_cx = group_bounds['ci_cx']
    cf_cx = group_bounds['cf_cx']
    ci_cy = group_bounds['ci_cy']
    cf_cy = group_bounds['cf_cy']
    ci_cz = group_bounds['ci_cz']
    cf_cz = group_bounds['cf_cz']
    method = 'hybr'

    # try:
        # sol = optimize.root(moment_eq, initial_guess, \
        #                                 args=(mu[1] / mu[0], mu[2] / mu[0], \
        #                                         mu[3] / mu[0], mu[4] / mu[0], \
        #                                         ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), method=method)

        # if sol.success:
        #     b = sol.x[0]
        #     wx = sol.x[1]
        #     wy = sol.x[2]
        #     wz = sol.x[3]
        
        # if sol.success == False: raise RuntimeError
        # if VELOCITY_SPACE['cx_range'][0] > wx or VELOCITY_SPACE['cx_range'][1] < wx: raise RuntimeError
        # if VELOCITY_SPACE['cy_range'][0] > wy or VELOCITY_SPACE['cy_range'][1] < wy: raise RuntimeError
        # if VELOCITY_SPACE['cz_range'][0] > wz or VELOCITY_SPACE['cz_range'][1] < wz: raise RuntimeError
    # except RuntimeError:
        # guesses = grid_search(group_bounds, mu[1] / mu[0], mu[2] / mu[0], \
                                                # mu[3] / mu[0], mu[4] / mu[0])

        # for guess in guesses:
            # if guess[1] > 1e-4:
    sol = optimize.least_squares(moment_eq, initial_guess, args=(mu[1] / mu[0], mu[2] / mu[0], \
                                    mu[3] / mu[0], mu[4] / mu[0], \
                                    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), \
                                        bounds=([0.0, -10, -10, -10], [np.inf, 10, 10, 10]), method='trf', loss='soft_l1')

    if sol.success:
        b = sol.x[0]
        wx = sol.x[1]
        wy = sol.x[2]
        wz = sol.x[3]
    
    I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cx'] - wx)) - special.erf(np.sqrt(b) * (group_bounds['ci_cx'] - wx)))
    I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cy'] - wy)) - special.erf(np.sqrt(b) * (group_bounds['ci_cy'] - wy)))
    I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cz'] - wz)) - special.erf(np.sqrt(b) * (group_bounds['ci_cz'] - wz)))
    A = mu[0] / (I0x * I0y * I0z)

    return A, b, wx, wy, wz
