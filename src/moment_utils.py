import numpy as np
from scipy import special
from scipy import optimize
from .config import GROUP_PARAMS, LOOKUP_TABLE
import matplotlib.pyplot as plt

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

def solve_equation(uk, ek, beta_list, w_list, uk_tab, ek_tab):
    """Solve for beta and w parameters given target moments.
    
    Args:
        uk: Target velocity
        ek: Target energy
        beta_list: List of beta values
        w_list: List of w values
        uk_tab: Table of velocity values
        ek_tab: Table of energy values
        
    Returns:
        Tuple of (beta, w) parameters
    """
    # Start looking in beta for uk and ek. Get each row in table out. 
    # Return the interpolated w values as a function of beta.
    w_val_list_uk = np.zeros(len(beta_list))
    for i, row in enumerate(uk_tab):
        mask = np.isnan(row)
        row = row[~mask]
        w_r = w_list[~mask]

        j1 = np.arange(0, len(w_r) - 1)
        j2 = np.arange(1, len(w_r))
        
        f_mask = np.logical_or(np.logical_and(uk > row[j1], uk < row[j2]), np.logical_and(uk < row[j1], uk > row[j2]))
        arr = f_mask.nonzero()[0]
        if len(arr) == 0:
            w_val_list_uk[i] = 0
        else:
            j = arr[0]
            w_val_list_uk[i] = (uk - row[j]) * (w_r[j+1] - w_r[j]) / (row[j+1] - row[j]) + w_r[j]

    w_val_list_ek = np.zeros(len(beta_list))
    for i, row in enumerate(ek_tab):
        mask = np.isnan(row)
        row = row[~mask]
        w_r = w_list[~mask]

        j1 = np.arange(0, len(w_r) - 1)
        j2 = np.arange(1, len(w_r))
        
        f_mask = np.logical_or(np.logical_and(ek > row[j1], ek < row[j2]), np.logical_and(ek < row[j1], ek > row[j2]))
        arr = f_mask.nonzero()[0]
        if len(arr) == 0:
            w_val_list_ek[i] = 0
        else:
            j = arr[0]
            w_val_list_ek[i] = (ek - row[j]) * (w_r[j+1] - w_r[j]) / (row[j+1] - row[j]) + w_r[j]

    # Scrub zeros from the returned w_val lists.
    umask = w_val_list_uk == 0.0
    w_val_list_uk = w_val_list_uk[~umask]
    uk_beta = beta_list[~umask]
    emask = w_val_list_ek == 0.0
    w_val_list_ek = w_val_list_ek[~emask]
    ek_beta = beta_list[~emask]

    # Find intersection between the two curves.
    lbkidx = np.where(beta_list == np.max(np.array([uk_beta[0], ek_beta[0]])))[0][0]
    rbkidx = np.where(beta_list == np.min(np.array([uk_beta[-1], ek_beta[-1]])))[0][0]
    restrict_beta = beta_list[lbkidx:rbkidx+1]
    uk_interp = np.interp(restrict_beta, uk_beta, w_val_list_uk)
    ek_interp = np.interp(restrict_beta, ek_beta, w_val_list_ek)
    idx = np.argwhere(np.diff(np.sign(ek_interp - uk_interp))).flatten()
    
    x1 = restrict_beta[idx]
    x2 = restrict_beta[idx+1]
    m1 = (uk_interp[idx+1] - uk_interp[idx])/(x2 - x1)
    m2 = (ek_interp[idx+1] - ek_interp[idx])/(x2 - x1)
    bk = (m1*x1 - uk_interp[idx] - m2*x1 + ek_interp[idx])/(m1 - m2)
    wk = np.interp(bk, uk_beta, w_val_list_uk)

    if bk.size == 0:
        b_val_list_uk = np.zeros(len(w_list))
        for i, col in enumerate(np.transpose(uk_tab)):
            mask = np.isnan(col)
            col = col[~mask]
            b_r = beta_list[~mask]

            j1 = np.arange(0, len(b_r) - 1)
            j2 = np.arange(1, len(b_r))
            
            f_mask = np.logical_or(np.logical_and(uk > col[j1], uk < col[j2]), np.logical_and(uk < col[j1], uk > col[j2]))
            arr = f_mask.nonzero()[0]
            if len(arr) == 0:
                b_val_list_uk[i] = 0
            else:
                j = arr[0]
                b_val_list_uk[i] = (uk - col[j]) * (b_r[j+1] - b_r[j]) / (col[j+1] - col[j]) + b_r[j]

        b_val_list_ek = np.zeros(len(w_list))
        for i, col in enumerate(np.transpose(ek_tab)):
            mask = np.isnan(col)
            col = col[~mask]
            b_r = beta_list[~mask]

            j1 = np.arange(0, len(b_r) - 1)
            j2 = np.arange(1, len(b_r))
            
            f_mask = np.logical_or(np.logical_and(ek > col[j1], ek < col[j2]), np.logical_and(ek < col[j1], ek > col[j2]))
            arr = f_mask.nonzero()[0]
            if len(arr) == 0:
                b_val_list_ek[i] = 0
            else:
                j = arr[0]
                b_val_list_ek[i] = (ek - col[j]) * (b_r[j+1] - b_r[j]) / (col[j+1] - col[j]) + b_r[j]

        # Scrub zeros from the returned w_val lists.
        umask = b_val_list_uk == 0.0
        b_val_list_uk = b_val_list_uk[~umask]
        uk_w = w_list[~umask]
        emask = b_val_list_ek == 0.0
        b_val_list_ek = b_val_list_ek[~emask]
        ek_w = w_list[~emask]

        # Find intersection between the two curves.
        lbkidx = np.where(w_list == np.max(np.array([uk_w[0], ek_w[0]])))[0][0]
        rbkidx = np.where(w_list == np.min(np.array([uk_w[-1], ek_w[-1]])))[0][0]
        restrict_w = w_list[lbkidx:rbkidx+1]
        uk_interp = np.interp(restrict_w, uk_w, b_val_list_uk)
        ek_interp = np.interp(restrict_w, ek_w, b_val_list_ek)
        idx = np.argwhere(np.diff(np.sign(ek_interp - uk_interp))).flatten()
        
        x1 = restrict_w[idx]
        x2 = restrict_w[idx+1]
        m1 = (uk_interp[idx+1] - uk_interp[idx])/(x2 - x1)
        m2 = (ek_interp[idx+1] - ek_interp[idx])/(x2 - x1)
        wk = (m1*x1 - uk_interp[idx] - m2*x1 + ek_interp[idx])/(m1 - m2)
        bk = np.interp(wk, uk_w, b_val_list_uk)

        return (bk[0], wk[0])

    return (bk[0], wk[0]) 

def calculate_group_moments(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, params=GROUP_PARAMS):
    mu = np.zeros((params['num_groups_cx'], params['num_groups_cy'], params['num_groups_cz'], 5))

    for i in range(params['num_groups_cx']):
        for j in range(params['num_groups_cy']):
            for k in range(params['num_groups_cz']):
                lb_cx = params['group_bounds_cx'][i, 0]
                ub_cx = params['group_bounds_cx'][i, 1]
                lb_cy = params['group_bounds_cy'][j, 0]
                ub_cy = params['group_bounds_cy'][j, 1]
                lb_cz = params['group_bounds_cz'][k, 0]
                ub_cz = params['group_bounds_cz'][k, 1]

                mu[i, j, k] = calc_moment(f0[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cx[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], \
                                cy[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], cz[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz], \
                                        cx_vec[lb_cx:ub_cx], cy_vec[lb_cy:ub_cy], cz_vec[lb_cz:ub_cz])

    return mu

def create_table(beta_list, w_list):
    table = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz'], 4, LOOKUP_TABLE['n_points'], LOOKUP_TABLE['n_points']))
    X, Y = np.meshgrid(beta_list, w_list, indexing='ij')

    for i in range(GROUP_PARAMS['num_groups_cx']):
        for j in range(GROUP_PARAMS['num_groups_cy']):
            for k in range(GROUP_PARAMS['num_groups_cz']):
                uxk_tab, uyk_tab, uzk_tab, ek_tab = np.array(moments(X, Y, GROUP_PARAMS['ci_cx'][i], GROUP_PARAMS['cf_cx'][i], \
                                                  GROUP_PARAMS['ci_cy'][j], GROUP_PARAMS['cf_cy'][j], GROUP_PARAMS['ci_cz'][k], GROUP_PARAMS['cf_cz'][k]))
                table[i, j, k, 0] = uxk_tab
                table[i, j, k, 1] = uyk_tab
                table[i, j, k, 2] = uzk_tab
                table[i, j, k, 3] = ek_tab

    return table

def invert(mu, b_guess, wx_guess, wy_guess, wz_guess, params=GROUP_PARAMS):
    A = np.zeros((params['num_groups_cx'], params['num_groups_cy'], params['num_groups_cz']))
    b = np.zeros((params['num_groups_cx'], params['num_groups_cy'], params['num_groups_cz']))
    wx = np.zeros((params['num_groups_cx'], params['num_groups_cy'], params['num_groups_cz']))
    wy = np.zeros((params['num_groups_cx'], params['num_groups_cy'], params['num_groups_cz']))
    wz = np.zeros((params['num_groups_cx'], params['num_groups_cy'], params['num_groups_cz']))

    for i in range(0, params['num_groups_cx']):
        for j in range(0, params['num_groups_cy']):
            for k in range(0, params['num_groups_cz']):
                b[i, j, k], wx[i, j, k], wy[i, j, k], wz[i, j, k] = optimize.fsolve(moment_eq, \
                                                                                    [1.0, 0.0, 0.0, 0.0], \
                                                                                        args=(mu[i, j, k, 1] / mu[i, j, k, 0], mu[i, j, k, 2] / mu[i, j, k, 0], \
                                                                                              mu[i, j, k, 3] / mu[i, j, k, 0], mu[i, j, k, 4] / mu[i, j, k, 0], \
                                                                                                params['ci_cx'][i], params['cf_cx'][i], \
                                                                                                    params['ci_cy'][j], params['cf_cy'][j], \
                                                                                                        params['ci_cz'][k], params['cf_cz'][k]))

                I0x = np.sqrt(np.pi / (4 * b[i, j, k])) * (special.erf(np.sqrt(b[i, j, k]) * (params['cf_cx'][i] - wx[i, j, k])) - special.erf(np.sqrt(b[i, j, k]) * (params['ci_cx'][i] - wx[i, j, k])))
                I0y = np.sqrt(np.pi / (4 * b[i, j, k])) * (special.erf(np.sqrt(b[i, j, k]) * (params['cf_cy'][j] - wy[i, j, k])) - special.erf(np.sqrt(b[i, j, k]) * (params['ci_cy'][j] - wy[i, j, k])))
                I0z = np.sqrt(np.pi / (4 * b[i, j, k])) * (special.erf(np.sqrt(b[i, j, k]) * (params['cf_cz'][k] - wz[i, j, k])) - special.erf(np.sqrt(b[i, j, k]) * (params['ci_cz'][k] - wz[i, j, k])))
                A[i, j, k] = mu[i, j, k, 0] / (I0x * I0y * I0z)

    return A, b, wx, wy, wz