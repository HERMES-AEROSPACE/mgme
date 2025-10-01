from numba import jit
import numpy as np
from scipy import optimize, special, interpolate
from .virtual_collisions import collide
import math
import cvxpy as cp


@jit(nopython=True)
def f(x, y, z, A, b, wx, wy, wz):
    return A * np.exp(-b * ((x - wx)**2 + (y - wy)**2 + (z - wz)**2))

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

@jit(nopython=True)
def initialize_maxwellian(m_hat, T_hat, v_hat, cx, cy, cz):
    A = (m_hat / (np.pi * T_hat))**1.5
    beta = m_hat / T_hat
    dist = A * np.exp(-beta * ((cx - v_hat)**2 + cy**2 + cz**2))

    return dist

@jit(nopython=True)
def calc_moment(f, n, cx, cy, cz, dcx, dcy, dcz):
    mu = np.zeros(5)

    tmp1 = np.trapz(np.trapz(n * f, dx=dcz), dx=dcy)
    mu[0] = np.trapz(tmp1, dx=dcx)

    uk = cx * n * f
    mu[1] = np.trapz(np.trapz(np.trapz(uk, dx=dcz), dx=dcy), dx=dcx)

    uk = cy * n * f
    mu[2] = np.trapz(np.trapz(np.trapz(uk, dx=dcz), dx=dcy), dx=dcx)

    uk = cz * n * f
    mu[3] = np.trapz(np.trapz(np.trapz(uk, dx=dcz), dx=dcy), dx=dcx)

    c2 = cx**2 + cy**2 + cz**2
    ek = c2 * n * f
    mu[4] = np.trapz(np.trapz(np.trapz(ek, dx=dcz), dx=dcy), dx=dcx)

    return mu

@jit(nopython=True)
def ic(cx, cy, cz, dcx, dcy, dcz, n_val, u_val, T_val, numCx, numCy, numCz, numXj, numGroups, group_bounds):
    f = np.zeros((numXj, numCx, numCy, numCz))
    U0 = np.zeros((numXj, numGroups, 5))

    for point in range(0, numXj):
        m_hat = 1.0
        f[point] = initialize_maxwellian(m_hat, T_val[point], u_val[point], cx, cy, cz)

        for i in range(0, numGroups):
            lbound = group_bounds[i][0]
            ubound = group_bounds[i][1]

            U0[point, i] = calc_moment(f[point, lbound:ubound, :, :], n_val[point], \
            cx[lbound:ubound, :, :], cy[lbound:ubound, :, :], cz[lbound:ubound, :, :], dcx, dcy, dcz)

    return (U0, f)

def invert(U_list, numXj, numGroups, group_bounds, input_list, output_list, input_list2, output_list2, input_list1, output_list1):
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

                ci_cx = group_bounds['ci_cx'][i]
                cf_cx = group_bounds['cf_cx'][i]
                ci_cy = group_bounds['ci_cy'][0]
                cf_cy = group_bounds['cf_cy'][0]
                ci_cz = group_bounds['ci_cz'][0]
                cf_cz = group_bounds['cf_cz'][0]

                target_point = np.array([ux / n, uy / n, uz / n, e / n])
                if i == 1:
                    interp = interpolate.griddata(
                                points=output_list,
                                values=input_list,
                                xi=target_point.reshape(1, -1),
                                method='nearest'
                            )
                elif i == 2:
                    interp = interpolate.griddata(
                                points=output_list2,
                                values=input_list2,
                                xi=target_point.reshape(1, -1),
                                method='nearest'
                            )
                else:
                    interp = interpolate.griddata(
                                points=output_list1,
                                values=input_list1,
                                xi=target_point.reshape(1, -1),
                                method='nearest'
                            )
                # print(interp[0])
                sol = optimize.least_squares(moment_eq, interp[0], args=(ux / n, uy / n, \
                                            uz / n, e / n, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), \
                                                bounds=([0.0, -20, -20, -20], [10.0, 20, 20, 20]), method='trf', loss='soft_l1')
                if np.linalg.norm(sol.fun) > 1e-6:
                    print('residual:', np.linalg.norm(sol.fun), point, i)

                if sol.success:
                    b = sol.x[0]
                    wx = sol.x[1]
                    wy = sol.x[2]
                    wz = sol.x[3]

                I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cx'][i] - wx)) - special.erf(np.sqrt(b) * (group_bounds['ci_cx'][i] - wx)))
                I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cy'][0] - wy)) - special.erf(np.sqrt(b) * (group_bounds['ci_cy'][0] - wy)))
                I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (group_bounds['cf_cz'][0] - wz)) - special.erf(np.sqrt(b) * (group_bounds['ci_cz'][0] - wz)))
                A = n / (I0x * I0y * I0z)

                Ak[point, i] = A
                bk[point, i] = b
                wxk[point, i] = wx
                wyk[point, i] = wy
                wzk[point, i] = wz

    return Ak, bk, wxk, wyk, wzk

# @jit(nopython=True)
def calc_integral(bk, wxk, wyk, wzk, group_bounds, numXj, numGroups):
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

                ci_cx = group_bounds['ci_cx'][i]
                cf_cx = group_bounds['cf_cx'][i]
                ci_cy = group_bounds['ci_cy'][0]
                cf_cy = group_bounds['cf_cy'][0]
                ci_cz = group_bounds['ci_cz'][0]
                cf_cz = group_bounds['cf_cz'][0]

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

@jit(nopython=True)
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
                        (I2z[point, i] + 2 * wz * I1z[point, i] + wz**2 + I0z[point, i]) * (I1x[point, i] + wx * I0x[point, i]) * I0y[point, i])
                F[point, i] = np.array([F1, F2x, 0.0, 0.0, F3])
            else:
                F[point, i] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    return F

@jit(nopython=True)
def RK_LF(U_list, F_list, numXj, numGroups, dx, delta_t):
    k = np.zeros((numXj, numGroups, 5))

    p = np.arange(1, numXj - 1, 1)

    for i in range(0, numGroups):
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

    if mu[0] < 1e-2: # wonder what a good threshold is.
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
        prob.solve()
        
        weights = x.value

    return num_sample_group, weights, mask

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, mu, GROUP_PARAMS, num_groups):
    weights = np.zeros(n_samples)
    num_sample_group = np.zeros(num_groups)

    for i in range(num_groups):
        ci_cx = GROUP_PARAMS['ci_cx'][i]
        cf_cx = GROUP_PARAMS['cf_cx'][i]
        ci_cy = GROUP_PARAMS['ci_cy'][0]
        cf_cy = GROUP_PARAMS['cf_cy'][0]
        ci_cz = GROUP_PARAMS['ci_cz'][0]
        cf_cz = GROUP_PARAMS['cf_cz'][0]
        
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