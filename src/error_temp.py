import numpy as np
from numba import njit
import time
from .config_1d import GROUP_PARAMS
import itertools
from scipy.stats import qmc
import cvxpy as cp
from matplotlib import pyplot as plt
import sys

def generate_grid(bounds_list, num_groups, factor):
    num_samples = np.zeros(num_groups)
    for i in range(0, num_groups):
        volume = (bounds_list[i, 1] - bounds_list[i, 0]) * \
            (bounds_list[i, 3] - bounds_list[i, 2]) * \
            (bounds_list[i, 5] - bounds_list[i, 4])
        num_samples[i] = np.max((300, int(np.ceil(factor * volume))))
    
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
                return False, None, f"flux_too_large_{predicted_flux:.3f}"
        else:
            return False, None, f"status_{prob.status}"
    except Exception as e:
        return False, None, str(e)

# warm up
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

bounds_list = np.zeros((num_groups, 6))
for i in range(num_groups):
    bounds_list[i] = np.array([ci_combo[i, 0], cf_combo[i, 0], ci_combo[i, 1], cf_combo[i, 1], ci_combo[i, 2], cf_combo[i, 2]])

# LHS grid.
x_s, y_s, z_s, offsets, num_samples = generate_grid(bounds_list, num_groups, 0.75)
print(bounds_list)
print(num_samples)

U = np.load('simulation_data/U0.npy')

n_sigma = 3.0
for point in range(0, 101):
    U_i = U[point]
    for i in range(num_groups):
        if U_i[i, 0] <= 1e-8:
            continue

        # Try CVXPY by selecting a subset of LHS samples. I wonder if regular samples are better honestly.
        ux  = U_i[i, 1] / U_i[i, 0]
        uy  = U_i[i, 2] / U_i[i, 0]
        uz  = U_i[i, 3] / U_i[i, 0]
        T   = max(2 * (U_i[i, 4] / U_i[i, 0] - ux**2 - uy**2 - uz**2) / 3.0, 1e-10)
        v_th = np.sqrt(T)

        xlo = np.max([ux - n_sigma * v_th, bounds_list[i, 0]])
        xhi = np.min([ux + n_sigma * v_th, bounds_list[i, 1]])
        ylo = np.max([uy - n_sigma * v_th, bounds_list[i, 2]])
        yhi = np.min([uy + n_sigma * v_th, bounds_list[i, 3]])
        zlo = np.max([uz - n_sigma * v_th, bounds_list[i, 4]])
        zhi = np.min([uz + n_sigma * v_th, bounds_list[i, 5]])

        gx = np.linspace(xlo, xhi, 5)
        gy = np.linspace(ylo, yhi, 5)
        gz = np.linspace(zlo, zhi, 5)

        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
        x_sub = GX.ravel()
        y_sub = GY.ravel()
        z_sub = GZ.ravel()
        
        start = time.perf_counter()
        success, solution_sub, dual_vals = solve_group_cvxpy(x_sub, y_sub, z_sub, U_i[i])
        cvxtime = time.perf_counter() - start

        dual_vals = -dual_vals
        dual_weights = np.exp(dual_vals[0] + dual_vals[1] * x_sub + dual_vals[2] * y_sub + dual_vals[3] * z_sub + \
                        dual_vals[4] * (x_sub**2 + y_sub**2 + z_sub**2))
        # dual_weights *= U_i[i, 0] / np.sum(dual_weights)   

        gx_fine = np.linspace(xlo, xhi, 21)
        gy_fine = np.linspace(ylo, yhi, 21)
        gz_fine = np.linspace(zlo, zhi, 21)
        GX, GY, GZ = np.meshgrid(gx_fine, gy_fine, gz_fine, indexing='ij')
        x_fine = GX.ravel()
        y_fine = GY.ravel()
        z_fine = GZ.ravel()

        start = time.perf_counter()
        phi_fine = np.stack([
            np.ones(len(x_s)),
            x_s, y_s, z_s,
            x_s**2 + y_s**2 + z_s**2
        ], axis=0)  # (5, N)

        print(dual_vals)
        # Newton refinement on fine grid starting from coarse lambdas
        lam = dual_vals.copy()
        w_init = np.exp(lam[1]*x_s + lam[2]*y_s + lam[3]*z_s + 
                lam[4]*(x_s**2 + y_s**2 + z_s**2))
        lam[0] = np.log(U_i[i, 0]) - np.log(np.sum(w_init))
        
        res_hist = np.array([])
        for k in range(50):
            w = np.exp(lam @ phi_fine)
            mu_hat = phi_fine @ w
            residual = mu_hat - U_i[i]
            res_norm = np.linalg.norm(residual)
            res_hist = np.append(res_hist, res_norm)
            if res_norm < 1e-10:
                break
            H = phi_fine @ (w[:, None] * phi_fine.T)
            lam -= np.linalg.solve(H, residual)
        print(lam)
        print('solve time:', time.perf_counter() - start + cvxtime)
        dual_weights = np.exp(lam @ phi_fine)   
        print(np.abs(np.sum(dual_weights) - U_i[i, 0]) / U_i[i, 0], np.sum(dual_weights), U_i[i, 0])
        print(np.abs(np.sum(x_s * dual_weights) - U_i[i, 1]) / U_i[i, 1], np.sum(x_s * dual_weights), U_i[i, 1])
        print(np.abs(np.sum(y_s * dual_weights) - U_i[i, 2]) / U_i[i, 2], np.sum(y_s * dual_weights), U_i[i, 2])
        print(np.abs(np.sum(z_s * dual_weights) - U_i[i, 3]) / U_i[i, 3], np.sum(z_s * dual_weights), U_i[i, 3])
        print(np.abs(np.sum((x_s**2 + y_s**2 + z_s**2) * dual_weights) - U_i[i, 4]) / U_i[i, 4], np.sum((x_s**2 + y_s**2 + z_s**2) * dual_weights), U_i[i, 4])
        print()