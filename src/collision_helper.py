from numba import jit, types
from numba.typed import Dict
import numpy as np
from scipy import optimize, special, interpolate
import math
import cvxpy as cp
import sys
from scipy.special import erf
from scipy.stats import norm, qmc


class GroupNode:
    def __init__(self, bounds):
        self.group_bounds = bounds
        self.children = []

    def set_mu(self, mu):
        self.mu = mu

    def set_entropy(self, entropy):
        self.entropy = entropy

    def add_child(self, child):
        self.children.append(child)

    def update_mu(self, dt, dn, dpx, dpy, dpz, de):
        self.mu[0] += dt * dn
        self.mu[1] += dt * dpx
        self.mu[2] += dt * dpy
        self.mu[3] += dt * dpz
        self.mu[4] += dt * de

    def generate_samples(self, sampler):
        x_sample, y_sample, z_sample, num_sample, dv = generate_grid(self.group_bounds, sampler)
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.z_sample = z_sample
        self.num_sample = num_sample
        self.dv = dv

    def optimize_weights(self):
        res = try_solve_group(self.x_sample, self.y_sample, self.z_sample, self.dv, self.mu)
        if res[0]:
            self.weights = res[1]
            self.entropy = res[2]
        else:
            print('Optimization failed, investigate further.')

def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)

    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)

    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu

def generate_grid(bounds, sampler, method='regular'):
    l_bounds = np.array([bounds[0], bounds[2], bounds[4]])
    u_bounds = np.array([bounds[1], bounds[3], bounds[5]])

    if np.any(l_bounds > u_bounds): return

    if method == 'regular':
        n = 13

        # Cell width in each dimension
        Lx = u_bounds[0] - l_bounds[0]
        Ly = u_bounds[1] - l_bounds[1]
        Lz = u_bounds[2] - l_bounds[2]
        
        dx = Lx / n
        dy = Ly / n
        dz = Lz / n

        # Cell centers: first at L/(2n), last at L - L/(2n)
        x_sample = l_bounds[0] + dx * (np.arange(n) + 0.5)
        y_sample = l_bounds[1] + dy * (np.arange(n) + 0.5)
        z_sample = l_bounds[2] + dz * (np.arange(n) + 0.5)

        dv = dx * dy * dz  # = Lx*Ly*Lz / n**3
        print(dv)
        num_samples = n**3
    elif method == 'lhc':
        volume = (bounds[1] - bounds[0]) * \
                (bounds[3] - bounds[2]) * \
                (bounds[5] - bounds[4])
        num_samples = np.max((300, int(np.ceil(20 * volume))))
        print(num_samples)

        sample = qmc.scale(sampler.random(n=int(num_samples)), l_bounds, u_bounds)
        x_sample = sample[:, 0]
        y_sample = sample[:, 1]
        z_sample = sample[:, 2]

        dv = 0

    return x_sample, y_sample, z_sample, num_samples, dv

def try_solve_group(x_sample, y_sample, z_sample, dv, U_i, flux_limit=10.0):
    """Attempt to solve for one group"""
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
                lambdas = np.array([c.dual_value for c in constraints])

                
                H = -np.dot(lambdas, U_i)
                return True, x.value, H
            else:
                return False, None, f"flux_too_large_{predicted_flux:.3f}"
        else:
            return False, None, f"status_{prob.status}"

    except Exception as e:
        return False, None, str(e)

def refine_group2(bounds):
    """
    Split a group into 8 octants.
    
    Args:
        bounds: array of shape (6,) -> [ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz]
                these are indices into the velocity grid
    Returns:
        list of 8 child bounds arrays, each of shape (6,)
    """
    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz = bounds

    mid_cx = (ci_cx + cf_cx) // 2
    mid_cy = (ci_cy + cf_cy) // 2
    mid_cz = (ci_cz + cf_cz) // 2

    # Each dimension splits into [ci, mid] and [mid, cf]
    # Note: mid is shared between children — left child is [ci, mid+1),
    # right child is [mid, cf) so they tile without overlap on the grid.
    cx_splits = [(ci_cx, mid_cx), (mid_cx, cf_cx)]
    cy_splits = [(ci_cy, mid_cy), (mid_cy, cf_cy)]
    cz_splits = [(ci_cz, mid_cz), (mid_cz, cf_cz)]

    child_bounds = []
    for (lx, ux), (ly, uy), (lz, uz) in product(cx_splits, cy_splits, cz_splits):
        child_bounds.append(np.array([lx, ux, ly, uy, lz, uz]))

    return child_bounds

def refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, node, max_depth=6, curr_depth=0):
    """
    Recursively refine the AMR tree using entropy.
    
    Refinement is triggered when H_parent - sum(H_children) > threshold,
    meaning the coarse group is suppressing entropy that finer groups reveal.
    """
    bounds = node.group_bounds  # shape (6,): [ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz]
    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz = bounds

    f0_slice = f0[ci_cx:cf_cx, ci_cy:cf_cy, ci_cz:cf_cz]
    cx_vec_slice = cx_vec[ci_cx:cf_cx]
    cy_vec_slice = cy_vec[ci_cy:cf_cy]
    cz_vec_slice = cz_vec[ci_cz:cf_cz]

    # Skip groups with negligible population
    mu = calc_moment(
        f0_slice,
        cx[ci_cx:cf_cx, ci_cy:cf_cy, ci_cz:cf_cz],
        cy[ci_cx:cf_cx, ci_cy:cf_cy, ci_cz:cf_cz],
        cz[ci_cx:cf_cx, ci_cy:cf_cy, ci_cz:cf_cz],
        cx_vec_slice, cy_vec_slice, cz_vec_slice
    )
    node.set_mu(mu)

    if mu[0] < 1e-4:
        node.set_dist_param(0.0, 0.0, 0.0, 0.0, 0.0)
        node.set_hellinger_distance(0.0)
        node.set_entropy(0.0)
        return

    # Compute entropy defect — this is the core refinement criterion
    entropy_defect, child_entropies = entropy_refinement_criterion(
        f0_slice, cx_vec_slice, cy_vec_slice, cz_vec_slice
    )
    node.set_entropy(compute_group_entropy(f0_slice, cx_vec_slice, cy_vec_slice, cz_vec_slice))

    # Refinement check
    if entropy_defect < AMR['entropy_threshold']:
        return
    if curr_depth >= max_depth:
        print(f"Warning: Maximum recursion depth {max_depth} reached")
        return

    # Split into 8 children
    child_bounds_list = refine_group2(bounds)

    for i, (child_bounds, H_child) in enumerate(zip(child_bounds_list, child_entropies)):
        lx, ux, ly, uy, lz, uz = child_bounds

        child = GroupNode(child_bounds)
        node.add_child(child)

        # Reuse entropy already computed during defect calculation
        child.set_entropy(H_child)

        refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, child, max_depth, curr_depth + 1)