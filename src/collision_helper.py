import numpy as np
from numba import njit
from scipy.stats import qmc
from scipy import special
from matplotlib import pyplot as plt
from .config_0d import AMR
import math


_BOUND_EPS = 1e-10  # matches the old fit_maxent_weights barely-interior offset

def _split_samples(node):
    """
    Return (left_mask, right_mask, mid) for node.split_dim.
    Shared by update_shadows, accumulate_h2, split.
    """
    dim = node.split_dim
    samples   = [node.x_s, node.y_s, node.z_s]
    all_bounds = [node.xbounds, node.ybounds, node.zbounds]
    mid        = (all_bounds[dim][0] + all_bounds[dim][1]) / 2.0
    left_mask  = samples[dim] < mid
    return left_mask, ~left_mask, mid
 
def _child_bounds(node, side, mid):
    """
    Return [xbounds, ybounds, zbounds] for child `side` (0=left, 1=right).
    Only the split dimension is halved; the other two are copied from parent.
    """
    dim = node.split_dim
    all_bounds = [list(node.xbounds), list(node.ybounds), list(node.zbounds)]
    if side == 0:
        all_bounds[dim][1] = mid      # [lo, mid]
    else:
        all_bounds[dim][0] = mid      # [mid, hi]
    return all_bounds                  # [xb, yb, zb] — compatible with VelocityGroup(bounds=...)

def _axis_grid_for_leaf(G_master, lo, hi):
    """
    Per-leaf 1-D sample grid: master-grid points strictly inside (lo, hi),
    plus a leaf-specific boundary point at each end (lo+eps, hi-eps).

    The boundary points restore full compact support of the leaf so the
    Newton solve sees samples reaching the leaf walls. Adjacent siblings
    pick up boundary points at mid+eps and mid-eps respectively, so they
    are disjoint by 2*eps — collide() never sees coincident samples.
    """
    interior = G_master[(G_master > lo + _BOUND_EPS) & (G_master < hi - _BOUND_EPS)]
    return np.concatenate(([lo + _BOUND_EPS], interior, [hi - _BOUND_EPS]))


class VelocityGroup:
    # Master sample grid — shared across all leaves. Set once via
    # set_master_grid(MASTER_GRID) before any VelocityGroup is created.
    # Built strictly interior to the global bounds; per-leaf slices add
    # leaf-local boundary points (see _axis_grid_for_leaf).
    _gx_master = None
    _gy_master = None
    _gz_master = None
    min_points_per_axis = None

    @classmethod
    def set_master_grid(cls, master_grid_cfg):
        def _build_axis(lo, hi, dx):
            n = max(int(np.floor((hi - lo) / dx)) - 1, 0)
            return lo + dx * (np.arange(n) + 1)
        (xb, yb, zb) = master_grid_cfg['bounds']
        (dx, dy, dz) = master_grid_cfg['spacing']
        cls._gx_master = _build_axis(xb[0], xb[1], dx)
        cls._gy_master = _build_axis(yb[0], yb[1], dy)
        cls._gz_master = _build_axis(zb[0], zb[1], dz)
        cls.min_points_per_axis = master_grid_cfg['min_points_per_axis']

    def __init__(self, bounds, depth=0, max_depth=1, created_at=0, split_axes=0):
        """
        bounds: (cx_lo, cx_hi) index bounds into the full grid
        """
        self.is_empty = False
        self.n_threshold = 1e-4  # below this, treat as empty

        self.xbounds      = bounds[0]        # (lo, hi) value bounds
        self.ybounds      = bounds[1]        # (lo, hi) value bounds
        self.zbounds      = bounds[2]        # (lo, hi) value bounds 

        self.depth       = depth
        self.max_depth   = max_depth
        self.created_at  = created_at

        self.children    = []            # empty = leaf node
        self.parent      = None

        self.split_axes  = list(split_axes) if split_axes is not None else [0]

        # State
        self.w           = None          # current fitted max entropy weights
        self.lam         = np.zeros(5)   # lagrange multipliers
        self.mu          = None          # moments
        self.x_s = None
        self.y_s = None
        self.z_s = None

        # Shadow children — trial split, always two halves in vx
        # self.shadow_w       = [None, None]   # weights for left/right shadow
        # self.shadow_lam     = [np.zeros(5), np.zeros(5)]   # multipliers for left/right shadow
        self.shadow_mu      = [None, None]
        self.shadow_bounds  = [None, None]   # value bounds for each shadow

        # do I need to store the shadow sample locations?

        # Accumulation
        self.h2_accum = 0.0

        # EMA of relative ||coll * dt|| / ||mu||. Initialized to inf so a
        # newly-created leaf never trips the coarsen criterion or the split
        # veto until it has actually been measured by collide() at least
        # once (the EMA replaces inf on first update_rate call).
        self.rate_ema = np.inf

    @property
    def split_dim(self):
        """Dimension to split along: split_axes[depth % len(split_axes)]."""
        return self.split_axes[self.depth % len(self.split_axes)]

    def is_leaf(self):
        return len(self.children) == 0

    def has_split_density(self):
        """
        True iff both halves of a split along split_dim would have at
        least min_points_per_axis points along that axis (counting the
        boundary aug from _axis_grid_for_leaf).
        """
        if VelocityGroup.min_points_per_axis is None:
            return True  # gate disabled when master grid not configured
        G_master = (VelocityGroup._gx_master,
                    VelocityGroup._gy_master,
                    VelocityGroup._gz_master)[self.split_dim]
        lo, hi = (self.xbounds, self.ybounds, self.zbounds)[self.split_dim]
        mid = (lo + hi) / 2.0
        n_left  = len(_axis_grid_for_leaf(G_master, lo, mid))
        n_right = len(_axis_grid_for_leaf(G_master, mid, hi))
        return (n_left  >= VelocityGroup.min_points_per_axis and
                n_right >= VelocityGroup.min_points_per_axis)

    def split_block_reason(self):
        """None if can_split; else 'max_depth' or 'insufficient_density'."""
        if self.depth >= self.max_depth:
            return 'max_depth'
        if not self.has_split_density():
            return 'insufficient_density'
        return None

    def can_split(self):
        return self.split_block_reason() is None

    def can_coarsen(self, current_t, min_lifetime=AMR['min_lifetime']):
        """
        Node must have existed for min_lifetime steps before coarsening.
        """
        return (current_t - self.created_at) > min_lifetime

    def update_rate(self, coll_vec, dt, mu_total_norm):
        """
        Update the EMA of the leaf's rate of change, expressed as the
        fraction of total system mass moving in this leaf per step:
        r = ||coll * dt|| / ||mu_total||.

        Normalizing by the GLOBAL total instead of the per-leaf ||mu||
        keeps the noise floor uniform across leaves of any size — small
        tail leaves don't get their rate signal blown up by a tiny
        denominator, so the coarsen criterion and split veto behave the
        same way for them as for big leaves.
        """
        if mu_total_norm < 1e-12:
            return
        r = np.linalg.norm(coll_vec * dt) / mu_total_norm
        if not np.isfinite(self.rate_ema):
            self.rate_ema = r          # first measurement: seed the EMA
        else:
            g = AMR['rate_ema_gamma']
            self.rate_ema = g * self.rate_ema + (1.0 - g) * r

    def get_sibling(self):
        """
        Return the sibling node — the other child of our parent.
        Returns None if we are root or parent has unexpected children.
        """
        if self.parent is None:
            return None
        siblings = [c for c in self.parent.children if c is not self]
        if len(siblings) == 1:
            return siblings[0]
        return None

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def update_shadows(self):
        """
        Updates shadow values based on current group weights and sample locations.
        """
        left_mask, right_mask, mid = _split_samples(self)

        for i, mask in enumerate([left_mask, right_mask]):
            cb = _child_bounds(self, i, mid)
            self.shadow_bounds[i] = cb

            if not np.any(mask):
                self.shadow_mu[i] = np.zeros(5)
                continue

            n  = np.sum(self.w[mask])
            if n < self.n_threshold:
                self.shadow_mu[i] = np.zeros(5)
                continue

            ux = np.sum(self.w[mask] * self.x_s[mask])
            uy = np.sum(self.w[mask] * self.y_s[mask])
            uz = np.sum(self.w[mask] * self.z_s[mask])
            r2 = (self.x_s[mask]**2 + self.y_s[mask]**2
                + self.z_s[mask]**2)
            e  = np.sum(self.w[mask] * r2)
            self.shadow_mu[i] = np.array([n, ux, uy, uz, e])

    def split(self, current_t=0):
        """
        Promote shadow children to real children.
        Samples are partitioned by vx midpoint.
        Shadow weights/lam used to initialize children.
        """
        left_mask, right_mask, mid = _split_samples(self)
 
        for i, mask in enumerate([left_mask, right_mask]):
            cb = self.shadow_bounds[i]         # [xb, yb, zb] set by update_shadows
 
            child = VelocityGroup(
                bounds      = cb,
                depth       = self.depth + 1,
                max_depth   = self.max_depth,
                created_at  = current_t,
                split_axes  = self.split_axes, # ← pass down
            )
            child.parent = self
 
            n  = np.sum(self.w[mask])
            ux = np.sum(self.w[mask] * self.x_s[mask])
            uy = np.sum(self.w[mask] * self.y_s[mask])
            uz = np.sum(self.w[mask] * self.z_s[mask])
            r2 = (self.x_s[mask]**2 + self.y_s[mask]**2
                  + self.z_s[mask]**2)
            e  = np.sum(self.w[mask] * r2)
            child.mu = np.array([n, ux, uy, uz, e])
            # print(child.mu, self.shadow_mu)
 
            if child.mu[0] < child.n_threshold:
                child.is_empty = True
            else:
                result = fit_maxent_weights(
                    child.mu, child.xbounds, child.ybounds, child.zbounds, child.lam)
                if result is None:
                    child.is_empty = True
                else:
                    child.w, child.lam, child.x_s, child.y_s, child.z_s = result
                    child.update_shadows()
 
            self.children.append(child)

    def accumulate_h2(self):
        """
        Calculate the KL divergence between the current leaf and its children. Add it to the current leaf.

        There are no samples on the shadow children currently. Just the moments. To get the KL divergence I can
        use the same samples from my current leaf, fit the weights according to shadow moments and then compare.
        """
        # If we couldn't act on the signal anyway (max depth or insufficient
        # density), skip the work and stop letting h2_accum drift upward
        # forever on a leaf that can never split.
        if not self.can_split():
            return

        # Rate-based split veto: at equilibrium, h2 noise would otherwise
        # eventually cross the accum threshold and force a needless split.
        # If the leaf's smoothed rate of change is already below the
        # coarsen threshold, the dynamics says nothing's happening here.
        if self.rate_ema < AMR['rate_coarsen_threshold']:
            return

        left_mask, right_mask, _ = _split_samples(self)
        masks = [left_mask, right_mask]
        p = np.zeros_like(self.w)
 
        for i, mask in enumerate(masks):
            # Pass a copy of self.lam: solve_group_newton mutates lam in
            # place, and the two shadow fits are independent — letting the
            # second inherit the first's lam corrupts both this fit and
            # self.lam for the parent's next refit.
            w, _, success, _ = solve_group_newton(
                self.x_s[mask], self.y_s[mask], self.z_s[mask],
                self.shadow_mu[i], self.lam.copy())
            if success:
                p[mask] = w

        # Skip empty halves so 0 * log(0/q) doesn't poison h2_accum with NaN.
        nz = p > 0
        if not np.any(nz):
            return
        p_nz = p[nz] / p[nz].sum()
        q_nz = self.w[nz] / self.w[nz].sum()
        kl = np.sum(p_nz * np.log(p_nz / q_nz))

        self.h2_accum += kl

    def merge_children(self, current_t=0):
        assert len(self.children) == 2
        left, right = self.children

        # Concatenate samples
        self.mu  = left.mu + right.mu
        self.w, self.lam, self.x_s, self.y_s, self.z_s = fit_maxent_weights(self.mu, self.xbounds, self.ybounds, self.zbounds, self.lam)
        self.update_shadows()

        self.children         = []
        self.h2_accum         = 0.0
        self.created_at       = current_t
        self.rate_ema         = np.inf  # reseed; new state, new measurement

    def reactivate(self, current_t=0):
        """
        Called when mu[0] rises above threshold after a moment update.
        Fits weights from scratch and resets shadow reference.
        """
        result = fit_maxent_weights(self.mu, self.xbounds, self.ybounds, self.zbounds, self.lam)
        if result is None:
            return  # still can't fit, stay empty
        self.w, self.lam, self.x_s, self.y_s, self.z_s = result
        self.update_shadows()
        self.is_empty = False
        self.created_at = current_t
        self.h2_accum = 0.0  # fresh start, don't trigger immediate re-splitq

def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 

def solve_group_newton(x_s, y_s, z_s, U, lam, max_iter=50, tol=1e-10):
    """
    Solve max-entropy weights directly via Newton iterations on dual.
    
    Dual problem: find lambda s.t. sum_i phi_i * exp(lam . phi_i) = U
    """
    n = x_s.shape[0]

    phi = np.empty((5, n))
    phi[0] = np.ones(n)
    phi[1] = x_s
    phi[2] = y_s
    phi[3] = z_s
    phi[4] = x_s**2 + y_s**2 + z_s**2

    converged = False
    for iteration in range(max_iter):
        # Compute weights from current lambda
        w = np.exp(lam @ phi)

        phi_w = phi @ w  # broadcast: each row of phi scaled by w
        # Gradient: g = phi @ w - U  (moment residuals)
        g = phi_w - U

        # Convergence check.
        if np.linalg.norm(g) < tol:
            converged = True
            break

        # Hessian: H_ij = sum(phi_i * phi_j * w)
        H = phi @ (w[:, None] * phi.T)
        # Newton step. Bail with success=False on a singular Hessian
        # rather than crashing the whole run.
        try:
            lam -= np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return w, lam, False, g

    w = np.exp(lam @ phi)

    return w, lam, converged, g

def fit_maxent_weights(mu, xbounds, ybounds, zbounds, lam0):
    """
    Fit max-entropy weights on the per-leaf sample grid.

    Sample grid = master grid points strictly inside leaf bounds,
    plus (lo+eps, hi-eps) boundary points on each axis. See
    _axis_grid_for_leaf for the rationale.
    """
    if VelocityGroup._gx_master is None:
        raise RuntimeError(
            "Master grid not initialized. Call VelocityGroup.set_master_grid"
            " before constructing or fitting any VelocityGroup.")

    gx = _axis_grid_for_leaf(VelocityGroup._gx_master, xbounds[0], xbounds[1])
    gy = _axis_grid_for_leaf(VelocityGroup._gy_master, ybounds[0], ybounds[1])
    gz = _axis_grid_for_leaf(VelocityGroup._gz_master, zbounds[0], zbounds[1])

    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    x_slice = GX.ravel()
    y_slice = GY.ravel()
    z_slice = GZ.ravel()

    solution, lam, success, _ = solve_group_newton(x_slice, y_slice, z_slice, mu, lam0)

    if success:
        return solution, lam, x_slice, y_slice, z_slice
    return None

def leaf_entropy(leaf):
    """
    Entropy of the leaf's max-entropy distribution computed directly from
    fitted weights and sample positions:

        S_leaf = -sum_i w_i * log(w_i / dv_i)

    The continuous density at sample i is f(v_i) = w_i / dv_i, where
    dv_i is the per-sample volume element. Using the 3-D trapezoidal
    rule on the leaf's per-axis grids (master-grid interior + boundary
    aug, see _axis_grid_for_leaf) gives non-uniform dv_i that handles
    the ε-from-wall boundary samples correctly.

    Total entropy across the tree is sum(leaf_entropy(l) for l in leaves)
    since leaves partition velocity space with no overlap.
    """
    if leaf.w is None or leaf.is_empty:
        return 0.0

    gx = _axis_grid_for_leaf(VelocityGroup._gx_master, leaf.xbounds[0], leaf.xbounds[1])
    gy = _axis_grid_for_leaf(VelocityGroup._gy_master, leaf.ybounds[0], leaf.ybounds[1])
    gz = _axis_grid_for_leaf(VelocityGroup._gz_master, leaf.zbounds[0], leaf.zbounds[1])

    def _trap(x):
        n = len(x)
        if n < 2:
            return np.zeros(n)
        dv = np.empty(n)
        dv[0]  = (x[1]  - x[0])  / 2.0
        dv[-1] = (x[-1] - x[-2]) / 2.0
        if n > 2:
            dv[1:-1] = (x[2:] - x[:-2]) / 2.0
        return dv

    dvx, dvy, dvz = _trap(gx), _trap(gy), _trap(gz)
    # 3-D volume tensor product, raveled to match leaf.w (indexing='ij')
    DV = (dvx[:, None, None] * dvy[None, :, None] * dvz[None, None, :]).ravel()

    w = leaf.w
    nz = w > 0
    return -np.sum(w[nz] * np.log(w[nz] / DV[nz]))


def coarsening_h2_analytic(left, right, parent):
    """
    KL divergence of child groups vs. their parent.

    Dealing with all real nodes, therefore they all have their own samples. The number of samples of left + right = 2 * parent.
    They do NOT have the same sample set so need to reconcile this fact.

    What if I combine the left and right child samples and evaluate parent moments on it?
    """
    total_xs = np.concatenate((left.x_s, right.x_s), axis=0)
    total_ys = np.concatenate((left.y_s, right.y_s), axis=0)
    total_zs = np.concatenate((left.z_s, right.z_s), axis=0)

    # Pass a copy of parent.lam — solve_group_newton mutates lam in place,
    # and a coarsening check that fails (or even one that succeeds on a
    # slightly extended sample set with extra boundary points from each
    # child) shouldn't drift the parent's stored lam away from what
    # describes its own distribution.
    q, _, success, _ = solve_group_newton(
        total_xs, total_ys, total_zs, parent.mu, parent.lam.copy())

    if not success:
        return np.inf  # can't evaluate — be conservative, don't merge

    p = np.concatenate((left.w, right.w))
    nz = (p > 0) & (q > 0)
    if not np.any(nz):
        return np.inf
    p_nz = p[nz] / p[nz].sum()
    q_nz = q[nz] / q[nz].sum()
    return np.sum(p_nz * np.log(p_nz / q_nz))

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)

    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)

    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu

def initial_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, dS_threshold, max_passes=10):
    """
    Refine the AMR tree based on entropy difference on each leaf. 
    Runs until no splits occur or max_passes is reached.
    """
    n_fine = 50  # increasing this will make the entropy difference more accurate (not negative near converged grouping).
    for pass_idx in range(max_passes):
        leaves = root.get_leaves()
        splits_this_pass = 0

        for leaf in leaves:
            cx_lo, cx_hi = leaf.xbounds
            cy_lo, cy_hi = leaf.ybounds
            cz_lo, cz_hi = leaf.zbounds
            
            # Evaluate f0 at unique grid to get a relatively accurate moment evaluation.
            cx_vec = np.linspace(cx_lo, cx_hi, n_fine)
            cy_vec = np.linspace(cy_lo, cy_hi, n_fine)
            cz_vec = np.linspace(cz_lo, cz_hi, n_fine) 
            cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
            f_slice = f0(cx, cy, cz)

            # Recompute moments from f0, calculate weights, and update shadow values.
            mu = calc_moment(f_slice, cx, cy, cz, cx_vec, cy_vec, cz_vec)
            leaf.mu = mu
            leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = fit_maxent_weights(mu, leaf.xbounds, leaf.ybounds, leaf.zbounds, leaf.lam)
            leaf.update_shadows()

            # Calculate the KL divergence.
            dcx = (cx_hi - cx_lo) / (n_fine - 1)
            dcy = (cy_hi - cy_lo) / (n_fine - 1)
            dcz = (cz_hi - cz_lo) / (n_fine - 1)
            dv  = dcx * dcy * dcz
            f0_weights = f0(leaf.x_s, leaf.y_s, leaf.z_s) * dv

            p = f0_weights / np.sum(f0_weights)
            q = leaf.w / np.sum(leaf.w)
            kl = np.sum(p * np.log(p / q))

            # If divergence is larger than threshold, split the current group (update shadow children to real children).
            if kl > dS_threshold and leaf.can_split():
                print(f'pass {pass_idx}: splitting depth={leaf.depth} '
                      f'bounds={leaf.xbounds}, KL={kl:.4f}')
                
                leaf.split(current_t=0)
                splits_this_pass += 1
            if kl > dS_threshold and not leaf.can_split():
                print(f'Warning: cannot split depth={leaf.depth} '
                      f'bounds={leaf.xbounds}, reason={leaf.split_block_reason()}')

        print(f'Pass {pass_idx}: {splits_this_pass} split(s), '
              f'{len(root.get_leaves())} leaves total')

        if splits_this_pass == 0:
            print(f'Initial refinement converged after {pass_idx + 1} pass(es).')
            break

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
        # C = 0.5 * W**2 / n_actual * g**(2 - 2*omega) * sigma_coeff_hat
        C = 0.5 * W**2 / n_actual * g**(2 - 2*omega)

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