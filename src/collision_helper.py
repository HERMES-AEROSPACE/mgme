import numpy as np
from numba import njit
from scipy.stats import qmc
from scipy import special
from matplotlib import pyplot as plt
from .config_0d import AMR
import math


_BOUND_EPS = 1e-10  # matches the old fit_maxent_weights barely-interior offset

def _split_samples(node, dim):
    """
    Return (left_mask, right_mask, mid) for the requested axis.
    Shared by update_shadows, accumulate_kl, split.
    """
    samples   = [node.x_s, node.y_s, node.z_s]
    all_bounds = [node.xbounds, node.ybounds, node.zbounds]
    mid        = (all_bounds[dim][0] + all_bounds[dim][1]) / 2.0
    left_mask  = samples[dim] < mid
    return left_mask, ~left_mask, mid

def _child_bounds(node, side, mid, dim):
    """
    Return [xbounds, ybounds, zbounds] for child `side` (0=left, 1=right)
    when splitting along `dim`. Only the split dimension is halved.
    """
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

    def __init__(self, bounds, depth=0, max_depth=1, created_at=0, split_axes=None):
        """
        bounds: (cx_lo, cx_hi) index bounds into the full grid
        split_axes: cycled by depth (split_axes[depth % len(split_axes)]).
                    initial_refine may override self.split_dim per leaf
                    after a per-axis f0 evaluation.
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

        # Cycled split-axis list (inherited by children); split_dim is the
        # default axis for this leaf, settable so initial_refine can pick
        # adaptively per leaf at startup.
        self.split_axes = list(split_axes) if split_axes is not None else [0]
        self.split_dim  = self.split_axes[self.depth % len(self.split_axes)]

        # State
        self.w           = None          # current fitted max entropy weights
        self.lam         = np.zeros(5)   # lagrange multipliers
        self.mu          = None          # moments
        self.x_s = None
        self.y_s = None
        self.z_s = None

        # Shadow children — trial split along self.split_dim.
        self.shadow_mu      = [None, None]
        self.shadow_bounds  = [None, None]

        # KL accumulator (drift signal — see accumulate_kl).
        self.kl_accum = 0.0

        # EMA of relative ||coll * dt|| / ||mu||. Initialized to inf so a
        # newly-created leaf never trips the coarsen criterion or the split
        # veto until it has actually been measured by collide() at least
        # once (the EMA replaces inf on first update_rate call).
        self.rate_ema = np.inf

    def is_leaf(self):
        return len(self.children) == 0

    def has_split_density(self, dim):
        """
        True iff both halves of a split along `dim` would have at least
        min_points_per_axis points along that axis (counting boundary aug
        from _axis_grid_for_leaf).
        """
        if VelocityGroup.min_points_per_axis is None:
            return True  # gate disabled when master grid not configured
        G_master = (VelocityGroup._gx_master,
                    VelocityGroup._gy_master,
                    VelocityGroup._gz_master)[dim]
        lo, hi = (self.xbounds, self.ybounds, self.zbounds)[dim]
        mid = (lo + hi) / 2.0
        n_left  = len(_axis_grid_for_leaf(G_master, lo, mid))
        n_right = len(_axis_grid_for_leaf(G_master, mid, hi))
        return (n_left  >= VelocityGroup.min_points_per_axis and
                n_right >= VelocityGroup.min_points_per_axis)

    def split_block_reason(self, dim):
        """None if can_split along dim; else 'max_depth' or 'insufficient_density'."""
        if self.depth >= self.max_depth:
            return 'max_depth'
        if not self.has_split_density(dim):
            return 'insufficient_density'
        return None

    def can_split(self, dim):
        return self.split_block_reason(dim) is None

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
        Update shadow_mu / shadow_bounds for a hypothetical split along
        self.split_dim. Called at split time, after merge, after reactivate,
        and from initial_refine after picking an axis.
        """
        d = self.split_dim
        left_mask, right_mask, mid = _split_samples(self, d)

        for i, mask in enumerate([left_mask, right_mask]):
            cb = _child_bounds(self, i, mid, d)
            self.shadow_bounds[i] = cb

            if not np.any(mask):
                self.shadow_mu[i] = np.zeros(5)
                continue

            n = np.sum(self.w[mask])
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
        Promote shadow children to real children along self.split_dim.
        update_shadows() must have been called for self.split_dim first
        (it is, as part of the normal lifecycle: split / merge / reactivate
        all call update_shadows, and initial_refine calls it after picking
        an axis).
        """
        d = self.split_dim
        left_mask, right_mask, mid = _split_samples(self, d)

        for i, mask in enumerate([left_mask, right_mask]):
            cb = self.shadow_bounds[i]         # set by update_shadows

            child = VelocityGroup(
                bounds      = cb,
                depth       = self.depth + 1,
                max_depth   = self.max_depth,
                created_at  = current_t,
                split_axes  = self.split_axes,
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

    def accumulate_kl(self):
        """
        Drift signal: KL between (refit on current samples to STALE
        shadow_mu set at split/merge time) and (current Maxent fit).
        Grows as leaf moments evolve away from the configuration captured
        at the last update_shadows() call. When kl_accum > threshold the
        driver triggers a split along self.split_dim.

        Note: shadow_mu is structurally tied to self.split_dim, so this
        signal is axis-agnostic — it measures total drift, not per-axis
        structural mismatch. The cycled split_axes list (or initial_refine's
        per-leaf override) chooses the axis externally.
        """
        if not self.can_split(self.split_dim):
            return

        # Rate-based split veto: at equilibrium, KL noise would otherwise
        # eventually cross the accum threshold and force a needless split.
        if self.rate_ema < AMR['rate_coarsen_threshold']:
            return

        left_mask, right_mask, _ = _split_samples(self, self.split_dim)
        p = np.zeros_like(self.w)

        for i, mask in enumerate([left_mask, right_mask]):
            w, _, success, _ = solve_group_newton(
                self.x_s[mask], self.y_s[mask], self.z_s[mask],
                self.shadow_mu[i], self.lam.copy())
            if success:
                p[mask] = w

        nz = p > 0
        if not np.any(nz):
            return
        p_nz = p[nz] / p[nz].sum()
        q_nz = self.w[nz] / self.w[nz].sum()
        kl = float(np.sum(p_nz * np.log(p_nz / q_nz)))

        self.kl_accum += kl

    def merge_children(self, current_t=0):
        assert len(self.children) == 2
        left, right = self.children

        # Conserve moments exactly; refit on parent box.
        self.mu  = left.mu + right.mu
        self.w, self.lam, self.x_s, self.y_s, self.z_s = fit_maxent_weights(self.mu, self.xbounds, self.ybounds, self.zbounds, self.lam)
        self.update_shadows()

        self.children         = []
        self.kl_accum         = 0.0
        self.created_at       = current_t
        self.rate_ema         = np.inf       # reseed; new state, new measurement

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
        self.kl_accum  = 0.0                 # fresh start, don't trigger immediate re-split

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

def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)

    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)

    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu

def _f0_half_kl(f0, xb, yb, zb, n_fine):
    """
    Fit Maxent on the (xb, yb, zb) half-leaf to f0-derived moments and
    return KL(f0 || maxent) on the leaf's master-grid samples for that
    half. Returns +inf if the fit fails or the half is empty.
    """
    v0 = np.linspace(xb[0], xb[1], n_fine)
    v1 = np.linspace(yb[0], yb[1], n_fine)
    v2 = np.linspace(zb[0], zb[1], n_fine)
    ccx, ccy, ccz = np.meshgrid(v0, v1, v2, indexing='ij')
    f_slice = f0(ccx, ccy, ccz)
    mu_h = calc_moment(f_slice, ccx, ccy, ccz, v0, v1, v2)
    if mu_h[0] < 1e-4:
        return np.inf
    res = fit_maxent_weights(mu_h, xb, yb, zb, np.zeros(5))
    if res is None:
        return np.inf
    w_h, _, xs, ys, zs = res
    dv = ((xb[1]-xb[0])*(yb[1]-yb[0])*(zb[1]-zb[0])) / (n_fine - 1)**3
    f0_w = f0(xs, ys, zs) * dv
    s = float(np.sum(f0_w))
    if s < 1e-12:
        return np.inf
    p = f0_w / s
    q = w_h / np.sum(w_h)
    nz = (p > 0) & (q > 0)
    if not np.any(nz):
        return np.inf
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))


def initial_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, dS_threshold, max_passes=10):
    """
    Refine the AMR tree based on entropy difference on each leaf.
    f0-vs-Maxent KL on the parent leaf is the trigger; the split axis is
    picked adaptively as argmin over candidate axes of the sum of per-half
    KL(f0 || half-Maxent). Runs until no splits occur or max_passes hit.
    """
    n_fine = 50  # accuracy of f0 moment integrals
    for pass_idx in range(max_passes):
        leaves = root.get_leaves()
        splits_this_pass = 0

        for leaf in leaves:
            xb = leaf.xbounds
            yb = leaf.ybounds
            zb = leaf.zbounds

            # f0 fine grid for parent moments + parent KL
            v0 = np.linspace(xb[0], xb[1], n_fine)
            v1 = np.linspace(yb[0], yb[1], n_fine)
            v2 = np.linspace(zb[0], zb[1], n_fine)
            ccx, ccy, ccz = np.meshgrid(v0, v1, v2, indexing='ij')
            f_slice = f0(ccx, ccy, ccz)

            mu = calc_moment(f_slice, ccx, ccy, ccz, v0, v1, v2)
            leaf.mu = mu
            leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = fit_maxent_weights(
                mu, leaf.xbounds, leaf.ybounds, leaf.zbounds, leaf.lam)

            # Parent f0-vs-Maxent KL — the split trigger.
            dv_parent = ((xb[1]-xb[0])*(yb[1]-yb[0])*(zb[1]-zb[0])) / (n_fine - 1)**3
            f0_weights = f0(leaf.x_s, leaf.y_s, leaf.z_s) * dv_parent
            p = f0_weights / np.sum(f0_weights)
            q = leaf.w / np.sum(leaf.w)
            nz = (p > 0) & (q > 0)   # 0 * log(0/q) → NaN; drop those samples
            kl = float(np.sum(p[nz] * np.log(p[nz] / q[nz])))

            if kl <= dS_threshold:
                # No split needed; still refresh shadows for runtime.
                leaf.update_shadows()
                continue

            # Adaptive axis pick: for each splittable axis, evaluate the
            # sum of KL(f0 || half-Maxent) over the two halves. Smallest
            # = the structural mismatch is most reduced by splitting along
            # this axis.
            axis_kls = np.full(3, np.inf)
            for d in range(3):
                if not leaf.can_split(d):
                    continue
                lo, hi = (xb, yb, zb)[d]
                mid = 0.5 * (lo + hi)
                hb_L = [list(xb), list(yb), list(zb)]; hb_L[d] = [lo, mid]
                hb_R = [list(xb), list(yb), list(zb)]; hb_R[d] = [mid, hi]
                kl_L = _f0_half_kl(f0, hb_L[0], hb_L[1], hb_L[2], n_fine)
                kl_R = _f0_half_kl(f0, hb_R[0], hb_R[1], hb_R[2], n_fine)
                axis_kls[d] = kl_L + kl_R

            d_pick = int(np.argmin(axis_kls))
            if not np.isfinite(axis_kls[d_pick]):
                reasons = {d: leaf.split_block_reason(d) for d in range(3)}
                print(f'Warning: cannot split depth={leaf.depth} '
                      f'bounds={leaf.xbounds}, reasons={reasons}')
                leaf.update_shadows()
                continue

            leaf.split_dim = d_pick
            leaf.update_shadows()  # for the chosen axis
            print(f'pass {pass_idx}: splitting depth={leaf.depth} '
                  f'bounds=(x={xb},y={yb},z={zb}), KL={kl:.4f}, '
                  f'axis={d_pick} (half_kl_sums={axis_kls})')
            leaf.split(current_t=0)
            splits_this_pass += 1

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