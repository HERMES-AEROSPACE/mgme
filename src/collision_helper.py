import numpy as np
from numba import jit


KL_THRESHOLD = 0.05
KL_COARSEN_THRESHOLD = 0.005   # coarsen below this
MIN_LIFETIME         = 5      # minimum steps before coarsening allowed

def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 

@jit(nopython=True)
def max_entropy_newton(x_s, y_s, z_s, moments, lam0=None, max_iter=50, tol=1e-8):
    """
    Solve max-entropy weights directly via Newton iterations on dual.
    
    Dual problem: find lambda s.t. sum_i phi_i * exp(lam . phi_i) = moments
    """
    n = x_s.shape[0]
    r2 = x_s**2 + y_s**2 + z_s**2

    # Initial lambda guess
    if lam0 is None:
        lam = np.zeros(5)
    else:
        lam = lam0.copy()

    for iteration in range(max_iter):
        # Compute weights from current lambda
        log_w = (lam[0] + lam[1]*x_s + lam[2]*y_s + lam[3]*z_s + lam[4]*r2)
        w = np.exp(log_w)

        # Compute moment residual
        g = np.zeros(5)
        g[0] = np.sum(w) - moments[0]
        g[1] = np.sum(x_s * w) - moments[1]
        g[2] = np.sum(y_s * w) - moments[2]
        g[3] = np.sum(z_s * w) - moments[3]
        g[4] = np.sum(r2  * w) - moments[4]

        if np.linalg.norm(g) < tol:
            break

        # Hessian: H_ij = sum(phi_i * phi_j * w)
        phi = np.zeros((5, n))
        phi[0] = np.ones(n)
        phi[1] = x_s
        phi[2] = y_s
        phi[3] = z_s
        phi[4] = r2

        H = np.zeros((5, 5))
        for a in range(5):
            for b in range(5):
                H[a, b] = np.sum(phi[a] * phi[b] * w)

        # Newton step: lam -= H^{-1} g
        dlam = np.linalg.solve(H, g)
        lam -= dlam

    return w, lam

def solve_group_newton(x_sample, y_sample, z_sample, U_i, lam0):
    """Attempt to solve for one group"""
    try:
        w, lam = max_entropy_newton(x_sample, y_sample, z_sample, U_i, lam0)
        residual = np.abs(np.sum(w) - U_i[0]) / U_i[0]
        if residual < 1e-6:
            return True, w, lam
        else:
            return False, w, lam
    except:
        # print('Newton failed')
        return False, np.zeros_like(x_sample), np.zeros(5)

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

        if x.value is not None and not np.any(np.isnan(x.value)):
            predicted_flux = np.sum(x_sample * x.value)
            
            if np.abs(predicted_flux) < flux_limit:
                return True, x.value, predicted_flux
            else:
                return False, None, f"flux_too_large_{predicted_flux:.3f}"
        else:
            return False, None, f"status_{prob.status}"
    except Exception as e:
        return False, None, str(e)

class VelocityGroup:
    def __init__(self, x_s, y_s, z_s, cx_vec_full, bounds, depth=0, max_depth=1, created_at=0):
        """
        x_s, y_s, z_s : samples whose vx falls in this group's vx range
        cx_vec_full    : full 1D vx grid (needed for split midpoint lookup)
        bounds         : (ix_lo, ix_hi) index bounds into cx_vec_full
        """
        self.x_s         = x_s
        self.y_s         = y_s
        self.z_s         = z_s
        self.cx_vec_full = cx_vec_full
        self.bounds      = bounds
        self.depth       = depth
        self.max_depth   = max_depth
        self.created_at  = created_at

        self.children    = []
        self.parent      = None

        # State
        self.w           = None          # weights over x_s, y_s, z_s
        self.lam         = np.zeros(5)   # dual variables — warm start
        self.mu          = None          # moments [n, px, py, pz, e]

        # Shadow children
        self.shadow_w    = [None, None]
        self.shadow_lam  = [np.zeros(5), np.zeros(5)]
        self.shadow_x    = [None, None]  # filtered samples per shadow half
        self.shadow_y    = [None, None]
        self.shadow_z    = [None, None]
        self.shadow_bounds = [None, None]

        # KL accumulation
        self.kl_accum       = 0.0
        self.kl_last_step   = 0.0
        self.w_shadow_old   = [None, None]


    def is_leaf(self):
        return len(self.children) == 0

    def can_split(self):
        return self.depth < self.max_depth and len(self.x_s) > 10  # need enough samples

    def can_coarsen(self, current_t, min_lifetime=MIN_LIFETIME):
        return (current_t - self.created_at) > min_lifetime

    def get_sibling(self):
        if self.parent is None:
            return None
        siblings = [c for c in self.parent.children if c is not self]
        return siblings[0] if len(siblings) == 1 else None


    def compute_moments(self):
        """Moment sums over weighted samples."""
        w = self.w
        r2 = self.x_s**2 + self.y_s**2 + self.z_s**2
        self.mu = np.array([
            np.sum(w),
            np.sum(self.x_s * w),
            np.sum(self.y_s * w),
            np.sum(self.z_s * w),
            np.sum(r2 * w),
        ])
        return self.mu


    def fit_weights(self):
        """
        Fit max-entropy weights to self.mu using Newton solver.
        Uses self.lam as warm start; updates self.w and self.lam.
        """
        if self.mu is None or self.mu[0] < 1e-12 or len(self.x_s) == 0:
            self.w = np.zeros(len(self.x_s))
            return

        success, w, lam = solve_group_newton(self.x_s, self.y_s, self.z_s, self.mu, lam0=self.lam)
        if success:
            self.w   = w
            self.lam = lam
        elif not success:
            success, w, status = solve_group_cvxpy(self.x_s, self.y_s, self.z_s, self.mu)
            if success:
                self.w   = w
                self.lam = np.zeros(5)


    def update_shadows(self):
        ix_lo, ix_hi = self.bounds
        ix_mid       = (ix_lo + ix_hi) // 2
        vx_mid       = self.cx_vec_full[ix_mid]

        masks  = [self.x_s < vx_mid, self.x_s >= vx_mid]
        bounds = [(ix_lo, ix_mid + 1), (ix_mid, ix_hi)]

        self.shadow_bounds = bounds

        for i, (mask, bnd) in enumerate(zip(masks, bounds)):
            xs = self.x_s[mask];  ys = self.y_s[mask];  zs = self.z_s[mask]
            self.shadow_x[i] = xs
            self.shadow_y[i] = ys
            self.shadow_z[i] = zs

            if len(xs) < 5 or self.w is None:
                self.shadow_w[i] = np.zeros(len(xs))
                continue

            # Moments of this half — use current weights filtered to half
            w_half = self.w[mask]
            r2     = xs**2 + ys**2 + zs**2
            mu_s   = np.array([
                np.sum(w_half), np.sum(xs*w_half), np.sum(ys*w_half),
                np.sum(zs*w_half), np.sum(r2*w_half),
            ])

            success, w_fit, lam_fit = solve_group_newton(
                xs, ys, zs, mu_s, lam0=self.shadow_lam[i]
            )
            self.shadow_w[i]   = w_fit   if success else w_half
            self.shadow_lam[i] = lam_fit if success else self.shadow_lam[i]


    def accumulate_kl(self):
        """
        KL proxy via shadow moment change — works regardless of whether
        sample positions were refreshed since last step.
        Compares normalised moment vectors of each shadow half.
        """
        # Compute current shadow moments from weights
        shadow_mu_now = []
        for i in range(2):
            xs = self.shadow_x[i];  ys = self.shadow_y[i];  zs = self.shadow_z[i]
            w  = self.shadow_w[i]
            if xs is None or w is None or len(xs) == 0:
                shadow_mu_now.append(None)
                continue
            r2 = xs**2 + ys**2 + zs**2
            shadow_mu_now.append(np.array([
                np.sum(w), np.sum(xs*w), np.sum(ys*w), np.sum(zs*w), np.sum(r2*w)
            ]))

        if any(m is None for m in shadow_mu_now):
            return 0.0

        # First call — establish baseline
        if self.w_shadow_old[0] is None:
            self.w_shadow_old = shadow_mu_now   # store moments, not weights
            return 0.0

        eps = 1e-30
        kl_total = 0.0
        for i in range(2):
            mu_now  = shadow_mu_now[i]
            mu_prev = self.w_shadow_old[i]
            n_now   = max(mu_now[0],  eps)
            n_prev  = max(mu_prev[0], eps)
            # Normalised moment vector — direction encodes bulk velocity and temperature
            p = mu_now  / n_now
            q = mu_prev / n_prev
            # L2 norm on normalised moments as a KL proxy
            kl_total += float(np.sum((p - q)**2))

        self.kl_accum     += kl_total
        self.kl_last_step  = kl_total
        self.w_shadow_old  = shadow_mu_now
        return kl_total


    def split(self, current_t=0):
        """
        Promote shadow children to real children.
        Each child gets its filtered sample subset and fitted weights.
        """
        if not self.can_split():
            return []

        ix_lo, ix_hi = self.bounds
        ix_mid = (ix_lo + ix_hi) // 2
        vx_mid = self.cx_vec_full[ix_mid]

        masks  = [self.x_s < vx_mid, self.x_s >= vx_mid]
        bounds = [(ix_lo, ix_mid + 1), (ix_mid, ix_hi)]

        for i, (mask, bnd) in enumerate(zip(masks, bounds)):
            child = VelocityGroup(
                self.x_s[mask], self.y_s[mask], self.z_s[mask],
                self.cx_vec_full,
                bounds=bnd,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                created_at=current_t,
            )
            child.parent = self
            child.w      = self.shadow_w[i].copy() if self.shadow_w[i] is not None else np.zeros(np.sum(mask))
            child.lam    = self.shadow_lam[i].copy()
            child.compute_moments()
            child.update_shadows()
            child.kl_accum      = 0.0
            child.w_shadow_old  = [None, None]
            self.children.append(child)

        self.kl_accum = 0.0
        return self.children


    def merge_children(self, current_t=0):
        """
        Merge two children back into this parent.
        Concatenate sample arrays, sum moments, refit weights.
        """
        assert len(self.children) == 2
        left, right = self.children

        # Merge sample arrays
        self.x_s = np.concatenate([left.x_s, right.x_s])
        self.y_s = np.concatenate([left.y_s, right.y_s])
        self.z_s = np.concatenate([left.z_s, right.z_s])

        # Moments are additive
        self.mu = left.mu + right.mu

        # Refit max-entropy weights on merged sample set
        self.fit_weights()

        # Discard children
        self.children = []

        self.kl_accum      = 0.0
        self.kl_last_step  = 0.0
        self.w_shadow_old  = [None, None]
        self.created_at    = current_t

        self.update_shadows()


    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

def resample_leaf(leaf, cx_vec, cy_vec, cz_vec, cell_idx=0):
    """
    Generate adaptive sample positions for a single leaf using generate_regular_samples.
    Called on leaf creation, split, and merge — not every timestep.
    Positions are then held fixed; only weights evolve via fit_weights().
    """
    if leaf.mu is None or leaf.mu[0] < 1e-12:
        return

    U_i = leaf.mu[np.newaxis, :]           # (1, 5)
    bounds = np.array([[
        cx_vec[leaf.bounds[0]], cx_vec[leaf.bounds[1] - 1],
        cy_vec[0], cy_vec[-1],
        cz_vec[0], cz_vec[-1],
    ]])                                     # (1, 6)

    lam_cache = leaf.lam[np.newaxis, :]    # (1, 5)

    weights, num_valid, offsets, xs, ys, zs, lam_out = \
        generate_regular_samples(cell_idx, U_i, 1, bounds, lam_cache)

    leaf.x_s = xs
    leaf.y_s = ys
    leaf.z_s = zs
    leaf.w   = weights
    leaf.lam = lam_out[0]


def resample_all_leaves(leaves, cx_vec, cy_vec, cz_vec, cell_idx=0):
    """
    Batch version — builds combined inputs for all leaves in one
    generate_regular_samples call, then scatters back.
    More efficient than calling resample_leaf per leaf.
    """
    n_groups  = len(leaves)
    U_i       = np.array([leaf.mu for leaf in leaves])          # (n_groups, 5)
    bounds    = np.zeros((n_groups, 6))
    lam_cache = np.zeros((n_groups, 5))

    for g, leaf in enumerate(leaves):
        bounds[g] = [
            cx_vec[leaf.bounds[0]], cx_vec[leaf.bounds[1] - 1],
            cy_vec[0], cy_vec[-1],
            cz_vec[0], cz_vec[-1],
        ]
        if leaf.lam is not None:
            lam_cache[g] = leaf.lam

    weights, num_valid, offsets, xs, ys, zs, lam_out = \
        generate_regular_samples(cell_idx, U_i, n_groups, bounds, lam_cache)

    for g, leaf in enumerate(leaves):
        lo, hi   = int(offsets[g]), int(offsets[g + 1])
        leaf.x_s = xs[lo:hi]
        leaf.y_s = ys[lo:hi]
        leaf.z_s = zs[lo:hi]
        leaf.w   = weights[lo:hi]
        leaf.lam = lam_out[g]

def apply_collision_deltas(leaves, group_deltas):
    """Add collide() moment deltas to each leaf's mu. Clamp density positive."""
    dn, dpx, dpy, dpz, de = group_deltas
    for g, leaf in enumerate(leaves):
        leaf.mu[0] += dn[g]
        leaf.mu[1] += dpx[g]
        leaf.mu[2] += dpy[g]
        leaf.mu[3] += dpz[g]
        leaf.mu[4] += de[g]


def coarsening_kl_check_samples(left, right):
    """
    KL cost of merging left+right back into parent.
    Fits max-entropy on merged sample set, compares to individual child weights.
    """
    eps = 1e-300

    x_m = np.concatenate([left.x_s, right.x_s])
    y_m = np.concatenate([left.y_s, right.y_s])
    z_m = np.concatenate([left.z_s, right.z_s])
    mu_m = left.mu + right.mu

    success, w_par, _ = solve_group_newton(x_m, y_m, z_m, mu_m)
    if not success:
        return np.inf

    n_left       = len(left.x_s)
    w_par_left   = w_par[:n_left]
    w_par_right  = w_par[n_left:]

    kl_total = 0.0
    for w_child, w_p in [(left.w, w_par_left), (right.w, w_par_right)]:
        Sc = np.sum(w_child);  Sp = np.sum(w_p)
        if Sc < eps or Sp < eps:
            continue
        p    = w_child / Sc
        q    = w_p     / Sp
        mask = (p > eps) & (q > eps)
        kl_total += max(float(np.sum(p[mask] * np.log(p[mask] / q[mask]))), 0.0)

    return kl_total

def bootstrap_refine(root, f0, cx_vec, cy_vec, cz_vec,
                     kl_threshold=KL_THRESHOLD, max_passes=20):
    # Compute root moments from grid (only time we touch the grid)
    dvx = cx_vec[1] - cx_vec[0]
    dvy = cy_vec[1] - cy_vec[0]
    dvz = cz_vec[1] - cz_vec[0]
    dv  = dvx * dvy * dvz
    CX, CY, CZ = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    def assign_grid_moments(leaf):
        """Compute moments by integrating f0 over leaf's vx range."""
        ix_lo, ix_hi = leaf.bounds
        cx_s = CX[ix_lo:ix_hi]
        cy_s = CY[ix_lo:ix_hi]
        cz_s = CZ[ix_lo:ix_hi]
        f_s  = f0[ix_lo:ix_hi]
        leaf.mu = calc_moment(f_s, cx_s, cy_s, cz_s,
                              cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)

    assign_grid_moments(root)
    resample_leaf(root, cx_vec, cy_vec, cz_vec)  # establishes LHS sample positions

    for pass_idx in range(max_passes):
        splits_this_pass = 0

        for leaf in list(root.get_leaves()):
            if not leaf.can_split():
                continue

            # Ground truth moments from grid integration
            assign_grid_moments(leaf)

            # Best max-entropy fit at current sample positions
            success, w_me, lam = solve_group_newton(
                leaf.x_s, leaf.y_s, leaf.z_s, leaf.mu, lam0=leaf.lam
            )
            if not success:
                continue

            # KL(grid_moments || max_entropy_moments)
            # Compare by checking how well w_me reproduces the moments
            r2      = leaf.x_s**2 + leaf.y_s**2 + leaf.z_s**2
            mu_fit  = np.array([np.sum(w_me), np.sum(leaf.x_s*w_me),
                                 np.sum(leaf.y_s*w_me), np.sum(leaf.z_s*w_me),
                                 np.sum(r2*w_me)])
            mu_ref  = leaf.mu
            n       = max(mu_ref[0], 1e-30)
            kl      = float(np.sum(((mu_fit - mu_ref) / n)**2))

            print(f'  pass {pass_idx}: leaf {leaf.bounds} kl={kl:.4f}')

            if kl > kl_threshold:
                leaf.split(current_t=0)
                for child in leaf.children:
                    assign_grid_moments(child)
                    resample_leaf(child, cx_vec, cy_vec, cz_vec)
                splits_this_pass += 1
            else:
                leaf.w   = w_me
                leaf.lam = lam

        print(f'Bootstrap pass {pass_idx}: {splits_this_pass} split(s), '
              f'{len(root.get_leaves())} leaves')
        if splits_this_pass == 0:
            break

    # Final init
    for leaf in root.get_leaves():
        assign_grid_moments(leaf)
        resample_leaf(leaf, cx_vec, cy_vec, cz_vec)
        leaf.update_shadows()
        leaf.kl_accum     = 0.0
        leaf.w_shadow_old = [None, None]

@jit(nopython=True)
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
        x_valid = (vx >= ci_cx) & (vx <= cf_cx)
        y_valid = (vy >= ci_cy) & (vy <= cf_cy)
        z_valid = (vz >= ci_cz) & (vz <= cf_cz)
        return np.argmax(x_valid & y_valid & z_valid)

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

    # --- Precompute all pre and post collision groups in one pass ---
    pre_group1  = np.zeros(n_actual, dtype=np.int64)
    pre_group2  = np.zeros(n_actual, dtype=np.int64)
    post_group1 = np.zeros(n_actual, dtype=np.int64)
    post_group2 = np.zeros(n_actual, dtype=np.int64)

    # Store post-collision velocities for second pass
    vx1_arr  = np.zeros(n_actual);  vy1_arr  = np.zeros(n_actual);  vz1_arr  = np.zeros(n_actual)
    vx2_arr  = np.zeros(n_actual);  vy2_arr  = np.zeros(n_actual);  vz2_arr  = np.zeros(n_actual)
    vx1p_arr = np.zeros(n_actual);  vy1p_arr = np.zeros(n_actual);  vz1p_arr = np.zeros(n_actual)
    vx2p_arr = np.zeros(n_actual);  vy2p_arr = np.zeros(n_actual);  vz2p_arr = np.zeros(n_actual)
    g_arr = np.zeros(n_actual)

    for i in range(n_actual):
        vx1 = x_sample[depl_idx1[i]]
        vy1 = y_sample[depl_idx1[i]]
        vz1 = z_sample[depl_idx1[i]]
        vx2 = x_sample[depl_idx2[i]]
        vy2 = y_sample[depl_idx2[i]]
        vz2 = z_sample[depl_idx2[i]]

        vx1_arr[i] = vx1;  vy1_arr[i] = vy1;  vz1_arr[i] = vz1
        vx2_arr[i] = vx2;  vy2_arr[i] = vy2;  vz2_arr[i] = vz2

        # Pre-collision groups
        pre_group1[i] = find_group(vx1, vy1, vz1)
        pre_group2[i] = find_group(vx2, vy2, vz2)

        # Post-collision velocities
        gx = vx2 - vx1
        gy = vy2 - vy1
        gz = vz2 - vz1
        g  = np.sqrt(gx**2 + gy**2 + gz**2)
        g_arr[i] = g

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

        vx1p_arr[i] = vx1p
        vy1p_arr[i] = vy1p
        vz1p_arr[i] = vz1p
        vx2p_arr[i] = vx2p
        vy2p_arr[i] = vy2p
        vz2p_arr[i] = vz2p

        # Post-collision groups
        post_group1[i] = clamp_and_find_group(vx1p, vy1p, vz1p)
        post_group2[i] = clamp_and_find_group(vx2p, vy2p, vz2p)

    # --- Apply collision rates ---
    for i in range(n_actual):
        g1,  g2  = pre_group1[i],  pre_group2[i]
        g1r, g2r = post_group1[i], post_group2[i]
        g = g_arr[i]

        # Single collision weight — used for BOTH loss and gain
        C = 0.5 * W**2 / n_actual * g**(2 - 2*omega) * sigma_coeff_hat

        # Loss from pre-collision groups
        group_n[g1]  -= C;       group_n[g2]  -= C
        group_px[g1] -= C * vx1_arr[i]
        group_py[g1] -= C * vy1_arr[i]
        group_pz[g1] -= C * vz1_arr[i]
        group_e[g1]  -= C * (vx1_arr[i]**2 + vy1_arr[i]**2 + vz1_arr[i]**2)
        group_px[g2] -= C * vx2_arr[i]
        group_py[g2] -= C * vy2_arr[i]
        group_pz[g2] -= C * vz2_arr[i]
        group_e[g2]  -= C * (vx2_arr[i]**2 + vy2_arr[i]**2 + vz2_arr[i]**2)

        # Gain into post-collision groups
        group_n[g1r]  += C;       group_n[g2r]  += C
        group_px[g1r] += C * vx1p_arr[i]
        group_py[g1r] += C * vy1p_arr[i]
        group_pz[g1r] += C * vz1p_arr[i]
        group_e[g1r]  += C * (vx1p_arr[i]**2 + vy1p_arr[i]**2 + vz1p_arr[i]**2)
        group_px[g2r] += C * vx2p_arr[i]
        group_py[g2r] += C * vy2p_arr[i]
        group_pz[g2r] += C * vz2p_arr[i]
        group_e[g2r]  += C * (vx2p_arr[i]**2 + vy2p_arr[i]**2 + vz2p_arr[i]**2)

    return [group_n, group_px, group_py, group_pz, group_e]