import numpy as np
from numba import jit
from scipy.stats import qmc
from scipy import special


KL_THRESHOLD = 0.005
KL_COARSEN_THRESHOLD = 0.01   # coarsen below this
MIN_LIFETIME         = 5      # minimum steps before coarsening allowed

def calculate_velocity_grid(velocity_space):
    # Helper function to get velocity space grid
    cx_vec = np.linspace(*velocity_space['cx_range'], velocity_space['num_cx'])
    cy_vec = np.linspace(*velocity_space['cy_range'], velocity_space['num_cy'])
    cz_vec = np.linspace(*velocity_space['cz_range'], velocity_space['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 

def generate_grid(bounds_list, num_groups):
    num_samples = np.zeros(num_groups)
    for i in range(0, num_groups):
        volume = (bounds_list[i, 1] - bounds_list[i, 0]) * \
            (bounds_list[i, 3] - bounds_list[i, 2]) * \
            (bounds_list[i, 5] - bounds_list[i, 4])
        num_samples[i] = np.max((300, int(np.ceil(15 * volume))))
    
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

def solve_group_newton(x_s, y_s, z_s, moments, lam0, max_iter=50, tol=1e-6):
    """
    Solve max-entropy weights directly via Newton iterations on dual.
    
    Dual problem: find lambda s.t. sum_i phi_i * exp(lam . phi_i) = moments
    """
    n = x_s.shape[0]
    r2 = x_s**2 + y_s**2 + z_s**2

    # Initial lambda guess
    lam = lam0.copy()

    converged = False
    for iteration in range(max_iter):
        # Compute weights from current lambda
        log_w = (lam[0] + lam[1]*x_s + lam[2]*y_s + lam[3]*z_s + lam[4]*r2)
        # log_w = np.clip(log_w, None, 500.0)
        w = np.exp(log_w)

        # Compute moment residual
        g = np.zeros(5)
        g[0] = np.sum(w) - moments[0]
        g[1] = np.sum(x_s * w) - moments[1]
        g[2] = np.sum(y_s * w) - moments[2]
        g[3] = np.sum(z_s * w) - moments[3]
        g[4] = np.sum(r2  * w) - moments[4]

        if np.linalg.norm(g) < tol:
            converged = True
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

    return w, lam, converged

def coarsening_kl_analytic(left, right, cx_vec, cy_vec, cz_vec):
    """
    KL cost of merging: KL(f_left || f_parent) + KL(f_right || f_parent)
    Parent Maxwellian fitted to merged moments analytically.
    """
    mu_m = left.mu + right.mu
    if mu_m[0] < 1e-12:
        return 0.0

    parent_bounds = (left.bounds[0], right.bounds[1])

    def lam_from_mu(mu):
        ux = mu[1]/mu[0]; uy = mu[2]/mu[0]; uz = mu[3]/mu[0]
        thermal = mu[4]/mu[0] - ux**2 - uy**2 - uz**2
        beta = 1.5 / max(thermal, 1e-10)
        return np.array([0.0, 2*beta*ux, 2*beta*uy, 2*beta*uz, -beta])

    lam_p = lam_from_mu(mu_m)

    kl_total = 0.0
    for child in [left, right]:
        if child.mu[0] < 1e-12:
            continue
        lam_o = lam_from_mu(child.mu)   # ← moments-derived, not child.lam
        kl_total += _kl_gaussian_static(
            lam_o, child.mu, lam_p, mu_m,
            cx_vec, cy_vec, cz_vec,
            bounds_old=child.bounds,
            bounds_new=parent_bounds)

    return kl_total


def _kl_gaussian_static(lam_old, mu_old, lam_new, mu_new,
                         cx_vec, cy_vec, cz_vec, bounds_old, bounds_new=None):
    """Standalone version of _kl_gaussian for use outside VelocityGroup."""
    if bounds_new is None:
        bounds_new = bounds_old
        
    beta_o = -lam_old[4];  beta_n = -lam_new[4]
    wx_o   = -lam_old[1] / (2*lam_old[4]);  wx_n = -lam_new[1] / (2*lam_new[4])
    wy_o   = -lam_old[2] / (2*lam_old[4]);  wy_n = -lam_new[2] / (2*lam_new[4])
    wz_o   = -lam_old[3] / (2*lam_old[4]);  wz_n = -lam_new[3] / (2*lam_new[4])

    n_o = mu_old[0];  n_n = mu_new[0]
    if n_o < 1e-12 or n_n < 1e-12:
        return 0.0

    sqb_o = np.sqrt(beta_o);  sqb_n = np.sqrt(beta_n)

    ix_lo_o, ix_hi_o = bounds_old
    ix_lo_n, ix_hi_n = bounds_new   #

    I0x_o = _erf_integral(cx_vec[ix_lo_o], cx_vec[ix_hi_o-1], wx_o, sqb_o)
    I0y_o = _erf_integral(cy_vec[0], cy_vec[-1], wy_o, sqb_o)
    I0z_o = _erf_integral(cz_vec[0], cz_vec[-1], wz_o, sqb_o)
    I0x_n = _erf_integral(cx_vec[ix_lo_n], cx_vec[ix_hi_n-1], wx_n, sqb_n)
    I0y_n = _erf_integral(cy_vec[0], cy_vec[-1], wy_n, sqb_n)
    I0z_n = _erf_integral(cz_vec[0], cz_vec[-1], wz_n, sqb_n)

    if (abs(I0x_o*I0y_o*I0z_o) < 1e-30 or
            abs(I0x_n*I0y_n*I0z_n) < 1e-30):
        return 0.0

    A_o = n_o / (I0x_o * I0y_o * I0z_o)
    A_n = n_n / (I0x_n * I0y_n * I0z_n)

    if A_o <= 0 or A_n <= 0:
        return 0.0

    ln_A_ratio = np.log(A_o / A_n)

    kl = (n_o * ln_A_ratio
          + (beta_n - beta_o) * mu_old[4]
          + 2*(beta_o*wx_o - beta_n*wx_n) * mu_old[1]
          + 2*(beta_o*wy_o - beta_n*wy_n) * mu_old[2]
          + 2*(beta_o*wz_o - beta_n*wz_n) * mu_old[3]
          - n_o*(beta_o*(wx_o**2+wy_o**2+wz_o**2)
               - beta_n*(wx_n**2+wy_n**2+wz_n**2)))

    return max(kl, 0.0)

def _erf_integral(lo, hi, w, sqb):
    """Truncated Gaussian integral: integral of exp(-beta*(v-w)^2) dv over [lo, hi]."""
    return 0.5 * np.sqrt(np.pi) / sqb * (special.erf(sqb*(hi-w)) - special.erf(sqb*(lo-w)))

def _warm_start_from_moments(mu):
    if mu[0] < 1e-12:
        return np.zeros(5)
    ux = mu[1]/mu[0]; uy = mu[2]/mu[0]; uz = mu[3]/mu[0]
    thermal = mu[4]/mu[0] - ux**2 - uy**2 - uz**2
    beta = 1.5 / max(thermal, 1e-10)
    return np.array([0.0, 2*beta*ux, 2*beta*uy, 2*beta*uz, -beta])

def kl_div(p, q, cx_vec, cy_vec, cz_vec):
    mask   = (p > 0) & (q > 1e-8)
    safe_p = np.where(mask, p, 1.0)   # avoids log(0)
    safe_q = np.where(mask, q, 1.0)   # avoids div by zero
    integrand = np.where(mask, safe_p * np.log(safe_p / safe_q), 0.0)
    kl = np.trapezoid(np.trapezoid(np.trapezoid(integrand, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return max(kl, 0.0)

class VelocityGroup:
    def __init__(self, x_s, y_s, z_s, cx_vec_slice, cy_vec, cz_vec, bounds, depth=0, max_depth=1, created_at=0):
        """
        bounds: (cx_lo, cx_hi) index bounds into the full grid
        """
        self.bounds      = bounds        # (ix_lo, ix_hi) index bounds in cx_vec
        self.cx_vec      = cx_vec_slice  # 1D velocity vector for this group
        self.cy_vec      = cy_vec
        self.cz_vec      = cz_vec
        self.x_s         = x_s   # sample arrays inside this group
        self.y_s         = y_s
        self.z_s         = z_s

        self.depth       = depth
        self.max_depth   = max_depth
        self.created_at  = created_at

        self.children    = []            # empty = leaf node
        self.parent      = None

        # State
        self.w           = None          # current fitted max entropy weights
        self.lam         = np.zeros(5)   # lagrange multipliers
        self.mu          = None          # moments

        # Shadow children — trial split, always two halves in vx
        self.shadow_lam     = [np.zeros(5), np.zeros(5)]   # multipliers for left/right shadow
        self.shadow_w       = [None, None]   # weights for left/right shadow
        self.shadow_bounds  = [None, None]   # index bounds for each shadow
        self.shadow_x = [None, None]
        self.shadow_y = [None, None]
        self.shadow_z = [None, None]

        # Accumulation
        self.kl_accum      = 0.0
        self.kl_last_step  = 0.0
        self.shadow_state_old = [(None, None), (None, None)]


    def is_leaf(self):
        return len(self.children) == 0

    
    def can_split(self):
        return self.depth < self.max_depth


    def can_coarsen(self, current_t, min_lifetime=MIN_LIFETIME):
        """
        Node must have existed for min_lifetime steps before coarsening.
        """
        return (current_t - self.created_at) > min_lifetime


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


    def compute_moments_from_grid(self, f_slice, cx_s, cy_s, cz_s, cx_vec_s, cy_vec, cz_vec):
        """
        Only used when calculating the initial refinement by integrating against a known f0.
        """
        mu = np.zeros(5)
        mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f_slice, cz_vec, axis=2), cy_vec, axis=1), cx_vec_s, axis=0)
        mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx_s * f_slice, cz_vec, axis=2), cy_vec, axis=1), cx_vec_s, axis=0)
        mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy_s * f_slice, cz_vec, axis=2), cy_vec, axis=1), cx_vec_s, axis=0)
        mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz_s * f_slice, cz_vec, axis=2), cy_vec, axis=1), cx_vec_s, axis=0)
        r2    = cx_s**2 + cy_s**2 + cz_s**2
        mu[4] = np.trapezoid(np.trapezoid(np.trapezoid(r2 * f_slice, cz_vec, axis=2), cy_vec, axis=1), cx_vec_s, axis=0)
        self.mu = mu


    def compute_moments(self, xs=None, ys=None, zs=None, ws=None):
        """Moments from current weighted samples."""
        xs = xs if xs is not None else self.x_s
        ys = ys if ys is not None else self.y_s
        zs = zs if zs is not None else self.z_s
        ws = ws if ws is not None else self.w

        if ws is None or len(xs) == 0:
            mu = np.zeros(5)
            if xs is self.x_s:
                self.mu = mu
            return mu

        r2 = xs**2 + ys**2 + zs**2
        mu = np.array([np.sum(ws), np.sum(xs*ws), np.sum(ys*ws), np.sum(zs*ws), np.sum(r2*ws)])
        
        if xs is self.x_s:
            self.mu = mu
        return mu


    def fit_maxent_weights(self):
        solution, lam, success = solve_group_newton(self.x_s, self.y_s, self.z_s, self.mu, self.lam)
        if success:
            self.w = solution
            self.lam = lam
        if not success:
            solution, lam, success = solve_group_newton(self.x_s, self.y_s, self.z_s, self.mu, np.zeros(5))
            if success:
                self.w = solution
                self.lam = lam
            else:
                print('die')


    def update_shadows(self, cx_vec_full):
        """
        Split samples at vx midpoint, fit Newton weights on each half.
        Shadow lam stored for KL checks and split initialization.
        """
        ix_lo, ix_hi = self.bounds
        ix_mid        = (ix_lo + ix_hi) // 2
        vx_mid        = cx_vec_full[ix_mid]

        self.shadow_bounds = [(ix_lo, ix_mid + 1), (ix_mid, ix_hi)]
        masks = [self.x_s < vx_mid, self.x_s >= vx_mid]

        for i, mask in enumerate(masks):
            xs = self.x_s[mask]
            ys = self.y_s[mask]
            zs = self.z_s[mask]

            self.shadow_x[i] = xs
            self.shadow_y[i] = ys
            self.shadow_z[i] = zs

            # Moments of this half from current weights
            w_half = self.w[mask]
            r2     = xs**2 + ys**2 + zs**2
            mu_s   = self.compute_moments(xs, ys, zs, w_half)

            if mu_s[0] < 1e-6:
                self.shadow_w[i]   = np.zeros(len(xs))
                self.shadow_lam[i] = np.zeros(5)
                continue
            lam_init = _warm_start_from_moments(mu_s)
            w_fit, lam_fit, success = solve_group_newton(xs, ys, zs, mu_s, lam_init)
            self.shadow_w[i]   = w_fit   if success else w_half
            self.shadow_lam[i] = lam_fit if success else np.zeros(5)


    def accumulate_kl(self, cx_vec, cy_vec, cz_vec):
        if self.shadow_x[0] is None or self.shadow_x[1] is None:
            return 0.0

        shadow_mu_now = [
            self.compute_moments(self.shadow_x[i], self.shadow_y[i],
                                self.shadow_z[i], self.shadow_w[i])
            for i in range(2)
        ]

        if self.shadow_state_old[0][0] is None:
            self.shadow_state_old = [
                (_warm_start_from_moments(shadow_mu_now[i]), shadow_mu_now[i].copy())
                for i in range(2)
            ]
            return 0.0

        kl_total = 0.0
        for i in range(2):
            mu_new = shadow_mu_now[i]
            lam_old, mu_old = self.shadow_state_old[i]

            if mu_old[0] < 1e-12 or mu_new[0] < 1e-12:
                continue

            # Derive lam from moments — no Newton noise
            lam_new = _warm_start_from_moments(mu_new)
            if lam_new[4] >= 0 or lam_old[4] >= 0:
                continue

            kl_total += self._kl_gaussian(
                lam_old, mu_old, lam_new, mu_new,
                cx_vec, cy_vec, cz_vec,
                self.shadow_bounds[i])

        self.kl_accum    += kl_total
        self.kl_last_step = kl_total
        self.shadow_state_old = [
            (_warm_start_from_moments(shadow_mu_now[i]), shadow_mu_now[i].copy())
            for i in range(2)
        ]
        return kl_total

    def _kl_gaussian(self, lam_old, mu_old, lam_new, mu_new, cx_vec, cy_vec, cz_vec, bounds):
        return _kl_gaussian_static(lam_old, mu_old, lam_new, mu_new, cx_vec, cy_vec, cz_vec, bounds)

    def split(self, cx_vec_full, current_t=0):
        """
        Promote shadow children to real children.
        Samples are partitioned by vx midpoint.
        Shadow weights/lam used to initialize children.
        """
        if not self.can_split():
            print(f'Warning: tried to split at max_depth={self.max_depth}')
            return []

        ix_lo, ix_hi = self.bounds
        ix_mid = (ix_lo + ix_hi) // 2
        vx_mid = cx_vec_full[ix_mid]

        masks  = [self.x_s < vx_mid, self.x_s >= vx_mid]
        bounds = [(ix_lo, ix_mid + 1), (ix_mid, ix_hi)]

        for i, (mask, bnd) in enumerate(zip(masks, bounds)):
            lo, hi   = bnd
            cx_vec_s = cx_vec_full[lo:hi]

            child = VelocityGroup(
                self.x_s[mask].copy(),
                self.y_s[mask].copy(),
                self.z_s[mask].copy(),
                cx_vec_s, self.cy_vec, self.cz_vec,
                bounds=bnd,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                created_at=current_t)
            child.parent = self

            # Initialize from shadow state
            child.w   = self.shadow_w[i].copy() if self.shadow_w[i] is not None \
                    else np.zeros(int(np.sum(mask)))
        
            # Reset lam to zeros if bad
            child.lam = self.shadow_lam[i].copy() if self.shadow_lam[i][4] < 0.0 else np.zeros(5)

            child.compute_moments()
            child.update_shadows(cx_vec_full)
            child.kl_accum         = 0.0
            child.shadow_state_old = [(None, None), (None, None)]
            self.children.append(child)

        self.kl_accum = 0.0
        return self.children
    

    def merge_children(self, cx_vec_full, current_t=0):
        assert len(self.children) == 2
        left, right = self.children

        # Concatenate samples
        self.x_s = np.concatenate([left.x_s, right.x_s])
        self.y_s = np.concatenate([left.y_s, right.y_s])
        self.z_s = np.concatenate([left.z_s, right.z_s])
        self.mu  = left.mu + right.mu
        self.lam = _warm_start_from_moments(self.mu)
        self.fit_maxent_weights()

        self.children         = []
        self.kl_accum         = 0.0
        self.kl_last_step     = 0.0
        self.shadow_state_old = [(None, None), (None, None)]
        self.created_at       = current_t
        self.update_shadows(cx_vec_full)


    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

def bootstrap_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, kl_threshold=KL_THRESHOLD, max_passes=20):
    """
    Refine the AMR tree based on goodness-of-fit KL(f0 || f_maxwellian)
    on each leaf. Runs until no splits occur or max_passes is reached.
    """
    for pass_idx in range(max_passes):
        leaves = root.get_leaves()
        splits_this_pass = 0

        for leaf in leaves:
            ix_lo, ix_hi = leaf.bounds

            # Recompute moments from f0 and calculate weights
            leaf.compute_moments_from_grid(f0[ix_lo:ix_hi], cx[ix_lo:ix_hi], cy[ix_lo:ix_hi], cz[ix_lo:ix_hi], cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)
            leaf.fit_maxent_weights()
            leaf.update_shadows(cx_vec)

            w_true = f0[ix_lo:ix_hi]

            beta = -leaf.lam[4]
            wx   = -leaf.lam[1] / (2*leaf.lam[4])
            wy   = -leaf.lam[2] / (2*leaf.lam[4])
            wz   = -leaf.lam[3] / (2*leaf.lam[4])

            cx_s = cx[ix_lo:ix_hi]
            cy_s = cy[ix_lo:ix_hi]
            cz_s = cz[ix_lo:ix_hi]

            w_fit_grid = np.exp(-beta * ((cx_s - wx)**2 + (cy_s - wy)**2 + (cz_s - wz)**2))

            n_true = np.trapezoid(np.trapezoid(np.trapezoid(w_true, cz_vec, axis=2), cy_vec, axis=1), cx_vec[ix_lo:ix_hi], axis=0)
            n_fit  = np.trapezoid(np.trapezoid(np.trapezoid(w_fit_grid, cz_vec, axis=2), cy_vec, axis=1), cx_vec[ix_lo:ix_hi], axis=0)
            w_fit_grid = w_fit_grid * (n_true / n_fit)

            kl = kl_div(w_true, w_fit_grid, cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)

            if kl > kl_threshold and leaf.can_split():
                print(f'  bootstrap pass {pass_idx}: splitting depth={leaf.depth} '
                      f'bounds={leaf.bounds}, kl={kl:.4f}')
                
                leaf.split(cx_vec, current_t=0)
                splits_this_pass += 1

        print(f'Bootstrap pass {pass_idx}: {splits_this_pass} split(s), '
              f'{len(root.get_leaves())} leaves total')

        if splits_this_pass == 0:
            print(f'Bootstrap converged after {pass_idx + 1} pass(es).')
            break

    # Final pass: initialize shadows on all leaves after tree is stable
    for leaf in root.get_leaves():
        ix_lo, ix_hi = leaf.bounds

        leaf.compute_moments_from_grid(f0[ix_lo:ix_hi], cx[ix_lo:ix_hi], cy[ix_lo:ix_hi], cz[ix_lo:ix_hi], cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)
        leaf.fit_maxent_weights()
        leaf.update_shadows(cx_vec)

        leaf.kl_accum = 0.0
        leaf.shadow_state_old = [(None, None), (None, None)]



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