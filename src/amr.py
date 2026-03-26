import numpy as np
from itertools import product
from scipy import optimize, special
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import qmc
from numba import jit
import time


MAX_DEPTH    = 4       # 0 = no splitting, 1 = one split etc.
KL_THRESHOLD = 0.02
KL_COARSEN_THRESHOLD = 0.005   # coarsen below this
MIN_LIFETIME         = 10      # minimum steps before coarsening allowed

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
        log_w = np.clip(log_w, None, 500.0)
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

def plot_amr_state(root, ft, cx_vec, cy_vec, cz_vec, t, ax_dist, ax_kl, kl_history):
    """
    Plot current distribution state and group structure.
    
    ax_dist: distribution + group boundaries
    ax_kl:   KL accumulation over time
    """
    ax_dist.cla()

    # True f — 1D marginal
    ft_1d = np.trapezoid(np.trapezoid(ft, cz_vec, axis=2), cy_vec, axis=1)
    ax_dist.plot(cx_vec, ft_1d, 'k-', linewidth=2, label=r'True $f$', zorder=5)

    # Each leaf — fitted Maxwellian and shadow distributions
    leaves = root.get_leaves()

    for i, leaf in enumerate(leaves):
        ix_lo, ix_hi = leaf.bounds
        cx_vec_leaf = cx_vec[ix_lo:ix_hi]

        # Recover continuous params from lam
        if leaf.lam[4] >= 0: continue
        beta = -leaf.lam[4]
        wx   = -leaf.lam[1] / (2*leaf.lam[4])
        wy   = -leaf.lam[2] / (2*leaf.lam[4])
        wz   = -leaf.lam[3] / (2*leaf.lam[4])

        # 1D marginal: integrate over vy, vz analytically
        sqb  = np.sqrt(beta)
        I0y  = 0.5*np.sqrt(np.pi/beta) * (special.erf(sqb*(cy_vec[-1]-wy))
                                           - special.erf(sqb*(cy_vec[0]-wy)))
        I0z  = 0.5*np.sqrt(np.pi/beta) * (special.erf(sqb*(cz_vec[-1]-wz))
                                           - special.erf(sqb*(cz_vec[0]-wz)))
        I0x  = 0.5*np.sqrt(np.pi/beta) * (special.erf(sqb*(cx_vec[ix_hi-1]-wx))
                                           - special.erf(sqb*(cx_vec[ix_lo]-wx)))

        if abs(I0x * I0y * I0z) < 1e-30:
            continue

        A        = leaf.mu[0] / (I0x * I0y * I0z)
        f_leaf_1d = A * I0y * I0z * np.exp(-beta * (cx_vec_leaf - wx)**2)
        ax_dist.plot(cx_vec_leaf, f_leaf_1d, '--', linewidth=1.8,
                     label=f'Leaf {i}', zorder=4)

        # Shadow boundaries and their Maxwellians
        for sb in leaf.shadow_bounds:
            if sb is None:
                continue
            lo, hi = sb
            ax_dist.axvline(x=cx_vec[lo], linewidth=0.8, linestyle=':',
                            alpha=0.7, color='gray')

        ax_dist.axvline(x=cx_vec[ix_hi-1], linewidth=1.5, linestyle='-',
                        alpha=0.9, color='black')

    ax_dist.set_xlabel(r'$v_x$', fontsize=18)
    ax_dist.set_ylabel(r'$f_{1D}(v_x)$', fontsize=18)
    ax_dist.set_title(f't = {t}', fontsize=14)
    ax_dist.set_xlim(cx_vec[0], cx_vec[-1])
    ax_dist.grid(True, alpha=0.2)

    # --- KL accumulation history ---
    ax_kl.cla()
    for bounds, data in kl_history.items():
        created_at = data['created_at']
        values     = data['values']
        timesteps  = np.arange(created_at, created_at + len(values))
        ax_kl.plot(timesteps, values, linewidth=1.8,
                   label=f'vx[{cx_vec[bounds[0]]:.1f}, {cx_vec[bounds[1]-1]:.1f}] '
                         f'depth={get_depth(root, bounds)}')

    ax_kl.axhline(y=KL_THRESHOLD, color='red', linestyle='--',
                  linewidth=1.5, label='Threshold')
    ax_kl.set_xlabel('Timestep', fontsize=18)
    ax_kl.set_ylabel('Accumulated KL', fontsize=18)
    ax_kl.set_title('KL accumulation per leaf', fontsize=14)
    ax_kl.grid(True, alpha=0.2)

def kl_div(p, q, cx_vec, cy_vec, cz_vec):
    kl = np.trapezoid(np.trapezoid(np.trapezoid(p * np.log(p / q), cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
    return max(kl, 0.0)

def get_depth(root, bounds):
    for leaf in root.get_leaves():
        if leaf.bounds == bounds:
            return leaf.depth
    return 0

def coarsening_kl_analytic(left, right, cx_vec, cy_vec, cz_vec):
    """
    KL cost of merging: KL(f_left || f_parent) + KL(f_right || f_parent)
    Parent Maxwellian fitted to merged moments analytically.
    """
    mu_m = left.mu + right.mu
    if mu_m[0] < 1e-12:
        return 0.0

    ux = mu_m[1]/mu_m[0]; uy = mu_m[2]/mu_m[0]; uz = mu_m[3]/mu_m[0]
    thermal = mu_m[4]/mu_m[0] - ux**2 - uy**2 - uz**2
    beta_p  = 1.5 / max(thermal, 1e-10)

    # Parent lam from merged moments
    lam_p    = np.zeros(5)
    lam_p[4] = -beta_p
    lam_p[1] = 2*beta_p*ux
    lam_p[2] = 2*beta_p*uy
    lam_p[3] = 2*beta_p*uz
    # lam_p[0] not needed — A recovered from density in _kl_gaussian

    kl_total = 0.0
    for child in [left, right]:
        if child.lam[4] >= 0 or not np.all(np.isfinite(child.lam)):
            return np.inf
        kl_total += _kl_gaussian_static(
            child.lam, child.mu, lam_p, mu_m,
            cx_vec, cy_vec, cz_vec, child.bounds)

    return kl_total


def _kl_gaussian_static(lam_old, mu_old, lam_new, mu_new,
                         cx_vec, cy_vec, cz_vec, bounds):
    """Standalone version of _kl_gaussian for use outside VelocityGroup."""
    beta_o = -lam_old[4];  beta_n = -lam_new[4]
    wx_o   = -lam_old[1] / (2*lam_old[4]);  wx_n = -lam_new[1] / (2*lam_new[4])
    wy_o   = -lam_old[2] / (2*lam_old[4]);  wy_n = -lam_new[2] / (2*lam_new[4])
    wz_o   = -lam_old[3] / (2*lam_old[4]);  wz_n = -lam_new[3] / (2*lam_new[4])

    n_o = mu_old[0];  n_n = mu_new[0]
    if n_o < 1e-12 or n_n < 1e-12:
        return 0.0

    ix_lo, ix_hi = bounds
    sqb_o = np.sqrt(beta_o);  sqb_n = np.sqrt(beta_n)

    I0x_o = _erf_integral(cx_vec[ix_lo], cx_vec[ix_hi-1], wx_o, sqb_o)
    I0y_o = _erf_integral(cy_vec[0], cy_vec[-1], wy_o, sqb_o)
    I0z_o = _erf_integral(cz_vec[0], cz_vec[-1], wz_o, sqb_o)
    I0x_n = _erf_integral(cx_vec[ix_lo], cx_vec[ix_hi-1], wx_n, sqb_n)
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

        # Compute current shadow moments
        shadow_mu_now = [
            self.compute_moments(self.shadow_x[i], self.shadow_y[i],
                                self.shadow_z[i], self.shadow_w[i])
            for i in range(2)
        ]

        if self.shadow_state_old[0][0] is None:
            self.shadow_state_old = [
                (self.shadow_lam[i].copy(), shadow_mu_now[i].copy())
                for i in range(2)
            ]
            return 0.0

        kl_total = 0.0
        for i in range(2):
            lam_new = self.shadow_lam[i]
            lam_old, mu_old = self.shadow_state_old[i]
            mu_new = shadow_mu_now[i]

            if lam_old is None: continue
            if lam_new[4] >= 0 or lam_old[4] >= 0: continue

            kl_total += self._kl_gaussian(
                lam_old, mu_old, lam_new, mu_new,
                cx_vec, cy_vec, cz_vec,
                self.shadow_bounds[i])

        self.kl_accum     += kl_total
        self.kl_last_step  = kl_total
        self.shadow_state_old = [
            (self.shadow_lam[i].copy(), shadow_mu_now[i].copy())
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
        
            # Reset lam to zeros — child will cold-start Newton against
            # grid moments on first time loop iteration, avoiding mismatch
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
            f_slice = f0[ix_lo:ix_hi]

            # Recompute moments from f0 and calculate weights
            leaf.compute_moments_from_grid(f_slice, cx[ix_lo:ix_hi], cy[ix_lo:ix_hi], cz[ix_lo:ix_hi], cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)
            leaf.fit_maxent_weights()
            leaf.update_shadows(cx_vec)

            w_true = f_slice

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
        f_slice = f0[ix_lo:ix_hi]
        leaf.compute_moments_from_grid(f_slice, cx[ix_lo:ix_hi], cy[ix_lo:ix_hi], cz[ix_lo:ix_hi], cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)

        leaf.fit_maxent_weights()
        leaf.update_shadows(cx_vec)
        leaf.kl_accum = 0.0
        leaf.shadow_state_old = [(None, None), (None, None)]

# ============================================================
# Setup
# ============================================================
cx_vec = np.linspace(-7, 7, 121)
cy_vec = np.linspace(-7, 7, 121)
cz_vec = np.linspace(-7, 7, 121)
cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

f0  = 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 3)**2 + cy**2 + cz**2))
f02 = 0.04 / (np.pi**1.5) * np.exp(-0.2 * ((cx + 1)**2 + cy**2 + cz**2))

# Build root node
bounds_list = np.array([[-7, 7, -7, 7, -7, 7]])
x_sample, y_sample, z_sample, _, _ = generate_grid(bounds_list, 1)
root = VelocityGroup(x_sample, y_sample, z_sample, cx_vec, cy_vec, cz_vec, bounds=(0, 121), depth=0, max_depth=MAX_DEPTH)

# Refine the root node using f0 knowledge.
bootstrap_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)

# ============================================================
# Time loop
# ============================================================
kl_history = {}   # {leaf_bounds_tuple: [kl_t0, kl_t1, ...]}
split_times = []  # record when splits happen
coarse_times = []

for t in range(0, 201):
    if t <= 100:
        # Phase 1: drift and mix toward non-Maxwellian
        alpha   = t / 100
        weight2 = 0.5 * alpha
        weight1 = 1 - weight2
        f01 = 1/(np.pi**1.5) * np.exp(-1*((cx - 3 + 0.05*t)**2 + cy**2 + cz**2))
        ft  = weight1 * f01 + weight2 * f02
    else:
        tau   = 30   # relaxation timescale in steps
        decay = np.exp(-(t - 100) / tau)
        f_eq  = 1/(np.pi**1.5) * np.exp(-1*(cx**2 + cy**2 + cz**2))
        ft    = decay * ft_end_phase1 + (1 - decay) * f_eq

    # Save ft at the end of phase 1 to use as starting point for phase 2
    if t == 100:
        ft_end_phase1 = ft.copy()

    # --- Update each leaf ---
    leaves = root.get_leaves()
    for leaf in leaves:
        ix_lo, ix_hi = leaf.bounds

        # Update moments from new ft on this leaf's domain
        f_slice = ft[ix_lo:ix_hi]
        leaf.compute_moments_from_grid(f_slice, cx[ix_lo:ix_hi], cy[ix_lo:ix_hi], cz[ix_lo:ix_hi], cx_vec[ix_lo:ix_hi], cy_vec, cz_vec)

        # Refit max entropy weights
        leaf.fit_maxent_weights()

        # Project onto shadow children
        leaf.update_shadows(cx_vec)

        # Accumulate KL
        leaf.accumulate_kl(cx_vec, cy_vec, cz_vec)

        # Track KL history per leaf
        key = leaf.bounds
        if key not in kl_history:
            kl_history[key] = {
                'created_at': leaf.created_at,
                'values':     []
            }
        kl_history[key]['values'].append(leaf.kl_accum)

    # --- Refinement check ---
    for leaf in list(root.get_leaves()):
        if leaf.kl_accum > KL_THRESHOLD:
            if leaf.can_split():
                print(f't={t}: splitting depth={leaf.depth} '
                    f'bounds={leaf.bounds}, kl={leaf.kl_accum:.4f}')
                split_times.append(t)
                leaf.split(cx_vec, t)
                for child in leaf.children:
                    print(f'child bounds={child.bounds} mu={child.mu}')
                
            else:
                # At max depth — log the signal but can't split
                print(f't={t}: max depth reached at bounds={leaf.bounds}, '
                    f'kl={leaf.kl_accum:.4f} — consider increasing MAX_DEPTH')

    # --- Coarsen check ---
    checked_parents = set()
    for leaf in list(root.get_leaves()):
        if leaf.parent is None:
            continue

        parent = leaf.parent
        if id(parent) in checked_parents:
            continue
        checked_parents.add(id(parent))

        if not (not parent.is_leaf() and len(parent.children) == 2):
            continue

        left_child, right_child = parent.children

        if not (left_child.is_leaf() and right_child.is_leaf()):
            continue

        if not (left_child.can_coarsen(t) and right_child.can_coarsen(t)):
            continue

        # Skip if either child has invalid lam
        if (left_child.lam is None  or left_child.lam[4]  >= 0 or
            right_child.lam is None or right_child.lam[4] >= 0):
            continue

        if (left_child.mu is None or left_child.mu[0]   < 1e-6 or
            right_child.mu is None or right_child.mu[0] < 1e-6):
            continue

        # Analytic KL cost of merging
        kl = coarsening_kl_analytic(left_child, right_child, cx_vec, cy_vec, cz_vec)

        if kl < KL_COARSEN_THRESHOLD:
            print(f't={t}: coarsening '
                f'{left_child.bounds}+{right_child.bounds}'
                f'->{parent.bounds}')
            coarse_times.append(t)
            parent.merge_children(cx_vec, current_t=t)

            key = parent.bounds
            kl_history[key] = {'created_at': t, 'values': [0.0]}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_amr_state(root, ft, cx_vec, cy_vec, cz_vec, t,
                    axes[0], axes[1], kl_history)

    for st in split_times:
        axes[1].axvline(x=st, color='black', linestyle='--',
                        linewidth=1.4, alpha=0.7)

    axes[0].tick_params(axis='both', labelsize=14)
    axes[1].tick_params(axis='both', labelsize=14)
    axes[0].set_ylim(0.0, 0.6)
    plt.tight_layout()
    plt.savefig(f'plots/amr_t{t:04d}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)  # important — free memory, don't accumulate figures

