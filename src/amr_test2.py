import numpy as np
from itertools import product
from scipy import optimize, special
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


MAX_DEPTH    = 4       # 0 = no splitting, 1 = one split etc.
KL_THRESHOLD = 0.05
KL_COARSEN_THRESHOLD = 0.005   # coarsen below this
MIN_LIFETIME         = 5      # minimum steps before coarsening allowed

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

        # Fitted Maxwellian — 1D marginal
        A, b, wx, wy, wz = leaf.params
        f_leaf_1d = A * (np.pi / b) * np.exp(-b * (cx_vec_leaf - wx)**2)
        ax_dist.plot(cx_vec_leaf, f_leaf_1d, '--', linewidth=1.8,
                     label=f'Leaf {i} Maxwellian', zorder=4)

        # Shadow boundaries and their Maxwellians
        for j, (sp, sb) in enumerate(zip(leaf.shadow_params, leaf.shadow_bounds)):
            if sp is None or sb is None:
                continue
            A_s, b_s, wx_s, _, _ = sp
            if A_s == 0: continue
            lo, hi = sb

            # Shadow boundary marker
            ax_dist.axvline(x=cx_vec[lo], linewidth=0.8, linestyle=':', alpha=0.7, color='black')

        ax_dist.axvline(x=cx_vec[ix_hi-1], linewidth=1.5, linestyle='-', alpha=0.9, color='black')

    ax_dist.set_xlabel(r'$v_x$', fontsize=18)
    ax_dist.set_ylabel(r'$f_{1D}(v_x)$', fontsize=18)
    ax_dist.set_title(f't = {t}', fontsize=14)
    # ax_dist.legend(fontsize=14, loc='upper left')
    ax_dist.set_xlim(cx_vec[0], cx_vec[-1])
    ax_dist.grid(True, alpha=0.2)

    # --- KL accumulation history ---
    ax_kl.cla()
    for i, (bounds, data) in enumerate(kl_history.items()):
        created_at = data['created_at']
        values     = data['values']

        timesteps = np.arange(created_at, created_at + len(values))

        ax_kl.plot(timesteps, values, linewidth=1.8, label=f'vx[{cx_vec[bounds[0]]:.1f}, {cx_vec[bounds[1]-1]:.1f}] '
                         f'depth={get_depth(root, bounds)}')

    ax_kl.axhline(y=KL_THRESHOLD, color='red', linestyle='--',
                  linewidth=1.5, label='Threshold')
    ax_kl.set_xlabel('Timestep', fontsize=18)
    ax_kl.set_ylabel('Accumulated KL', fontsize=18)
    ax_kl.set_title('KL accumulation per leaf', fontsize=14)
    # ax_kl.legend(fontsize=14)
    ax_kl.grid(True, alpha=0.2)

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
    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Momentum moment
    uk = cx * f
    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cy * f
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    uk = cz * f
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(uk, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    # Energy moment
    c2 = cx**2 + cy**2 + cz**2
    ek = c2 * f
    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid(ek, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return mu 

def invert(mu, initial_guess, group_bounds):
    A, b, wx, wy, wz = 0.0, 0.0, 0.0, 0.0, 0.0
    ci_cx = group_bounds['ci_cx']
    cf_cx = group_bounds['cf_cx']
    ci_cy = group_bounds['ci_cy']
    cf_cy = group_bounds['cf_cy']
    ci_cz = group_bounds['ci_cz']
    cf_cz = group_bounds['cf_cz']

    sol = optimize.least_squares(moment_eq, initial_guess, args=(mu[1] / mu[0], mu[2] / mu[0], \
                                    mu[3] / mu[0], mu[4] / mu[0], \
                                    ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), \
                                        bounds=([0.0, -10, -10, -10], [np.inf, 10, 10, 10]), method='trf', loss='soft_l1')
    # print(sol)
    # print('residual:', np.linalg.norm(sol.fun))
    
    # sol = optimize.root(moment_eq, initial_guess, args=(mu[1] / mu[0], mu[2] / mu[0], \
    #                                 mu[3] / mu[0], mu[4] / mu[0], \
    #                                     ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz), method='lm')
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

def kl_div(p, q, cx_vec, cy_vec, cz_vec):
    kl = np.trapezoid(np.trapezoid(np.trapezoid(p * np.log(p / q), cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)
    return max(kl, 0.0)

def get_depth(root, bounds):
    for leaf in root.get_leaves():
        if leaf.bounds == bounds:
            return leaf.depth
    return 0

def coarsening_kl_check(left, right, cx_full, cy_full, cz_full, cx_vec_full, cy_vec_full, cz_vec_full, invert_fn):
    """
    KL = integral_{left domain}  f_L * ln(f_L / f_parent) dv
       + integral_{right domain} f_R * ln(f_R / f_parent) dv
    """
    U_parent = left.mu + right.mu
    M0_total = U_parent[0]

    ix_lo = min(left.bounds[0],  right.bounds[0])
    ix_hi = max(left.bounds[1],  right.bounds[1])

    # Fit parent Maxwellian to merged moments
    group_bounds_dict = {
        'ci_cx': cx_vec_full[ix_lo],  'cf_cx': cx_vec_full[ix_hi - 1],
        'group_bounds_cx': np.array([ix_lo, ix_hi]),
        'ci_cy': cy_vec_full[0],      'cf_cy': cy_vec_full[-1],
        'group_bounds_cy': np.array([0, len(cy_vec_full)]),
        'ci_cz': cz_vec_full[0],      'cf_cz': cz_vec_full[-1],
        'group_bounds_cz': np.array([0, len(cz_vec_full)]),
    }

    try:
        A_p, b_p, wx_p, wy_p, wz_p = invert_fn(
            U_parent, [1.0, 0.0, 0.0, 0.0], group_bounds_dict
        )
    except Exception:
        return np.inf, None, U_parent

    params_parent = (A_p, b_p, wx_p, wy_p, wz_p)

    def kl_on_subdomain(child):
        """
        Compute integral_{child domain} f_child * ln(f_child / f_parent) dv
        """
        lo, hi = child.bounds
        cx_s     = cx_full[lo:hi]
        cy_s     = cy_full[lo:hi]
        cz_s     = cz_full[lo:hi]
        cx_vec_s = cx_vec_full[lo:hi]

        A_c, b_c, wx_c, wy_c, wz_c = child.params

        f_child = A_c * np.exp(-b_c * ((cx_s - wx_c)**2 +
                                         (cy_s - wy_c)**2 +
                                         (cz_s - wz_c)**2))

        # Parent evaluated on this child's subdomain
        f_par_s = A_p * np.exp(-b_p * ((cx_s - wx_p)**2 +
                                         (cy_s - wy_p)**2 +
                                         (cz_s - wz_p)**2))

        integrand = f_child * np.log(f_child / f_par_s)
        kl = np.trapezoid(np.trapezoid(np.trapezoid(integrand, cz_vec_full, axis=2), cy_vec_full, axis=1), cx_vec_s, axis=0)

        return max(kl, 0.0)

    kl_left  = kl_on_subdomain(left)
    kl_right = kl_on_subdomain(right)
    kl_total = kl_left + kl_right

    return kl_total, params_parent, U_parent

class VelocityGroup:
    def __init__(self, cx_slice, cy, cz, cx_vec_slice, cy_vec, cz_vec, bounds, depth=0, max_depth=1, created_at=0):
        """
        bounds: (cx_lo, cx_hi) index bounds into the full grid
        """
        self.bounds      = bounds        # (ix_lo, ix_hi) index bounds in cx_vec
        self.cx_vec      = cx_vec_slice  # 1D velocity vector for this group
        self.cy_vec      = cy_vec
        self.cz_vec      = cz_vec
        self.cx          = cx_slice      # meshgrid slices
        self.cy          = cy
        self.cz          = cz

        self.depth       = depth
        self.max_depth   = max_depth
        self.created_at  = created_at

        self.children    = []            # empty = leaf node
        self.parent      = None

        # State
        self.f           = None          # current fitted Maxwellian on full grid
        self.params      = None          # (A, b, wx, wy, wz)
        self.mu          = None          # moments

        # Shadow children — trial split, always two halves in vx
        self.shadow_params = [None, None]   # params for left/right shadow
        self.shadow_f      = [None, None]   # distributions for left/right shadow
        self.shadow_bounds = [None, None]   # index bounds for each shadow

        # Accumulation
        self.kl_accum    = 0.0
        self.kl_last_step   = 0.0
        self.f_shadow_old = [None, None]


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


    def compute_moments(self, f):
        self.mu = calc_moment(f, self.cx, self.cy, self.cz,
                              self.cx_vec, self.cy_vec, self.cz_vec)
        return self.mu


    def fit_maxwellian(self, invert, group_bounds_dict):
        if self.mu[0] < 1e-6:
            self.params = (0, 0, 0, 0, 0)
            self.f = np.zeros_like(self.cx)
            return self.params

        A, b, wx, wy, wz = invert(self.mu, [1.0, 0.0, 0.0, 0.0], group_bounds_dict)
        self.params = (A, b, wx, wy, wz)
        self.f = A * np.exp(-b * ((self.cx - wx)**2 + (self.cy - wy)**2 + (self.cz - wz)**2))

        return self.params


    def update_shadows(self, invert, cx_full, cy_full, cz_full, cx_vec_full, cy_vec_full, cz_vec_full):
        """
        Project current fitted Maxwellian onto two shadow children.
        Split at midpoint of vx domain.
        """
        ix_lo, ix_hi = self.bounds
        ix_mid = (ix_lo + ix_hi) // 2
        shadow_index_bounds = [(ix_lo, ix_mid + 1), (ix_mid, ix_hi)]

        for i, (lo, hi) in enumerate(shadow_index_bounds):
            self.shadow_bounds[i] = (lo, hi)

            cx_s    = cx_full[lo:hi]
            cy_s    = cy_full[lo:hi]
            cz_s    = cz_full[lo:hi]
            cx_vec_s = cx_vec_full[lo:hi]

            # Moments of the fitted Maxwellian over shadow subdomain
            mu_s = calc_moment(self.f[lo - ix_lo : hi - ix_lo], cx_s, cy_s, cz_s,
                                cx_vec_s, cy_vec_full, cz_vec_full)
                                
            group_bounds_dict = {
                'ci_cx': cx_vec_full[lo], 'cf_cx': cx_vec_full[hi - 1],
                'group_bounds_cx': np.array([lo, hi]),
                'ci_cy': cy_vec_full[0],  'cf_cy': cy_vec_full[-1],
                'group_bounds_cy': np.array([0, len(cy_vec_full)]),
                'ci_cz': cz_vec_full[0],  'cf_cz': cz_vec_full[-1],
                'group_bounds_cz': np.array([0, len(cz_vec_full)]),
            }

            if mu_s[0] > 1e-6: 
                A, b, wx, wy, wz = invert(mu_s, [1.0, 0.0, 0.0, 0.0], group_bounds_dict)
            else:
                A, b, wx, wy, wz = 0, 0, 0, 0, 0
            self.shadow_params[i] = (A, b, wx, wy, wz)
            self.shadow_f[i] = A * np.exp(-b * ((cx_s - wx)**2 + (cy_s - wy)**2 + (cz_s - wz)**2))


    def accumulate_kl(self, kl_fn):
        """
        Accumulate intra-shadow KL: how much has each shadow changed.
        Returns total accumulated KL across both shadows.
        """
        # if not self.can_split():
        #     return 0.0
        if self.shadow_f[0] is None or self.shadow_f[1] is None:
            return 0.0
        if self.f_shadow_old[0] is None:
            self.f_shadow_old = [s.copy() for s in self.shadow_f]
            return 0.0

        kl_total = 0.0
        ix_lo = self.bounds[0]
        for i, (lo, hi) in enumerate(self.shadow_bounds):
            # Relative index within this group's array
            kl = kl_fn(self.shadow_f[i],
                        self.f_shadow_old[i],
                        self.cx_vec[lo - ix_lo : hi - ix_lo],
                        self.cy_vec, self.cz_vec)
            kl_total += kl

        self.kl_accum += kl_total
        self.kl_last_step = kl_total
        self.f_shadow_old = [s.copy() for s in self.shadow_f]
        return kl_total


    def split(self, cx_full, cy_full, cz_full, cx_vec_full, cy_vec_full, cz_vec_full, invert, f_current, current_t=0):
        """
        Promote shadow children to real children.
        Returns list of two child VelocityGroup nodes.
        """
        if not self.can_split():
            print(f'Warning: tried to split at max_depth={self.max_depth}')
            return []

        ix_lo, ix_hi = self.bounds
        ix_mid = (ix_lo + ix_hi) // 2

        for (lo, hi) in [(ix_lo, ix_mid + 1), (ix_mid, ix_hi)]:
            cx_s     = cx_full[lo:hi]
            cy_s     = cy_full[lo:hi]
            cz_s     = cz_full[lo:hi]
            cx_vec_s = cx_vec_full[lo:hi]

            child = VelocityGroup(cx_s, cy_s, cz_s,
                                   cx_vec_s, cy_vec_full, cz_vec_full,
                                   bounds=(lo, hi),
                                   depth=self.depth+1, max_depth=self.max_depth,
                                   created_at=current_t)
            child.parent = self

            # Initialize child moments from the current f in its subdomain
            f_slice = f_current[lo - ix_lo : hi - ix_lo]
            child.compute_moments(f_slice)

            group_bounds_dict = {
                'ci_cx': cx_vec_full[lo], 'cf_cx': cx_vec_full[hi - 1],
                'group_bounds_cx': np.array([lo, hi]),
                'ci_cy': cy_vec_full[0],  'cf_cy': cy_vec_full[-1],
                'group_bounds_cy': np.array([0, len(cy_vec_full)]),
                'ci_cz': cz_vec_full[0],  'cf_cz': cz_vec_full[-1],
                'group_bounds_cz': np.array([0, len(cz_vec_full)]),
            }
            print(group_bounds_dict)
            child.fit_maxwellian(invert, group_bounds_dict)

            # Initialize shadows for child (no further splitting)
            child.update_shadows(invert, cx_full, cy_full, cz_full,
                                  cx_vec_full, cy_vec_full, cz_vec_full)
            child.kl_accum = 0.0
            self.children.append(child)

        self.kl_accum = 0.0
        return self.children
    

    def merge_children(self, cx_full, cy_full, cz_full, cx_vec_full, cy_vec_full, cz_vec_full, invert_fn, current_t=0):
        """
        Merge two children back into this parent node.
        Moments are summed, Maxwellian refit, shadows reinitialised.
        """
        assert len(self.children) == 2, "Can only merge exactly 2 children"

        left, right = self.children

        # Merged moments — exact, just sum
        self.mu = left.mu + right.mu

        # Refit Maxwellian to merged moments
        ix_lo, ix_hi = self.bounds
        group_bounds_dict = {
            'ci_cx': cx_vec_full[ix_lo],  'cf_cx': cx_vec_full[ix_hi - 1],
            'group_bounds_cx': np.array([ix_lo, ix_hi]),
            'ci_cy': cy_vec_full[0],      'cf_cy': cy_vec_full[-1],
            'group_bounds_cy': np.array([0, len(cy_vec_full)]),
            'ci_cz': cz_vec_full[0],      'cf_cz': cz_vec_full[-1],
            'group_bounds_cz': np.array([0, len(cz_vec_full)]),
        }
        self.fit_maxwellian(invert_fn, group_bounds_dict)

        # Discard children
        self.children = []

        # Reset accumulator — fresh start after merge
        self.kl_accum       = 0.0
        self.kl_last_step   = 0.0
        self.f_shadow_old   = [None, None]
        self.created_at     = current_t   # reset lifetime for future coarsen check

        # Reinitialise shadows from merged Maxwellian
        self.update_shadows(invert_fn, cx_full, cy_full, cz_full,
                             cx_vec_full, cy_vec_full, cz_vec_full)


    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves


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
root = VelocityGroup(cx, cy, cz, cx_vec, cy_vec, cz_vec, bounds=(0, 121), depth=0, max_depth=MAX_DEPTH)
root.compute_moments(f0)

group_bounds_root = {
    'ci_cx': -7, 'cf_cx': 7, 'group_bounds_cx': np.array([0, 121]),
    'ci_cy': -7, 'cf_cy': 7, 'group_bounds_cy': np.array([0, 121]),
    'ci_cz': -7, 'cf_cz': 7, 'group_bounds_cz': np.array([0, 121]),
}
root.fit_maxwellian(invert, group_bounds_root)
root.update_shadows(invert, cx, cy, cz, cx_vec, cy_vec, cz_vec)

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
        leaf.compute_moments(f_slice)

        # Refit Maxwellian
        cx_lo_v = cx_vec[ix_lo]
        cx_hi_v = cx_vec[ix_hi - 1]
        group_bounds = {
            'ci_cx': cx_lo_v, 'cf_cx': cx_hi_v,
            'group_bounds_cx': np.array([ix_lo, ix_hi]),
            'ci_cy': cy_vec[0],  'cf_cy': cy_vec[-1],
            'group_bounds_cy': np.array([0, 121]),
            'ci_cz': cz_vec[0],  'cf_cz': cz_vec[-1],
            'group_bounds_cz': np.array([0, 121]),
        }
        leaf.fit_maxwellian(invert, group_bounds)

        # Project onto shadow children
        leaf.update_shadows(invert, cx, cy, cz, cx_vec, cy_vec, cz_vec)

        # Accumulate KL
        leaf.accumulate_kl(kl_div)

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
                leaf.split(cx, cy, cz, cx_vec, cy_vec, cz_vec, invert, leaf.f, t)
            else:
                # At max depth — log the signal but can't split
                print(f't={t}: max depth reached at bounds={leaf.bounds}, '
                    f'kl={leaf.kl_accum:.4f} — consider increasing MAX_DEPTH')

    # --- Coarsen check ---
    checked_parents = set()
    for leaf in list(root.get_leaves()):
        if leaf.parent is None:
            continue  # root — never coarsen

        parent = leaf.parent
        if id(parent) in checked_parents:
            continue  # already checked this pair
        checked_parents.add(id(parent))

        if not parent.is_leaf() and len(parent.children) == 2:
            left_child, right_child = parent.children

            # Both must be leaves
            if not (left_child.is_leaf() and right_child.is_leaf()):
                continue

            # Both must have lived long enough
            if not (left_child.can_coarsen(t, MIN_LIFETIME) and
                    right_child.can_coarsen(t, MIN_LIFETIME)):
                continue

            # KL of merged state — coarsen if information loss is small
            kl, params_p, U_p = coarsening_kl_check(left_child, right_child, cx, cy, cz, cx_vec, cy_vec, cz_vec, invert)

            if kl < KL_COARSEN_THRESHOLD:
                print(f't={t}: coarsening '
                      f'{left_child.bounds} + {right_child.bounds} '
                      f'-> {parent.bounds}, kl={kl:.6f}')
                coarse_times.append(t)

                parent.merge_children(cx, cy, cz, cx_vec, cy_vec, cz_vec, invert, current_t=t)

                # Update kl_history for parent — restart from merge time
                key = parent.bounds
                kl_history[key] = {
                    'created_at': t,
                    'values':     [0.0]   # fresh accumulator after merge
                }

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

