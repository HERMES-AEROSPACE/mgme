import numpy as np
from numba import njit
from scipy.stats import qmc
from scipy import special
from matplotlib import pyplot as plt
from .config_0d import AMR


class VelocityGroup:
    def __init__(self, bounds, depth=0, max_depth=1, created_at=0):
        """
        bounds: (cx_lo, cx_hi) index bounds into the full grid
        """
        self.is_empty = False
        self.n_threshold = 1e-5  # below this, treat as empty

        self.xbounds      = bounds[0]        # (lo, hi) value bounds
        self.ybounds      = bounds[1]        # (lo, hi) value bounds
        self.zbounds      = bounds[2]        # (lo, hi) value bounds 

        self.depth       = depth
        self.max_depth   = max_depth
        self.created_at  = created_at

        self.children    = []            # empty = leaf node
        self.parent      = None

        # State
        self.w           = None          # current fitted max entropy weights
        self.lam         = np.zeros(5)   # lagrange multipliers
        self.mu          = None          # moments
        self.x_s = None
        self.y_s = None
        self.z_s = None

        # Shadow children — trial split, always two halves in vx
        self.shadow_w       = [None, None]   # weights for left/right shadow
        self.shadow_lam     = [np.zeros(5), np.zeros(5)]   # multipliers for left/right shadow
        self.shadow_mu      = [None, None]
        self.shadow_bounds  = [None, None]   # value bounds for each shadow

        # do I need to store the shadow sample locations?

        # Accumulation
        self.h2_accum = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def can_split(self):
        return self.depth < self.max_depth

    def can_coarsen(self, current_t, min_lifetime=AMR['min_lifetime']):
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
        cx_lo, cx_hi = self.xbounds
        cx_mid        = (cx_lo + cx_hi) / 2.0

        self.shadow_bounds[0] = [cx_lo, cx_mid]
        self.shadow_bounds[1] = [cx_mid, cx_hi]

        if cx_mid in self.x_s:
            print('The group split velocity is contained in x_s.')
            sys.exit()

        left_idx = self.x_s < cx_mid
        right_idx = ~left_idx
        masks = np.array([left_idx, right_idx])

        for i, mask in enumerate(masks):
            # Moments of this half from current weights.
            n = np.sum(self.w[mask])

            if n < self.n_threshold:
                # Shadow is empty — leave as None so split() can detect it
                self.shadow_mu[i]  = np.zeros(5)
                self.shadow_w[i]   = None
                self.shadow_lam[i] = np.zeros(5)
                continue

            ux = np.sum(self.w[mask] * self.x_s[mask])
            uy = np.sum(self.w[mask] * self.y_s[mask])
            uz = np.sum(self.w[mask] * self.z_s[mask])
            r2 = self.x_s[mask]**2 + self.y_s[mask]**2 + self.z_s[mask]**2
            e = np.sum(self.w[mask] * r2)  
            mu = np.array([n, ux, uy, uz, e])

            # Fit weights to these shadows.
            results = fit_maxent_weights(mu, self.xbounds, self.ybounds, self.zbounds)
            if results is None:
                self.shadow_mu[i]  = mu
                self.shadow_w[i]   = None
                self.shadow_lam[i] = np.zeros(5)
                continue

            self.shadow_mu[i]  = mu
            self.shadow_w[i]   = results[0]
            self.shadow_lam[i] = results[1]

    def split(self, current_t=0):
        """
        Promote shadow children to real children.
        Samples are partitioned by vx midpoint.
        Shadow weights/lam used to initialize children.
        """
        self.update_shadows()
        
        for i in range(2):
            bnd = np.array([self.shadow_bounds[i], [-7, 7], [-7, 7]])
            child = VelocityGroup(
                bounds=bnd,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                created_at=current_t)
            child.parent = self
            child.mu = self.shadow_mu[i].copy()

            if child.mu[0] < child.n_threshold:
                child.is_empty = True
            else:
                result = fit_maxent_weights(child.mu, child.xbounds, child.ybounds, child.zbounds)
                child.w, child.lam, child.x_s, child.y_s, child.z_s = result
                child.update_shadows()

            self.children.append(child)

    def accumulate_h2(self):
        """
        Calculate the squared Hellinger distance between the current leaves and its children. Add it to the current leaves.

        Evaluate the children shadow groups on the current group sample locations.
        """
        cx_lo, cx_hi = self.xbounds
        cx_mid = (cx_lo + cx_hi) / 2.0

        left_idx  = self.x_s < cx_mid
        right_idx = ~left_idx

        masks = np.array([left_idx, right_idx])
        q = np.zeros_like(self.w)

        for i, mask in enumerate(masks):
            phi = np.array([
                np.ones(np.sum(mask)), 
                self.x_s[mask], 
                self.y_s[mask], 
                self.z_s[mask], 
                self.x_s[mask]**2 + self.y_s[mask]**2 + self.z_s[mask]**2
            ])

            # Need to solve for the correct lambdas using the current group sample locations.
            _, lam, success, _ = solve_group_newton(self.x_s[mask], self.y_s[mask], self.z_s[mask], self.shadow_mu[i], np.zeros(5))
            if success: 
                q[mask] = np.exp(lam @ phi)

        p_norm = self.w / np.sum(self.w)
        q_norm = q / np.sum(q)
        mask = (p_norm > 0) & (q_norm > 0)
        hellinger_sq = 1.0 - np.sum(np.sqrt(p_norm[mask] * q_norm[mask]))
        hellinger_sq = np.clip(hellinger_sq, 0.0, 1.0)  # guard against float noise

        self.h2_accum += hellinger_sq

    def merge_children(self, current_t=0):
        assert len(self.children) == 2
        left, right = self.children

        # Concatenate samples
        self.mu  = left.mu + right.mu
        self.w, self.lam, self.x_s, self.y_s, self.z_s = fit_maxent_weights(self.mu, self.xbounds, self.ybounds, self.zbounds)
        self.update_shadows()

        self.children         = []
        self.h2_accum         = 0.0
        self.created_at       = current_t

    def reactivate(self, current_t=0):
        """
        Called when mu[0] rises above threshold after a moment update.
        Fits weights from scratch and resets shadow reference.
        """
        result = fit_maxent_weights(self.mu, self.xbounds, self.ybounds, self.zbounds)
        if result is None:
            return  # still can't fit, stay empty
        self.w, self.lam, self.x_s, self.y_s, self.z_s = result
        self.update_shadows()
        self.is_empty = False
        self.created_at = current_t
        self.h2_accum = 0.0  # fresh start, don't trigger immediate re-split

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
        # Newton step
        lam -= np.linalg.solve(H, g)
    
    w = np.exp(lam @ phi) 
        
    return w, lam, converged, g

def fit_maxent_weights(mu, xbounds, ybounds, zbounds, n_sigma=3.0):
    n_fine = 30  # would probably like to change this so this scales based on bounds

    ux  = mu[1] / mu[0]
    uy  = mu[2] / mu[0]
    uz  = mu[3] / mu[0]
    T   = max(2 * (mu[4] / mu[0] - ux**2 - uy**2 - uz**2) / 3.0, 1e-10)
    v_th = np.sqrt(T)

    # Since there is only one group in y and z, no slight factor. This is to make collision routine work.
    xlo = np.max([ux - n_sigma * v_th, xbounds[0]]) + 1e-10
    xhi = np.min([ux + n_sigma * v_th, xbounds[1]]) - 1e-10
    ylo = np.max([uy - n_sigma * v_th, ybounds[0]])
    yhi = np.min([uy + n_sigma * v_th, ybounds[1]])
    zlo = np.max([uz - n_sigma * v_th, zbounds[0]])
    zhi = np.min([uz + n_sigma * v_th, zbounds[1]])

    gx_fine = np.linspace(xlo, xhi, n_fine)
    gy_fine = np.linspace(ylo, yhi, n_fine)
    gz_fine = np.linspace(zlo, zhi, n_fine)
    GX, GY, GZ = np.meshgrid(gx_fine, gy_fine, gz_fine, indexing='ij')
    x_slice = GX.ravel()
    y_slice = GY.ravel()
    z_slice = GZ.ravel()

    lam = np.zeros(5)  # will likely need a better way of keeping track of initial starts.
    solution, lam, success, rel_err = solve_group_newton(x_slice, y_slice, z_slice, mu, lam)

    if success:
        return solution, lam, x_slice, y_slice, z_slice

def coarsening_h2_analytic(left, right, parent):
    """
    Square Hellinger distance of merging: H^2(f_left + f_right | f_parent).
    
    Evaluate the child left and right groups on the parent sample locations.
    """
    children = np.array([left, right])
    cx_lo, cx_hi = parent.xbounds
    cx_mid = (cx_lo + cx_hi) / 2.0

    left_idx  = parent.x_s < cx_mid
    right_idx = ~left_idx

    masks = np.array([left_idx, right_idx])
    hellinger_sq = 0

    for mask, child in zip(masks, children):
        n = np.sum(parent.w[mask])
        ux = np.sum(parent.w[mask] * parent.x_s[mask])
        uy = np.sum(parent.w[mask] * parent.y_s[mask])
        uz = np.sum(parent.w[mask] * parent.z_s[mask])
        r2 = parent.x_s[mask]**2 + parent.y_s[mask]**2 + parent.z_s[mask]**2
        e = np.sum(parent.w[mask] * r2)  
        mu = np.array([n, ux, uy, uz, e])

        sub_parent_w, _, _, _, _ = fit_maxent_weights(mu, child.xbounds, child.ybounds, child.zbounds)

        p = child.w / np.sum(child.w)
        q = sub_parent_w / np.sum(sub_parent_w)
        hellinger_sq += np.clip(1.0 - np.sum(np.sqrt(p * q)), 0.0, 1.0)

    return hellinger_sq
    
def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)

    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)

    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu

def initial_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, h2_threshold=AMR['h2_threshold'], max_passes=10):
    """
    Refine the AMR tree based on goodness-of-fit H^2(f0 || f_maxent)
    on each leaf. Runs until no splits occur or max_passes is reached.
    """
    for pass_idx in range(max_passes):
        leaves = root.get_leaves()
        splits_this_pass = 0

        for leaf in leaves:
            cx_lo, cx_hi = leaf.xbounds
            cy_lo, cy_hi = leaf.ybounds
            cz_lo, cz_hi = leaf.zbounds
            
            # Evaluate f0 at unique grid to get a relatively accurate moment evaluation.
            cx_vec = np.linspace(cx_lo, cx_hi, 30)
            cy_vec = np.linspace(cy_lo, cy_hi, 30)
            cz_vec = np.linspace(cz_lo, cz_hi, 30) 
            cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
            f_slice = f0(cx, cy, cz)

            # Recompute moments from f0, calculate weights, and update shadow values.
            mu = calc_moment(f_slice, cx, cy, cz, cx_vec, cy_vec, cz_vec)
            leaf.mu = mu
            leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = fit_maxent_weights(mu, leaf.xbounds, leaf.ybounds, leaf.zbounds)
            leaf.update_shadows()

            # Calculate the squared Hellinger distance between the current group and true f0.
            dcx = (cx_hi - cx_lo) / (30 - 1)
            dcy = (cy_hi - cy_lo) / (30 - 1)
            dcz = (cz_hi - cz_lo) / (30 - 1)
            dv  = dcx * dcy * dcz
            f0_weights = f0(leaf.x_s, leaf.y_s, leaf.z_s) * dv

            p = f0_weights / np.sum(f0_weights)
            q = leaf.w / np.sum(leaf.w)
            hellinger_sq = 1.0 - np.sum(np.sqrt(p * q))
            hellinger_sq = np.clip(hellinger_sq, 0.0, 1.0)  # guard against float noise
            
            # If divergence is larger than threshold, split the current group (update shadow children to real children).
            if hellinger_sq > h2_threshold and leaf.can_split():
                print(f'pass {pass_idx}: splitting depth={leaf.depth} '
                      f'bounds={leaf.xbounds}, h2={hellinger_sq:.4f}')
                
                leaf.split(current_t=0)
                splits_this_pass += 1
            if not leaf.can_split():
                print(f'Warning: tried to split at max_depth={leaf.max_depth}')

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
        C = 0.5 * W**2 / n_actual * g**(2 - 2*omega) * sigma_coeff_hat

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