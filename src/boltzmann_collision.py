import numpy as np
from scipy import special
from scipy.stats import qmc, norm
from .config_0d import (
    VELOCITY_SPACE,
    GROUP_PARAMS,
    MASTER_GRID,
    AMR,
    COLLISION_PARAMS
)
from .collision_helper import calculate_velocity_grid, VelocityGroup, initial_refine, collide, fit_maxent_weights, calc_moment
from .banner import print_banner
from .moment_utils import invert
import copy
import sys
from numba import types
from matplotlib import pyplot as plt
from scipy.special import erf as scipy_erf


def plot_and_entropy(root, invert, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()

    # Skip is_empty leaves: their fits failed and leaf.mu can be unphysical
    # (e.g. negative effective T from a forward-Euler step) — invert() on
    # those moments produces the weird tail-region blobs in the marginal.
    all_leaves = [l for l in root.get_leaves() if l.mu[0] >= 1e-4 and not l.is_empty]
    if not all_leaves:
        return ax, 0.0

    vx_lo = min(l.xbounds[0] for l in all_leaves)
    vx_hi = max(l.xbounds[1] for l in all_leaves)

    # Odd number so 0 is always included for symmetric distributions;
    # np.linspace with endpoint guarantees vx_lo and vx_hi are hit exactly.
    n_vx = 201
    vx_global = np.linspace(vx_lo, vx_hi, n_vx)
    f_marg = np.zeros_like(vx_global)
    entropy = 0.0

    # Sort leaves by cx_lo so boundary ownership is well-defined
    all_leaves_sorted = sorted(all_leaves, key=lambda l: l.xbounds[0])

    for k, leaf in enumerate(all_leaves_sorted):
        cx_lo, cx_hi = leaf.xbounds
        cy_lo, cy_hi = leaf.ybounds
        cz_lo, cz_hi = leaf.zbounds

        A, b, wx, wy, wz = invert(
            [leaf.mu[0], leaf.mu[1], leaf.mu[2], leaf.mu[3], leaf.mu[4]],
            [0.1, 0.0, 0.0, 0.0],
            {'ci_cx': cx_lo, 'cf_cx': cx_hi,
             'ci_cy': cy_lo, 'cf_cy': cy_hi,
             'ci_cz': cz_lo, 'cf_cz': cz_hi}
        )

        sqrt_b = np.sqrt(b)
        I0x = (np.sqrt(np.pi / (4 * b))
               * (scipy_erf(sqrt_b * (cx_hi - wx))
                  - scipy_erf(sqrt_b * (cx_lo - wx))))
        I0y = (np.sqrt(np.pi / (4 * b))
               * (scipy_erf(sqrt_b * (cy_hi - wy))
                  - scipy_erf(sqrt_b * (cy_lo - wy))))
        I0z = (np.sqrt(np.pi / (4 * b))
               * (scipy_erf(sqrt_b * (cz_hi - wz))
                  - scipy_erf(sqrt_b * (cz_lo - wz))))

        # Half-open [lo, hi) for all but the last leaf to avoid double-counting
        # boundary points shared between adjacent leaves
        is_last = (k == len(all_leaves_sorted) - 1)
        if is_last:
            mask = (vx_global >= cx_lo) & (vx_global <= cx_hi)
        else:
            mask = (vx_global >= cx_lo) & (vx_global < cx_hi)

        vx_in = vx_global[mask]
        f_marg[mask] += A * np.exp(-b * (vx_in - wx)**2) * I0y * I0z

        # Entropy from analytic integration of -f log f over the leaf box
        # using the inverted truncated Maxwellian f = A exp(-b * |c-w|^2):
        #   S_leaf = -log(A) * mu[0] + b * mu[0] * (J2x/I0x + J2y/I0y + J2z/I0z)
        # where J2u = integral of (c-wu)^2 exp(-b(c-wu)^2) dc over [ci,cf]
        #            = [(ci-wu) e^{-b(ci-wu)^2} - (cf-wu) e^{-b(cf-wu)^2}] / (2b)
        #              + I0u / (2b)
        def _J2(c_lo, c_hi, w):
            d_lo = c_lo - w
            d_hi = c_hi - w
            return (d_lo * np.exp(-b * d_lo**2)
                    - d_hi * np.exp(-b * d_hi**2)) / (2 * b)
        J2x = _J2(cx_lo, cx_hi, wx) + I0x / (2 * b)
        J2y = _J2(cy_lo, cy_hi, wy) + I0y / (2 * b)
        J2z = _J2(cz_lo, cz_hi, wz) + I0z / (2 * b)
        n_leaf = leaf.mu[0]
        entropy += (-np.log(A) * n_leaf
                    + b * n_leaf * (J2x / I0x + J2y / I0y + J2z / I0z))

    ax.plot(vx_global, f_marg, color='red')
    return ax, entropy


def run_simulation():
    print_banner()
    key_type = types.UniTuple(types.int64, 2)
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    dx = np.abs(cx_vec[1] - cx_vec[0])
    dy = np.abs(cy_vec[1] - cy_vec[0])
    dz = np.abs(cz_vec[1] - cz_vec[0])

    CX_LB = cx_vec[0];  CX_UB = cx_vec[-1]
    CY_LB = cy_vec[0];  CY_UB = cy_vec[-1]
    CZ_LB = cz_vec[0];  CZ_UB = cz_vec[-1]

    n_coll = COLLISION_PARAMS['n_coll']
    omega = COLLISION_PARAMS['omega']
    alpha = COLLISION_PARAMS['alpha']
    m_r = 0.5
    gamma_omega = special.gamma(5/2 - omega)
    sigma_coeff_hat = 1/gamma_omega * (1 / m_r)**(0.5 - omega)

    MAX_DEPTH = AMR['max_depth']       # 0 = no splitting, 1 = one split etc.
    KL_THRESHOLD = AMR['KL_threshold']
    KL_ACCUM_THRESHOLD = AMR['KL_accum_threshold']
    MIN_LIFETIME         = AMR['min_lifetime']    # minimum steps before coarsening allowed

    # Initial distribution function. Still have to uncomment the correct one.
    K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    # print(np.trapezoid(np.trapezoid(np.trapezoid(-f0 * np.log(f0), cz_vec, axis=2), cy_vec, axis=1), cx_vec))
    # f0  = 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 3)**2 + cy**2 + cz**2))
    
    def f0(cx, cy, cz):
        # return 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))
        return 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
        # return 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 0)**2 + cy**2 + cz**2))

    # Initialize the shared master sample grid before any leaves are built.
    VelocityGroup.set_master_grid(MASTER_GRID)

    # Create the root node of the AMR tree.
    bounds_list = np.array([[-3, 3, -3, 3, -3, 3]])
    root = VelocityGroup(bounds=np.array([[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]), depth=0,
                         max_depth=MAX_DEPTH, split_axes=AMR['split_axes'])
    
    print('Running AMR to get initial groups...\n')
    # Choose between using custom groups or AMR to get initial groups.
    # custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    initial_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, KL_THRESHOLD, MAX_DEPTH+1)

    entropy_list = np.zeros(COLLISION_PARAMS['n_t'])
    bkw_entropy  = np.zeros(COLLISION_PARAMS['n_t'])
    # MAIN SIMULATION LOOP.
    for t in range(0, COLLISION_PARAMS['n_t']):
        # ------------------ PLOTTING STUFF ----------------------
        fig, ax = plt.subplots()
        ax.set_ylim(0, 0.6)
        ax.set_xlim(-4, 4)

        ax, entropy_list[t] = plot_and_entropy(root, invert, ax=ax)
        
        # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f0, cz_vec, axis=2), cy_vec, axis=1), '--', color='black')
        K = 1 - 0.4 * np.exp(-t*COLLISION_PARAMS['dt']/6)
        f = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
        bkw_entropy[t] = np.trapezoid(np.trapezoid(np.trapezoid(-f * np.log(f, where=f>0), cz_vec, axis=2), cy_vec, axis=1), cx_vec)
        # f = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))
        # f = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
        plt.plot(cx_vec, np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1), '--', color='black')
        # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f2, cz_vec, axis=2), cy_vec, axis=1), '-.', color='black')

        # ------------------------- BEGIN COLLISION ROUTINE ----------------------------        
        leaves   = root.get_leaves()
        n_groups = len(leaves)

        # Calculate arrays necessary to run collision step.
        active_leaves = [leaf for leaf in leaves if not leaf.is_empty]

        bounds_list = np.array([
            [leaf.xbounds[0], leaf.xbounds[1],
             leaf.ybounds[0], leaf.ybounds[1],
             leaf.zbounds[0], leaf.zbounds[1]]
            for leaf in active_leaves
        ])
        n_total  = sum(len(leaf.x_s) for leaf in active_leaves)
        x_flat   = np.zeros(n_total)
        y_flat   = np.zeros(n_total)
        z_flat   = np.zeros(n_total)
        w_flat   = np.zeros(n_total)
        offset   = 0
        for leaf in active_leaves:
            n = len(leaf.x_s)
            x_flat[offset:offset + n] = leaf.x_s
            y_flat[offset:offset + n] = leaf.y_s
            z_flat[offset:offset + n] = leaf.z_s
            w_flat[offset:offset + n] = leaf.w
            offset += n

        if t % 10 == 0:
            print('Time step: ', t)
        print(n_groups)

        # Run collision step.
        coll = collide(
            x_flat, y_flat, z_flat, w_flat,
            len(w_flat),
            bounds_list, n_groups, n_coll,
            CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB,
            key_type, sigma_coeff_hat, omega, alpha
        )

        # Define test distribution if using to simulate forcing/flux terms.
        # def ft(cx, cy, cz):
        #     f01 = 1/(np.pi**1.5) * np.exp(-1*((cx - 3 + 0.05*t)**2 + cy**2 + cz**2))
        #     f02 = 0.04 / (np.pi**1.5) * np.exp(-0.2 * ((cx + 1)**2 + cy**2 + cz**2))
        #     f_eq  = 1/(np.pi**1.5) * np.exp(-1*(cx**2 + cy**2 + cz**2))

        #     if t <= 100:
        #         # Phase 1: drift and mix toward non-Maxwellian
        #         alpha   = t / 100
        #         weight2 = 0.5 * alpha
        #         weight1 = 1 - weight2
        #         ft  = weight1 * f01 + weight2 * f02
        #     else:
        #         tau   = 30   # relaxation timescale in steps
        #         decay = np.exp(-(t - 100) / tau)
        #         ft_end_phase1 = 0.5 * f01 + 0.5 * f02
        #         ft    = decay * ft_end_phase1 + (1 - decay) * f_eq

        #     return ft

        # ── update moments, refit weights, update shadows ───────────────────
        coll_arr = np.array(coll)
        dt_step  = COLLISION_PARAMS['dt']
        # Global mu-norm is the denominator for every leaf's rate signal,
        # so the noise floor is uniform across leaves of any size. Computed
        # against pre-update moments since coll is the dt=0 RHS estimate.
        mu_total_norm = np.linalg.norm(sum(leaf.mu for leaf in active_leaves))
        for i, leaf in enumerate(active_leaves):
            coll_vec = coll_arr[:, i]
            leaf.mu += dt_step * coll_vec
            leaf.update_rate(coll_vec, dt_step, mu_total_norm)

        for i, leaf in enumerate(leaves):
            # Arbitrary test distribution.
            # cx_lo, cx_hi = leaf.xbounds
            # cy_lo, cy_hi = leaf.ybounds
            # cz_lo, cz_hi = leaf.zbounds
            # cx_vec = np.linspace(cx_lo, cx_hi, 30)
            # cy_vec = np.linspace(cy_lo, cy_hi, 30)
            # cz_vec = np.linspace(cz_lo, cz_hi, 30) 
            # cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
            # f_slice = ft(cx, cy, cz)
            # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f_slice, cz_vec, axis=2), cy_vec, axis=1), '--', color='black')
            # mu = calc_moment(f_slice, cx, cy, cz, cx_vec, cy_vec, cz_vec)
            # leaf.mu = mu

            if leaf.is_empty:
                if leaf.mu[0] >= leaf.n_threshold:
                    leaf.reactivate(current_t=t)
                continue  # skip fit/shadow/h2 until reactivated

            # Refit weights and update shadows.
            result = fit_maxent_weights(leaf.mu, leaf.xbounds, leaf.ybounds, leaf.zbounds, leaf.lam)
            if result is None:
                leaf.is_empty = True
                continue

            leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = result

            # Drift signal for split decision
            leaf.accumulate_kl()
        
        plt.title(f't = {t}')
        plt.xlabel('Cx', fontsize=18)
        plt.ylabel('f', fontsize=18)
        plt.savefig(f'plots/amr/f_{t:04d}.png', dpi=300)
        plt.close(fig)

        # --- Refinement check ---
        # Single axis per leaf, cycled by depth via split_axes (set in
        # __init__). Adaptive axis pick happens only at initial_refine.
        for leaf in list(root.get_leaves()):
            if leaf.is_empty:
                continue

            if (leaf.kl_accum > KL_ACCUM_THRESHOLD and
                    leaf.rate_ema > AMR['rate_coarsen_threshold']):
                if leaf.can_split(leaf.split_dim):
                    print(f't={t}: splitting depth={leaf.depth} axis={leaf.split_dim} '
                          f'bounds=(x={leaf.xbounds},y={leaf.ybounds},z={leaf.zbounds}), '
                          f'KL={leaf.kl_accum:.4f}, rate={leaf.rate_ema:.4f}')
                    leaf.split(t)
                else:
                    print(f't={t}: cannot split depth={leaf.depth} axis={leaf.split_dim} '
                          f'KL={leaf.kl_accum:.4f}, '
                          f'reason={leaf.split_block_reason(leaf.split_dim)}')

        # --- Coarsen check (rate-based) ---
        # Merge a sibling pair when both children's smoothed rate-of-change
        # has dropped below the coarsen threshold.
        rate_threshold = AMR['rate_coarsen_threshold']
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
            if not (left_child.can_coarsen(t) and right_child.can_coarsen(t)):
                continue
            if not (left_child.is_leaf() and right_child.is_leaf()):
                continue
            if left_child.is_empty or right_child.is_empty:
                continue  # asymmetric — wait until both have decided

            if (left_child.rate_ema < rate_threshold and
                    right_child.rate_ema < rate_threshold):
                print(f't={t}: coarsening '
                      f'bounds=(x={left_child.xbounds},y={left_child.ybounds},z={left_child.zbounds}), '
                      f'bounds=(x={right_child.xbounds},y={right_child.ybounds},z={right_child.zbounds}), '
                      f'(rate_L={left_child.rate_ema:.4f}, '
                      f'rate_R={right_child.rate_ema:.4f})')
                print(left_child.mu + right_child.mu)
                parent.merge_children(current_t=t)
                print(parent.mu)
        
    # --- Final plot of leaf distributions ---
    plt.figure(2)
    plt.plot(range(0, COLLISION_PARAMS['n_t']), entropy_list, color='black')
    plt.hlines(3.2162536221141895, 0, COLLISION_PARAMS['n_t'], color='red', linestyles='dashed')
    plt.plot(range(0, COLLISION_PARAMS['n_t']), bkw_entropy, '--', color='purple')
    plt.savefig('plots/0d_entropy.pdf')

if __name__ == '__main__':
    run_simulation()
