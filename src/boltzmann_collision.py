import numpy as np
from scipy import special
from scipy.stats import qmc, norm
from .config_0d import (
    VELOCITY_SPACE,
    GROUP_PARAMS, 
    AMR,
    COLLISION_PARAMS
)
from .collision_helper import calculate_velocity_grid, VelocityGroup, initial_refine, collide, fit_maxent_weights, coarsening_h2_analytic, calc_moment
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

    all_leaves = [l for l in root.get_leaves() if l.mu[0] >= 1e-4]
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

        # ── Entropy: analytic vy/vz integrals, numerical vx ──────────────────
        # Per-leaf vx grid for entropy — use leaf's own bounds
        n_vx_leaf = 101
        vx_leaf = np.linspace(cx_lo, cx_hi, n_vx_leaf)
        f_vx    = A * np.exp(-b * (vx_leaf - wx)**2) * I0y * I0z

        # Entropy of the marginal: -integral f_marg * log(f_3d) dvx
        # For a separable Maxwellian: log(f_3d) = log(A) - b*|v-w|^2
        # so we can keep entropy exact without needing the full 3D grid
        # S_leaf = -integral_{vx} integral_{vy} integral_{vz} f log f dv
        #        = -integral_{vx} f_vx * [log(A) - b*(vx-wx)^2
        #                                  + log(I0y/I0y) ...] dvx
        # Cleanest: just do full 3D numerically on a modest grid
        n_s = 21
        cvx = np.linspace(cx_lo, cx_hi, n_s)
        cvy = np.linspace(cy_lo, cy_hi, n_s)
        cvz = np.linspace(cz_lo, cz_hi, n_s)
        gx, gy, gz = np.meshgrid(cvx, cvy, cvz, indexing='ij')
        f3  = A * np.exp(-b * ((gx - wx)**2 + (gy - wy)**2 + (gz - wz)**2))
        safe_f    = np.where(f3 > 0, f3, 1.0)
        integrand = np.where(f3 > 0, -f3 * np.log(safe_f), 0.0)
        entropy  += np.trapezoid(
                        np.trapezoid(
                            np.trapezoid(integrand, cvz, axis=2),
                        cvy, axis=1),
                    cvx, axis=0)

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
    DS_THRESHOLD = AMR['dS_threshold']
    DS_ACCUM_THRESHOLD = AMR['dS_accum_threshold']
    DS_COARSEN_THRESHOLD = AMR['dS_coarsen_threshold']  # coarsen below this
    MIN_LIFETIME         = AMR['min_lifetime']    # minimum steps before coarsening allowed

    # Initial distribution function. Still have to uncomment the correct one.
    K = 1 - 0.4 * np.exp(-0/6)
    # f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    # print(np.trapezoid(np.trapezoid(np.trapezoid(-f0 * np.log(f0), cz_vec, axis=2), cy_vec, axis=1), cx_vec))
    # f0  = 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 3)**2 + cy**2 + cz**2))
    
    def f0(cx, cy, cz):
        return 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))
        # return 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
        # return 1 / (np.pi**1.5) * np.exp(-1 * ((cx - 0)**2 + cy**2 + cz**2))

    # Create the root node of the AMR tree.
    bounds_list = np.array([[-3, 3, -3, 3, -3, 3]])
    root = VelocityGroup(bounds=np.array([[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]), depth=0, 
                         max_depth=MAX_DEPTH, split_axes=AMR['split_axes'])
    
    print('Running AMR to get initial groups...\n')
    # Choose between using custom groups or AMR to get initial groups.
    # custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    initial_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, DS_THRESHOLD)

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
        # K = 1 - 0.4 * np.exp(-t*COLLISION_PARAMS['dt']/6)
        # f = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
        # bkw_entropy[t] = np.trapezoid(np.trapezoid(np.trapezoid(-f * np.log(f, where=f>0), cz_vec, axis=2), cy_vec, axis=1), cx_vec)
        # f = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))
        f = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
        plt.plot(cx_vec, np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1), '--', color='black')
        # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f2, cz_vec, axis=2), cy_vec, axis=1), '-.', color='black')

        # ------------------------- BEGIN COLLISION ROUTINE ----------------------------        
        leaves   = root.get_leaves()
        n_groups = len(leaves)

        # Calculate arrays necessary to run collision step.
        # Need to filter for active leaves since building the arrays will break if we try to use sample arrays that are empty.
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
        for i, leaf in enumerate(active_leaves):
            # Update moments.
            leaf.mu += COLLISION_PARAMS['dt'] * np.array(coll)[:, i]

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

            # Accumulate squared Hellinger distance
            leaf.accumulate_h2()
        
        plt.title(f't = {t}')
        plt.xlabel('Cx', fontsize=18)
        plt.ylabel('f', fontsize=18)
        plt.savefig(f'plots/amr/f_{t:04d}.png', dpi=300)
        plt.close(fig)
        # --- Refinement check ---
        for leaf in list(root.get_leaves()):
            if leaf.is_empty:
                continue

            if leaf.h2_accum > DS_ACCUM_THRESHOLD:
                if leaf.can_split():
                    print(f't={t}: splitting depth={leaf.depth} '
                        f'bounds={leaf.xbounds}, h2={leaf.h2_accum:.4f}')

                    leaf.split(t)
                else:
                    pass
                    # At max depth — log the signal but can't split
                    # print(f't={t}: max depth reached at bounds={leaf.xbounds}, '
                    #     f'h2={leaf.h2_accum:.4f} — consider increasing MAX_DEPTH')

        # --- Coarsen check ---
        checked_parents = set()
        for leaf in list(root.get_leaves()):
            # Checks to find valid parent and children triangles.
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
            if left_child.is_empty or right_child.is_empty:
                continue  # asymmetric — wait until both have decided
            if not (left_child.is_leaf() and right_child.is_leaf()):
                continue
            if left_child.is_empty and right_child.is_empty:
                parent.merge_children(current_t=t)  # parent will also likely be empty
                continue

            # Analytic H^2 cost of merging
            h2 = coarsening_h2_analytic(left_child, right_child, parent)

            if h2 < DS_COARSEN_THRESHOLD:
                print(f't={t}: coarsening '
                    f'{left_child.xbounds}+{right_child.xbounds}'
                    f'->{parent.xbounds}')

                parent.merge_children(current_t=t)
        
    # --- Final plot of leaf distributions ---
    plt.figure(2)
    plt.plot(range(0, COLLISION_PARAMS['n_t']), entropy_list, color='black')
    # plt.plot(range(0, COLLISION_PARAMS['n_t']), bkw_entropy, '--', color='purple')
    plt.savefig('plots/0d_entropy.pdf')

if __name__ == '__main__':
    run_simulation()
