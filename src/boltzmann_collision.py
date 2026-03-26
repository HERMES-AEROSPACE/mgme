import numpy as np
from scipy import special
from scipy.stats import qmc, norm
from .config_0d import (
    VELOCITY_SPACE,
    GROUP_PARAMS, 
    AMR,
    COLLISION_PARAMS
)
from .collision_helper import calculate_velocity_grid, VelocityGroup, bootstrap_refine, collide, generate_grid, coarsening_kl_analytic
from .banner import print_banner
from .moment_utils import invert
import copy
import sys
from numba import types
from matplotlib import pyplot as plt


def run_simulation():
    print_banner()
    key_type = types.UniTuple(types.int64, 2)
    
    # Get velocity space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    dx = np.abs(cx_vec[1] - cx_vec[0])
    dy = np.abs(cy_vec[1] - cy_vec[0])
    dz = np.abs(cz_vec[1] - cz_vec[0])

    # Initial distribution function. Still have to uncomment the correct one.
    K = 1 - 0.4 * np.exp(-0/6)
    f0 = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    # f0 = 1 / (np.pi**1.5) * np.exp(-1 * (cx**2 + cy**2 + cz**2))
    # f0 = 0.5 * (3 / np.pi)**1.5 * (np.exp(-3.0 * (cx - 1)**2) + np.exp(-3.0 * (cx + 1)**2)) * np.exp(-3 * (cy**2 + cz**2))

    # Create the root node of the AMR tree.
    MAX_DEPTH    = 4       # 0 = no splitting, 1 = one split etc.
    KL_THRESHOLD = 0.005
    KL_COARSEN_THRESHOLD = 0.01   # coarsen below this
    MIN_LIFETIME         = 5      # minimum steps before coarsening allowed
    bounds_list = np.array([[-3, 3, -3, 3, -3, 3]])
    x_sample, y_sample, z_sample, _, _ = generate_grid(bounds_list, 1)
    root = VelocityGroup(x_sample, y_sample, z_sample, cx_vec, cy_vec, cz_vec, bounds=(0, 241), depth=0, max_depth=MAX_DEPTH)

    print('Running AMR to get initial groups...\n')
    # Choose between using custom groups or AMR to get initial groups.
    # custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    bootstrap_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec)

    kl_history   = {}
    split_times  = []
    coarse_times = []

    CX_LB = cx_vec[0];  CX_UB = cx_vec[-1]
    CY_LB = cy_vec[0];  CY_UB = cy_vec[-1]
    CZ_LB = cz_vec[0];  CZ_UB = cz_vec[-1]

    n_coll = COLLISION_PARAMS['n_coll']
    omega = COLLISION_PARAMS['omega']
    alpha = COLLISION_PARAMS['alpha']
    m_r = 0.5
    gamma_omega = special.gamma(5/2 - omega)
    sigma_coeff_hat = 1/gamma_omega * (1 / m_r)**(0.5 - omega)

    entropy_list = np.zeros(100)
    bkw_entropy  = np.zeros(100)
    # MAIN SIMULATION LOOP.
    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
        
        leaves   = root.get_leaves()
        n_groups = len(leaves)

        bounds_list = np.array([
            [cx_vec[leaf.bounds[0]], cx_vec[leaf.bounds[1] - 1],
            cy_vec[0], cy_vec[-1],
            cz_vec[0], cz_vec[-1]]
            for leaf in leaves
        ])
        n_total  = sum(len(leaf.x_s) for leaf in leaves)
        x_flat   = np.zeros(n_total)
        y_flat   = np.zeros(n_total)
        z_flat   = np.zeros(n_total)
        w_flat   = np.zeros(n_total)
        offset   = 0
        for leaf in leaves:
            n = len(leaf.x_s)
            x_flat[offset:offset + n] = leaf.x_s
            y_flat[offset:offset + n] = leaf.y_s
            z_flat[offset:offset + n] = leaf.z_s
            w_flat[offset:offset + n] = leaf.w
            offset += n

        # Run collision step.
        coll = collide(
            x_flat, y_flat, z_flat, w_flat,
            len(w_flat),
            bounds_list, n_groups, n_coll,
            CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB,
            key_type, sigma_coeff_hat, omega, alpha
        )

        # ── update moments, refit weights, update shadows ───────────────────
        for i, leaf in enumerate(leaves):
            # update
            leaf.mu += COLLISION_PARAMS['dt'] * np.array(coll)[:, i]

            # Refit max entropy weights
            leaf.fit_maxent_weights()

            # Project onto shadow children
            leaf.update_shadows(cx_vec)

            # Accumulate KL
            leaf.accumulate_kl(cx_vec, cy_vec, cz_vec)
            if leaf.kl_last_step < 1e-5:
                leaf.kl_accum = 0.0

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
        
        fig = plt.figure()
        entropy = 0
        for leaf in root.get_leaves():
            ix_lo, ix_hi = leaf.bounds

            ci = cx_vec[ix_lo]
            cf = cx_vec[ix_hi-1]
            A, b, wx, wy, wz = invert(
                [leaf.mu[0], leaf.mu[1], 0.0, 0.0, leaf.mu[4]],
                [1.0, 0.0, 0.0, 0.0],
                {'ci_cx': ci, 'cf_cx': cf,
                'ci_cy': -3, 'cf_cy': 3,
                'ci_cz': -3, 'cf_cz': 3}
            )

            fx = np.trapezoid(
                    np.trapezoid(
                        A * np.exp(-b * ((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2)),
                    cz_vec, axis=2),
                cy_vec, axis=1)
            plt.plot(cx_vec[ix_lo:ix_hi], fx[ix_lo:ix_hi])
            entropy += np.trapezoid(-fx * np.log(fx), cx_vec)
        entropy_list[t-1] = entropy
        
        # plt.plot(cx_vec, np.trapezoid(np.trapezoid(f0, cz_vec, axis=2), cy_vec, axis=1), '--', color='black')
        K = 1 - 0.4 * np.exp(-t*COLLISION_PARAMS['dt']/6)
        f = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
        bkw_entropy[t-1] = np.trapezoid(np.trapezoid(np.trapezoid(-f * np.log(f), cz_vec, axis=2), cy_vec, axis=1), cx_vec)
        plt.plot(cx_vec, np.trapezoid(np.trapezoid(f, cz_vec, axis=2), cy_vec, axis=1), '--', color='black')
        plt.title(f't = {t}')
        plt.savefig(f'plots/f_{t:04d}.png', dpi=300)
        plt.close(fig)

    
    plt.figure()
    plt.plot(np.linspace(1, 101, 100), entropy_list, color='black')
    plt.plot(np.linspace(1, 101, 100), bkw_entropy, '--', color='black')
    plt.savefig('plots/entropy.pdf')
    # --- Final plot of leaf distributions ---
    # fig, ax = plt.subplots(figsize=(5, 5))
    # leaves  = root.get_leaves()
    # colors  = plt.cm.tab10(np.linspace(0, 1, len(leaves)))

    # for i, leaf in enumerate(leaves):
    #     ix_lo, ix_hi = leaf.bounds
    #     cx_leaf = cx_vec[ix_lo:ix_hi]

    #     if leaf.mu is None or leaf.mu[0] < 1e-12:
    #         continue
    #     if leaf.lam[4] >= 0 or not np.all(np.isfinite(leaf.lam)):
    #         continue

    #     ux = leaf.mu[1] / leaf.mu[0]
    #     uy = leaf.mu[2] / leaf.mu[0]
    #     uz = leaf.mu[3] / leaf.mu[0]
    #     thermal = leaf.mu[4] / leaf.mu[0] - ux**2 - uy**2 - uz**2
    #     beta = 1.5 / max(thermal, 1e-10)
    #     sqb  = np.sqrt(beta)

    #     def erf_integral(lo, hi, w):
    #         return 0.5 * np.sqrt(np.pi) / sqb * (
    #             special.erf(sqb * (hi - w)) - special.erf(sqb * (lo - w)))

    #     i0y = erf_integral(cy_vec[0], cy_vec[-1], uy)
    #     i0z = erf_integral(cz_vec[0], cz_vec[-1], uz)
    #     i0x = erf_integral(cx_vec[ix_lo], cx_vec[ix_hi - 1], ux)

    #     if abs(i0x * i0y * i0z) < 1e-30:
    #         continue

    #     A    = leaf.mu[0] / (i0x * i0y * i0z)
    #     f_1d = A * i0y * i0z * np.exp(-beta * (cx_leaf - ux)**2)

    #     ax.plot(cx_leaf, f_1d, color=colors[i], linewidth=1.8,
    #             label=f'leaf {i}  vx=[{cx_vec[ix_lo]:.2f}, {cx_vec[ix_hi-1]:.2f}]  depth={leaf.depth}')
    #     ax.axvline(x=cx_vec[ix_lo],   color=colors[i], linewidth=0.8, linestyle='--', alpha=0.5)
    #     ax.axvline(x=cx_vec[ix_hi-1], color=colors[i], linewidth=0.8, linestyle='--', alpha=0.5)

    # K = 1 - 0.4 * np.exp(-20/6)
    # fend = 1 / (2 * K * (np.pi * K)**1.5) * (5 * K - 3 + 2 * (1 - K) / K * (cx**2 + cy**2 + cz**2)) * np.exp(-(cx**2 + cy**2 + cz**2) / K)
    # fI = np.trapezoid(np.trapezoid(fend, cz_vec, axis=2), cy_vec, axis=1)
    # ax.plot(cx_vec, fI, color='black')
    # ax.set_xlabel(r'$v_x$', fontsize=14)
    # ax.set_ylabel(r'$f_{1D}(v_x)$', fontsize=14)
    # ax.set_xlim(cx_vec[0], cx_vec[-1])
    # ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.2)
    # plt.tight_layout()
    # plt.savefig('final_leaves.pdf', bbox_inches='tight')
    # plt.close(fig)


if __name__ == '__main__':
    run_simulation()
