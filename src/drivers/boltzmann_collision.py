import os
import numpy as np
from scipy import special
from scipy.stats import qmc, norm
from ..config_0d import (
    VELOCITY_SPACE,
    GROUP_PARAMS,
    MASTER_GRID,
    AMR,
    COLLISION_PARAMS
)
from ..amr import VelocityGroup, initial_refine, fit_maxent_weights
from ..physics.grid import calculate_velocity_grid
from ..physics.collide import collide
from ..physics.moments import calc_moment
from ..banner import print_banner
import copy
import sys
from numba import types


DATA_DIR = 'simulation_data_0d'


def dump_leaves(root, t):
    """Per-step record for the cplot post-processor: each non-empty leaf
    contributes [xlo, xhi, ylo, yhi, zlo, zhi, mu0..mu4]. is_empty leaves
    are skipped — they're not used in the marginal/entropy reconstruction."""
    rows = []
    for leaf in root.get_leaves():
        if leaf.is_empty or leaf.mu[0] <= 0:
            continue
        rows.append([leaf.xbounds[0], leaf.xbounds[1],
                     leaf.ybounds[0], leaf.ybounds[1],
                     leaf.zbounds[0], leaf.zbounds[1],
                     leaf.mu[0], leaf.mu[1], leaf.mu[2], leaf.mu[3], leaf.mu[4]])
    np.save(os.path.join(DATA_DIR, f'leaves_{t:04d}.npy'),
            np.array(rows, dtype=np.float64))


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
    # 0-D uses the σ_T * g = 1 simplification for BKW comparison.
    # The physical value would be: 1/gamma_omega * (1 / m_r)**(0.5 - omega).
    sigma_coeff_hat = 1.0

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
                         max_depth=MAX_DEPTH, split_axes=AMR['split_axes'],
                         split_mode=AMR.get('split_mode', 'binary'))
    
    print('Running AMR to get initial groups...\n')
    # Choose between using custom groups or AMR to get initial groups.
    # custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    initial_refine(root, f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, KL_THRESHOLD, MAX_DEPTH+1)

    os.makedirs(DATA_DIR, exist_ok=True)

    # MAIN SIMULATION LOOP.
    for t in range(0, COLLISION_PARAMS['n_t']):
        # Snapshot the current tree for cplot post-processing. Done at the
        # *start* of step t so that step 0 captures the initial condition
        # and step n_t-1 captures the state going into the final collision.
        dump_leaves(root, t)

        # ------------------------- BEGIN COLLISION ROUTINE ----------------------------
        leaves   = root.get_leaves()
        n_groups = len(leaves)

        # bounds_list covers ALL leaves (including is_empty) so that
        # find_group() inside collide() can route post-collision velocities
        # into empty octants. If an empty leaf is omitted, its octant
        # becomes an absorption hole: find_group falls back to leaf 0 and
        # corrupts its moments within a few steps. Sample arrays are still
        # built only from non-empty leaves — empty leaves contribute zero
        # weight and never get picked as collision partners. After collide,
        # any post-collision gain landing in an empty leaf accumulates into
        # its mu and the existing reactivate() branch handles it once
        # mu[0] >= n_threshold.
        bounds_list = np.array([
            [leaf.xbounds[0], leaf.xbounds[1],
             leaf.ybounds[0], leaf.ybounds[1],
             leaf.zbounds[0], leaf.zbounds[1]]
            for leaf in leaves
        ])

        active_leaves = [leaf for leaf in leaves if not leaf.is_empty]
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

        # ── update moments, refit weights, update shadows ───────────────────
        coll_arr = np.array(coll)
        dt_step  = COLLISION_PARAMS['dt']
        # Global mu-norm is the denominator for every leaf's rate signal,
        # so the noise floor is uniform across leaves of any size. Computed
        # against pre-update moments since coll is the dt=0 RHS estimate.
        # Iterate over ALL leaves: bounds_list now covers the full domain,
        # so coll_arr[:, i] aligns with leaves[i] (not active_leaves[i]).
        # Empty leaves accumulate gain too — reactivation is handled below.
        mu_total_norm = np.linalg.norm(sum(leaf.mu for leaf in leaves))
        for i, leaf in enumerate(leaves):
            coll_vec = coll_arr[:, i]
            leaf.mu += dt_step * coll_vec
            leaf.update_rate(coll_vec, dt_step, mu_total_norm)

        for i, leaf in enumerate(leaves):
            if leaf.is_empty:
                if leaf.mu[0] >= leaf.n_threshold:
                    leaf.reactivate(current_t=t)
                continue  # skip fit/shadow/h2 until reactivated

            # Refit weights and update shadows.
            result = fit_maxent_weights(leaf.mu, leaf.xbounds, leaf.ybounds, leaf.zbounds, leaf.lam)
            if result is None:
                # Fit failed — moments are geometrically inconsistent with
                # the leaf box (e.g. mu[1]/mu[0] outside cx-bounds because
                # asymmetric collisional gain/loss drifted the mean past
                # the wall). Keep the previous w/lam/samples instead of
                # marking is_empty, so the leaf's mass stays visible to
                # the entropy/marginal sums. The stale fit is a bounded
                # approximation; next step's mu update may restore
                # feasibility.
                continue

            leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = result

            # Drift signal for split decision
            leaf.accumulate_kl()

        # --- Refinement check ---
        # Octree: each split halves all 3 axes (1->8 children).
        # Binary: single axis per leaf, cycled by depth via split_axes —
        # adaptive axis pick only happens at initial_refine.
        for leaf in list(root.get_leaves()):
            if leaf.is_empty:
                continue

            if (leaf.kl_accum > KL_ACCUM_THRESHOLD and
                    leaf.rate_ema > AMR['rate_coarsen_threshold']):
                if leaf.split_mode == 'octree':
                    if leaf.can_split():
                        print(f't={t}: octree-splitting depth={leaf.depth} '
                              f'bounds=(x={leaf.xbounds},y={leaf.ybounds},z={leaf.zbounds}), '
                              f'KL={leaf.kl_accum:.4f}, rate={leaf.rate_ema:.4f}')
                        leaf.split(t)
                    else:
                        print(f't={t}: cannot split depth={leaf.depth} '
                              f'KL={leaf.kl_accum:.4f}, '
                              f'reason={leaf.split_block_reason()}')
                else:
                    if leaf.can_split(leaf.split_dim):
                        print(f't={t}: splitting depth={leaf.depth} axis={leaf.split_dim} '
                              f'bounds=(x={leaf.xbounds},y={leaf.ybounds},z={leaf.zbounds}), '
                              f'KL={leaf.kl_accum:.4f}, rate={leaf.rate_ema:.4f}')
                        leaf.split(t)
                    else:
                        print(f't={t}: cannot split depth={leaf.depth} axis={leaf.split_dim} '
                              f'KL={leaf.kl_accum:.4f}, '
                              f'reason={leaf.split_block_reason(leaf.split_dim)}')

        # --- Coarsen check (rate-based + structure-based) ---
        # Merge a sibling group when ALL children's smoothed rate-of-change
        # has dropped below the coarsen threshold AND the resulting Maxent
        # entropy increase is below the structure threshold (i.e. the
        # children weren't capturing structure the parent box can't
        # represent — important during smooth but non-Maxwellian phases
        # like BKW relaxation, where the rate criterion alone would
        # prematurely coarsen and inflate the entropy display).
        rate_threshold = AMR['rate_coarsen_threshold']
        entropy_threshold = AMR.get('entropy_coarsen_threshold', np.inf)
        expected_children = 8 if AMR.get('split_mode', 'binary') == 'octree' else 2
        checked_parents = set()
        for leaf in list(root.get_leaves()):
            if leaf.parent is None:
                continue

            parent = leaf.parent
            if id(parent) in checked_parents:
                continue
            checked_parents.add(id(parent))

            if parent.is_leaf() or len(parent.children) != expected_children:
                continue

            children = parent.children
            if not all(c.is_leaf() for c in children):
                continue
            if not all(c.can_coarsen(t) for c in children):
                continue

            # Merge once all siblings are quiet, including is_empty ones —
            # they get update_rate every step (so rate_ema is meaningful)
            # and a stuck-empty corner (Maxent fit fails on its tiny box)
            # would otherwise block this group from ever coarsening. Their
            # mu still gets summed into the merged parent (mass conserved),
            # and the parent's larger box almost always fits cleanly.
            if all(c.rate_ema < rate_threshold for c in children):
                dS = parent.coarsen_entropy_increase()
                if dS > entropy_threshold:
                    # Structure veto: children captured non-Maxwellian
                    # detail that the parent box would lose. Wait.
                    continue
                rates_str = ', '.join(f'{c.rate_ema:.4f}' for c in children)
                print(f't={t}: coarsening {len(children)} children -> '
                      f'(x={parent.xbounds},y={parent.ybounds},z={parent.zbounds}), '
                      f'rates=[{rates_str}], dS={dS:.5f}')
                      
                parent.merge_children(current_t=t)


if __name__ == '__main__':
    run_simulation()
