import numpy as np
from scipy import special
from scipy.stats import qmc, norm
from .config_0d import (
    VELOCITY_SPACE,
    GROUP_PARAMS, 
    AMR,
    COLLISION_PARAMS
)
from .collision_helper import calculate_velocity_grid, VelocityGroup, bootstrap_refine, collide
from .banner import print_banner
import copy
import sys


def run_simulation():
    print_banner()
    
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
    root = VelocityGroup(
        x_s=np.array([]), y_s=np.array([]), z_s=np.array([]),
        cx_vec_full=cx_vec,
        bounds=(0, len(cx_vec)),
        depth=0, max_depth=MAX_DEPTH
    )
    print('Running AMR to get initial groups...\n')
    # Choose between using custom groups or AMR to get initial groups.
    # custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, GROUP_PARAMS)
    bootstrap_refine(root, f0, cx_vec, cy_vec, cz_vec)

    kl_history   = {}
    split_times  = []
    coarse_times = []

    CX_LB = cx_vec[0];  CX_UB = cx_vec[-1]
    CY_LB = cy_vec[0];  CY_UB = cy_vec[-1]
    CZ_LB = cz_vec[0];  CZ_UB = cz_vec[-1]

    # MAIN SIMULATION LOOP.
    for t in range(1, COLLISION_PARAMS['n_t'] + 1):
        if t % 10 == 0:
            print('Time step: ', t)
        
        leaves   = root.get_leaves()
        n_groups = len(leaves)

        # Run collision step.
        x_flat, y_flat, z_flat, w_flat, bounds_list, group_slices = \
        build_collision_inputs(leaves, cx_vec, cy_vec, cz_vec)

        coll = collide(
            x_flat, y_flat, z_flat, w_flat,
            len(w_flat),
            bounds_list, n_groups, n_coll,
            CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB,
            key_type, sigma_coeff_hat, omega, alpha
        )

        # ── update moments, refit weights, update shadows ───────────────────
        apply_collision_deltas(leaves, group_deltas)

        for leaf in leaves:
            leaf.fit_weights()      # positions fixed — Newton only updates weights
            leaf.update_shadows()
            leaf.accumulate_kl()

            key = leaf.bounds
            if key not in kl_history:
                kl_history[key] = {'created_at': leaf.created_at, 'values': []}
            kl_history[key]['values'].append(leaf.kl_accum)
        
        # ── refinement check ─────────────────────────────────────────────────
        for leaf in list(root.get_leaves()):
            if leaf.kl_accum > KL_THRESHOLD:
                if leaf.can_split():
                    leaf.split(current_t=t)
                    for child in leaf.children:
                        resample_leaf(child, cx_vec, cy_vec, cz_vec)
                        child.update_shadows()

        # ── coarsen check ────────────────────────────────────────────────────
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

            kl = coarsening_kl_check_samples(left_child, right_child)
            if kl < KL_COARSEN_THRESHOLD:
                parent.merge_children(current_t=t)
                resample_leaf(parent, cx_vec, cy_vec, cz_vec)
                parent.update_shadows()


if __name__ == '__main__':
    run_simulation()
