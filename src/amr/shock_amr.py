"""Shock-side AMR helpers.

Bridges the global VelocityGroup tree (which carries the partition + AMR
state aggregated across cells) and the per-cell U / lam_cache arrays
(which carry physical moments and Newton warm-starts for collide+flux).

The tree's per-leaf w/x_s live on the master grid (via fit_maxent_weights);
the per-cell view here lives on the same clipped fine grid that
generate_regular_samples uses inside the joblib step. They never share
samples — they're two different bookkeeping layers wired together via
leaf.mu = U[:, g, :].sum(axis=0) at the top of every step.
"""
import numpy as np


def bounds_list_from_leaves(leaves):
    """Pack the tree's leaf bounds into the (n_leaves, 6) layout that
    collide() expects: [xlo, xhi, ylo, yhi, zlo, zhi] per row."""
    return np.array([
        [l.xbounds[0], l.xbounds[1],
         l.ybounds[0], l.ybounds[1],
         l.zbounds[0], l.zbounds[1]]
        for l in leaves
    ], dtype=np.float64)


def _compute_child_bounds(leaf):
    """Geometric child bounds for the split that leaf.split() will produce.
    Order matches what get_leaves() yields after the split: binary mode
    is [left, right] along leaf.split_dim; octree mode follows
    _split_octants' canonical ix*4+iy*2+iz ordering."""
    xb = list(leaf.xbounds)
    yb = list(leaf.ybounds)
    zb = list(leaf.zbounds)

    if leaf.split_mode == 'octree':
        mx = 0.5 * (xb[0] + xb[1])
        my = 0.5 * (yb[0] + yb[1])
        mz = 0.5 * (zb[0] + zb[1])
        out = []
        for ix in (0, 1):
            xb_c = [xb[0], mx] if ix == 0 else [mx, xb[1]]
            for iy in (0, 1):
                yb_c = [yb[0], my] if iy == 0 else [my, yb[1]]
                for iz in (0, 1):
                    zb_c = [zb[0], mz] if iz == 0 else [mz, zb[1]]
                    out.append((xb_c, yb_c, zb_c))
        return out

    d = leaf.split_dim
    all_b = [xb, yb, zb]
    mid = 0.5 * (all_b[d][0] + all_b[d][1])
    cb_L = [list(xb), list(yb), list(zb)]
    cb_R = [list(xb), list(yb), list(zb)]
    cb_L[d] = [all_b[d][0], mid]
    cb_R[d] = [mid, all_b[d][1]]
    return [(cb_L[0], cb_L[1], cb_L[2]),
            (cb_R[0], cb_R[1], cb_R[2])]


def partition_leaf_moments_per_cell(U_g, lam_g, leaf_bounds,
                                    child_bounds_list,
                                    n_fine=31, n_sigma=3.0,
                                    rho_floor=None):
    """Split one cell's moments for one parent leaf into k child moments.

    Reconstructs w = exp(lam · phi) on the *same clipped fine grid* that
    generate_regular_samples used during the most recent joblib step, then
    masks by each child's bounds and accumulates moments. Mass is
    approximately conserved (w is the Newton-converged form of the lam fit;
    any partition error is ≤ a single-cell-wall sample's contribution and
    gets cleaned up by the next step's refit).

    Returns array shape (k_children, 5). All-zero rows for children whose
    intersection with the clipped grid is empty, or whenever the parent
    cell has no mass / no fitted lam.
    """
    n_children = len(child_bounds_list)
    out = np.zeros((n_children, 5))

    if rho_floor is None:
        from ..config_1d import THRESHOLDS
        rho_floor = THRESHOLDS['cell_rho_floor']

    if U_g[0] < rho_floor or np.all(lam_g == 0):
        return out

    ux = U_g[1] / U_g[0]
    uy = U_g[2] / U_g[0]
    uz = U_g[3] / U_g[0]
    T  = max(2.0 * (U_g[4] / U_g[0] - ux**2 - uy**2 - uz**2) / 3.0, 1e-10)
    v_th = np.sqrt(T)

    xb, yb, zb = leaf_bounds
    xlo = max(ux - n_sigma * v_th, xb[0])
    xhi = min(ux + n_sigma * v_th, xb[1])
    ylo = max(uy - n_sigma * v_th, yb[0])
    yhi = min(uy + n_sigma * v_th, yb[1])
    zlo = max(uz - n_sigma * v_th, zb[0])
    zhi = min(uz + n_sigma * v_th, zb[1])

    if xhi <= xlo or yhi <= ylo or zhi <= zlo:
        return out

    gx = np.linspace(xlo, xhi, n_fine)
    gy = np.linspace(ylo, yhi, n_fine)
    gz = np.linspace(zlo, zhi, n_fine)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    x_s = GX.ravel(); y_s = GY.ravel(); z_s = GZ.ravel()

    w = np.exp(lam_g[0]
               + lam_g[1] * x_s + lam_g[2] * y_s + lam_g[3] * z_s
               + lam_g[4] * (x_s**2 + y_s**2 + z_s**2))

    # Closed-open intervals match split()'s convention (samples on the
    # midpoint go to the right sibling). The very last sample at the
    # parent's outer wall would otherwise fall through; nudge the
    # rightmost child's upper bound by a tiny eps so it still picks up
    # the wall sample.
    eps = 1e-12
    for k, (cb_x, cb_y, cb_z) in enumerate(child_bounds_list):
        x_hi_eff = cb_x[1] + (eps if cb_x[1] >= xhi - eps else 0.0)
        y_hi_eff = cb_y[1] + (eps if cb_y[1] >= yhi - eps else 0.0)
        z_hi_eff = cb_z[1] + (eps if cb_z[1] >= zhi - eps else 0.0)
        mask = ((x_s >= cb_x[0]) & (x_s < x_hi_eff) &
                (y_s >= cb_y[0]) & (y_s < y_hi_eff) &
                (z_s >= cb_z[0]) & (z_s < z_hi_eff))
        if not mask.any():
            continue
        wm = w[mask]
        out[k, 0] = wm.sum()
        out[k, 1] = (x_s[mask] * wm).sum()
        out[k, 2] = (y_s[mask] * wm).sum()
        out[k, 3] = (z_s[mask] * wm).sum()
        out[k, 4] = ((x_s[mask]**2 + y_s[mask]**2 + z_s[mask]**2) * wm).sum()

    return out


def apply_splits(root, U, lam_cache, AMR_cfg, t):
    """Decide and apply per-leaf splits. Returns (U_new, lam_cache_new),
    possibly with an expanded second axis. Per-cell U columns for split
    leaves are partitioned via the converged lam from lam_cache; new
    children warm-start their lam from the parent's row (Newton on the
    next step refines)."""
    leaves = root.get_leaves()
    numXj  = U.shape[0]

    decisions = []
    for g, leaf in enumerate(leaves):
        gate = (leaf.kl_accum > AMR_cfg['KL_accum_threshold']
                and leaf.rate_ema > AMR_cfg['rate_split_threshold'])
        if not gate:
            decisions.append(('keep', g, None, None, None))
            continue

        if leaf.split_mode == 'octree':
            allowed = leaf.can_split()
            block_dim = None
        else:
            allowed = leaf.can_split(leaf.split_dim)
            block_dim = leaf.split_dim

        if not allowed:
            reason = leaf.split_block_reason(block_dim)
            print(f't={t}: cannot split depth={leaf.depth} '
                  f'{("axis=" + str(block_dim) + " ") if block_dim is not None else ""}'
                  f'reason={reason}')
            decisions.append(('keep', g, None, None, None))
            continue

        child_bounds = _compute_child_bounds(leaf)
        n_children = len(child_bounds)
        leaf_bounds = (leaf.xbounds, leaf.ybounds, leaf.zbounds)

        child_U   = np.zeros((numXj, n_children, 5))
        child_lam = np.zeros((numXj, n_children, 5))
        for c in range(numXj):
            child_U[c] = partition_leaf_moments_per_cell(
                U[c, g, :], lam_cache[c, g, :], leaf_bounds, child_bounds)
            for k in range(n_children):
                child_lam[c, k, :] = lam_cache[c, g, :]

        kind = 'octree-splitting' if leaf.split_mode == 'octree' else 'splitting'
        axis = '' if leaf.split_mode == 'octree' else f'axis={leaf.split_dim} '
        print(f't={t}: {kind} depth={leaf.depth} {axis}'
              f'bounds=(x={leaf.xbounds},y={leaf.ybounds},z={leaf.zbounds}), '
              f'KL={leaf.kl_accum:.4f}, rate={leaf.rate_ema:.4f}')

        decisions.append(('split', g, child_U, child_lam, leaf))

    # Apply tree-side splits once decisions are finalized — leaf.split()
    # mutates the tree, so don't interleave with the decision pass.
    any_split = False
    for kind, _, _, _, leaf in decisions:
        if kind == 'split':
            leaf.split(t)
            any_split = True

    if not any_split:
        return U, lam_cache

    new_U = []
    new_lam = []
    for kind, g_old, child_U, child_lam, _ in decisions:
        if kind == 'keep':
            new_U.append(U[:, g_old:g_old+1, :])
            new_lam.append(lam_cache[:, g_old:g_old+1, :])
        else:
            new_U.append(child_U)
            new_lam.append(child_lam)

    return np.concatenate(new_U, axis=1), np.concatenate(new_lam, axis=1)


def apply_coarsens(root, U, lam_cache, AMR_cfg, t):
    """Decide and apply per-parent merges. Returns (U_new, lam_cache_new)
    with a contracted second axis. Children's U columns are summed into
    the parent (mass-conservative). Children's lam_cache rows are
    discarded — the parent warm-starts from zeros, matching
    merge_children() which fits Maxent on the parent box from zero lam."""
    leaves = root.get_leaves()
    leaf_idx = {id(l): i for i, l in enumerate(leaves)}
    numXj = U.shape[0]

    expected_n        = 8 if AMR_cfg.get('split_mode', 'binary') == 'octree' else 2
    rate_threshold    = AMR_cfg['rate_coarsen_threshold']
    entropy_threshold = AMR_cfg.get('entropy_coarsen_threshold', np.inf)

    merges = []
    checked = set()
    for leaf in leaves:
        if leaf.parent is None:
            continue
        parent = leaf.parent
        if id(parent) in checked:
            continue
        checked.add(id(parent))

        if parent.is_leaf() or len(parent.children) != expected_n:
            continue
        children = parent.children
        if not all(c.is_leaf() for c in children):
            continue
        if not all(c.can_coarsen(t) for c in children):
            continue
        if not all(c.rate_ema < rate_threshold for c in children):
            continue

        dS = parent.coarsen_entropy_increase()
        if dS > entropy_threshold:
            continue

        rates_str = ', '.join(f'{c.rate_ema:.4f}' for c in children)
        print(f't={t}: coarsening {len(children)} children -> '
              f'(x={parent.xbounds},y={parent.ybounds},z={parent.zbounds}), '
              f'rates=[{rates_str}], dS={dS:.5f}')
        merges.append((parent, [leaf_idx[id(c)] for c in children]))

    if not merges:
        return U, lam_cache

    # Apply tree merges first — merge_children() can fail (refit on parent
    # box rejects); skip those at the array-rebuild stage.
    successful = []
    for parent, child_idxs in merges:
        if parent.merge_children(t):
            successful.append((parent, child_idxs))

    if not successful:
        return U, lam_cache

    merged_indices = set()
    leftmost = {}      # leftmost old idx -> child idx list
    for _, child_idxs in successful:
        merged_indices.update(child_idxs)
        leftmost[min(child_idxs)] = child_idxs

    new_U = []
    new_lam = []
    skip_until = -1
    n_old = U.shape[1]
    for g in range(n_old):
        if g < skip_until:
            continue
        if g in leftmost:
            child_idxs = leftmost[g]
            parent_U   = U[:, child_idxs, :].sum(axis=1, keepdims=True)
            parent_lam = np.zeros((numXj, 1, 5))
            new_U.append(parent_U)
            new_lam.append(parent_lam)
            skip_until = max(child_idxs) + 1
        elif g in merged_indices:
            # Non-leftmost child of an already-handled merge.
            continue
        else:
            new_U.append(U[:, g:g+1, :])
            new_lam.append(lam_cache[:, g:g+1, :])

    return np.concatenate(new_U, axis=1), np.concatenate(new_lam, axis=1)
