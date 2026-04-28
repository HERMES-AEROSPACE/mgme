import numpy as np


# Velocity space grid parameters
VELOCITY_SPACE = {
    'num_cx': 241,
    'num_cy': 241,
    'num_cz': 241,
    'cx_range': (-3.0, 3.0),
    'cy_range': (-3.0, 3.0),
    'cz_range': (-3.0, 3.0)
}

# Group parameters
GROUP_PARAMS = {
    'ci_cx': np.array([-5.0, -2.0, -0.5, 1.0, 2.5]),
    'cf_cx': np.array([-2.0, -0.5, 1.0, 2.5, 5.5]),
    'group_bounds_cx': np.array([[0, 31], [30, 46], [45, 61], [60, 76], [75, 106]]),
    'ci_cy': np.array([-5.0, 0]), 
    'cf_cy': np.array([0, 5.5]),
    'group_bounds_cy': np.array([[0, 51], [50, 106]]),
    'ci_cz': np.array([-5.0, 0]), 
    'cf_cz': np.array([0, 5.5]),
    'group_bounds_cz': np.array([[0, 51], [50, 106]])
}

# Master sample grid — shared by every leaf. Each leaf's samples are the
# subset of master-grid points falling inside its bounds, so per-leaf point
# count tracks leaf size automatically.
MASTER_GRID = {
    'bounds':  ((-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)),
    'spacing': (0.1, 0.1, 0.1),     # per-axis (dx, dy, dz)
    'min_points_per_axis': 3,       # split-denial gate (per axis, per child)
}

# AMR parameters
AMR = {
    'KL_threshold': 0.04,
    'KL_accum_threshold': 0.5,
    # Rate-based coarsen criterion: relative dmu/dt smoothed by EMA across
    # steps. At equilibrium the rate signal collapses below this floor (set
    # to live above the n_coll-driven noise floor ~ 1/sqrt(n_coll)) and the
    # tree coarsens cleanly. Doubles as a split veto inside accumulate_kl.
    'rate_coarsen_threshold': 0.0001,
    'rate_ema_gamma': 0.9,
    # Structure-based veto on coarsening: if merging 8 children into the
    # parent would raise the Maxent entropy ceiling by more than this
    # (in nats), refuse the merge — the children captured non-Maxwellian
    # structure (e.g. BKW polynomial factor) that the parent's bigger
    # box can't represent. Catches premature coarsening during smooth
    # but non-Maxwellian relaxation phases that the rate criterion misses.
    'entropy_coarsen_threshold': 0.005,
    'min_lifetime': 1,      # minimum steps before coarsening allowed
    'max_depth': 4,
    # 'octree' (default): each split halves all 3 axes simultaneously, 1->8 children.
    # 'binary': halve one axis per split, cycled by depth via split_axes (legacy mode).
    'split_mode': 'octree',
    'split_axes': [0, 1, 2],  # binary mode only — axes cycled by depth
}

# Collision parameters
COLLISION_PARAMS = {
    'n_coll': 200000,
    'n_t': 100,
    'omega': 1.0,
    'alpha': 1.0,
    'dt': 0.1
}

CONSTANTS = {
    'm': 6.633521421485196e-26,
    'k': 1.380649e-23,
    'R': 208.12055672374083,
    'gamma': 1.6666666666666667,
    'd': 3.974e-10
}
