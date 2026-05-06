import numpy as np


# Velocity space grid parameters
VELOCITY_SPACE = {
    # 'num_cx': 106,
    # 'num_cy': 106,
    # 'num_cz': 106,
    # 'cx_range': (-5.0, 5.5),
    # 'cy_range': (-5.0, 5.5),
    # 'cz_range': (-5.0, 5.5)
    'num_cx': 321,
    'num_cy': 106,
    'num_cz': 106,
    'cx_range': (-15.0, 17),
    'cy_range': (-15.0, 15),
    'cz_range': (-15.0, 15)
}

PHYS_SPACE = {
    'num_xj': 101,
    'xj_range': [-25, 15]
}

# Group parameters
GROUP_PARAMS = {
    # 'ci_cx': np.array([-5.0, -1.0, 0.0, 1.0, 2.0]),
    # 'cf_cx': np.array([-1.0, 0.0, 1.0, 2.0, 5.5]),
    # 'group_bounds_cx': np.array([[0, 41], [40, 51], [50, 61], [60, 71], [70, 106]]),
    # 'ci_cy': np.array([-5.0]), 
    # 'cf_cy': np.array([5.5]),
    # 'group_bounds_cy': np.array([[0, 106]]),
    # 'ci_cz': np.array([-5.0]), 
    # 'cf_cz': np.array([5.5]),
    # 'group_bounds_cz': np.array([[0, 106]])
    'ci_cx': np.array([-15.0, -7.0, 0.0, 6.5]),
    'cf_cx': np.array([-7.0, 0.0, 6.5, 17.0]),
    'group_bounds_cx': np.array([[0, 81], [80, 151], [150, 216], [215, 321]]),
    'ci_cy': np.array([-15.0]), 
    'cf_cy': np.array([15.0]),
    'group_bounds_cy': np.array([[0, 106]]),
    'ci_cz': np.array([-15.0]), 
    'cf_cz': np.array([15.0]),
    'group_bounds_cz': np.array([[0, 106]])
}

# Random simulation parameters
SIMULATION_PARAMS = {
    'n_coll': 100000,
    'cfl': 0.7,
    't_end': 25.0,
    'alpha': 1.0
}

CONSTANTS = {
    'm': 6.633521421485196e-26,
    'k': 1.380649e-23,
    'R': 208.12055672374083,
    'gamma': 1.6666666666666667,
    'd': 3.974e-10
}

FREESTREAM_PARAMS = {
    'T1': 300,
    'P1': 6.667,
    'Ma1': 9.0,
    'omega': 0.72  # Variable Hard Sphere Model: 1.0 - Pseudo-Maxwell, 0.5 - Hard Sphere, 0.811 - VHS Argon
}

# Single source of truth for the per-cell, per-group low-density gate used
# by sample fits, flux closure, and AMR partition. Distinct from
# VelocityGroup.n_threshold (1e-6), which is the cell-summed leaf-empty
# floor — this is the per-cell scale, typically 10x larger.
THRESHOLDS = {
    'cell_rho_floor': 1e-5,
}

# AMR (velocity-space refinement) parameters. Mirrors config_0d.AMR but
# tuned for the shock's velocity-space scale. Knob meanings: see config_0d.
AMR = {
    'KL_accum_threshold':       0.1,
    'rate_coarsen_threshold':   0.0001,
    'rate_split_threshold':     0.001,
    'rate_ema_gamma':           0.9,
    'entropy_coarsen_threshold': 0.005,
    'min_lifetime':             5,
    # max_depth=0 disables runtime AMR — useful for the smoke test.
    'max_depth':                4,
    'split_mode':               'binary',
    'split_axes':               [0],
    # Number of times to bisect the root along axis 0 before the main loop
    # starts. A value of 0 keeps the simulation single-group, which makes
    # collide() return zero (every collision pair stays in the only group),
    # so kl_accum / rate_ema can never grow and runtime AMR never fires.
    # Set to >=1 so collide redistributes between leaves and the AMR
    # signal pipeline can build drift.
    'initial_splits':           2,
}

# Master sample grid for the AMR signal machinery (NOT for collide samples).
# VelocityGroup.set_master_grid populates the per-leaf weights used by
# accumulate_kl, update_shadows, and coarsen_entropy_increase. The shock's
# collide+flux samples are still built per-cell by generate_regular_samples
# on its own 31^3 clipped grid.
MASTER_GRID = {
    'bounds':              ((-15.0, 17.0), (-15.0, 15.0), (-15.0, 15.0)),
    'spacing':             (1.0, 1.0, 1.0),
    'min_points_per_axis': 3,
}
