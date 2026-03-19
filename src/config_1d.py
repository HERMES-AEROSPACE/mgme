import numpy as np


# Velocity space grid parameters
VELOCITY_SPACE = {
    'num_cx': 106,
    'num_cy': 106,
    'num_cz': 106,
    'cx_range': (-5.0, 5.5),
    'cy_range': (-5.0, 5.5),
    'cz_range': (-5.0, 5.5)
}

PHYS_SPACE = {
    'num_xj': 101,
    'xj_range': [-15, 15]
}

# Group parameters
GROUP_PARAMS = {
    'ci_cx': np.array([-5.0, -1.0, 0.0, 1.0, 2.0]),
    'cf_cx': np.array([-1.0, 0.0, 1.0, 2.0, 5.5]),
    'group_bounds_cx': np.array([[0, 41], [40, 51], [50, 61], [60, 71], [70, 106]]),
    'ci_cy': np.array([-5.0]), 
    'cf_cy': np.array([5.5]),
    'group_bounds_cy': np.array([[0, 106]]),
    'ci_cz': np.array([-5.0]), 
    'cf_cz': np.array([5.5]),
    'group_bounds_cz': np.array([[0, 106]])
}

# Collision parameters
SIMULATION_PARAMS = {
    'n_coll': 40000,
    'cfl': 0.7,
    't_end': 35.0
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
    'P1': 6.6667,
    'Ma1': 2.05,
    'omega': 0.811  # Variable Hard Sphere Model: 1.0 - Pseudo-Maxwell, 0.5 - Hard Sphere, 0.811 - VHS Argon
}
