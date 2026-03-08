import numpy as np


# Velocity space grid parameters
VELOCITY_SPACE = {
    'num_cx': 121,
    'num_cy': 121,
    'num_cz': 121,
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

# AMR parameters
AMR = {
    'entropy_threshold': 0.01
}

# Collision parameters
COLLISION_PARAMS = {
    'n_coll': 20000
}

CONSTANTS = {
    'm': 6.633521421485196e-26,
    'k': 1.380649e-23,
    'R': 208.12055672374083,
    'gamma': 1.6666666666666667,
    'd': 3.974e-10
}
