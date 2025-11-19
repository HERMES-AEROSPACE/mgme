import numpy as np


# Velocity space grid parameters
VELOCITY_SPACE = {
    'num_cx': 121,
    'num_cy': 121,
    'num_cz': 121,
    'cx_range': (-7.0, 7.0),
    'cy_range': (-7.0, 7.0),
    'cz_range': (-7.0, 7.0)
}

PHYS_SPACE = {
    'num_xj': 471,
    'xj_range': [-14, 10]
}

# Group parameters
GROUP_PARAMS = {
    # 'ci_cx': np.array([-5.0, -0.5, 0.0, 0.5]),
    # 'cf_cx': np.array([-0.5, 0.0, 0.5, 5.0]),
    # 'group_bounds_cx': np.array([[0, 109], [108, 121], [120, 133], [132, 241]]),
    # 'ci_cy': np.array([-5.0, -0.5, 0.0, 0.5]),
    # 'cf_cy': np.array([-0.5, 0.0, 0.5, 5.0]),
    # 'group_bounds_cy': np.array([[0, 109], [108, 121], [120, 133], [132, 241]]),
    # 'ci_cz': np.array([-5.0, -0.5, 0.0, 0.5]),
    # 'cf_cz': np.array([-0.5, 0.0, 0.5, 5.0]),
    # 'group_bounds_cz': np.array([[0, 109], [108, 121], [120, 133], [132, 241]])
    'ci_cx': np.array([-7, 0.4666666666666668]),
    'cf_cx': np.array([0.4666666666666668, 7.0]),
    'group_bounds_cx': np.array([[0, 65], [64, 121]]),
    'ci_cy': np.array([-7.0, 0.0]),
    'cf_cy': np.array([0.0, 7.0]),
    'group_bounds_cy': np.array([[0, 61], [60, 121]]),
    'ci_cz': np.array([-7.0, 0.0]),
    'cf_cz': np.array([0.0, 7.0]),
    'group_bounds_cz': np.array([[0, 61], [60, 121]])
    # 'ci_cx': np.array([-3, -1.0, 0.0, 1.0]),
    # 'cf_cx': np.array([-1.0, 0.0, 1.0, 3.0]),
    # 'group_bounds_cx': np.array([[0, 81], [80, 121], [120, 161], [160, 241]]),
    # 'ci_cy': np.array([-3, 0.0]),
    # 'cf_cy': np.array([0.0, 3.0]),
    # 'group_bounds_cy': np.array([[0, 121], [120, 241]]),
    # 'ci_cz': np.array([-3, 0.0]),
    # 'cf_cz': np.array([0.0, 3.0]),
    # 'group_bounds_cz': np.array([[0, 121], [120, 241]])
}

# AMR parameters
AMR = {
    'threshold': 0.01
}

# Collision parameters
COLLISION_PARAMS = {
    'n_coll': 200000,
    'n_t': 100
}

LOOKUP_TABLE = {
    'n_points': 600
}

SAMPLING_PARAMS = {
    'n_samples_x': 19,
    'n_samples_y': 14,
    'n_samples_z': 14
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
    'P1': 0.415,
    'Ma1': 2
}
