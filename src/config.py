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
    # 'num_groups_cx': 4,
    # 'num_groups_cy': 4,
    # 'num_groups_cz': 4,
    # 'ci_cx': np.array([-5.0, -0.5, 0.0, 0.5]),
    # 'cf_cx': np.array([-0.5, 0.0, 0.5, 5.0]),
    # 'group_bounds_cx': np.array([[0, 109], [108, 121], [120, 133], [132, 241]]),
    # 'ci_cy': np.array([-5.0, -0.5, 0.0, 0.5]),
    # 'cf_cy': np.array([-0.5, 0.0, 0.5, 5.0]),
    # 'group_bounds_cy': np.array([[0, 109], [108, 121], [120, 133], [132, 241]]),
    # 'ci_cz': np.array([-5.0, -0.5, 0.0, 0.5]),
    # 'cf_cz': np.array([-0.5, 0.0, 0.5, 5.0]),
    # 'group_bounds_cz': np.array([[0, 109], [108, 121], [120, 133], [132, 241]])
    'num_groups_cx': 4,
    'num_groups_cy': 1,
    'num_groups_cz': 1,
    'ci_cx': np.array([-3.0, -0.5, 0.0, 0.5]),
    'cf_cx': np.array([-0.5, 0.0, 0.5, 3.0]),
    'group_bounds_cx': np.array([[0, 101], [100, 121], [120, 141], [140, 241]]),
    'ci_cy': np.array([-3.0]),
    'cf_cy': np.array([3.0]),
    'group_bounds_cy': np.array([[0, 241]]),
    'ci_cz': np.array([-3.0]),
    'cf_cz': np.array([3.0]),
    'group_bounds_cz': np.array([[0, 241]])
}

# AMR parameters
AMR = {
    'threshold': 0.01
}

# Collision parameters
COLLISION_PARAMS = {
    'n_coll': 200000,
    'dt': 0.2,
    'n_t': 30
}

LOOKUP_TABLE = {
    'n_points': 600
}

SAMPLING_PARAMS = {
    'n_samples_x': 34,
    'n_samples_y': 34,
    'n_samples_z': 34
}
