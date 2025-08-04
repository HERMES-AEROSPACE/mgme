import numpy as np


# Velocity space grid parameters
VELOCITY_SPACE = {
    'num_cx': 241,
    'num_cy': 241,
    'num_cz': 241,
    'cx_range': (-5.0, 5.0),
    'cy_range': (-5.0, 5.0),
    'cz_range': (-5.0, 5.0)
}

# Group parameters
GROUP_PARAMS = {
    'num_groups_cx': 4,
    'num_groups_cy': 4,
    'num_groups_cz': 4,
    'ci_cx': np.array([-5.0, -0.5, 0.0, 0.5]),
    'cf_cx': np.array([-0.5, 0.0, 0.5, 5.0]),
    'group_bounds_cx': np.array([[0, 109], [108, 121], [120, 133], [132, 241]]),
    'ci_cy': np.array([-5.0, -0.5, 0.0, 0.5]),
    'cf_cy': np.array([-0.5, 0.0, 0.5, 5.0]),
    'group_bounds_cy': np.array([[0, 109], [108, 121], [120, 133], [132, 241]]),
    'ci_cz': np.array([-5.0, -0.5, 0.0, 0.5]),
    'cf_cz': np.array([-0.5, 0.0, 0.5, 5.0]),
    'group_bounds_cz': np.array([[0, 109], [108, 121], [120, 133], [132, 241]])
    # 'num_groups_cx': 2,
    # 'num_groups_cy': 1,
    # 'num_groups_cz': 1,
    # 'ci_cx': np.array([-3.0, 0.0]),
    # 'cf_cx': np.array([0.0, 3.0]),
    # 'group_bounds_cx': np.array([[0, 121], [120, 241]]),
    # 'ci_cy': np.array([-3.0]),
    # 'cf_cy': np.array([3.0]),
    # 'group_bounds_cy': np.array([[0, 241]]),
    # 'ci_cz': np.array([-3.0]),
    # 'cf_cz': np.array([3.0]),
    # 'group_bounds_cz': np.array([[0, 241]])
}

# AMR parameters
AMR = {
    'threshold': 0.01
}

# Collision parameters
COLLISION_PARAMS = {
    'n_coll': 2000000,
    'dt': 0.2,
    'n_t': 100
}

LOOKUP_TABLE = {
    'n_points': 600
}

SAMPLING_PARAMS = {
    'n_samples_x': 48,
    'n_samples_y': 48,
    'n_samples_z': 48
}

# Helper function to get velocity space grid
def calculate_velocity_grid():
    cx_vec = np.linspace(*VELOCITY_SPACE['cx_range'], VELOCITY_SPACE['num_cx'])
    cy_vec = np.linspace(*VELOCITY_SPACE['cy_range'], VELOCITY_SPACE['num_cy'])
    cz_vec = np.linspace(*VELOCITY_SPACE['cz_range'], VELOCITY_SPACE['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 
