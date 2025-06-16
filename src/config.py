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
    'num_groups_cx': 2,
    'num_groups_cy': 2,
    'num_groups_cz': 2,
    'ci_cx': np.array([-3.0, 0.0]),
    'cf_cx': np.array([0.0, 3.0]),
    'group_bounds_cx': np.array([[0, 121], [120, 241]]),
    'ci_cy': np.array([-3.0, 0.0]),
    'cf_cy': np.array([0.0, 3.0]),
    'group_bounds_cy': np.array([[0, 121], [120, 241]]),
    'ci_cz': np.array([-3.0, 0.0]),
    'cf_cz': np.array([0.0, 3.0]),
    'group_bounds_cz': np.array([[0, 121], [120, 241]])
}

# AMR parameters
AMR = {
    'threshold': 0.01
}

# Collision parameters
COLLISION_PARAMS = {
    'n_coll': 500000,
    'dt': 0.2,
    'n_t': 100
}

LOOKUP_TABLE = {
    'n_points': 600
}

SAMPLING_PARAMS = {
    'n_samples_x': 98 * 2,
    'n_samples_y': 98 * 2,
    'n_samples_z': 98 * 2
}

# Helper function to get velocity space grid
def calculate_velocity_grid():
    cx_vec = np.linspace(*VELOCITY_SPACE['cx_range'], VELOCITY_SPACE['num_cx'])
    cy_vec = np.linspace(*VELOCITY_SPACE['cy_range'], VELOCITY_SPACE['num_cy'])
    cz_vec = np.linspace(*VELOCITY_SPACE['cz_range'], VELOCITY_SPACE['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 
