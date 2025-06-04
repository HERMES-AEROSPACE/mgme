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
    'n_coll': 200000,
    'dt': 0.2,
    'n_t': 100
}

# Beta and w lists for calculations
BETA_W_LISTS = {
    'beta_list1': np.linspace(0.001, 1.1, 225),
    'beta_list2': np.linspace(1.1, 2.0, 225),
    'beta_list3': np.linspace(2.0, 4.5, 150),
    'w_list1': np.linspace(-4.5, -0.2, 200),
    'w_list2': np.linspace(-0.2, 0.2, 200),
    'w_list3': np.linspace(0.2, 4.5, 200)
}

LOOKUP_TABLE = {
    'n_points': 600
}

SAMPLING_PARAMS = {
    'n_samples_x': 98,
    'n_samples_y': 98,
    'n_samples_z': 98
}

# Helper function to get combined beta and w lists
def calculate_beta_w_lists():
    beta_list = np.append(np.append(BETA_W_LISTS['beta_list1'], 
                                   BETA_W_LISTS['beta_list2']), 
                         BETA_W_LISTS['beta_list3'])
    w_list = np.append(np.append(BETA_W_LISTS['w_list1'], 
                                BETA_W_LISTS['w_list2']), 
                      BETA_W_LISTS['w_list3'])
    
    return beta_list, w_list

# Helper function to get velocity space grid
def calculate_velocity_grid():
    cx_vec = np.linspace(*VELOCITY_SPACE['cx_range'], VELOCITY_SPACE['num_cx'])
    cy_vec = np.linspace(*VELOCITY_SPACE['cy_range'], VELOCITY_SPACE['num_cy'])
    cz_vec = np.linspace(*VELOCITY_SPACE['cz_range'], VELOCITY_SPACE['num_cz'])
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')

    return cx_vec, cy_vec, cz_vec, cx, cy, cz 
