import numpy as np
import os
from datetime import datetime
from .config import GROUP_PARAMS, VELOCITY_SPACE, COLLISION_PARAMS, SAMPLING_PARAMS

def save_simulation_data(t, curr_groups_list, save_dir='simulation_data'):
    """Save simulation data at time step t.
    
    Args:
        t: Current time step
        curr_groups_list: Array of group nodes.
        save_dir: Directory to save data
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save data
    np.save(f'{save_dir}/groups{t}_{timestamp}.npy', curr_groups_list)
    
    # Save metadata
    metadata = {
        'time_step': t,
        # 'num_groups_cx': GROUP_PARAMS['num_groups_cx'],
        # 'num_groups_cy': GROUP_PARAMS['num_groups_cy'],
        # 'num_groups_cz': GROUP_PARAMS['num_groups_cz'],
        # 'group_bounds_cx': GROUP_PARAMS['group_bounds_cx'],
        # 'group_bounds_cy': GROUP_PARAMS['group_bounds_cy'],
        # 'group_bounds_cz': GROUP_PARAMS['group_bounds_cz'],
        # 'ci_cx': GROUP_PARAMS['ci_cx'],
        # 'cf_cx': GROUP_PARAMS['cf_cx'],
        # 'ci_cy': GROUP_PARAMS['ci_cy'],
        # 'cf_cy': GROUP_PARAMS['cf_cy'],
        # 'ci_cz': GROUP_PARAMS['ci_cz'],
        # 'cf_cz': GROUP_PARAMS['cf_cz'],
        'num_cx': VELOCITY_SPACE['num_cx'],
        'num_cy': VELOCITY_SPACE['num_cy'],
        'num_cz': VELOCITY_SPACE['num_cz'],
        'cx_range': VELOCITY_SPACE['cx_range'],
        'cy_range': VELOCITY_SPACE['cy_range'],
        'cz_range': VELOCITY_SPACE['cz_range'],
        'n_coll': COLLISION_PARAMS['n_coll'],
        'dt': COLLISION_PARAMS['dt'],
        'n_t': COLLISION_PARAMS['n_t'],
        'n_samples_x': SAMPLING_PARAMS['n_samples_x'],
        'n_samples_y': SAMPLING_PARAMS['n_samples_y'],
        'n_samples_z': SAMPLING_PARAMS['n_samples_z']
    }
    np.save(f'{save_dir}/metadata_t{t}_{timestamp}.npy', metadata)
