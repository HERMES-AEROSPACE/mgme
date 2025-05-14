import numpy as np
import os
from datetime import datetime
from .config import GROUP_PARAMS, VELOCITY_SPACE, COLLISION_PARAMS, SAMPLING_PARAMS

def save_simulation_data(t, Ak_list, bk_list, wk_list, save_dir='simulation_data'):
    """Save simulation data at time step t.
    
    Args:
        t: Current time step
        Ak_list, bk_list, wk_list: Parameter arrays
        save_dir: Directory to save data
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save data
    np.save(f'{save_dir}/Ak_t{t}_{timestamp}.npy', Ak_list[:t+1])
    np.save(f'{save_dir}/bk_t{t}_{timestamp}.npy', bk_list[:t+1])
    np.save(f'{save_dir}/wk_t{t}_{timestamp}.npy', wk_list[:t+1])
    
    # Save metadata
    metadata = {
        'time_step': t,
        'num_groups': GROUP_PARAMS['num_groups'],
        'group_bounds': GROUP_PARAMS['group_bounds'],
        'ci': GROUP_PARAMS['ci'],
        'cf': GROUP_PARAMS['cf'],
        'num_cx': VELOCITY_SPACE['num_cx'],
        'num_cy': VELOCITY_SPACE['num_cy'],
        'num_cz': VELOCITY_SPACE['num_cz'],
        'cx_range': VELOCITY_SPACE['cx_range'],
        'cy_range': VELOCITY_SPACE['cy_range'],
        'cz_range': VELOCITY_SPACE['cz_range'],
        'n_coll': COLLISION_PARAMS['n_coll'],
        'dt': COLLISION_PARAMS['dt'],
        'n_t': COLLISION_PARAMS['n_t'],
        'n_samples_dir': SAMPLING_PARAMS['n_samples_dir']
    }
    np.save(f'{save_dir}/metadata_t{t}_{timestamp}.npy', metadata)
