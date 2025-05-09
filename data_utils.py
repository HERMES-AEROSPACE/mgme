import numpy as np
import os
from datetime import datetime


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
        'timestamp': timestamp
    }
    np.save(f'{save_dir}/metadata_t{t}_{timestamp}.npy', metadata)

def load_simulation_data(t, timestamp, save_dir='simulation_data'):
    """Load simulation data for a specific time step and timestamp.
    
    Args:
        t: Time step to load
        timestamp: Timestamp of the saved data
        save_dir: Directory containing the saved data
        
    Returns:
        Dictionary containing the loaded data
    """
    data = {}
    data['Ak'] = np.load(f'{save_dir}/Ak_t{t}_{timestamp}.npy')
    data['bk'] = np.load(f'{save_dir}/bk_t{t}_{timestamp}.npy')
    data['wk'] = np.load(f'{save_dir}/wk_t{t}_{timestamp}.npy')
    data['metadata'] = np.load(f'{save_dir}/metadata_t{t}_{timestamp}.npy', allow_pickle=True).item()
    
    return data 