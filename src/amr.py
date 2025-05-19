import numpy as np
from .config import GROUP_PARAMS


def calculate_hellinger_distance(f1_group, f2_group, cx_vec, cy_vec, cz_vec, group_idx_x, group_idx_y, group_idx_z, params=GROUP_PARAMS):
    """
    Calculate the Hellinger distance between two distributions in a specific group.
    
    Args:
        f1_group, f2_group: Distribution functions
        cx_vec, cy_vec, cz_vec: Velocity space vectors
        group_idx_x, group_idx_y, group_idx_z: Group indices
        
    Returns:
        Hellinger distance between f1 and f2 in the specified group
    """
    # Get group bounds
    lb_cx = params['group_bounds_cx'][group_idx_x, 0]
    ub_cx = params['group_bounds_cx'][group_idx_x, 1]
    lb_cy = params['group_bounds_cy'][group_idx_y, 0]
    ub_cy = params['group_bounds_cy'][group_idx_y, 1]
    lb_cz = params['group_bounds_cz'][group_idx_z, 0]
    ub_cz = params['group_bounds_cz'][group_idx_z, 1]
    
    
    # Calculate Hellinger distance
    # H(P,Q) = √(1/2) * √(∫(√P(x) - √Q(x))² dx)
    diff = np.sqrt(f1_group) - np.sqrt(f2_group)
    squared_diff = diff * diff
    
    # Integrate over the group volume
    integral = np.trapz(np.trapz(np.trapz(squared_diff, 
                                         cz_vec[lb_cz:ub_cz], axis=2),
                                cy_vec[lb_cy:ub_cy], axis=1),
                       cx_vec[lb_cx:ub_cx], axis=0)
    
    return np.sqrt(0.5 * integral)

def refine_group(group_idx_x, group_idx_y, group_idx_z):
    """
    Refine a specific group by splitting it into smaller groups.
    
    Args:
        group_idx_x, group_idx_y, group_idx_z: Indices of the group to refine
        
    Returns:
        Updated group parameters after refinement
    """
    # Get current group bounds
    ci_cx = GROUP_PARAMS['ci_cx'][group_idx_x]
    cf_cx = GROUP_PARAMS['cf_cx'][group_idx_x]
    ci_cy = GROUP_PARAMS['ci_cy'][group_idx_y]
    cf_cy = GROUP_PARAMS['cf_cy'][group_idx_y]
    ci_cz = GROUP_PARAMS['ci_cz'][group_idx_z]
    cf_cz = GROUP_PARAMS['cf_cz'][group_idx_z]
    
    # Calculate midpoints
    mid_cx = (ci_cx + cf_cx) / 2
    mid_cy = (ci_cy + cf_cy) / 2
    mid_cz = (ci_cz + cf_cz) / 2
    
    # Create new group bounds
    new_ci_cx = np.array([ci_cx, mid_cx])
    new_cf_cx = np.array([mid_cx, cf_cx])
    new_ci_cy = np.array([ci_cy, mid_cy])
    new_cf_cy = np.array([mid_cy, cf_cy])
    new_ci_cz = np.array([ci_cz, mid_cz])
    new_cf_cz = np.array([mid_cz, cf_cz])
    
    # Update group bounds in velocity space
    new_group_bounds_cx = np.array([
        [GROUP_PARAMS['group_bounds_cx'][group_idx_x, 0], 
         GROUP_PARAMS['group_bounds_cx'][group_idx_x, 0] + (GROUP_PARAMS['group_bounds_cx'][group_idx_x, 1] - GROUP_PARAMS['group_bounds_cx'][group_idx_x, 0])//2 + 1],
        [GROUP_PARAMS['group_bounds_cx'][group_idx_x, 0] + (GROUP_PARAMS['group_bounds_cx'][group_idx_x, 1] - GROUP_PARAMS['group_bounds_cx'][group_idx_x, 0])//2,
         GROUP_PARAMS['group_bounds_cx'][group_idx_x, 1]]
    ])
    
    new_group_bounds_cy = np.array([
        [GROUP_PARAMS['group_bounds_cy'][group_idx_y, 0],
         GROUP_PARAMS['group_bounds_cy'][group_idx_y, 0] + (GROUP_PARAMS['group_bounds_cy'][group_idx_y, 1] - GROUP_PARAMS['group_bounds_cy'][group_idx_y, 0])//2 + 1],
        [GROUP_PARAMS['group_bounds_cy'][group_idx_y, 0] + (GROUP_PARAMS['group_bounds_cy'][group_idx_y, 1] - GROUP_PARAMS['group_bounds_cy'][group_idx_y, 0])//2,
         GROUP_PARAMS['group_bounds_cy'][group_idx_y, 1]]
    ])
    
    new_group_bounds_cz = np.array([
        [GROUP_PARAMS['group_bounds_cz'][group_idx_z, 0],
         GROUP_PARAMS['group_bounds_cz'][group_idx_z, 0] + (GROUP_PARAMS['group_bounds_cz'][group_idx_z, 1] - GROUP_PARAMS['group_bounds_cz'][group_idx_z, 0])//2 + 1],
        [GROUP_PARAMS['group_bounds_cz'][group_idx_z, 0] + (GROUP_PARAMS['group_bounds_cz'][group_idx_z, 1] - GROUP_PARAMS['group_bounds_cz'][group_idx_z, 0])//2,
         GROUP_PARAMS['group_bounds_cz'][group_idx_z, 1]]
    ])
    
    return {
        'ci_cx': new_ci_cx,
        'cf_cx': new_cf_cx,
        'ci_cy': new_ci_cy,
        'cf_cy': new_cf_cy,
        'ci_cz': new_ci_cz,
        'cf_cz': new_cf_cz,
        'group_bounds_cx': new_group_bounds_cx,
        'group_bounds_cy': new_group_bounds_cy,
        'group_bounds_cz': new_group_bounds_cz,
        'num_groups_cx': len(new_group_bounds_cx),
        'num_groups_cy': len(new_group_bounds_cy),
        'num_groups_cz': len(new_group_bounds_cz)
    }