import numpy as np
from .config import GROUP_PARAMS, AMR


class GroupNode:
    def __init__(self, data: dict):
        self.group_bounds = data
        self.children = []

    def set_mu(self, mu):
        self.mu = mu

    def set_hellinger_distance(self, dist):
        self.hellinger_dist = dist

    def add_child(self, child):
        self.children.append(child)


def calculate_hellinger_distance(f1_group, f2_group, cx_vec, cy_vec, cz_vec, params=GROUP_PARAMS):
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
    lb_cx = params['group_bounds_cx'][0]
    ub_cx = params['group_bounds_cx'][1]
    lb_cy = params['group_bounds_cy'][0]
    ub_cy = params['group_bounds_cy'][1]
    lb_cz = params['group_bounds_cz'][0]
    ub_cz = params['group_bounds_cz'][1]
    
    
    # Calculate Hellinger distance
    # H(P,Q) = √(1/2) * √(∫(√P(x) - √Q(x))² dx)
    diff = np.sqrt(f1_group) - np.sqrt(f2_group)
    squared_diff = diff**2
    
    # Integrate over the group volume
    integral = np.trapz(np.trapz(np.trapz(squared_diff, cz_vec[lb_cz:ub_cz], axis=2), cy_vec[lb_cy:ub_cy], axis=1), cx_vec[lb_cx:ub_cx], axis=0)

    return np.sqrt(0.5 * integral)

def refine_group(node):
    if node.hellinger_dist < AMR['threshold']:
        return
    else:
        # Split into octree data structure. Create 8 subnodes off of input node.
        test_children = refine_group2(node.group_bounds)
        c1 = GroupNode({'ci_cx': np.array([test_children['ci_cx'][0]]), 'cf_cx': np.array([test_children['cf_cx'][0]]), \
                        'ci_cy': np.array([test_children['ci_cy'][0]]), 'cf_cy': np.array([test_children['cf_cy'][0]]), \
                        'ci_cz': np.array([test_children['ci_cz'][0]]), 'cf_cz': np.array([test_children['cf_cz'][0]]), \
                        'group_bounds_cx': np.array(test_children['group_bounds_cx'][0]), 'group_bounds_cy': np.array(test_children['group_bounds_cy'][0]), 'group_bounds_cz': np.array(test_children['group_bounds_cz'][0])})
        print(test_children)
        print(c1.group_bounds)

        # Calculate mu in each sub-node and invert.

        # Calculate Hellinger distance.

        # Refine the children cells if needed.
        # refine_group()


def refine_group2(group_bounds):
    """
    Refine a specific group by splitting it into smaller groups.
    
    Args:
        group_bounds
        
    Returns:
        Updated group bounds after refinement
    """
    # Get current group bounds
    ci_cx = group_bounds['ci_cx']
    cf_cx = group_bounds['cf_cx']
    ci_cy = group_bounds['ci_cy']
    cf_cy = group_bounds['cf_cy']
    ci_cz = group_bounds['ci_cz']
    cf_cz = group_bounds['cf_cz']
    
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
        [group_bounds['group_bounds_cx'][0], 
         group_bounds['group_bounds_cx'][0] + (group_bounds['group_bounds_cx'][1] - group_bounds['group_bounds_cx'][0])//2 + 1],
        [group_bounds['group_bounds_cx'][0] + (group_bounds['group_bounds_cx'][1] - group_bounds['group_bounds_cx'][0])//2,
         group_bounds['group_bounds_cx'][1]]
    ])
    
    new_group_bounds_cy = np.array([
        [group_bounds['group_bounds_cy'][0],
         group_bounds['group_bounds_cy'][0] + (group_bounds['group_bounds_cy'][1] - group_bounds['group_bounds_cy'][0])//2 + 1],
        [group_bounds['group_bounds_cy'][0] + (group_bounds['group_bounds_cy'][1] - group_bounds['group_bounds_cy'][0])//2,
         group_bounds['group_bounds_cy'][1]]
    ])
    
    new_group_bounds_cz = np.array([
        [group_bounds['group_bounds_cz'][0],
         group_bounds['group_bounds_cz'][0] + (group_bounds['group_bounds_cz'][1] - group_bounds['group_bounds_cz'][0])//2 + 1],
        [group_bounds['group_bounds_cz'][0] + (group_bounds['group_bounds_cz'][1] - group_bounds['group_bounds_cz'][0])//2,
         group_bounds['group_bounds_cz'][1]]
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
        'group_bounds_cz': new_group_bounds_cz
    }