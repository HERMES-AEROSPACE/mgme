import numpy as np
from itertools import product
from .moment_utils import calc_moment, invert
from .config import GROUP_PARAMS, AMR
from contextlib import contextmanager


@contextmanager
def suppress_fsolve_warnings():
    old_settings = np.seterr(invalid='ignore', divide='ignore')
    try:
        yield
    finally:
        np.seterr(**old_settings)

class GroupNode:
    def __init__(self, data: dict):
        self.group_bounds = data
        self.children = []

    def set_mu(self, mu):
        self.mu = mu

    def set_dist_param(self, A, b, wx, wy, wz):
        self.A = A
        self.b = b
        self.wx = wx
        self.wy = wy
        self.wz = wz

    def set_hellinger_distance(self, dist):
        self.hellinger_dist = dist

    def add_child(self, child):
        self.children.append(child)

    def update_parameters(self, dt, dn, dpx, dpy, dpz, de):
        self.mu[0] += dt * dn
        self.mu[1] += dt * dpx
        self.mu[2] += dt * dpy
        self.mu[3] += dt * dpz
        self.mu[4] += dt * de

        if self.mu[0] > 1e-4:
            self._update_group_dist_params([self.b, self.wx, self.wy, self.wz])
    
    def _update_group_dist_params(self, initial_guess):
        self.A, self.b, self.wx, self.wy, self.wz = invert(self.mu, initial_guess, self.group_bounds)


def calculate_hellinger_distance(f1, f2, cx_vec, cy_vec, cz_vec, params=GROUP_PARAMS):
    """
    Calculate the Hellinger distance between two distributions in a specific group.
    Make sure distributions are normalized to 1!!!!!!
    
    Args:
        f1, f2: Group distribution functions
        cx_vec, cy_vec, cz_vec: Velocity space vectors
        
    Returns:
        Hellinger distance between f1 and f2
    """
    # Calculate Hellinger distance
    # H(P,Q) = √(1/2) * √(∫(√P(x) - √Q(x))² dx)
    diff = np.sqrt(f1) - np.sqrt(f2)
    squared_diff = diff**2
    
    # Integrate over the group volume
    integral = np.trapz(np.trapz(np.trapz(squared_diff, cz_vec, axis=2), cy_vec, axis=1), cx_vec, axis=0)

    return np.sqrt(0.5 * integral)

def refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, node, max_depth=6, curr_depth=0):
    if node.hellinger_dist < AMR['threshold']:
        return
    if curr_depth >= max_depth:
        print(f"Warning: Maximum recursion depth {max_depth} reached")
        return
    else:
        # Split into octree data structure. Create 8 subnodes off of input node.
        ref_child = refine_group2(node.group_bounds)

        children = []
        for cx_idx, cy_idx, cz_idx in product([0, 1], repeat=3):
            # Get important group bounds as they will be used everywhere.
            lb_cx, ub_cx = ref_child['group_bounds_cx'][cx_idx]
            lb_cy, ub_cy = ref_child['group_bounds_cy'][cy_idx]
            lb_cz, ub_cz = ref_child['group_bounds_cz'][cz_idx]

            # Create child node.
            child = GroupNode({'ci_cx': ref_child['ci_cx'][cx_idx], 'cf_cx': ref_child['cf_cx'][cx_idx], 'group_bounds_cx': np.array([lb_cx, ub_cx]), \
                        'ci_cy': ref_child['ci_cy'][cy_idx], 'cf_cy': ref_child['cf_cy'][cy_idx], 'group_bounds_cy': np.array([lb_cy, ub_cy]), \
                        'ci_cz': ref_child['ci_cz'][cz_idx], 'cf_cz': ref_child['cf_cz'][cz_idx], 'group_bounds_cz': np.array([lb_cz, ub_cz])})

            node.add_child(child)

            f0_slice = f0[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
            cx_slice = cx[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
            cy_slice = cy[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
            cz_slice = cz[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
            cx_vec_slice = cx_vec[lb_cx:ub_cx]
            cy_vec_slice = cy_vec[lb_cy:ub_cy]
            cz_vec_slice = cz_vec[lb_cz:ub_cz]

            # Calculate mu in each sub-node and invert.
            mu = calc_moment(f0_slice, cx_slice, cy_slice, cz_slice, cx_vec_slice, cy_vec_slice, cz_vec_slice)

            child.set_mu(mu)
            
            if mu[0] > 1e-4:
                A, b, wx, wy, wz = invert(mu, [1.0, 0.0, 0.0, 0.0], child.group_bounds)
                child.set_dist_param(A, b, wx, wy, wz)
            
                # Calculate Hellinger distance.
                f = A * np.exp(-b * ((cx_slice - wx)**2 + (cy_slice - wy)**2 + (cz_slice - wz)**2))
                dist = calculate_hellinger_distance(f0_slice, f, cx_vec_slice, cy_vec_slice, cz_vec_slice, child.group_bounds)
                child.set_hellinger_distance(dist)

                children.append(child)
            else:
                child.set_dist_param(0.0, 0.0, 0.0, 0.0, 0.0)
                child.set_hellinger_distance(0.0)

                children.append(child)

        # Refine the children cells if needed.
        for child in children:
            refine_init(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, child, max_depth, curr_depth + 1)


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

def print_tree_structure(root, prefix="", is_last=True):
    """
    Print tree structure with visual formatting
    
    Args:
        root: TreeNode - the root of the tree/subtree
        prefix: str - prefix for current line (used for indentation)
        is_last: bool - whether this is the last child of its parent
    """
    if root is None:
        return
    
    # Print current node with proper connector
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}O")
    
    # Update prefix for children
    child_prefix = prefix + ("    " if is_last else "│   ")
    
    # Print all children
    for i, child in enumerate(root.children):
        is_child_last = (i == len(root.children) - 1)
        print_tree_structure(child, child_prefix, is_child_last)

def get_current_groups(node):
    leaves = []

    def _get_leaves(node, leaves):
        if node:
            if len(node.children) == 0:
                leaves.append(node)
            for n in node.children:
                _get_leaves(n, leaves)

    _get_leaves(node, leaves)
    return leaves

def custom_groups(f0, cx, cy, cz, cx_vec, cy_vec, cz_vec, root, group_params):
    group_bounds_cx = group_params['group_bounds_cx']
    group_bounds_cy = group_params['group_bounds_cy']
    group_bounds_cz = group_params['group_bounds_cz']
    ci_cx = group_params['ci_cx']
    cf_cx = group_params['cf_cx']
    ci_cy = group_params['ci_cy']
    cf_cy = group_params['cf_cy']
    ci_cz = group_params['ci_cz']
    cf_cz = group_params['cf_cz']

    idx_x, idx_y, idx_z = np.meshgrid(
        np.arange(len(ci_cx)),
        np.arange(len(ci_cy)),
        np.arange(len(ci_cz)),
        indexing='ij'
    )

    # Flatten the grids
    idx_x_flat = idx_x.flatten()
    idx_y_flat = idx_y.flatten()
    idx_z_flat = idx_z.flatten()

    velocity_bounds_cx = np.column_stack([ci_cx[idx_x_flat], cf_cx[idx_x_flat]])
    velocity_bounds_cy = np.column_stack([ci_cy[idx_y_flat], cf_cy[idx_y_flat]])
    velocity_bounds_cz = np.column_stack([ci_cz[idx_z_flat], cf_cz[idx_z_flat]])

    group_bounds_cx_selected = group_bounds_cx[idx_x_flat]
    group_bounds_cy_selected = group_bounds_cy[idx_y_flat]
    group_bounds_cz_selected = group_bounds_cz[idx_z_flat]

    for i in range(len(idx_x_flat)):
        data_dict = {
            'ci_cx': velocity_bounds_cx[i][0], 'cf_cx': velocity_bounds_cx[i][1], 'group_bounds_cx': group_bounds_cx_selected[i],
            'ci_cy': velocity_bounds_cy[i][0], 'cf_cy': velocity_bounds_cy[i][1], 'group_bounds_cy': group_bounds_cy_selected[i],
            'ci_cz': velocity_bounds_cz[i][0], 'cf_cz': velocity_bounds_cz[i][1], 'group_bounds_cz': group_bounds_cz_selected[i],
        }
        child = GroupNode(data_dict)

        lb_cx, ub_cx = group_bounds_cx_selected[i]
        lb_cy, ub_cy = group_bounds_cy_selected[i]
        lb_cz, ub_cz = group_bounds_cz_selected[i]
        f0_slice = f0[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
        cx_slice = cx[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
        cy_slice = cy[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
        cz_slice = cz[lb_cx:ub_cx, lb_cy:ub_cy, lb_cz:ub_cz]
        cx_vec_slice = cx_vec[lb_cx:ub_cx]
        cy_vec_slice = cy_vec[lb_cy:ub_cy]
        cz_vec_slice = cz_vec[lb_cz:ub_cz]

        mu = calc_moment(f0_slice, cx_slice, cy_slice, cz_slice, cx_vec_slice, cy_vec_slice, cz_vec_slice)
        child.set_mu(mu)
            
        if mu[0] > 1e-4:
            A, b, wx, wy, wz = invert(mu, [1.0, 0.0, 0.0, 0.0], child.group_bounds)
            child.set_dist_param(A, b, wx, wy, wz)
        
            # Calculate Hellinger distance.
            f = A * np.exp(-b * ((cx_slice - wx)**2 + (cy_slice - wy)**2 + (cz_slice - wz)**2))
            dist = calculate_hellinger_distance(f0_slice, f, cx_vec_slice, cy_vec_slice, cz_vec_slice, child.group_bounds)
            child.set_hellinger_distance(dist)
        else:
            child.set_dist_param(0.0, 0.0, 0.0, 0.0, 0.0)
            child.set_hellinger_distance(0.0)
        
        root.add_child(child)