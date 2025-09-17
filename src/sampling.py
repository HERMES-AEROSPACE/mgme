import numpy as np
from .config import VELOCITY_SPACE, GROUP_PARAMS
from numba import jit
from scipy import optimize
import sys
from matplotlib import pyplot as plt


@jit(nopython=True)
def f(x, y, z, A, b, wx, wy, wz):
    return A * np.exp(-b * ((x - wx)**2 + (y - wy)**2 + (z - wz)**2))

def func(x, M, Q):
    return np.matmul(M, x) - Q

def calculate_volume_elements(x_centers, y_centers, z_centers):
    """
    Calculate volume elements for discretization with given center points.
    Boundary points get full spacing to neighbors (extending past centers).
    """
    def get_spacings(centers, boundary_positions):
        # Calculate spacings between adjacent centers
        center_spacings = centers[1:] - centers[:-1]
        
        spacings = np.zeros_like(centers)

        # Interior points: average of adjacent spacings
        spacings[1:-1] = (center_spacings[:-1] + center_spacings[1:]) / 2

        # Boundary points: full spacing to neighbor
        spacings[0] = center_spacings[0]   # Full distance to next center
        spacings[-1] = center_spacings[-1]  # Full distance to previous center

        for boundary in boundary_positions:
            left_of_boundary = centers < boundary
            right_of_boundary = centers > boundary

            if np.any(left_of_boundary) and np.any(right_of_boundary):
                # Find the indices of points closest to boundary on each side
                left_boundary_idx = np.where(left_of_boundary)[0][-1]  # rightmost point left of boundary
                right_boundary_idx = np.where(right_of_boundary)[0][0]  # leftmost point right of boundary
                
                # Adjust spacing for point just left of boundary
                distance_to_boundary = boundary - centers[left_boundary_idx]
                if left_boundary_idx > 0:
                    spacings[left_boundary_idx] = center_spacings[left_boundary_idx-1]/2 + distance_to_boundary
                else:
                    spacings[left_boundary_idx] = distance_to_boundary
                    
                # Adjust spacing for point just right of boundary  
                distance_from_boundary = centers[right_boundary_idx] - boundary
                if right_boundary_idx < len(centers) - 1:
                    spacings[right_boundary_idx] = distance_from_boundary + center_spacings[right_boundary_idx]/2
                else:
                    spacings[right_boundary_idx] = distance_from_boundary
        
        return spacings
    
    dx = get_spacings(x_centers, [-1, 0, 1])
    dy = get_spacings(y_centers, [-1, 0, 1]) 
    dz = get_spacings(z_centers, [-1, 0, 1])
    print(dx)
    
    # Create 3D mesh of volume elements
    DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing='ij')
    volume_elements = DX * DY * DZ
    
    return volume_elements.flatten()

def generate_grid(n_samples_x, n_samples_y, n_samples_z):
    sample_loc_x = np.linspace(*VELOCITY_SPACE['cx_range'], n_samples_x)
    sample_loc_y = np.linspace(*VELOCITY_SPACE['cy_range'], n_samples_y)
    sample_loc_z = np.linspace(*VELOCITY_SPACE['cz_range'], n_samples_z)
    print(sample_loc_x)

    [xgrid, ygrid, zgrid] = np.meshgrid(sample_loc_x, sample_loc_y, sample_loc_z, indexing='ij')

    x_sample = xgrid.flatten()
    y_sample = ygrid.flatten()
    z_sample = zgrid.flatten()

    return x_sample, y_sample, z_sample, sample_loc_x, sample_loc_y, sample_loc_z

@jit(nopython=True)
def generate_regular_samples_helper(mu, x_sample, y_sample, z_sample, vol_elem, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, Ak, bk, wxk, wyk, wzk):
    mask = (x_sample >= ci_cx) & (x_sample <= cf_cx) & \
    (y_sample >= ci_cy) & (y_sample <= cf_cy) & \
    (z_sample >= ci_cz) & (z_sample <= cf_cz)

    x_sample_slice = x_sample[mask]
    y_sample_slice = y_sample[mask]
    z_sample_slice = z_sample[mask]

    sum_f_group = np.sum(f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk) * vol_elem[mask])
    num_group_sample = len(x_sample_slice)
    if Ak == 0.0 and bk == 0.0  and wxk == 0.0 and wyk == 0.0 and wzk == 0.0:
        weights = np.zeros(len(x_sample_slice))
    else:
        weights = mu[0] * f(x_sample_slice, y_sample_slice, z_sample_slice, Ak, bk, wxk, wyk, wzk) * vol_elem[mask] / sum_f_group

    return num_group_sample, weights, mask

def generate_regular_samples(n_samples, x_sample, y_sample, z_sample, curr_groups, vol_elem):
    weights = np.zeros(n_samples)
    num_group_sample = np.zeros(len(curr_groups))

    for i, group in enumerate(curr_groups):
        ci_cx, cf_cx = group.group_bounds['ci_cx'], group.group_bounds['cf_cx']
        ci_cy, cf_cy = group.group_bounds['ci_cy'], group.group_bounds['cf_cy']
        ci_cz, cf_cz = group.group_bounds['ci_cz'], group.group_bounds['cf_cz']

        Ak, bk, wxk, wyk, wzk = group.A, group.b, group.wx, group.wy, group.wz
        mu = group.mu
        n_group_sample, group_weights, mask = generate_regular_samples_helper(mu, x_sample, y_sample, z_sample, vol_elem, \
                                                                                       ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz, Ak, bk, wxk, wyk, wzk)
        
        num_group_sample[i] = n_group_sample
        weights[mask] = group_weights

    return weights, num_group_sample

# def reweight_samples(x_sample, y_sample, z_sample, weights, num_group_sample, mu):
#     new_weights = np.zeros(int(np.sum(num_group_sample)))

#     for i in range(0, NUM_GROUPS_CX):
#         for j in range(0, NUM_GROUPS_CY):
#             for k in range(0, NUM_GROUPS_CZ):
#                 M = np.zeros((5, int(num_group_sample[i, j, k])))
#                 Q = np.zeros((5,))

#                 x_mask = np.asarray(np.logical_and(x_sample >= CI_CX[i], x_sample <= CF_CX[i])).nonzero()
#                 y_mask = np.asarray(np.logical_and(y_sample >= CI_CY[j], y_sample <= CF_CY[j])).nonzero()
#                 z_mask = np.asarray(np.logical_and(z_sample >= CI_CZ[k], z_sample <= CF_CZ[k])).nonzero()
#                 mask = np.array(list(set(x_mask[0].flatten()) & set(y_mask[0].flatten()) & set(z_mask[0].flatten())))

#                 M[0, :] = 1
#                 M[1, :] = x_sample[mask]
#                 M[2, :] = y_sample[mask]
#                 M[3, :] = z_sample[mask]
#                 M[4, :] = (x_sample[mask]**2 + y_sample[mask]**2 + z_sample[mask]**2)

#                 Q[0] = mu[i, j, k, 0]
#                 Q[1] = mu[i, j, k, 1]
#                 Q[2] = mu[i, j, k, 2]
#                 Q[3] = mu[i, j, k, 3]
#                 Q[4] = mu[i, j, k, 4]
                
#                 sol = optimize.least_squares(func, weights[mask], args=(M, Q), bounds=(0.0, 1.0), loss='soft_l1')
#                 new_weights[mask] = sol.x

#     return new_weights
