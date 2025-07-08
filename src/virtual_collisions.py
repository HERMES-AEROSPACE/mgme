import numpy as np
from .config import GROUP_PARAMS, VELOCITY_SPACE, COLLISION_PARAMS
from numba import jit, types
from numba.typed import Dict
import sys
from matplotlib import pyplot as plt


CX_LB, CX_UB = VELOCITY_SPACE['cx_range']
CY_LB, CY_UB = VELOCITY_SPACE['cy_range']
CZ_LB, CZ_UB = VELOCITY_SPACE['cz_range']

n_coll = COLLISION_PARAMS['n_coll']

key_type = types.UniTuple(types.int64, 2)


@jit(nopython=True)
def collide(x_sample, y_sample, z_sample, weights, num_group_sample, bounds_list, n_samples, n_groups, Rf1, Rf2, depl_idx1, depl_idx2):
    group_n_d = np.zeros(n_groups)
    group_n_r = np.zeros(n_groups)
    group_px = np.zeros(n_groups)
    group_py = np.zeros(n_groups)
    group_pz = np.zeros(n_groups)
    group_e = np.zeros(n_groups)

    # depl_idx1 = np.random.randint(0, n_samples, n_coll)
    # depl_idx2 = np.random.randint(0, n_samples, n_coll)

    mask = depl_idx1 != depl_idx2

    depl_idx1 = depl_idx1[mask]
    depl_idx2 = depl_idx2[mask]

    # Group the prospective collisions into which group they end up in.
    depl_tracker = Dict.empty(key_type=key_type, \
                              value_type=types.int64)
    
    ci_cx = bounds_list[:, 0]
    cf_cx = bounds_list[:, 1]
    ci_cy = bounds_list[:, 2]
    cf_cy = bounds_list[:, 3]
    ci_cz = bounds_list[:, 4]
    cf_cz = bounds_list[:, 5]

    d_group_count = np.zeros(n_groups)
    r_group_count = np.zeros(n_groups)

    vx_collector = np.zeros(depl_idx1.size)
    vxp_collector = np.zeros(depl_idx1.size)
    
    for i in range(0, depl_idx1.size):
        x_valid = (x_sample[depl_idx1[i]] >= ci_cx) & (x_sample[depl_idx1[i]] <= cf_cx)
        y_valid = (y_sample[depl_idx1[i]] >= ci_cy) & (y_sample[depl_idx1[i]] <= cf_cy)
        z_valid = (z_sample[depl_idx1[i]] >= ci_cz) & (z_sample[depl_idx1[i]] <= cf_cz)
        depl_group1 = np.argmax(x_valid & y_valid & z_valid)

        x_valid = (x_sample[depl_idx2[i]] >= ci_cx) & (x_sample[depl_idx2[i]] <= cf_cx)
        y_valid = (y_sample[depl_idx2[i]] >= ci_cy) & (y_sample[depl_idx2[i]] <= cf_cy)
        z_valid = (z_sample[depl_idx2[i]] >= ci_cz) & (z_sample[depl_idx2[i]] <= cf_cz)
        depl_group2 = np.argmax(x_valid & y_valid & z_valid)

        if depl_group1 < depl_group2: key = (depl_group1, depl_group2)
        else: key = (depl_group2, depl_group1)

        if key in depl_tracker: depl_tracker[key] += 1
        else: depl_tracker[key] = 1

    for i in range(0, depl_idx1.size):
        d_idx1 = depl_idx1[i]
        d_idx2 = depl_idx2[i]

        vx1 = x_sample[d_idx1]
        vy1 = y_sample[d_idx1]
        vz1 = z_sample[d_idx1]
        vx2 = x_sample[d_idx2]
        vy2 = y_sample[d_idx2]
        vz2 = z_sample[d_idx2]

        # Simulate virtual collisions.
        gx = np.abs(vx2 - vx1)
        gy = np.abs(vy2 - vy1)
        gz = np.abs(vz2 - vz1)
        g = np.sqrt(gx**2 + gy**2 + gz**2)

        Rf = Rf1[i]
        phi = 2 * np.pi * Rf
        Rf = Rf2[i]
        cos_theta = 2 * Rf - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        gx_p = 0.5 * g * sin_theta * np.cos(phi)
        gy_p = 0.5 * g * sin_theta * np.sin(phi)
        gz_p = 0.5 * g * cos_theta

        V_x = 0.5 * (vx1 + vx2)
        V_y = 0.5 * (vy1 + vy2)
        V_z = 0.5 * (vz1 + vz2)

        vx1p = V_x - gx_p
        vy1p = V_y - gy_p
        vz1p = V_z - gz_p
        vx2p = V_x + gx_p
        vy2p = V_y + gy_p
        vz2p = V_z + gz_p

        vx_collector[i] = vy1p
        vxp_collector[i] = vy2p

        # Calculate loss rate for mass, momentum, and energy.
        x_valid = (vx1 >= ci_cx) & (vx1 <= cf_cx)
        y_valid = (vy1 >= ci_cy) & (vy1 <= cf_cy)
        z_valid = (vz1 >= ci_cz) & (vz1 <= cf_cz)
        group_idx1 = np.argmax(x_valid & y_valid & z_valid)

        x_valid = (vx2 >= ci_cx) & (vx2 <= cf_cx)
        y_valid = (vy2 >= ci_cy) & (vy2 <= cf_cy)
        z_valid = (vz2 >= ci_cz) & (vz2 <= cf_cz)
        group_idx2 = np.argmax(x_valid & y_valid & z_valid)

        d_group_count[group_idx1] += 1
        d_group_count[group_idx2] += 1

        # Calculate gain rate for mass, momentum, and energy.
        vx1p_clamped = np.minimum(np.maximum(vx1p, CX_LB), CX_UB)
        vy1p_clamped = np.minimum(np.maximum(vy1p, CY_LB), CY_UB)
        vz1p_clamped = np.minimum(np.maximum(vz1p, CZ_LB), CZ_UB)
        x_valid = (vx1p_clamped >= ci_cx) & (vx1p_clamped <= cf_cx)
        y_valid = (vy1p_clamped >= ci_cy) & (vy1p_clamped <= cf_cy)
        z_valid = (vz1p_clamped >= ci_cz) & (vz1p_clamped <= cf_cz)
        if np.count_nonzero(x_valid & y_valid & z_valid) != 1: print("uho h")
        group_idx1r = np.argmax(x_valid & y_valid & z_valid)

        vx2p_clamped = np.minimum(np.maximum(vx2p, CX_LB), CX_UB)
        vy2p_clamped = np.minimum(np.maximum(vy2p, CY_LB), CY_UB)
        vz2p_clamped = np.minimum(np.maximum(vz2p, CZ_LB), CZ_UB)
        x_valid = (vx2p_clamped >= ci_cx) & (vx2p_clamped <= cf_cx)
        y_valid = (vy2p_clamped >= ci_cy) & (vy2p_clamped <= cf_cy)
        z_valid = (vz2p_clamped >= ci_cz) & (vz2p_clamped <= cf_cz)
        if np.count_nonzero(x_valid & y_valid & z_valid) != 1: print("uho h")
        group_idx2r = np.argmax(x_valid & y_valid & z_valid)

        r_group_count[group_idx1r] += 1
        r_group_count[group_idx2r] += 1

        if group_idx1 < group_idx2: key = (group_idx1, group_idx2)
        else: key = (group_idx2, group_idx1)
        n_coll_group = depl_tracker[key]

        Li = weights[d_idx1] * weights[d_idx2] * num_group_sample[group_idx1] * num_group_sample[group_idx2] / n_coll_group
        group_n_d[group_idx1] -= Li
        group_px[group_idx1] -= Li * vx1
        group_py[group_idx1] -= Li * vy1
        group_pz[group_idx1] -= Li * vz1
        group_e[group_idx1] -= Li * (vx1**2 + vy1**2 + vz1**2)
        
        group_n_d[group_idx2] -= Li
        group_px[group_idx2] -= Li * vx2
        group_py[group_idx2] -= Li * vy2
        group_pz[group_idx2] -= Li * vz2
        group_e[group_idx2] -= Li * (vx2**2 + vy2**2 + vz2**2)

        Gi = Li
        group_n_r[group_idx1r] += Gi
        group_px[group_idx1r] += Gi * vx1p
        group_py[group_idx1r] += Gi * vy1p
        group_pz[group_idx1r] += Gi * vz1p
        group_e[group_idx1r] += Gi * (vx1p**2 + vy1p**2 + vz1p**2)

        group_n_r[group_idx2r] += Gi
        group_px[group_idx2r] += Gi * vx2p
        group_py[group_idx2r] += Gi * vy2p
        group_pz[group_idx2r] += Gi * vz2p
        group_e[group_idx2r] += Gi * (vx2p**2 + vy2p**2 + vz2p**2)
    
    return group_n_d, group_n_r, group_px, group_py, group_pz, group_e, d_group_count, r_group_count, vx_collector, vxp_collector
