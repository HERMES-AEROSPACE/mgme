import numpy as np
from .config import GROUP_PARAMS, VELOCITY_SPACE, COLLISION_PARAMS
from numba import jit, types
from numba.typed import Dict


NUM_GROUPS_CX = GROUP_PARAMS['num_groups_cx']
NUM_GROUPS_CY = GROUP_PARAMS['num_groups_cy']
NUM_GROUPS_CZ = GROUP_PARAMS['num_groups_cz']

CI_CX = GROUP_PARAMS['ci_cx']
CF_CX = GROUP_PARAMS['cf_cx']
CI_CY = GROUP_PARAMS['ci_cy']
CF_CY = GROUP_PARAMS['cf_cy']
CI_CZ = GROUP_PARAMS['ci_cz']
CF_CZ = GROUP_PARAMS['cf_cz']

CX_UB = VELOCITY_SPACE['cx_range'][1]
CY_UB = VELOCITY_SPACE['cy_range'][1]
CZ_UB = VELOCITY_SPACE['cz_range'][1]

n_coll = COLLISION_PARAMS['n_coll']

key_type = types.UniTuple(types.UniTuple(types.int64, 3), 2)


@jit(nopython=True)
def collide(x_sample, y_sample, z_sample, weights, num_group_sample, n_samples):
    group_n = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    group_px = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    group_py = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    group_pz = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))
    group_e = np.zeros((NUM_GROUPS_CX, NUM_GROUPS_CY, NUM_GROUPS_CZ))

    depl_idx1 = np.floor(np.random.uniform(0, n_samples, n_coll)).astype(np.int32)
    depl_idx2 = np.floor(np.random.uniform(0, n_samples, n_coll)).astype(np.int32)

    mask = depl_idx1 != depl_idx2

    depl_idx1 = depl_idx1[mask]
    depl_idx2 = depl_idx2[mask]

    # Group the prospective collisions into which group they end up in.
    depl_tracker = Dict.empty(key_type=key_type, \
                              value_type=types.int64)

    for i in range(0, depl_idx1.size):
        depl_group1_x = np.argmax(np.logical_and(x_sample[depl_idx1[i]] >= CI_CX, x_sample[depl_idx1[i]] <= CF_CX))
        depl_group1_y = np.argmax(np.logical_and(y_sample[depl_idx1[i]] >= CI_CY, y_sample[depl_idx1[i]] <= CF_CY))
        depl_group1_z = np.argmax(np.logical_and(z_sample[depl_idx1[i]] >= CI_CZ, z_sample[depl_idx1[i]] <= CF_CZ))

        depl_group2_x = np.argmax(np.logical_and(x_sample[depl_idx2[i]] >= CI_CX, x_sample[depl_idx2[i]] <= CF_CX))
        depl_group2_y = np.argmax(np.logical_and(y_sample[depl_idx2[i]] >= CI_CY, y_sample[depl_idx2[i]] <= CF_CY))
        depl_group2_z = np.argmax(np.logical_and(z_sample[depl_idx2[i]] >= CI_CZ, z_sample[depl_idx2[i]] <= CF_CZ))

        depl_1 = (depl_group1_x, depl_group1_y, depl_group1_z)
        depl_2 = (depl_group2_x, depl_group2_y, depl_group2_z)

        if depl_1 < depl_2: key = (depl_1, depl_2)
        else: key = (depl_2, depl_1)

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

        Rf = np.random.uniform(0.0, 1.0)
        phi = 2 * np.pi * Rf
        Rf = np.random.uniform(0.0, 1.0)
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

        # Calculate loss rate for mass, momentum, and energy.
        group_idx1_x = np.argmax(np.logical_and(vx1 >= CI_CX, vx1 <= CF_CX))
        group_idx1_y = np.argmax(np.logical_and(vy1 >= CI_CY, vy1 <= CF_CY))
        group_idx1_z = np.argmax(np.logical_and(vz1 >= CI_CZ, vz1 <= CF_CZ))
        group_idx2_x = np.argmax(np.logical_and(vx2 >= CI_CX, vx2 <= CF_CX))
        group_idx2_y = np.argmax(np.logical_and(vy2 >= CI_CY, vy2 <= CF_CY))
        group_idx2_z = np.argmax(np.logical_and(vz2 >= CI_CZ, vz2 <= CF_CZ))

        tmp1 = (group_idx1_x, group_idx1_y, group_idx1_z)
        tmp2 = (group_idx2_x, group_idx2_y, group_idx2_z)
        if tmp1 < tmp2: key = (tmp1, tmp2)
        else: key = (tmp2, tmp1)
        n_coll_group = depl_tracker[key]

        Li = weights[d_idx1] * weights[d_idx2] * num_group_sample[group_idx1_x, group_idx1_y, group_idx1_z] * num_group_sample[group_idx2_x, group_idx2_y, group_idx2_z] / n_coll_group
        group_n[group_idx1_x, group_idx1_y, group_idx1_z] -= Li
        group_px[group_idx1_x, group_idx1_y, group_idx1_z] -= Li * vx1
        group_py[group_idx1_x, group_idx1_y, group_idx1_z] -= Li * vy1
        group_pz[group_idx1_x, group_idx1_y, group_idx1_z] -= Li * vz1
        group_e[group_idx1_x, group_idx1_y, group_idx1_z] -= Li * (vx1**2 + vy1**2 + vz1**2)
        
        group_n[group_idx2_x, group_idx2_y, group_idx2_z] -= Li
        group_px[group_idx2_x, group_idx2_y, group_idx2_z] -= Li * vx2
        group_py[group_idx2_x, group_idx2_y, group_idx2_z] -= Li * vy2
        group_pz[group_idx2_x, group_idx2_y, group_idx2_z] -= Li * vz2
        group_e[group_idx2_x, group_idx2_y, group_idx2_z] -= Li * (vx2**2 + vy2**2 + vz2**2)

        # Calculate gain rate for mass, momentum, and energy.
        group_idx1_x = np.argmax(np.logical_and(vx1p >= CI_CX, vx1p <= CF_CX))
        group_idx1_y = np.argmax(np.logical_and(vy1p >= CI_CY, vy1p <= CF_CY))
        group_idx1_z = np.argmax(np.logical_and(vz1p >= CI_CZ, vz1p <= CF_CZ))
        group_idx2_x = np.argmax(np.logical_and(vx2p >= CI_CX, vx2p <= CF_CX))
        group_idx2_y = np.argmax(np.logical_and(vy2p >= CI_CY, vy2p <= CF_CY))
        group_idx2_z = np.argmax(np.logical_and(vz2p >= CI_CZ, vz2p <= CF_CZ))

        if vx1p > CX_UB:
            group_idx1_x = NUM_GROUPS_CX - 1
        if vx2p > CX_UB:
            group_idx2_x = NUM_GROUPS_CX - 1

        if vy1p > CY_UB:
            group_idx1_y = NUM_GROUPS_CY - 1
        if vy2p > CY_UB:
            group_idx2_y = NUM_GROUPS_CY - 1

        if vz1p > CZ_UB:
            group_idx1_z = NUM_GROUPS_CZ - 1
        if vz2p > CZ_UB:
            group_idx2_z = NUM_GROUPS_CZ - 1

        Gi = Li
        group_n[group_idx1_x, group_idx1_y, group_idx1_z] += Gi
        group_px[group_idx1_x, group_idx1_y, group_idx1_z] += Gi * vx1p
        group_py[group_idx1_x, group_idx1_y, group_idx1_z] += Gi * vy1p
        group_pz[group_idx1_x, group_idx1_y, group_idx1_z] += Gi * vz1p
        group_e[group_idx1_x, group_idx1_y, group_idx1_z] += Gi * (vx1p**2 + vy1p**2 + vz1p**2)

        group_n[group_idx2_x, group_idx2_y, group_idx2_z] += Gi
        group_px[group_idx2_x, group_idx2_y, group_idx2_z] += Gi * vx2p
        group_py[group_idx2_x, group_idx2_y, group_idx2_z] += Gi * vy2p
        group_pz[group_idx2_x, group_idx2_y, group_idx2_z] += Gi * vz2p
        group_e[group_idx2_x, group_idx2_y, group_idx2_z] += Gi * (vx2p**2 + vy2p**2 + vz2p**2)
    
    return group_n, group_px, group_py, group_pz, group_e
