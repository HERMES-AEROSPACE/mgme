import numpy as np
from .config import GROUP_PARAMS, VELOCITY_SPACE


def collide(x_sample, y_sample, z_sample, weights, num_group_sample, n_samples, n_coll):
    group_n = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    group_px = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    group_py = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    group_pz = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))
    group_e = np.zeros((GROUP_PARAMS['num_groups_cx'], GROUP_PARAMS['num_groups_cy'], GROUP_PARAMS['num_groups_cz']))

    for i in range(0, n_coll):
        # Draw random depletion velocities.
        depl_idx1 = int(np.floor(np.random.uniform(0, n_samples)))
        depl_idx2 = int(np.floor(np.random.uniform(0, n_samples)))
        if depl_idx1 == depl_idx2: continue

        vx1 = x_sample[depl_idx1]
        vy1 = y_sample[depl_idx1]
        vz1 = z_sample[depl_idx1]
        vx2 = x_sample[depl_idx2]
        vy2 = y_sample[depl_idx2]
        vz2 = z_sample[depl_idx2]

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
        group_idx1_x = np.argmax(np.logical_and(vx1 >= GROUP_PARAMS['ci_cx'], vx1 <= GROUP_PARAMS['cf_cx']))
        group_idx1_y = np.argmax(np.logical_and(vy1 >= GROUP_PARAMS['ci_cy'], vy1 <= GROUP_PARAMS['cf_cy']))
        group_idx1_z = np.argmax(np.logical_and(vz1 >= GROUP_PARAMS['ci_cz'], vz1 <= GROUP_PARAMS['cf_cz']))
        group_idx2_x = np.argmax(np.logical_and(vx2 >= GROUP_PARAMS['ci_cx'], vx2 <= GROUP_PARAMS['cf_cx']))
        group_idx2_y = np.argmax(np.logical_and(vy2 >= GROUP_PARAMS['ci_cy'], vy2 <= GROUP_PARAMS['cf_cy']))
        group_idx2_z = np.argmax(np.logical_and(vz2 >= GROUP_PARAMS['ci_cz'], vz2 <= GROUP_PARAMS['cf_cz']))

        Li = weights[depl_idx1] * weights[depl_idx2] * num_group_sample[group_idx1_x, group_idx1_y, group_idx1_z] * num_group_sample[group_idx2_x, group_idx2_y, group_idx2_z] / n_coll
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
        group_idx1_x = np.argmax(np.logical_and(vx1p >= GROUP_PARAMS['ci_cx'], vx1p <= GROUP_PARAMS['cf_cx']))
        group_idx1_y = np.argmax(np.logical_and(vy1p >= GROUP_PARAMS['ci_cy'], vy1p <= GROUP_PARAMS['cf_cy']))
        group_idx1_z = np.argmax(np.logical_and(vz1p >= GROUP_PARAMS['ci_cz'], vz1p <= GROUP_PARAMS['cf_cz']))
        group_idx2_x = np.argmax(np.logical_and(vx2p >= GROUP_PARAMS['ci_cx'], vx2p <= GROUP_PARAMS['cf_cx']))
        group_idx2_y = np.argmax(np.logical_and(vy2p >= GROUP_PARAMS['ci_cy'], vy2p <= GROUP_PARAMS['cf_cy']))
        group_idx2_z = np.argmax(np.logical_and(vz2p >= GROUP_PARAMS['ci_cz'], vz2p <= GROUP_PARAMS['cf_cz']))

        if vx1p > VELOCITY_SPACE['cx_range'][1]:
            group_idx1_x = GROUP_PARAMS['num_groups_cx'] - 1
        if vx2p > VELOCITY_SPACE['cx_range'][1]:
            group_idx2_x = GROUP_PARAMS['num_groups_cx'] - 1

        if vy1p > VELOCITY_SPACE['cy_range'][1]:
            group_idx1_y = GROUP_PARAMS['num_groups_cy'] - 1
        if vy2p > VELOCITY_SPACE['cy_range'][1]:
            group_idx2_y = GROUP_PARAMS['num_groups_cy'] - 1

        if vz1p > VELOCITY_SPACE['cz_range'][1]:
            group_idx1_z = GROUP_PARAMS['num_groups_cz'] - 1
        if vz2p > VELOCITY_SPACE['cz_range'][1]:
            group_idx2_z = GROUP_PARAMS['num_groups_cz'] - 1

        Gi = weights[depl_idx1] *  weights[depl_idx2] * num_group_sample[group_idx1_x, group_idx1_y, group_idx1_z] * num_group_sample[group_idx2_x, group_idx2_y, group_idx2_z] / n_coll
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