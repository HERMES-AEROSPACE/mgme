import numpy as np
from ..banner import print_banner
from ..shock_helper import ic, KT_central2, generate_regular_samples
from ..shock_helper import calc_flux_analytical, calc_flux_int
from ..physics.grid import calculate_velocity_grid
from ..physics.collide import collide
from ..config_1d import (CONSTANTS, FREESTREAM_PARAMS, PHYS_SPACE, GROUP_PARAMS,
                          VELOCITY_SPACE, SIMULATION_PARAMS, AMR, MASTER_GRID)
from ..amr import VelocityGroup, fit_maxent_weights
from ..amr.shock_amr import (bounds_list_from_leaves, apply_splits,
                              apply_coarsens)
import itertools
from scipy import special
from joblib import Parallel, delayed, parallel_backend
import time, sys, cProfile, io, pstats
from numba import types
from scipy.stats import qmc
from matplotlib import pyplot as plt


def run_simulation():
    print_banner()
    print("---------------------------CALCULATING SHOCK QUANTITIES---------------------------")

    gamma = CONSTANTS['gamma']
    R = CONSTANTS['R']
    d = CONSTANTS['d']
    T1 = FREESTREAM_PARAMS['T1']
    P1 = FREESTREAM_PARAMS['P1']
    Ma1 = FREESTREAM_PARAMS['Ma1']
    m = CONSTANTS['m']
    k = CONSTANTS['k']
    omega = FREESTREAM_PARAMS['omega']

    alpha = SIMULATION_PARAMS['alpha']
    n_coll = SIMULATION_PARAMS['n_coll']
    CX_LB, CX_UB = VELOCITY_SPACE['cx_range']
    CY_LB, CY_UB = VELOCITY_SPACE['cy_range']
    CZ_LB, CZ_UB = VELOCITY_SPACE['cz_range']
    key_type = types.UniTuple(types.int64, 2)

    # Some pre-shock quantities.
    a1 = np.sqrt(gamma * R * T1)
    u1 = Ma1 * a1
    rho1 = P1/(R * T1)
    n1 = P1/(R * T1) * 1/m
    m_r = 0.5  # reduced mass

    # Post shock quantities.
    T2 = T1 * (((gamma - 1) * Ma1**2 + 2) * (2 * gamma * Ma1**2 - (gamma - 1)))/((gamma + 1)**2 * Ma1**2)
    P2 = P1 * ((2 * gamma * Ma1**2) - (gamma - 1))/(gamma + 1)
    rho2 = P2/(R * T2)
    a2 = np.sqrt(gamma * R * T2)
    u2 = u1 * rho1/rho2
    Ma2 = Ma1 * u2/u1 * (T1/T2)**0.5
    n2 = P2/(R * T2) * 1/m
    print(n1)

    # Set up reference variables.
    m_ref = m
    n_ref = n1
    T_ref = T1
    d_ref = d
    c_ref = np.sqrt((2 * k * T_ref)/m_ref)
    sigma_ref = np.pi * d_ref**2
    lam_ref = 1/(n_ref * sigma_ref)
    L_ref = lam_ref
    Kn = lam_ref / L_ref
    t_ref = L_ref / c_ref
    gamma_omega = special.gamma(5/2 - omega)
    sigma_coeff_hat = m_r**(0.5 - omega) / gamma_omega
    print('Reference mean free path:', lam_ref, '[m]')
    print('Characteristic length:', L_ref, '[m]')
    print('Knudsen number:', Kn)
    print('Collision cross section omega:', omega)

    print("---------------------------SETTING UP INITIAL CONDITION---------------------------")
    # Set up velocity and physical space grid.
    cx_vec, cy_vec, cz_vec, cx, cy, cz = calculate_velocity_grid(VELOCITY_SPACE)
    xj_vec = np.linspace(PHYS_SPACE['xj_range'][0], PHYS_SPACE['xj_range'][1], PHYS_SPACE['num_xj'])
    dx = np.abs(xj_vec[1] - xj_vec[0])
    dcx = np.abs(cx_vec[1] - cx_vec[0])
    dcy = np.abs(cy_vec[1] - cy_vec[0])
    dcz = np.abs(cz_vec[1] - cz_vec[0])
    Xj_l = PHYS_SPACE['xj_range'][0]
    Xj_u = PHYS_SPACE['xj_range'][1]
    numXj = PHYS_SPACE['num_xj']

    # Set up time step and simulation parameters.
    cfl = SIMULATION_PARAMS['cfl']
    t_end = SIMULATION_PARAMS['t_end']
    T_local = T2 / T_ref   # downstream temperature, nondimensional
    T_d = 1.0          # reference temperature used to calculate species diameter
    v_max = np.max([np.abs(CX_UB), np.abs(CX_LB)])

    tc_vhs = 1 / (2 * (d/d_ref)**2 * (n2/n_ref) * np.sqrt(2 * T_d / np.pi) * (T_local/T_d)**(1-omega))
    tc_conv = dx / v_max

    dt = np.round(cfl * np.min([tc_vhs, tc_conv]), 3)

    print('CFL number:', cfl)
    print('Collision time scale:', tc_vhs * t_ref, '[s]')
    print('Time step:', dt)
    print('dx:', dx)

    # Calculate initial condition on physical grid (Maxwellian everywhere,
    # cosine ramped from freestream to post-shock conditions).
    transition_start = -25
    transition_end = 15
    ramp_length = transition_end - transition_start

    t_ramp = (xj_vec - transition_start) / ramp_length
    t_ramp = np.clip(t_ramp, 0, 1)
    cosine_factor = 0.5 * (1 - np.cos(np.pi * t_ramp))

    T_val = (T1 + cosine_factor * (T2 - T1)) / T_ref
    u_val = (u1 + cosine_factor * (u2 - u1)) / c_ref
    n_val = (n1 + cosine_factor * (n2 - n1)) / n_ref

    # ── AMR setup: install shock-side knobs and the master grid, then build
    # a single-leaf root spanning the entire velocity domain. Runtime AMR
    # develops the partition as drift signals build up; we skip
    # initial_refine since the IC is Maxwellian everywhere (KL ≈ 0).
    VelocityGroup.set_amr_config(AMR)
    VelocityGroup.set_master_grid(MASTER_GRID)
    root = VelocityGroup(
        bounds=np.array([[CX_LB, CX_UB], [CY_LB, CY_UB], [CZ_LB, CZ_UB]]),
        depth=0, max_depth=AMR['max_depth'],
        split_axes=AMR['split_axes'],
        split_mode=AMR['split_mode'],
    )

    # Single-group IC integration: ic() expects integer index slices into
    # the velocity grid; for the root leaf those are just [0, num_c?] for
    # each axis.
    single_group_bounds = np.array([[[0, VELOCITY_SPACE['num_cx']],
                                     [0, VELOCITY_SPACE['num_cy']],
                                     [0, VELOCITY_SPACE['num_cz']]]])
    restart = 0
    U0, f = ic(cx, cy, cz, cx_vec, cy_vec, cz_vec,
               n_val, u_val, T_val,
               VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'], VELOCITY_SPACE['num_cz'],
               numXj, 1, single_group_bounds)
    if restart:
        data = np.load('simulation_data/U2000.npy')
        print('Restarting from...')
        U = data
    else:
        U = U0.copy()

    initial_f = 'simulation_data/U{}.npy'.format(0)
    with open(initial_f, 'wb') as file:
        np.save(file, U)

    # ── Seed the tree's per-leaf state from the IC. The first
    # update_shadows() seeds the drift baseline used by accumulate_kl;
    # without it, kl_accum would stay at 0 and the root would never split.
    def _refit_and_seed_leaves(U_now):
        """Set leaf.mu = cell-summed moments and refit Maxent on the
        master grid. Tolerant: leaves whose moments are infeasible inside
        their bounds (e.g., tail leaves with near-zero cell-summed mass)
        are marked is_empty rather than raising. This matches the 0-D
        driver's fallback behavior."""
        ll = root.get_leaves()
        for g, leaf in enumerate(ll):
            leaf.mu = U_now[:, g, :].sum(axis=0)
            if leaf.mu[0] < leaf.n_threshold:
                leaf.is_empty = True
                continue
            result = fit_maxent_weights(leaf.mu, leaf.xbounds, leaf.ybounds,
                                        leaf.zbounds, leaf.lam)
            if result is None:
                # Geometrically infeasible (e.g., cell-summed mean velocity
                # outside leaf bounds) or Newton diverged. Mark empty so
                # collide() can still route particles into this leaf and
                # reactivate() picks it up later.
                print(f'  warning: fit failed on initial leaf {g} '
                      f'bounds=(x={leaf.xbounds},y={leaf.ybounds},z={leaf.zbounds}), '
                      f'mu={leaf.mu} — marking is_empty')
                leaf.is_empty = True
                continue
            leaf.is_empty = False
            leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = result
            leaf.update_shadows()
        return ll

    leaves_list = _refit_and_seed_leaves(U)

    # ── Pre-loop bisection. With a single root leaf collide() always
    # returns zero (every collision pair stays in the only group, so
    # loss and gain cancel exactly) — kl_accum/rate_ema would never
    # build drift and runtime AMR would never fire. Bisect the root a
    # few times along axis 0 to give collide somewhere to scatter to,
    # then runtime AMR takes over.
    #
    # Each pass: bisect every current leaf at its midpoint. NO refit is
    # done inside the loop — leaf.split() already populates each child's
    # mu / w / lam / x_s / shadows from its share of the parent's
    # master-grid Maxent fit, which is geometrically consistent with the
    # child's bounds. (Refitting on cell-summed mu inside the loop would
    # try to fit FULL-domain moments into a HALF-domain leaf, which is
    # geometrically infeasible whenever the cell-summed mean velocity
    # falls outside the half-leaf — Newton fails.)
    n_initial = AMR.get('initial_splits', 0)
    for _ in range(n_initial):
        for leaf in root.get_leaves():
            if leaf.can_split(leaf.split_dim):
                leaf.split(0)

    # Build leaf-bounds → grid-index slices for ic(), which expects integer
    # index ranges. Boundary leaves whose upper bound matches the grid wall
    # need to be promoted to the full count (searchsorted returns the
    # interior index otherwise).
    def _leaf_index_bounds(leaves):
        out = []
        for leaf in leaves:
            xlo_i = int(np.searchsorted(cx_vec, leaf.xbounds[0]))
            xhi_i = int(np.searchsorted(cx_vec, leaf.xbounds[1]))
            ylo_i = int(np.searchsorted(cy_vec, leaf.ybounds[0]))
            yhi_i = int(np.searchsorted(cy_vec, leaf.ybounds[1]))
            zlo_i = int(np.searchsorted(cz_vec, leaf.zbounds[0]))
            zhi_i = int(np.searchsorted(cz_vec, leaf.zbounds[1]))
            if leaf.xbounds[1] >= cx_vec[-1] - 1e-9:
                xhi_i = VELOCITY_SPACE['num_cx']
            if leaf.ybounds[1] >= cy_vec[-1] - 1e-9:
                yhi_i = VELOCITY_SPACE['num_cy']
            if leaf.zbounds[1] >= cz_vec[-1] - 1e-9:
                zhi_i = VELOCITY_SPACE['num_cz']
            out.append([[xlo_i, xhi_i], [ylo_i, yhi_i], [zlo_i, zhi_i]])
        return np.array(out)

    def _reintegrate_ic(leaves):
        """Re-integrate the (cosine-ramped Maxwellian) IC onto the current
        partition. Cell 0 of the result is the freestream-Maxwellian
        inflow used by the per-step BC; the remaining cells aren't read,
        but the full array keeps shapes aligned with U."""
        gb = _leaf_index_bounds(leaves)
        U_new, _ = ic(cx, cy, cz, cx_vec, cy_vec, cz_vec, n_val, u_val, T_val,
                      VELOCITY_SPACE['num_cx'], VELOCITY_SPACE['num_cy'],
                      VELOCITY_SPACE['num_cz'],
                      numXj, len(leaves), gb)
        return U_new

    # After all initial splits: re-integrate the IC over the new bounds,
    # then refit leaves on the freshly-integrated cell-summed moments.
    leaves_list = root.get_leaves()
    if len(leaves_list) > 1:
        U0 = _reintegrate_ic(leaves_list)
        U = U0.copy()
        with open(initial_f, 'wb') as file:
            np.save(file, U)
        leaves_list = _refit_and_seed_leaves(U)

    bounds_list = bounds_list_from_leaves(leaves_list)
    num_groups  = len(leaves_list)
    print(f'Initial partition: {num_groups} leaves')
    for g, leaf in enumerate(leaves_list):
        empty = ' (EMPTY)' if leaf.is_empty else ''
        print(f'  leaf {g}: x={leaf.xbounds}, y={leaf.ybounds}, z={leaf.zbounds}, '
              f'cell-summed rho={leaf.mu[0]:.3f}{empty}')

    print("--------------------------------BEGIN SIMULATION----------------------------------")
    # Cache distribution parameters for Newton solver.
    lam_cache = np.zeros((numXj, num_groups, 5))
    # k1_c from the previous step — feeds the next step's AMR signal pass.
    # Initialized to zeros so step 1's update_rate sees no drive (rate_ema
    # gets seeded on step 2).
    k1_c_prev = np.zeros((numXj, num_groups, 5))
    pr = cProfile.Profile()

    def step(i, U_i, bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, sigma_coeff_hat, omega, alpha, lam_cache_i):
        # Calculate weights through optimization.
        weights, num_valid_samples, lam_out, x_s, y_s, z_s, offsets = generate_regular_samples(i, U_i, num_groups, bounds_list, lam_cache_i)

        # Advance the collision and flux forward.
        coll = collide(x_s, y_s, z_s, weights, num_valid_samples, bounds_list, num_groups, \
                        n_coll, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, key_type, sigma_coeff_hat, omega, alpha)
        flux = calc_flux_int(num_groups, weights, offsets, x_s, y_s, z_s)

        second_moment_x = 0.0
        rho  = np.sum(U_i[:, 0])          # cell total density
        rhou = np.sum(U_i[:, 1])          # cell total x-momentum
        ux   = rhou / rho if rho > 0 else 0.0
        for j in range(num_groups):
            start = int(offsets[j])
            end   = int(offsets[j+1])
            w_i   = weights[start:end]     # f * dv for this group
            vx_i  = x_s[start:end]
            second_moment_x += np.sum(vx_i**2 * w_i)
        Tx = second_moment_x / rho - ux**2 if rho > 0 else 0.0

        return i, coll, flux, lam_out, Tx

    n_iters = int(np.ceil(int(t_end / dt) / 100) * 100) + 1
    for t in range(1, n_iters):
        # 1. Boundary conditions (use current U shape).
        U[0, :] = U0[0, :]
        U[-1, :] = U[-2, :]

        # 2. AMR signal update + decisions, BEFORE collide. Tree's view
        # of the system is cell-summed moments; refit on master grid so
        # accumulate_kl has up-to-date leaf.w / leaf.lam to work with.
        #
        # Rate signal: max over cells of per-cell rate, where each cell's
        # contribution is leaf g's collision change normalized by THAT
        # cell's total moment norm. The most-active cell (shock layer)
        # drives the AMR gate; equilibrium freestream / post-shock cells
        # don't dilute it via cross-cell cancellation. Excess groups in
        # quiet cells are accepted as the cost of shock-layer fidelity.
        leaves_list = root.get_leaves()
        cell_mu_norm = np.linalg.norm(U.sum(axis=1), axis=1)  # (numXj,)
        g_ema = AMR['rate_ema_gamma']
        for g, leaf in enumerate(leaves_list):
            leaf.mu = U[:, g, :].sum(axis=0)
            result = fit_maxent_weights(leaf.mu, leaf.xbounds, leaf.ybounds,
                                        leaf.zbounds, leaf.lam)
            if result is not None:
                leaf.w, leaf.lam, leaf.x_s, leaf.y_s, leaf.z_s = result
            # else: keep previous fit; matches 0-D fallback behavior.

            num = np.linalg.norm(k1_c_prev[:, g, :] * dt, axis=1)  # (numXj,)
            safe = cell_mu_norm > 1e-12
            per_cell_r = np.zeros_like(num)
            per_cell_r[safe] = num[safe] / cell_mu_norm[safe]
            r = float(per_cell_r.max()) if per_cell_r.size else 0.0

            if not np.isfinite(leaf.rate_ema):
                leaf.rate_ema = r
            else:
                leaf.rate_ema = g_ema * leaf.rate_ema + (1.0 - g_ema) * r

            leaf.accumulate_kl()
            # print(f't={t} g={g} rate_ema={leaf.rate_ema:.3e} '
            #       f'kl_accum={leaf.kl_accum:.3e} '
            #       f'r_cellmax={r:.3e} '
            #       f'argmax_cell={int(per_cell_r.argmax()) if per_cell_r.size else -1}')

        n_leaves_before = U.shape[1]
        U, lam_cache = apply_splits(root, U, lam_cache, AMR, t)
        U, lam_cache = apply_coarsens(root, U, lam_cache, AMR, t)

        # 3. Refresh dynamic shape after AMR.
        leaves_list = root.get_leaves()
        num_groups  = len(leaves_list)
        bounds_list = bounds_list_from_leaves(leaves_list)
        # If AMR changed the partition, U0 (the inflow BC reference) needs
        # to be re-partitioned onto the new leaf set; otherwise next step's
        # `U[0, :] = U0[0, :]` broadcast-fails. (k1_c_prev is reassigned
        # to the new-shape k1_c at the bottom of the loop, so no resize
        # needed for it here.)
        if num_groups != n_leaves_before:
            U0 = _reintegrate_ic(leaves_list)

        k1_c = np.zeros((numXj, num_groups, 5))
        F1   = np.zeros((numXj, num_groups, 5))

        # 4. Per-cell parallel collide + flux on the new partition.
        with parallel_backend('loky', inner_max_num_threads=1):
            step_dt = Parallel(n_jobs=32)(
                delayed(step)(i, U[i], bounds_list, num_groups, CX_LB, CX_UB, CY_LB, CY_UB, CZ_LB, CZ_UB, \
                            key_type, sigma_coeff_hat, omega, alpha, lam_cache[i])
                for i in range(0, numXj)
            )

        Tx_hist = np.zeros(numXj)

        for i, coll, flux, lam_out, Tx in step_dt:
            k1_c[i, :, 0] = coll[0]
            k1_c[i, :, 1] = coll[1]
            k1_c[i, :, 2] = coll[2]
            k1_c[i, :, 3] = coll[3]
            k1_c[i, :, 4] = coll[4]
            F1[i] = flux
            lam_cache[i] = lam_out
            Tx_hist[i] = Tx

        # 5. Transport on the same new partition.
        k1_f = KT_central2(U, F1, numXj, num_groups, dt, dx, CX_LB, CX_UB)
        U += (k1_f + k1_c) * dt

        # Carry k1_c forward for next step's AMR signal update.
        k1_c_prev = k1_c

        if t % 50 == 0:
            f1 = 'simulation_data/U{}.npy'.format(t + 0)
            with open(f1, 'wb') as file:
                np.save(file, U)

            f2 = 'simulation_data/Tx{}.npy'.format(t + 0)
            with open(f2, 'wb') as file:
                np.save(file, Tx_hist)
            print(t * dt, t + 0, f'num_groups={num_groups}')

if __name__ == '__main__':
    run_simulation()
