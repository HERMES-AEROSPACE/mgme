import math

import numpy as np
from numba import jit
from scipy import optimize, special


@jit(nopython=True)
def moments(beta, wx, wy, wz, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    I0x = np.sqrt(np.pi / (4 * beta)) * (math.erf(np.sqrt(beta) * (cf_cx - wx)) - math.erf(np.sqrt(beta) * (ci_cx - wx)))
    I0y = np.sqrt(np.pi / (4 * beta)) * (math.erf(np.sqrt(beta) * (cf_cy - wy)) - math.erf(np.sqrt(beta) * (ci_cy - wy)))
    I0z = np.sqrt(np.pi / (4 * beta)) * (math.erf(np.sqrt(beta) * (cf_cz - wz)) - math.erf(np.sqrt(beta) * (ci_cz - wz)))

    I1x = (np.exp(-beta * (ci_cx - wx)**2) - np.exp(-beta * (cf_cx - wx)**2)) / (2 * beta)
    I1y = (np.exp(-beta * (ci_cy - wy)**2) - np.exp(-beta * (cf_cy - wy)**2)) / (2 * beta)
    I1z = (np.exp(-beta * (ci_cz - wz)**2) - np.exp(-beta * (cf_cz - wz)**2)) / (2 * beta)

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cx - wx)**2) * (cf_cx - wx))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cx - wx)**2) * (ci_cx - wx))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (math.erf(np.sqrt(beta) * (cf_cx - wx)) - math.erf(np.sqrt(beta) * (ci_cx - wx)))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cy - wy)**2) * (cf_cy - wy))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cy - wy)**2) * (ci_cy - wy))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (math.erf(np.sqrt(beta) * (cf_cy - wy)) - math.erf(np.sqrt(beta) * (ci_cy - wy)))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(beta)) * \
        ((np.exp(-beta * (cf_cz - wz)**2) * (cf_cz - wz))/np.sqrt(np.pi * beta) - (np.exp(-beta * (ci_cz - wz)**2) * (ci_cz - wz))/np.sqrt(np.pi * beta)) + \
            np.sqrt(np.pi)/(4 * np.sqrt(beta**3)) * (math.erf(np.sqrt(beta) * (cf_cz - wz)) - math.erf(np.sqrt(beta) * (ci_cz - wz)))

    return [(I1x + wx*I0x) / I0x, (I1y + wy*I0y) / I0y, (I1z + wz*I0z) / I0z,
            (I2x + 2 * wx * I1x + wx**2 * I0x) / I0x + (I2y + 2 * wy * I1y + wy**2 * I0y) / I0y + (I2z + 2 * wz * I1z + wz**2 * I0z) / I0z]


def moment_eq(x, ux, uy, uz, e, ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz):
    I0x = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cx - x[1])) - special.erf(np.sqrt(x[0]) * (ci_cx - x[1])))
    I0y = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cy - x[2])) - special.erf(np.sqrt(x[0]) * (ci_cy - x[2])))
    I0z = np.sqrt(np.pi / (4 * x[0])) * (special.erf(np.sqrt(x[0]) * (cf_cz - x[3])) - special.erf(np.sqrt(x[0]) * (ci_cz - x[3])))

    I1x = (np.exp(-x[0] * (ci_cx - x[1])**2) - np.exp(-x[0] * (cf_cx - x[1])**2)) / (2 * x[0])
    I1y = (np.exp(-x[0] * (ci_cy - x[2])**2) - np.exp(-x[0] * (cf_cy - x[2])**2)) / (2 * x[0])
    I1z = (np.exp(-x[0] * (ci_cz - x[3])**2) - np.exp(-x[0] * (cf_cz - x[3])**2)) / (2 * x[0])

    I2x = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cx - x[1])**2) * (cf_cx - x[1]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cx - x[1])**2) * (ci_cx - x[1])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cx - x[1])) - special.erf(np.sqrt(x[0]) * (ci_cx - x[1])))
    I2y = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cy - x[2])**2) * (cf_cy - x[2]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cy - x[2])**2) * (ci_cy - x[2])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cy - x[2])) - special.erf(np.sqrt(x[0]) * (ci_cy - x[2])))
    I2z = -np.sqrt(np.pi) / (2 * np.sqrt(x[0])) * \
        ((np.exp(-x[0] * (cf_cz - x[3])**2) * (cf_cz - x[3]))/np.sqrt(np.pi * x[0]) - (np.exp(-x[0] * (ci_cz - x[3])**2) * (ci_cz - x[3])) / np.sqrt(np.pi * x[0])) + \
            np.sqrt(np.pi)/(4 * np.sqrt(x[0]**3)) * (special.erf(np.sqrt(x[0]) * (cf_cz - x[3])) - special.erf(np.sqrt(x[0]) * (ci_cz - x[3])))

    return [(I1x + x[1] * I0x) / I0x - ux, (I1y + x[2] * I0y) / I0y - uy, (I1z + x[3] * I0z) / I0z - uz,
            (I2x + 2 * x[1] * I1x + x[1]**2 * I0x) / (I0x) + (I2y + 2 * x[2] * I1y + x[2]**2 * I0y) / (I0y) + (I2z + 2 * x[3] * I1z + x[3]**2 * I0z) / (I0z) - e]


def calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(5)

    mu[0] = np.trapezoid(np.trapezoid(np.trapezoid(f, cz_vec), cy_vec), cx_vec)
    mu[1] = np.trapezoid(np.trapezoid(np.trapezoid(cx * f, cz_vec), cy_vec), cx_vec)
    mu[2] = np.trapezoid(np.trapezoid(np.trapezoid(cy * f, cz_vec), cy_vec), cx_vec)
    mu[3] = np.trapezoid(np.trapezoid(np.trapezoid(cz * f, cz_vec), cy_vec), cx_vec)
    mu[4] = np.trapezoid(np.trapezoid(np.trapezoid((cx**2 + cy**2 + cz**2) * f, cz_vec), cy_vec), cx_vec)

    return mu


def invert(mu, initial_guess, group_bounds):
    """Recover (A, beta, wx, wy, wz) of a truncated Maxwellian from its moments.

    On any degenerate input (zero/negative mass, failed least-squares,
    collapsed b, vanishing I0u), returns zeros for the parameters that
    can't be recovered. The caller's
        if not (np.isfinite(A) and np.isfinite(b) and A > 0 and b > 0): ...
    guard then catches these cleanly.
    """
    ci_cx = group_bounds['ci_cx']
    cf_cx = group_bounds['cf_cx']
    ci_cy = group_bounds['ci_cy']
    cf_cy = group_bounds['cf_cy']
    ci_cz = group_bounds['ci_cz']
    cf_cz = group_bounds['cf_cz']

    # Guard mu[i]/mu[0] divide-by-zero. A leaf with no mass has no
    # Maxent fit to recover.
    if mu[0] <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Suppress divide/invalid warnings raised inside moment_eq while
    # least_squares explores infeasible points (I0u → 0 when wu wanders
    # far outside the box). The solver handles inf/NaN residuals
    # internally — we just don't want the warning spam.
    with np.errstate(divide='ignore', invalid='ignore'):
        sol = optimize.least_squares(
            moment_eq, initial_guess,
            args=(mu[1] / mu[0], mu[2] / mu[0], mu[3] / mu[0], mu[4] / mu[0],
                  ci_cx, cf_cx, ci_cy, cf_cy, ci_cz, cf_cz),
            # Lower bound on b kept slightly positive so moment_eq doesn't
            # divide by zero during the solve (it has 1/(2*x[0]) terms).
            bounds=([1e-12, -10, -10, -10], [np.inf, 10, 10, 10]),
            method='trf', loss='soft_l1',
        )

    if not sol.success:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    b, wx, wy, wz = sol.x[0], sol.x[1], sol.x[2], sol.x[3]

    # b collapsed to (effectively) the lower bound: Maxent fit failed,
    # the moments are infeasible in this box. Skip the I0 computation
    # so we don't emit a sqrt(pi/0) warning or produce inf.
    if b <= 0:
        return 0.0, 0.0, wx, wy, wz

    I0x = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cx - wx)) - special.erf(np.sqrt(b) * (ci_cx - wx)))
    I0y = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cy - wy)) - special.erf(np.sqrt(b) * (ci_cy - wy)))
    I0z = np.sqrt(np.pi / (4 * b)) * (special.erf(np.sqrt(b) * (cf_cz - wz)) - special.erf(np.sqrt(b) * (ci_cz - wz)))

    # I0u collapses to ~0 when wu is far outside [ci_cu, cf_cu] and b
    # is large (erf saturates to the same value at both ends).
    denom = I0x * I0y * I0z
    if not np.isfinite(denom) or denom <= 0:
        return 0.0, b, wx, wy, wz

    A = mu[0] / denom
    return A, b, wx, wy, wz
