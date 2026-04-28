"""Maximum-entropy dual solver shared by the 0-D and 1-D drivers."""
import numpy as np


def solve_group_newton(x_s, y_s, z_s, U, lam, max_iter=50, tol=1e-10):
    """
    Solve max-entropy weights directly via Newton iterations on dual.

    Dual problem: find lambda s.t. sum_i phi_i * exp(lam . phi_i) = U.

    np.errstate suppresses overflow warnings that fire when a Newton
    step lands at an extreme lam (exp argument > 709). The convergence
    check on the gradient norm naturally rejects those iterations: if
    w becomes inf/nan, g and H propagate it, np.linalg.solve raises,
    and we return success=False — the caller's stale-fit fallback keeps
    the leaf alive without poisoning subsequent steps.
    """
    n = x_s.shape[0]

    phi = np.empty((5, n))
    phi[0] = np.ones(n)
    phi[1] = x_s
    phi[2] = y_s
    phi[3] = z_s
    phi[4] = x_s**2 + y_s**2 + z_s**2

    # Copy lam so we never mutate the caller's array. Critical: the
    # `lam -= ...` step inside the loop modifies in place, and on a
    # diverged fit (success=False) the caller would otherwise see its
    # original leaf.lam corrupted to garbage even though we returned
    # nothing. That corruption silently breaks the next step's
    # warm-start and any code that reads leaf.lam (e.g. entropy via
    # the Maxent identity S = -lam . mu).
    lam = lam.copy()

    converged = False
    with np.errstate(over='ignore', invalid='ignore'):
        for iteration in range(max_iter):
            # Compute weights from current lambda
            w = np.exp(lam @ phi)

            phi_w = phi @ w  # broadcast: each row of phi scaled by w
            # Gradient: g = phi @ w - U  (moment residuals)
            g = phi_w - U

            # Convergence check. Also bail if w went non-finite — Newton
            # has wandered into the overflow region and won't recover.
            if not np.all(np.isfinite(w)):
                return w, lam, False, g
            if np.linalg.norm(g) < tol:
                converged = True
                break

            # Hessian: H_ij = sum(phi_i * phi_j * w)
            H = phi @ (w[:, None] * phi.T)

            try:
                lam -= np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return w, lam, False, g

        w = np.exp(lam @ phi)

    return w, lam, converged, g
