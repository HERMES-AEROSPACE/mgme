"""Maximum-entropy dual solver shared by the 0-D and 1-D drivers."""
import numpy as np


def solve_group_newton(x_s, y_s, z_s, U, lam, max_iter=50, tol=1e-10,
                       accept_floor=1e-4):
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
        # Initial state.
        w = np.exp(lam @ phi)
        if not np.all(np.isfinite(w)):
            return w, lam, False, np.full(5, np.inf)
        g = phi @ w - U
        g_norm = np.linalg.norm(g)

        for iteration in range(max_iter):
            if g_norm < tol:
                converged = True
                break

            # Hessian: H_ij = sum(phi_i * phi_j * w)
            H = phi @ (w[:, None] * phi.T)

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return w, lam, False, g

            # Backtracking line search on ||g||. Pure Newton (step=1) often
            # overshoots into the overflow region (lam @ phi > 709) when
            # warm-started from a stale cache; halving accepts the Newton
            # direction but tames the magnitude. Sufficient-decrease on the
            # gradient norm is appropriate here because the dual is convex.
            step = 1.0
            accepted = False
            for _ in range(20):
                lam_trial = lam - step * delta
                w_trial = np.exp(lam_trial @ phi)
                if np.all(np.isfinite(w_trial)):
                    g_trial = phi @ w_trial - U
                    if np.all(np.isfinite(g_trial)):
                        gt_norm = np.linalg.norm(g_trial)
                        if gt_norm < g_norm:
                            lam = lam_trial
                            w = w_trial
                            g = g_trial
                            g_norm = gt_norm
                            accepted = True
                            break
                step *= 0.5

            if not accepted:
                # Newton direction can't make progress even at step 2^-20.
                # Either the local Hessian is misleading (saddle / near-
                # singular) or we're at a discrete-sample-induced floor of
                # the gradient norm. Accept the fit if the residual is
                # already small enough for moment closure; bail otherwise.
                if g_norm < accept_floor:
                    return w, lam, True, g
                return w, lam, False, g

    # Loop exited via tol (converged) or via max_iter without convergence.
    # When max_iter is hit, the absolute gradient is sometimes stuck at
    # 1e-7..1e-4 on noisy small-mass cells — well below moment scale and
    # adequate for closure. Accept those instead of forcing the caller to
    # discard a usable fit.
    if not converged and g_norm < accept_floor:
        converged = True

    return w, lam, converged, g
