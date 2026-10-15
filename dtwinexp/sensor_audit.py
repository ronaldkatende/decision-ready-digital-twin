from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd

from .theory import alpha_task

REL_ALPHA_TOL = 1e-8
PSD_VERIFY_TOL = 5e-8
TASK_RESIDUAL_TOL = 1e-10

def alpha_feasible(alpha: float, target_alpha: float) -> bool:
    if not np.isfinite(alpha) or alpha < 0:
        return False
    if target_alpha <= 0:
        return True

    abs_tol = (
        100.0
        * np.finfo(float).eps
        * max(abs(target_alpha), 1e-12)
    )
    return (
        alpha + abs_tol
        >= target_alpha * (1.0 - REL_ALPHA_TOL)
    )


def exact_binary_optimum(domain, target_alpha: float):
    """Exhaustively compute the exact binary sensor-cost optimum."""
    best = None

    for k in range(1, domain.m + 1):
        for S in combinations(range(domain.m), k):
            a = alpha_task(domain.A(S), domain.B)

            if not alpha_feasible(a, target_alpha):
                continue

            cost = float(domain.sensor_costs[list(S)].sum())

            if (
                best is None
                or cost < best["cost"] - 1e-12
                or (
                    abs(cost - best["cost"]) <= 1e-12
                    and a > best["alpha"]
                )
            ):
                best = {
                    "subset": tuple(S),
                    "cost": cost,
                    "alpha": float(a),
                }

    return best


def rounded_binary_solution(domain, target_alpha: float, w):
    """Rank sensors by fractional weight and add until alpha_min is truly met."""
    order = list(np.argsort(-np.asarray(w, float)))
    S = []

    for i in order:
        S.append(int(i))
        a = alpha_task(domain.A(S), domain.B)

        if alpha_feasible(a, target_alpha):
            return {
                "subset": tuple(sorted(S)),
                "cost": float(domain.sensor_costs[S].sum()),
                "alpha": float(a),
            }

    return None


def svd_whiten_information(domain):
    """
    Stable observable-subspace whitening using the SVD of A_full itself.

    This deliberately avoids eigendecomposing G_full=A_full^T A_full.
    Forming G_full squares the condition number.  In the water benchmark the
    an eigenvalue cutoff of 1e-12 corresponded to discarding singular
    directions of A_full below about 1e-6 relative scale, which removed a
    small but real task-relevant component.

    Here:
        A_full = U diag(s) V^T.

    We retain as many right-singular directions as needed so that the discarded
    component of B is <= TASK_RESIDUAL_TOL.  Whitening uses
        T = V_r diag(1/s_r),
    giving
        (A_full T)^T(A_full T) = I
    up to floating-point error.

    Each block information matrix is then formed as
        Ghat_i = (A_i T)^T(A_i T),
    rather than T^T(A_i^T A_i)T, which further avoids cancellation.
    """
    Aall = np.asarray(domain.all_A(), float)
    B = np.atleast_2d(np.asarray(domain.B, float))

    U, s, Vt = np.linalg.svd(Aall, full_matrices=False)

    if len(s) == 0 or not np.isfinite(s[0]) or s[0] <= 0:
        raise RuntimeError(f"{domain.name}: A_full has no observable directions.")

    # Numerical floor based on the SVD of A, not eigenvalues of A^T A.
    svd_floor = (
        np.finfo(float).eps
        * max(Aall.shape)
        * s[0]
        * 10.0
    )

    bnorm = float(np.linalg.norm(B, ord="fro"))
    if bnorm == 0:
        raise RuntimeError(f"{domain.name}: decision map B is zero.")

    chosen_r = None
    residuals = []

    max_r = min(len(s), Aall.shape[1])

    for r in range(1, max_r + 1):
        Vr = Vt[:r, :].T
        P = Vr @ Vr.T

        residual = float(
            np.linalg.norm(B - B @ P, ord="fro")
            / bnorm
        )
        residuals.append(residual)

        if residual <= TASK_RESIDUAL_TOL:
            chosen_r = r
            break

    if chosen_r is None:
        chosen_r = max_r

    s_keep = s[:chosen_r]

    # If a task-relevant retained direction is below the numerical SVD floor,
    # the model itself is not numerically resolvable; fail loudly.
    if float(np.min(s_keep)) <= svd_floor:
        raise RuntimeError(
            f"{domain.name}: task-relevant singular direction is below the "
            f"numerical SVD floor. min retained s={np.min(s_keep):.3e}, "
            f"floor={svd_floor:.3e}."
        )

    Vr = Vt[:chosen_r, :].T
    T = Vr @ np.diag(1.0 / s_keep)

    Ghat = []
    for i in range(domain.m):
        Ai = np.atleast_2d(np.asarray(domain.sensor_blocks[i], float))
        Zi = Ai @ T
        H = Zi.T @ Zi
        Ghat.append(0.5 * (H + H.T))

    BT = B @ T
    Fhat = BT.T @ BT
    Fhat = 0.5 * (Fhat + Fhat.T)

    full_hat = np.sum(Ghat, axis=0)

    identity_residual = float(
        np.linalg.norm(
            full_hat - np.eye(chosen_r),
            ord="fro",
        )
    )

    task_residual = float(residuals[chosen_r - 1])

    lamF = float(max(np.linalg.eigvalsh(Fhat).max(), 0.0))
    white_full_alpha = (
        float(1.0 / np.sqrt(lamF))
        if lamF > 0
        else float("inf")
    )

    cond_retained = float(s_keep[0] / s_keep[-1])

    return {
        "Ghat": Ghat,
        "Fhat": Fhat,
        "rank": int(chosen_r),
        "task_residual": task_residual,
        "identity_residual": identity_residual,
        "white_full_alpha": white_full_alpha,
        "svd_floor": float(svd_floor),
        "smallest_retained_singular_value": float(s_keep[-1]),
        "largest_singular_value": float(s_keep[0]),
        "retained_condition_number": cond_retained,
    }


def min_psd_eigenvalue(w, Ghat, Fhat, eta):
    M = (
        sum(float(w[i]) * Ghat[i] for i in range(len(w)))
        - eta * Fhat
    )
    M = 0.5 * (M + M.T)
    return float(np.linalg.eigvalsh(M).min())


def repair_fractional_primal(w, Ghat, Fhat, eta, costs):
    """
    Convert a nearly feasible numerical primal vector into an explicitly
    feasible fractional point by the smallest convex mixing with all sensors.

    This point is an upper bracket on the continuous optimum and supplies the
    sensor ranking for rounding.  It is NOT used as the lower bound.
    """
    w0 = np.clip(np.asarray(w, float).reshape(-1), 0.0, 1.0)

    e0 = min_psd_eigenvalue(w0, Ghat, Fhat, eta)

    if e0 >= -PSD_VERIFY_TOL:
        return {
            "w": w0,
            "cost": float(costs @ w0),
            "min_eig": e0,
            "repair_mix": 0.0,
        }

    one = np.ones_like(w0)
    e1 = min_psd_eigenvalue(one, Ghat, Fhat, eta)

    if e1 < -PSD_VERIFY_TOL:
        raise RuntimeError(
            "Even the all-sensor point is infeasible after SVD whitening."
        )

    lo, hi = 0.0, 1.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        wm = (1.0 - mid) * w0 + mid * one

        if min_psd_eigenvalue(wm, Ghat, Fhat, eta) >= 0.0:
            hi = mid
        else:
            lo = mid

    wfix = (1.0 - hi) * w0 + hi * one

    return {
        "w": wfix,
        "cost": float(costs @ wfix),
        "min_eig": min_psd_eigenvalue(wfix, Ghat, Fhat, eta),
        "repair_mix": float(hi),
    }


def solve_rescaled_primal_dual(domain, eta: float):
    """
    Solve both primal and dual SDPs after SVD-based exact whitening.

    Primal:
        min c^T w
        s.t. sum_i w_i Ghat_i - eta Fhat >= 0,
             0 <= w_i <= 1.

    Dual:
        max eta <Fhat,Z> - 1^T mu
        s.t. Z >= 0, mu >= 0,
             <Ghat_i,Z> - mu_i <= c_i.

    The reported lower bound is evaluated at an explicitly repaired dual-
    feasible point.
    """
    import cvxpy as cp

    white = svd_whiten_information(domain)

    Ghat = white["Ghat"]
    Fhat = white["Fhat"]
    costs = np.asarray(domain.sensor_costs, float)
    m = domain.m

    lamF = float(max(np.linalg.eigvalsh(Fhat).max(), 0.0))
    normalized_target_strength = float(eta * lamF)

    # ---------- Primal ----------
    w = cp.Variable(m)

    Gsum = 0
    for i in range(m):
        Gsum = Gsum + w[i] * Ghat[i]

    Gsum = 0.5 * (Gsum + Gsum.T)

    primal = cp.Problem(
        cp.Minimize(costs @ w),
        [
            w >= 0,
            w <= 1,
            Gsum - eta * Fhat >> 0,
        ],
    )

    primal_status = None

    for solver, kwargs in [
        ("CLARABEL", {}),
        ("SCS", {"eps": 1e-8, "max_iters": 300000}),
    ]:
        try:
            primal.solve(solver=solver, verbose=False, **kwargs)
            primal_status = primal.status

            if (
                primal.status in ("optimal", "optimal_inaccurate")
                and w.value is not None
            ):
                break
        except Exception:
            continue

    if w.value is None:
        raise RuntimeError(
            f"{domain.name}: primal SDP failed; status={primal_status}"
        )

    primal_fix = repair_fractional_primal(
        w.value, Ghat, Fhat, eta, costs
    )

    # ---------- Dual ----------
    r = Fhat.shape[0]

    Z = cp.Variable((r, r), symmetric=True)
    mu = cp.Variable(m)

    constraints = [Z >> 0, mu >= 0]

    for i in range(m):
        constraints.append(
            cp.trace(Ghat[i] @ Z) - mu[i] <= costs[i]
        )

    dual = cp.Problem(
        cp.Maximize(
            eta * cp.trace(Fhat @ Z) - cp.sum(mu)
        ),
        constraints,
    )

    dual_status = None

    for solver, kwargs in [
        ("CLARABEL", {}),
        ("SCS", {"eps": 1e-8, "max_iters": 300000}),
    ]:
        try:
            dual.solve(solver=solver, verbose=False, **kwargs)
            dual_status = dual.status

            if (
                dual.status in ("optimal", "optimal_inaccurate")
                and Z.value is not None
            ):
                break
        except Exception:
            continue

    if Z.value is None:
        raise RuntimeError(
            f"{domain.name}: dual SDP failed; status={dual_status}"
        )

    # Explicitly repair dual feasibility.
    Zraw = np.asarray(Z.value, float)
    Zraw = 0.5 * (Zraw + Zraw.T)

    zeig, zvec = np.linalg.eigh(Zraw)
    zeig = np.maximum(zeig, 0.0)

    Zpsd = (zvec * zeig) @ zvec.T
    Zpsd = 0.5 * (Zpsd + Zpsd.T)

    pairings = np.array(
        [
            float(np.trace(Ghat[i] @ Zpsd))
            for i in range(m)
        ],
        dtype=float,
    )

    margin = 1e-10 * max(1.0, float(np.max(np.abs(costs))))

    mu_fix = np.maximum(0.0, pairings - costs) + margin

    dual_slack = costs + mu_fix - pairings

    dual_lower = float(
        eta * np.trace(Fhat @ Zpsd) - np.sum(mu_fix)
    )

    primal_cost = float(primal_fix["cost"])

    return {
        "w": primal_fix["w"],
        "dual_lower_cost": dual_lower,
        "primal_feasible_cost": primal_cost,
        "primal_dual_gap": float(primal_cost - dual_lower),

        "primal_min_eig": float(primal_fix["min_eig"]),
        "primal_repair_mix": float(primal_fix["repair_mix"]),

        "dual_min_slack": float(np.min(dual_slack)),
        "dual_min_eig": float(np.linalg.eigvalsh(Zpsd).min()),

        "primal_status": str(primal_status),
        "dual_status": str(dual_status),

        "whitened_rank": int(white["rank"]),
        "whitening_identity_residual": float(white["identity_residual"]),
        "task_residual": float(white["task_residual"]),

        "svd_floor": float(white["svd_floor"]),
        "smallest_retained_singular_value": float(
            white["smallest_retained_singular_value"]
        ),
        "largest_singular_value": float(white["largest_singular_value"]),
        "retained_condition_number": float(white["retained_condition_number"]),

        "white_full_alpha": float(white["white_full_alpha"]),
        "normalized_target_strength": normalized_target_strength,
    }


def run_domain_e7(domain, cfg):
    full_alpha = alpha_task(domain.all_A(), domain.B)

    if not (
        full_alpha > 1e-8
        and np.isfinite(full_alpha)
    ):
        raise RuntimeError(
            f"{domain.name}: full design is not task-observable."
        )

    fracs = (
        [0.15, 0.30, 0.50, 0.70, 0.90]
        if cfg.mode == "full"
        else [0.25, 0.55, 0.85]
    )
    rows = []

    for frac in fracs:
        target_alpha = float(frac * full_alpha)
        eta = float(target_alpha ** 2)

        exact = exact_binary_optimum(domain, target_alpha)

        if exact is None:
            raise RuntimeError(
                f"{domain.name}: no exact binary design meets "
                f"alpha_min={target_alpha:.6e}"
            )

        sdp = solve_rescaled_primal_dual(domain, eta)

        rounded = rounded_binary_solution(
            domain, target_alpha, sdp["w"]
        )

        if rounded is None:
            raise RuntimeError(
                f"{domain.name}: rounded design does not meet "
                f"alpha_min={target_alpha:.6e}"
            )

        rows.append(
            {
                "domain": domain.name,
                "frac_full_alpha": frac,
                "full_alpha": full_alpha,
                "target_alpha": target_alpha,
                "eta": eta,

                "sdp_dual_lower_cost": sdp["dual_lower_cost"],
                "sdp_primal_feasible_cost": sdp["primal_feasible_cost"],
                "sdp_primal_dual_gap": sdp["primal_dual_gap"],

                "exact_cost": exact["cost"],
                "exact_alpha": exact["alpha"],
                "exact_subset": "|".join(map(str, exact["subset"])),

                "rounded_cost": rounded["cost"],
                "rounded_alpha": rounded["alpha"],
                "rounded_subset": "|".join(map(str, rounded["subset"])),

                "exact_alpha_feasible": alpha_feasible(
                    exact["alpha"], target_alpha
                ),
                "rounded_alpha_feasible": alpha_feasible(
                    rounded["alpha"], target_alpha
                ),

                "dual_lower_le_exact": (
                    sdp["dual_lower_cost"] <= exact["cost"] + 1e-7
                ),
                "exact_le_rounded": (
                    exact["cost"] <= rounded["cost"] + 1e-8
                ),

                "primal_min_eig": sdp["primal_min_eig"],
                "primal_repair_mix": sdp["primal_repair_mix"],
                "dual_min_slack": sdp["dual_min_slack"],
                "dual_min_eig": sdp["dual_min_eig"],

                "primal_status": sdp["primal_status"],
                "dual_status": sdp["dual_status"],

                "whitened_rank": sdp["whitened_rank"],
                "whitening_identity_residual": sdp[
                    "whitening_identity_residual"
                ],
                "task_residual": sdp["task_residual"],

                "svd_floor": sdp["svd_floor"],
                "smallest_retained_singular_value": sdp[
                    "smallest_retained_singular_value"
                ],
                "largest_singular_value": sdp["largest_singular_value"],
                "retained_condition_number": sdp[
                    "retained_condition_number"
                ],

                "white_full_alpha": sdp["white_full_alpha"],
                "normalized_target_strength": sdp[
                    "normalized_target_strength"
                ],
            }
        )

    return pd.DataFrame(rows)
