from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from numpy.linalg import norm
from scipy.linalg import null_space


def kernel_residual(A: np.ndarray, B: np.ndarray) -> float:
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    if A.size == 0:
        return float(norm(B, 2))
    P = np.linalg.pinv(A) @ A
    return float(norm(B @ (np.eye(A.shape[1]) - P), 2))


def kernel_contained(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> bool:
    scale = max(1.0, float(norm(B, 2)))
    return kernel_residual(A, B) <= tol * scale


def minimum_gain_reconstructor(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if not kernel_contained(A, B, tol=tol):
        raise ValueError("kernel(A) is not contained in kernel(B)")
    return np.atleast_2d(B) @ np.linalg.pinv(np.atleast_2d(A))


def alpha_task(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> float:
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))

    if norm(B) == 0:
        return float("inf")

    if not kernel_contained(A, B, tol=tol):
        return 0.0

    L = B @ np.linalg.pinv(A)
    op = float(norm(L, 2))
    if op == 0:
        return float("inf")
    return 1.0 / op


def exact_minimax(A: np.ndarray, B: np.ndarray, eps: float) -> float:
    a = alpha_task(A, B)
    if a == 0:
        return float("inf")
    if math.isinf(a):
        return 0.0
    return float(eps / a)


def worst_case_noise(A: np.ndarray, B: np.ndarray, eps: float):
    L = minimum_gain_reconstructor(A, B)
    _, _, vt = np.linalg.svd(L, full_matrices=False)
    v = vt[0]
    e = eps * v / max(norm(v), 1e-15)
    attained = float(norm(L @ e))
    return e, attained


def bad_null_direction(A: np.ndarray, B: np.ndarray, tol: float = 1e-10):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    N = null_space(A)
    if N.size == 0:
        return None, 0.0

    M = B @ N
    if norm(M) <= tol:
        return None, 0.0

    _, _, vt = np.linalg.svd(M, full_matrices=False)
    coeff = vt[0]
    d = N @ coeff
    d = d / norm(d)

    return d, float(norm(B @ d))


def bounded_pair_from_null(A: np.ndarray, B: np.ndarray, halfwidth: np.ndarray):
    d, _ = bad_null_direction(A, B)
    if d is None:
        return None

    halfwidth = np.asarray(halfwidth, float)

    ratios = np.divide(
        halfwidth,
        np.abs(d),
        out=np.full_like(halfwidth, np.inf),
        where=np.abs(d) > 1e-14,
    )

    t = 0.80 * float(np.min(ratios))
    c1 = t * d
    c2 = -t * d
    gap = float(norm(B @ (c1 - c2)))

    return c1, c2, gap


def scalar_box_ambiguity_socp(A, b, eps, halfwidth):
    import cvxpy as cp

    A = np.atleast_2d(np.asarray(A, float))
    b = np.asarray(b, float).reshape(-1)
    h = np.asarray(halfwidth, float).reshape(-1)
    d = len(h)

    c1 = cp.Variable(d)
    c2 = cp.Variable(d)

    constraints = [
        c1 <= h,
        c1 >= -h,
        c2 <= h,
        c2 >= -h,
        cp.norm(A @ (c1 - c2), 2) <= 2.0 * eps,
    ]

    prob = cp.Problem(
        cp.Maximize(b @ (c1 - c2)),
        constraints,
    )

    solved = False
    for solver, kwargs in [
        ("CLARABEL", {}),
        ("SCS", {"eps": 1e-6, "max_iters": 50000}),
    ]:
        try:
            prob.solve(
                solver=solver,
                verbose=False,
                **kwargs,
            )
            if prob.status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
        except Exception:
            pass

    if not solved:
        raise RuntimeError(f"SOCP did not solve: {prob.status}")

    omega = float(prob.value)

    return {
        "omega": omega,
        "minimax": 0.5 * omega,
        "status": prob.status,
        "c1": np.asarray(c1.value).reshape(-1),
        "c2": np.asarray(c2.value).reshape(-1),
    }


def subset_alpha(domain, subset) -> float:
    return alpha_task(domain.A(subset), domain.B)


def enumerate_subsets(m: int, max_count: int | None = None):
    max_count = m if max_count is None else min(max_count, m)
    for k in range(1, max_count + 1):
        yield from combinations(range(m), k)


def choose_good_subset(domain):
    best = None

    for k in range(1, domain.m + 1):
        for S in combinations(range(domain.m), k):
            a = subset_alpha(domain, S)

            if a > 1e-8 and np.isfinite(a):
                cost = float(domain.sensor_costs[list(S)].sum())
                key = (cost, -a, len(S))

                if best is None or key < best[0]:
                    best = (key, S, a)

        if best is not None:
            break

    if best is None:
        S = tuple(range(domain.m))
        return S, subset_alpha(domain, S)

    return tuple(best[1]), float(best[2])


def choose_bad_subset(domain):
    best = None
    maxk = max(1, min(domain.d - 1, domain.m))

    for k in range(1, maxk + 1):
        for S in combinations(range(domain.m), k):
            pair = bounded_pair_from_null(
                domain.A(S),
                domain.B,
                domain.box_halfwidth,
            )

            if pair is None:
                continue

            gap = pair[2]

            if best is None or gap > best[0]:
                best = (gap, S, pair)

    if best is None:
        return None

    return tuple(best[1]), best[2]



def frozen_features(Y, p, seed, R=1.0):
    Y = np.asarray(Y, float)
    rng = np.random.default_rng(seed)

    W = rng.normal(
        scale=1.0 / max(1.0, np.sqrt(Y.shape[1])),
        size=(Y.shape[1], p),
    )
    b = rng.uniform(-np.pi, np.pi, size=p)

    Phi = np.tanh(Y @ W + b)

    nrm = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi = Phi / np.maximum(1.0, nrm / R)

    return Phi


def fit_constrained_ls(Phi, Y, Lambda):
    Phi = np.asarray(Phi, float)
    Y = np.asarray(Y, float)

    if Y.ndim == 1:
        Y = Y[:, None]

    G = Phi.T @ Phi
    C = Y.T @ Phi
    p = Phi.shape[1]

    W0 = (np.linalg.pinv(Phi) @ Y).T

    if np.linalg.norm(W0, "fro") <= Lambda:
        return W0

    def solve(lam):
        return np.linalg.solve(
            G + lam * np.eye(p),
            C.T,
        ).T

    lo, hi = 0.0, 1.0

    while np.linalg.norm(solve(hi), "fro") > Lambda:
        hi *= 10.0

        if hi > 1e16:
            raise RuntimeError(
                "Could not bracket constrained-LS multiplier."
            )

    for _ in range(80):
        mid = 0.5 * (lo + hi)

        if np.linalg.norm(solve(mid), "fro") > Lambda:
            lo = mid
        else:
            hi = mid

    return solve(hi)


def vector_rademacher_mc(Phi, Q, Lambda, draws, seed):
    Phi = np.asarray(Phi, float)
    Q = np.atleast_2d(np.asarray(Q, float))

    N = Phi.shape[0]
    m = Q.shape[0]

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(draws):
        sigma = rng.choice(
            [-1.0, 1.0],
            size=(N, m),
        )

        M = Q.T @ sigma.T @ Phi

        vals.append(
            Lambda * np.linalg.norm(M, "fro") / N
        )

    empirical = float(np.mean(vals))
    R = float(
        np.max(np.linalg.norm(Phi, axis=1))
    )

    bound = float(
        Lambda
        * R
        * np.linalg.norm(Q, "fro")
        / np.sqrt(N)
    )

    return empirical, bound


def exact_box_target_norm_bound(Q, halfwidth):
    Q = np.atleast_2d(np.asarray(Q, float))
    h = np.asarray(halfwidth, float).reshape(-1)
    d = len(h)

    best = 0.0

    for mask in range(1 << d):
        signs = np.ones(d)

        for j in range(d):
            if (mask >> j) & 1:
                signs[j] = -1.0

        c = signs * h
        best = max(
            best,
            float(np.linalg.norm(Q @ c)),
        )

    return best


def normalized_squared_loss_constants(
    Q,
    Lambda,
    R,
    target_norm_bound,
):
    Q = np.atleast_2d(np.asarray(Q, float))

    pred_bound = float(
        np.linalg.norm(Q, 2) * Lambda * R
    )

    S = pred_bound + float(target_norm_bound)

    if S <= 0:
        raise ValueError("Loss scale S must be positive.")

    L_l = 2.0 / S
    M = 1.0

    return L_l, M, S


def excess_risk_bound(
    N,
    Q,
    Lambda,
    R,
    L_l,
    M,
    delta,
):
    return float(
        4.0
        * np.sqrt(2.0)
        * L_l
        * Lambda
        * R
        * np.linalg.norm(Q, "fro")
        / np.sqrt(N)
        + 2.0
        * M
        * np.sqrt(
            np.log(2.0 / delta) / (2.0 * N)
        )
    )
