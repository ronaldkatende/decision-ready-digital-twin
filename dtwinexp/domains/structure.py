from __future__ import annotations

import numpy as np
from scipy.linalg import expm, eigh
from ..domain import DecisionDomain


def build_structure_domain(cfg):
    # Six-storey shear building: 12 state variables [displacement, velocity].
    n_floor = 6
    m_story = 2.0e5
    k_story = 4.0e8
    M = np.eye(n_floor) * m_story
    K = np.zeros((n_floor, n_floor))
    for i in range(n_floor):
        if i == 0:
            K[i, i] += k_story
        else:
            K[i, i] += k_story
            K[i - 1, i - 1] += k_story
            K[i, i - 1] -= k_story
            K[i - 1, i] -= k_story

    # Rayleigh damping fitted to 2% at first and fourth modes.
    evals, modes = eigh(K, M)
    omega = np.sqrt(np.maximum(evals, 1e-12))
    zeta = 0.02
    w1, w4 = omega[0], omega[min(3, len(omega) - 1)]
    mat = np.array([[1.0 / (2*w1), w1/2], [1.0 / (2*w4), w4/2]])
    alpha_r, beta_r = np.linalg.solve(mat, np.array([zeta, zeta]))
    C = alpha_r * M + beta_r * K

    Z = np.zeros_like(M)
    I = np.eye(n_floor)
    Minv = np.linalg.inv(M)
    Fcont = np.block([[Z, I], [-Minv @ K, -Minv @ C]])

    dt = 0.02
    F = expm(Fcont * dt)

    # Reduced initial-state basis: first four displacement modes + first four velocity modes.
    r = 4
    Phi = np.zeros((2*n_floor, 2*r))
    Phi[:n_floor, :r] = modes[:, :r] * 0.002  # about mm-scale displacement
    Phi[n_floor:, r:] = modes[:, :r] * 0.02  # velocity scale
    d = Phi.shape[1]

    # Candidate sensor = one floor accelerometer observed over a short history.
    Acc = np.hstack([-Minv @ K, -Minv @ C])
    L = 4 if cfg.mode == "full" else 3
    blocks = []
    names = []
    costs = []
    accel_scale = 0.05
    for floor in range(n_floor):
        rows = []
        P = np.eye(2*n_floor)
        for ell in range(L):
            rows.append((Acc[[floor], :] @ P @ Phi) / accel_scale)
            P = F @ P
        blocks.append(np.vstack(rows))
        names.append(f"accel_floor_{floor+1}")
        costs.append(1.0)

    # Additional displacement channels give heterogeneous sensor costs.
    for floor in [1, 3, 5]:
        H = np.zeros((1, 2*n_floor))
        H[0, floor] = 1.0
        rows = []
        P = np.eye(2*n_floor)
        for ell in range(L):
            rows.append((H @ P @ Phi) / 0.001)
            P = F @ P
        blocks.append(np.vstack(rows))
        names.append(f"disp_floor_{floor+1}")
        costs.append(1.5)

    horizon = 20
    Cq = np.zeros((1, 2*n_floor))
    Cq[0, n_floor - 1] = 1.0
    B = (Cq @ np.linalg.matrix_power(F, horizon) @ Phi) / 0.001

    max_m = min(cfg.max_candidates, len(blocks))
    blocks = blocks[:max_m]
    names = names[:max_m]
    costs = np.asarray(costs[:max_m])

    return DecisionDomain(
        name="structure",
        d=d,
        sensor_blocks=blocks,
        sensor_names=names,
        sensor_costs=costs,
        B=B,
        box_halfwidth=np.ones(d),
        eps=cfg.eps,
        metadata={
            "model": "six-storey shear building",
            "floors": n_floor,
            "dt": dt,
            "history_L": L,
            "horizon": horizon,
            "decision": "future normalized top-floor displacement",
        },
        nonlinear_all=None,
    )
