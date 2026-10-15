from __future__ import annotations
import numpy as np
from ..domain import DecisionDomain


def build_thermal_domain(cfg):
    # Low-order stable electrothermal state: [surface T, core T, SOC-like, polarization].
    F = np.array([
        [0.970, 0.020, 0.000, 0.005],
        [0.012, 0.982, 0.000, 0.010],
        [0.000, 0.000, 0.996, 0.000],
        [0.005, 0.010, 0.015, 0.920],
    ], dtype=float)
    d = 4
    L = 3 if cfg.mode == "full" else 2
    raw_sensors = [
        ("surface_temperature", np.array([[1.0, 0.0, 0.0, 0.0]]), 1.0),
        ("terminal_voltage", np.array([[0.01, 0.02, 0.80, -1.0]]), 1.0),
        ("impedance_proxy", np.array([[0.00, 0.05, 0.10, -0.80]]), 1.4),
        ("soc_estimator", np.array([[0.00, 0.00, 1.00, 0.10]]), 1.4),
        ("surface_gradient_proxy", np.array([[1.00, -1.00, 0.0, 0.0]]), 1.2),
        ("voltage_temp_fusion", np.array([[0.20, 0.10, 0.65, -0.50]]), 1.6),
    ]
    blocks, names, costs = [], [], []
    for name, H, cost in raw_sensors:
        rows = []
        P = np.eye(d)
        for ell in range(L):
            rows.append(H @ P)
            P = F @ P
        block = np.vstack(rows)
        # Channel whitening by its nominal row norm.
        scale = max(np.linalg.norm(block, axis=1).mean(), 1e-8)
        blocks.append(block / scale)
        names.append(name)
        costs.append(cost)

    horizon = 12
    qrow = np.array([[0.0, 1.0, 0.0, 0.0]])
    B = qrow @ np.linalg.matrix_power(F, horizon)
    B = B / max(np.linalg.norm(B), 1e-12)

    return DecisionDomain(
        name="thermal",
        d=d,
        sensor_blocks=blocks[:cfg.max_candidates],
        sensor_names=names[:cfg.max_candidates],
        sensor_costs=np.asarray(costs[:cfg.max_candidates]),
        B=B,
        box_halfwidth=np.ones(d),
        eps=cfg.eps,
        metadata={
            "model": "four-state electrothermal RC surrogate",
            "history_L": L,
            "horizon": horizon,
            "decision": "future normalized core-temperature quantity",
            "external_dataset": "NASA Randomized Battery Usage 1 is optional and not required for theorem identities.",
        },
        nonlinear_all=None,
    )
