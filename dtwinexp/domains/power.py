from __future__ import annotations

import copy
import numpy as np

from ..domain import DecisionDomain


def build_power_domain(cfg):
    """
    Power benchmark design.

    q is the loading of a HELD-OUT critical line.
    The critical line is excluded from candidate line-loading sensors.
    """
    import pandapower as pp
    from pandapower.networks import create_cigre_network_mv

    base = create_cigre_network_mv(with_der="all")
    pp.runpp(base, calculate_voltage_angles=True, init="auto")

    load_order = list(base.load.p_mw.abs().sort_values(ascending=False).index)
    d = min(8 if cfg.mode == "full" else 6, len(load_order))
    load_idx = load_order[:d]
    p0 = base.load.loc[load_idx, "p_mw"].to_numpy(float)
    q0 = base.load.loc[load_idx, "q_mvar"].to_numpy(float)
    frac = 0.10

    slack_buses = set(base.ext_grid.bus.astype(int).tolist())

    crit_line = int(base.res_line.loading_percent.abs().idxmax())
    crit_loading0 = float(base.res_line.loc[crit_line, "loading_percent"])

    vm0 = base.res_bus.vm_pu.copy()
    bus_candidates = [
        int(i) for i in vm0.sort_values().index
        if int(i) not in slack_buses
    ]
    n_v = 5 if cfg.mode == "full" else 4
    bus_sel = bus_candidates[:n_v]

    line_candidates = [
        int(i) for i in base.res_line.loading_percent.abs().sort_values(ascending=False).index
        if int(i) != crit_line
    ]
    n_l = 5 if cfg.mode == "full" else 3
    line_sel = line_candidates[:n_l]

    n_p = max(0, min(cfg.max_candidates - n_v - n_l, 4))
    meter_pos = list(range(n_p))

    sensor_names = (
        [f"bus_vm_{b}" for b in bus_sel]
        + [f"line_loading_{l}" for l in line_sel]
        + [f"load_injection_meter_{j}" for j in meter_pos]
    )
    sensor_costs = np.array(
        [1.0] * len(bus_sel)
        + [1.4] * len(line_sel)
        + [0.8] * len(meter_pos),
        float,
    )

    def simulate(c):
        net = copy.deepcopy(base)
        c = np.asarray(c, float).reshape(-1)

        net.load.loc[load_idx, "p_mw"] = p0 * (1.0 + frac * c)
        net.load.loc[load_idx, "q_mvar"] = q0 * (1.0 + frac * c)

        pp.runpp(
            net,
            calculate_voltage_angles=True,
            init="auto",
            max_iteration=30,
        )

        vals = []
        for b in bus_sel:
            vals.append(
                (float(net.res_bus.loc[b, "vm_pu"]) - float(vm0.loc[b])) / 0.005
            )
        for l in line_sel:
            vals.append(
                (
                    float(net.res_line.loc[l, "loading_percent"])
                    - float(base.res_line.loc[l, "loading_percent"])
                ) / 2.0
            )
        for j in meter_pos:
            vals.append(float(c[j]))

        q = (
            float(net.res_line.loc[crit_line, "loading_percent"])
            - crit_loading0
        ) / 5.0

        return np.asarray(vals, float), float(q)

    h = 1e-3
    m = len(sensor_names)
    A = np.zeros((m, d))
    B = np.zeros((1, d))

    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        yp, qp = simulate(e)
        ym, qm = simulate(-e)
        A[:, j] = (yp - ym) / (2.0 * h)
        B[0, j] = (qp - qm) / (2.0 * h)

    keep = min(cfg.max_candidates, m)
    blocks = [A[[i], :] for i in range(keep)]
    names = sensor_names[:keep]
    costs = sensor_costs[:keep]

    def nonlinear_all(c):
        y, q = simulate(c)
        return y[:keep], q

    return DecisionDomain(
        name="power",
        d=d,
        sensor_blocks=blocks,
        sensor_names=names,
        sensor_costs=costs,
        B=B,
        box_halfwidth=np.ones(d),
        eps=cfg.eps,
        metadata={
            "benchmark": "CIGRE MV with all DER via pandapower",
            "load_dimensions": d,
            "load_perturbation_fraction": frac,
            "held_out_critical_line": crit_line,
            "decision": "held-out critical-line loading change, normalized by 5 percentage points",
            "important": "critical decision line excluded from candidate line-loading sensors",
        },
        nonlinear_all=nonlinear_all,
    )
