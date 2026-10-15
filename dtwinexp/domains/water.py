from __future__ import annotations

import numpy as np

from ..domain import DecisionDomain
from ..config import NET3_PATH


def _base_demand(node):
    try:
        return float(node.demand_timeseries_list[0].base_value)
    except Exception:
        try:
            return float(node.base_demand)
        except Exception:
            return 0.0


def _set_base_demand(node, value):
    try:
        node.demand_timeseries_list[0].base_value = float(value)
    except Exception:
        node.base_demand = float(value)


def build_water_domain(cfg):
    """
    Water benchmark design.

    q is pressure at a HELD-OUT critical node.
    The critical node is excluded from candidate pressure sensors.
    """
    import wntr

    inp = NET3_PATH
    if not inp.exists():
        raise FileNotFoundError(
            "Net3.inp is missing. Run `python download_benchmarks.py`."
        )

    base = wntr.network.WaterNetworkModel(str(inp))
    base.options.time.duration = 0
    base.options.time.hydraulic_timestep = 3600
    base.options.time.report_timestep = 3600

    junctions = list(base.junction_name_list)
    demands = [(name, abs(_base_demand(base.get_node(name)))) for name in junctions]
    demands.sort(key=lambda x: x[1], reverse=True)

    d = min(8 if cfg.mode == "full" else 6, sum(v > 0 for _, v in demands))
    demand_nodes = [name for name, v in demands if v > 0][:d]
    demand0 = np.array(
        [_base_demand(base.get_node(name)) for name in demand_nodes],
        float,
    )
    frac = 0.15

    def run_model(c):
        wn = wntr.network.WaterNetworkModel(str(inp))
        wn.options.time.duration = 0
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep = 3600

        c = np.asarray(c, float).reshape(-1)
        for j, name in enumerate(demand_nodes):
            _set_base_demand(
                wn.get_node(name),
                demand0[j] * (1.0 + frac * float(c[j])),
            )

        sim = wntr.sim.WNTRSimulator(wn)
        res = sim.run_sim()
        p = res.node["pressure"].iloc[0]
        f = res.link["flowrate"].iloc[0]
        return p, f

    p0, f0 = run_model(np.zeros(d))
    demand_junctions = [name for name in junctions if name in p0.index]

    crit_node = str(min(demand_junctions, key=lambda n: float(p0[n])))
    q0_phys = float(p0[crit_node])

    pressure_candidates = [
        n for n in sorted(demand_junctions, key=lambda n: float(p0[n]))
        if str(n) != crit_node
    ]
    n_p = 6 if cfg.mode == "full" else 4
    pressure_sel = pressure_candidates[:n_p]

    flow_candidates = list(f0.abs().sort_values(ascending=False).index)
    n_f = 4 if cfg.mode == "full" else 3
    flow_sel = flow_candidates[:n_f]

    n_dmeter = max(0, min(cfg.max_candidates - n_p - n_f, 4))
    demand_meter_pos = list(range(n_dmeter))

    names = (
        [f"pressure_{n}" for n in pressure_sel]
        + [f"flow_{n}" for n in flow_sel]
        + [f"demand_meter_{j}" for j in demand_meter_pos]
    )

    costs = np.array(
        [1.0] * len(pressure_sel)
        + [1.5] * len(flow_sel)
        + [0.8] * len(demand_meter_pos),
        float,
    )

    median_flow = float(np.median(np.abs(f0.to_numpy())))
    flow_scale = max(0.02 * median_flow, 1e-5)

    def observe_q(c):
        c = np.asarray(c, float).reshape(-1)
        p, f = run_model(c)

        vals = []
        for n in pressure_sel:
            vals.append((float(p[n]) - float(p0[n])) / 0.5)
        for n in flow_sel:
            vals.append((float(f[n]) - float(f0[n])) / flow_scale)
        for j in demand_meter_pos:
            vals.append(float(c[j]))

        q = (float(p[crit_node]) - q0_phys) / 2.0
        return np.asarray(vals, float), float(q)

    h = 5e-3
    m = len(names)
    A = np.zeros((m, d))
    B = np.zeros((1, d))

    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        yp, qp = observe_q(e)
        ym, qm = observe_q(-e)
        A[:, j] = (yp - ym) / (2.0 * h)
        B[0, j] = (qp - qm) / (2.0 * h)

    keep = min(cfg.max_candidates, m)
    blocks = [A[[i], :] for i in range(keep)]

    return DecisionDomain(
        name="water",
        d=d,
        sensor_blocks=blocks,
        sensor_names=names[:keep],
        sensor_costs=costs[:keep],
        B=B,
        box_halfwidth=np.ones(d),
        eps=cfg.eps,
        metadata={
            "benchmark": "EPA EPANET Net3 via WNTR",
            "demand_dimensions": d,
            "demand_perturbation_fraction": frac,
            "held_out_critical_node": crit_node,
            "decision": "held-out critical-node pressure change, normalized by 2 m",
            "important": "critical decision node excluded from candidate pressure sensors",
        },
        nonlinear_all=observe_q,
    )
