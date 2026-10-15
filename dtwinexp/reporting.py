from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import OUT, TAB

TABLE_FILES = {
    "e1": "e1_exact_minimax.csv",
    "e2": "e2_bounded_ambiguity.csv",
    "e3": "e3_no_compensation.csv",
    "e4": "e4_sensor_geometry.csv",
    "e5": "e5_gaussian_propagation.csv",
    "e6": "e6_task_projected_learning.csv",
    "e7": "e7_sensor_resource_frontier.csv",
    "e8": "e8_nonlinear_scope_stress.csv",
}


def _concat(result_by_domain, key):
    frames = []

    for dom, res in result_by_domain.items():
        df = res.get(key)

        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def collect_tables(result_by_domain):
    tables = {
        k: _concat(result_by_domain, k)
        for k in [f"e{i}" for i in range(1, 9)]
    }

    for key, df in tables.items():
        df.to_csv(
            TAB / TABLE_FILES[key],
            index=False,
        )

    return tables


def theorem_checks(tables):
    checks = {}

    e1 = tables["e1"]

    checks["E1_exact_minimax"] = {
        "n": int(len(e1)),
        "max_relative_identity_error": (
            float(e1.relative_identity_error.max())
            if not e1.empty
            else None
        ),
        "pass": (
            bool(
                (e1.relative_identity_error < 1e-8).all()
            )
            if not e1.empty
            else False
        ),
    }

    e2 = tables["e2"]

    if not e2.empty:
        witness_err = np.abs(
            e2["exact_minimax"]
            - e2["half_witness_gap"]
        )

        checks["E2_bounded_exact_witness"] = {
            "n": int(len(e2)),
            "max_abs_gap": float(witness_err.max()),
            "pass": bool(
                (witness_err < 2e-5).all()
            ),
        }

    e3 = tables["e3"]

    if (
        not e3.empty
        and "barrier_respected" in e3
    ):
        bad = e3[
            (e3.sensing == "bad")
            & e3.barrier_respected.notna()
        ]

        checks["E3_no_compensation"] = {
            "n_bad_model_fits": int(len(bad)),
            "violations": int(
                (
                    ~bad.barrier_respected.astype(bool)
                ).sum()
            ),
            "pass": (
                bool(
                    bad.barrier_respected.astype(bool).all()
                )
                if len(bad)
                else False
            ),
        }

    e4 = tables["e4"]

    if not e4.empty:
        by_dom = {}

        for dom, g in e4.groupby("domain"):
            rho = (
                spearmanr(
                    g["inverse_alpha"],
                    g["attained"],
                ).statistic
                if len(g) > 2
                else np.nan
            )

            by_dom[dom] = (
                float(rho)
                if np.isfinite(rho)
                else None
            )

        checks["E4_alpha_law"] = {
            "spearman_by_domain": by_dom
        }

    e5 = tables["e5"]

    if not e5.empty:
        ratio_dev = np.abs(
            e5["ratio"] - 1.0
        )

        checks["E5_gaussian"] = {
            "n": int(len(e5)),
            "median_ratio": float(
                e5["ratio"].median()
            ),
            "max_abs_ratio_error": float(
                ratio_dev.max()
            ),
            "pass": bool(
                (ratio_dev < 0.10).mean()
                >= 0.90
            ),
        }

    e6 = tables["e6"]

    if not e6.empty:
        checks["E6_task_learning"] = {
            "n": int(len(e6)),
            "rad_bound_violations": int(
                (
                    e6["empirical_rademacher"]
                    > e6["theorem_rademacher_bound"]
                    * 1.03
                ).sum()
            ),
            "excess_bound_violations": int(
                (
                    ~e6["certificate_respected"]
                    .astype(bool)
                ).sum()
            ),
            "pass": bool(
                (
                    e6["empirical_rademacher"]
                    <= e6["theorem_rademacher_bound"]
                    * 1.03
                ).all()
                and e6["certificate_respected"]
                .astype(bool)
                .all()
            ),
        }

    e7 = tables["e7"]

    if not e7.empty:
        exact_feas = e7["exact_alpha_feasible"].astype(bool)
        rounded_feas = e7["rounded_alpha_feasible"].astype(bool)
        lower_ok = e7["dual_lower_le_exact"].astype(bool)
        order_ok = e7["exact_le_rounded"].astype(bool)
        primal_ok = e7["primal_min_eig"] >= -5e-8
        dual_ok = e7["dual_min_slack"] >= -1e-9
        task_ok = e7["task_residual"] <= 1.01e-10

        checks["E7_resource_frontier"] = {
            "n": int(len(e7)),
            "exact_feasibility_violations": int((~exact_feas).sum()),
            "rounded_feasibility_violations": int((~rounded_feas).sum()),
            "sdp_lower_bound_violations": int((~lower_ok).sum()),
            "rounding_order_violations": int((~order_ok).sum()),
            "primal_psd_violations": int((~primal_ok).sum()),
            "dual_feasibility_violations": int((~dual_ok).sum()),
            "task_residual_violations": int((~task_ok).sum()),
            "max_primal_dual_gap": float(e7["sdp_primal_dual_gap"].max()),
            "pass": bool(
                exact_feas.all()
                and rounded_feas.all()
                and lower_ok.all()
                and order_ok.all()
                and primal_ok.all()
                and dual_ok.all()
                and task_ok.all()
            ),
        }

    e8 = tables["e8"]

    if not e8.empty:
        checks["E8_scope_stress"] = {
            "note": (
                "No pass/fail criterion: this experiment maps where "
                "the local linear decision slice ceases to approximate "
                "the nonlinear simulator."
            ),
            "domains": sorted(
                e8.domain.unique().tolist()
            ),
        }

    return checks


def write_reports(result_by_domain, tables, cfg):
    checks = theorem_checks(tables)

    (OUT / "theorem_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )

    summary_rows = []

    for name, c in checks.items():
        summary_rows.append(
            {
                "check": name,
                "pass": c.get("pass", ""),
                "details": json.dumps(c),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUT / "evidence_summary.csv",
        index=False,
    )

    lines = [
        "# Decision-ready digital twin experiment report",
        "",
        f"Mode: **{cfg.mode}**",
        f"Seed: `{cfg.seed}`",
        "",
        "## Domain runtimes",
        "",
    ]

    for dom, res in result_by_domain.items():
        lines.append(
            f"- {dom}: "
            f"{res.get('runtime_seconds', float('nan'))/60:.2f} min"
        )

    lines += [
        "",
        "## Theorem checks",
        "",
    ]

    for key, val in checks.items():
        lines.append(f"### {key}")
        lines.append("```json")
        lines.append(
            json.dumps(
                val,
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")

    lines += [
        "## Interpretation rule",
        "",
        "Exact identities are checked numerically rather than judged visually.",
        "E3 uses an explicitly constructed observation-indistinguishable witness pair.",
        "E8 is a scope-stress test and is allowed to show breakdown outside the local linear envelope.",
        "",
        "See `results/tables/` for the data behind every figure.",
    ]

    (OUT / "RUN_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return checks
