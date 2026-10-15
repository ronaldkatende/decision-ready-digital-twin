from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import FIG


def _domains(df):
    return [
        d for d in ["power", "water", "structure", "thermal"]
        if d in set(df["domain"])
    ]


def _grid_axes():
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8.2), constrained_layout=True
    )
    return fig, axes.ravel()


def fig01_exact_minimax(df):
    if df.empty:
        return
    fig, axes = _grid_axes()
    for ax, dom in zip(axes, _domains(df)):
        g = df[df.domain == dom]
        x = g["predicted_minimax"].to_numpy()
        y = g["attained_worst_case"].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.75)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
        ax.set_title(dom.capitalize())
        ax.set_xlabel("Theory: epsilon / alpha")
        ax.set_ylabel("Attained worst-case error")
        ax.set_xscale("log")
        ax.set_yscale("log")
    fig.suptitle("E1. Exact minimax identity across sensor subsets")
    fig.savefig(FIG / "fig01_exact_minimax.png", dpi=220)
    plt.close(fig)


def fig02_bounded(df):
    if df.empty:
        return
    fig, axes = _grid_axes()
    for ax, dom in zip(axes, _domains(df)):
        g = df[df.domain == dom]
        for eps, h in g.groupby("eps"):
            ax.plot(
                h["width"], h["exact_minimax"],
                marker="o", label=f"eps={eps:g}"
            )
        ax.set_title(dom.capitalize())
        ax.set_xlabel("Operational-envelope half-width")
        ax.set_ylabel("Exact bounded minimax = 0.5 Omega")
        ax.legend(fontsize=8)
    fig.suptitle("E2. Exact bounded-envelope ambiguity")
    fig.savefig(FIG / "fig02_bounded_ambiguity.png", dpi=220)
    plt.close(fig)


def fig03_no_comp(df):
    if df.empty:
        return
    fig, axes = _grid_axes()
    for ax, dom in zip(axes, _domains(df)):
        g = df[(df.domain == dom) & df["max_pair_error"].notna()]
        if g.empty:
            continue
        med = (
            g.groupby(["N", "model", "sensing"], as_index=False)["max_pair_error"]
            .median()
        )
        lower = float(g["theorem_pair_lower_bound"].dropna().iloc[0])
        for (model, sensing), h in med.groupby(["model", "sensing"]):
            ax.plot(
                h["N"], h["max_pair_error"], marker="o",
                label=f"{model} ({sensing})"
            )
        ax.axhline(
            lower, linestyle="--", linewidth=1.2,
            label="bad-pair lower bound"
        )
        ax.set_xscale("log")
        ax.set_title(dom.capitalize())
        ax.set_xlabel("Training trajectories N")
        ax.set_ylabel("Max error on fixed witness pair")
        ax.legend(fontsize=6)
    fig.suptitle(
        "E3. Learning cannot repair an observation-indistinguishable pair"
    )
    fig.savefig(FIG / "fig03_no_compensation.png", dpi=220)
    plt.close(fig)


def fig04_alpha(df):
    if df.empty:
        return
    fig, axes = _grid_axes()
    for ax, dom in zip(axes, _domains(df)):
        g = df[df.domain == dom]
        if g.empty:
            continue
        ax.scatter(
            g["inverse_alpha"], g["attained"], s=18, alpha=0.65
        )
        x = np.linspace(
            g["inverse_alpha"].min(),
            g["inverse_alpha"].max(), 100
        )
        eps_hat = np.median(
            g["attained"] / g["inverse_alpha"]
        )
        ax.plot(x, eps_hat * x, "--", linewidth=1.2)
        best = g[g.is_best]
        worst = g[g.is_worst]
        if not best.empty:
            ax.scatter(
                best["inverse_alpha"], best["attained"],
                marker="*", s=100, label="best same-count"
            )
        if not worst.empty:
            ax.scatter(
                worst["inverse_alpha"], worst["attained"],
                marker="x", s=70, label="worst same-count"
            )
        k = int(g["k"].iloc[0])
        spread = float(g["alpha"].max()) / max(
            float(g["alpha"].min()), 1e-15
        )
        ax.set_title(
            f"{dom.capitalize()} (k={k}, alpha spread={spread:.2f}x)"
        )
        ax.set_xlabel("1 / alpha (sensor count fixed)")
        ax.set_ylabel("Attained robust error")
        ax.legend(fontsize=7)
    fig.suptitle("E4. Equal sensor count, unequal decision information")
    fig.savefig(FIG / "fig04_alpha_law.png", dpi=220)
    plt.close(fig)


def fig05_gaussian(df):
    if df.empty:
        return
    fig, axes = _grid_axes()
    for ax, dom in zip(axes, _domains(df)):
        g = df[df.domain == dom]
        x = g["theory_mse"].to_numpy()
        y = g["empirical_mse"].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.75)
        if len(x):
            lo = min(np.min(x), np.min(y))
            hi = max(np.max(x), np.max(y))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
        ax.set_title(dom.capitalize())
        ax.set_xlabel("Theory: sigma^2 ||B A^dagger||_F^2")
        ax.set_ylabel("Monte-Carlo MSE")
        ax.set_xscale("log")
        ax.set_yscale("log")
    fig.suptitle("E5. Gaussian noise propagation")
    fig.savefig(FIG / "fig05_gaussian_propagation.png", dpi=220)
    plt.close(fig)


def fig06_learning(df):
    if df.empty:
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8.4), constrained_layout=True
    )
    ax = axes.ravel()

    x = df["theorem_rademacher_bound"].to_numpy()
    y = df["empirical_rademacher"].to_numpy()
    ax[0].scatter(x, y, s=10, alpha=0.30)
    lo = max(min(np.min(x), np.min(y)), 1e-6)
    hi = max(np.max(x), np.max(y))
    ax[0].plot([lo, hi], [lo, hi], "--", linewidth=1.2)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Theorem Rademacher upper bound")
    ax[0].set_ylabel("Monte-Carlo empirical complexity")
    ax[0].set_title("A. Empirical complexity stays below theorem bound")

    tmp = df.copy()
    tmp["scaled_complexity"] = (
        tmp["empirical_rademacher"]
        * np.sqrt(tmp["N"])
        / tmp["lambda"]
    )
    med = (
        tmp.groupby("rank", as_index=False)["scaled_complexity"]
        .median()
    )
    ax[1].plot(
        np.sqrt(med["rank"]),
        med["scaled_complexity"],
        marker="o",
    )
    ax[1].set_xlabel("sqrt(rank(Q))")
    ax[1].set_ylabel(
        "median empirical complexity * sqrt(N) / Lambda"
    )
    ax[1].set_title("B. Decision-rank scaling")

    x = df["corollary_excess_bound"].to_numpy()
    y = np.maximum(
        df["observed_excess_risk"].to_numpy(),
        1e-12,
    )
    ax[2].scatter(x, y, s=10, alpha=0.30)
    lo = max(min(np.min(x), np.min(y)), 1e-12)
    hi = max(np.max(x), np.max(y))
    ax[2].plot([lo, hi], [lo, hi], "--", linewidth=1.2)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("High-probability excess-risk bound")
    ax[2].set_ylabel("Observed normalized excess risk")
    ax[2].set_title("C. Auditable certificate")

    med2 = (
        df.groupby(["N", "rank"], as_index=False)["corollary_excess_bound"]
        .median()
    )
    for rank, h in med2.groupby("rank"):
        ax[3].plot(
            h["N"],
            h["corollary_excess_bound"],
            marker="o",
            label=f"rank={rank}",
        )
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("N")
    ax[3].set_ylabel("Median certificate bound")
    ax[3].set_title("D. Certificate contracts with N")
    ax[3].legend(fontsize=7)

    fig.suptitle(
        "E6. Task-projected learning: complexity, rank, and certification"
    )
    fig.savefig(
        FIG / "fig06_task_projected_learning.png",
        dpi=220,
    )
    plt.close(fig)


def fig07_frontier(df):
    if df.empty:
        return
    fig, axes = _grid_axes()
    for ax, dom in zip(axes, _domains(df)):
        g = df[df.domain == dom].sort_values("target_alpha")
        ax.plot(
            g["target_alpha"], g["sdp_dual_lower_cost"],
            marker="o", label="certified SDP dual lower bound"
        )
        ax.plot(
            g["target_alpha"], g["sdp_primal_feasible_cost"],
            marker=".", linestyle="--", label="feasible SDP primal cost"
        )
        ax.plot(
            g["target_alpha"], g["exact_cost"],
            marker="s", label="exact binary optimum"
        )
        ax.plot(
            g["target_alpha"], g["rounded_cost"],
            marker="^", label="rounded feasible binary cost"
        )
        ax.set_title(dom.capitalize())
        ax.set_xlabel("Required task observability alpha_min")
        ax.set_ylabel("Normalized sensing cost")
        ax.legend(fontsize=6.5)
    fig.suptitle("E7. SVD-whitened and dual-audited sensor-resource frontier")
    fig.savefig(FIG / "fig07_sensor_resource_frontier.png", dpi=240)
    plt.close(fig)


def fig08_nonlinear(df):
    if df.empty:
        return
    domains = _domains(df)
    if not domains:
        return
    fig, axes = plt.subplots(
        1, len(domains),
        figsize=(5.3 * len(domains), 4.2),
        constrained_layout=True,
    )
    if len(domains) == 1:
        axes = [axes]
    for ax, dom in zip(axes, domains):
        g = df[df.domain == dom]
        ax.plot(
            g["radius"], g["median_decision_error"],
            marker="o", label="median decision error"
        )
        ax.plot(
            g["radius"], g["p95_decision_error"],
            marker="s", label="95th percentile"
        )
        ax.plot(
            g["radius"], g["median_q_linearization_error"],
            linestyle="--", label="q linearization error"
        )
        ax.set_title(dom.capitalize())
        ax.set_xlabel("Fraction of declared operating envelope")
        ax.set_ylabel("Decision error")
        ax.legend(fontsize=8)
    fig.suptitle(
        "E8. Nonlinear scope stress with the decision channel held out"
    )
    fig.savefig(FIG / "fig08_nonlinear_scope_stress.png", dpi=220)
    plt.close(fig)


def make_all(tables):
    fig01_exact_minimax(tables.get("e1", pd.DataFrame()))
    fig02_bounded(tables.get("e2", pd.DataFrame()))
    fig03_no_comp(tables.get("e3", pd.DataFrame()))
    fig04_alpha(tables.get("e4", pd.DataFrame()))
    fig05_gaussian(tables.get("e5", pd.DataFrame()))
    fig06_learning(tables.get("e6", pd.DataFrame()))
    fig07_frontier(tables.get("e7", pd.DataFrame()))
    fig08_nonlinear(tables.get("e8", pd.DataFrame()))
