from __future__ import annotations

from itertools import combinations
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor

DOMAIN_SEED = {
    "power": 11,
    "water": 23,
    "structure": 37,
    "thermal": 53,
}

from ..theory import (
    alpha_task,
    worst_case_noise,
    scalar_box_ambiguity_socp,
    choose_good_subset,
    choose_bad_subset,
    frozen_features,
    vector_rademacher_mc,
    fit_constrained_ls,
    exact_box_target_norm_bound,
    normalized_squared_loss_constants,
    excess_risk_bound,
)
from ..sensor_audit import run_domain_e7


def _random_subsets(domain, count, rng):
    all_s = []
    max_k = min(domain.m, domain.d + 2)

    for k in range(1, max_k + 1):
        all_s.extend(
            combinations(range(domain.m), k)
        )

    if len(all_s) <= count:
        return [tuple(x) for x in all_s]

    idx = rng.choice(
        len(all_s),
        size=count,
        replace=False,
    )

    return [
        tuple(all_s[int(i)])
        for i in idx
    ]


def exp_e1_exact(domain, cfg, rng):
    rows = []

    subsets = _random_subsets(
        domain,
        cfg.e1_random_subsets,
        rng,
    )

    goodS, _ = choose_good_subset(domain)

    forced = {
        tuple(goodS),
        tuple(range(domain.m)),
    }

    subsets = list(
        dict.fromkeys(
            [*subsets, *forced]
        )
    )

    for S in subsets:
        A = domain.A(S)
        a = alpha_task(A, domain.B)

        if not (
            a > 1e-9
            and np.isfinite(a)
        ):
            continue

        predicted = domain.eps / a
        _, attained = worst_case_noise(
            A,
            domain.B,
            domain.eps,
        )

        rows.append(
            {
                "domain": domain.name,
                "subset": "|".join(map(str, S)),
                "k": len(S),
                "alpha": a,
                "predicted_minimax": predicted,
                "attained_worst_case": attained,
                "relative_identity_error": (
                    abs(attained - predicted)
                    / max(abs(predicted), 1e-14)
                ),
            }
        )

    return pd.DataFrame(rows)


def exp_e2_bounded(domain, cfg):
    good, a = choose_good_subset(domain)
    A = domain.A(good)
    b = domain.B.reshape(-1)

    widths = (
        [0.20, 0.40, 0.70, 1.00]
        if cfg.mode == "full"
        else [0.30, 0.70, 1.00]
    )

    epss = (
        [0.10, 0.25, 0.50, 0.80]
        if cfg.mode == "full"
        else [0.20, 0.50]
    )

    rows = []

    for width in widths:
        half = domain.box_halfwidth * width

        for eps in epss:
            sol = scalar_box_ambiguity_socp(
                A,
                b,
                eps,
                half,
            )

            c1, c2 = sol["c1"], sol["c2"]

            obs_sep = float(
                np.linalg.norm(
                    A @ (c1 - c2)
                )
            )

            q_sep = float(
                abs(
                    b @ (c1 - c2)
                )
            )

            rows.append(
                {
                    "domain": domain.name,
                    "width": width,
                    "eps": eps,
                    "subset": "|".join(map(str, good)),
                    "alpha": a,
                    "omega": sol["omega"],
                    "exact_minimax": sol["minimax"],
                    "witness_q_separation": q_sep,
                    "witness_obs_separation": obs_sep,
                    "half_witness_gap": 0.5 * q_sep,
                    "solver_status": sol["status"],
                }
            )

    return pd.DataFrame(rows)


def _make_models(seed, mode):
    return {
        "ridge": Ridge(alpha=1e-6),
        "random_forest": RandomForestRegressor(
            n_estimators=120 if mode == "full" else 40,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=160 if mode == "full" else 60,
            max_leaf_nodes=31,
            learning_rate=0.06,
            random_state=seed,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32) if mode == "full" else (24,),
            max_iter=600 if mode == "full" else 250,
            early_stopping=True,
            random_state=seed,
        ),
    }


def exp_e3_no_compensation(domain, cfg):
    bad = choose_bad_subset(domain)
    goodS, good_alpha = choose_good_subset(domain)

    if bad is None:
        return pd.DataFrame()

    badS, (c1, c2, pair_gap) = bad
    A_bad = domain.A(badS)
    A_good = domain.A(goodS)

    lower = 0.5 * pair_gap
    rows = []

    for seed_idx in range(cfg.no_comp_seeds):
        rng = np.random.default_rng(
            cfg.seed
            + 1000 * seed_idx
            + DOMAIN_SEED.get(domain.name, 97)
        )

        for N in cfg.train_sizes:
            C = domain.sample_c(N, rng)
            q = (C @ domain.B.T).reshape(-1)

            for sensing, A in [
                ("bad", A_bad),
                ("good", A_good),
            ]:
                Y = C @ A.T

                y_pair = np.vstack([
                    c1 @ A.T,
                    c2 @ A.T,
                ])

                q_pair = np.array([
                    (domain.B @ c1)[0],
                    (domain.B @ c2)[0],
                ])

                for model_name, model in _make_models(
                    cfg.seed + seed_idx,
                    cfg.mode,
                ).items():
                    try:
                        model.fit(Y, q)

                        pred = np.asarray(
                            model.predict(y_pair)
                        ).reshape(-1)

                        max_pair_err = float(
                            np.max(
                                np.abs(
                                    pred - q_pair
                                )
                            )
                        )

                        same_input_gap = float(
                            np.linalg.norm(
                                y_pair[0]
                                - y_pair[1]
                            )
                        )

                        rows.append(
                            {
                                "domain": domain.name,
                                "seed": seed_idx,
                                "N": N,
                                "model": model_name,
                                "sensing": sensing,
                                "bad_subset": "|".join(map(str, badS)),
                                "good_subset": "|".join(map(str, goodS)),
                                "good_alpha": good_alpha,
                                "pair_q_gap": pair_gap,
                                "theorem_pair_lower_bound": lower,
                                "pair_observation_gap": same_input_gap,
                                "max_pair_error": max_pair_err,
                                "barrier_respected": (
                                    max_pair_err + 1e-8
                                    >= lower
                                )
                                if sensing == "bad"
                                else True,
                            }
                        )

                    except Exception as exc:
                        rows.append(
                            {
                                "domain": domain.name,
                                "seed": seed_idx,
                                "N": N,
                                "model": model_name,
                                "sensing": sensing,
                                "error": repr(exc),
                            }
                        )

    return pd.DataFrame(rows)


def exp_e4_alpha_law(domain, cfg):
    required = 10 if cfg.mode == "full" else 2
    selected = None

    for k in range(1, domain.m + 1):
        vals = []

        for S in combinations(range(domain.m), k):
            a = alpha_task(
                domain.A(S),
                domain.B,
            )

            if (
                a > 1e-9
                and np.isfinite(a)
            ):
                pred = domain.eps / a

                _, att = worst_case_noise(
                    domain.A(S),
                    domain.B,
                    domain.eps,
                )

                vals.append(
                    {
                        "domain": domain.name,
                        "subset": "|".join(map(str, S)),
                        "k": k,
                        "alpha": a,
                        "inverse_alpha": 1.0 / a,
                        "predicted": pred,
                        "attained": att,
                        "is_best": False,
                        "is_worst": False,
                    }
                )

        if len(vals) >= required:
            selected = vals
            break

    if selected is None:
        return pd.DataFrame()

    selected[
        np.argmin(
            [x["alpha"] for x in selected]
        )
    ]["is_worst"] = True

    selected[
        np.argmax(
            [x["alpha"] for x in selected]
        )
    ]["is_best"] = True

    return pd.DataFrame(selected)


def exp_e5_gaussian(domain, cfg, rng):
    rows = []

    subsets = _random_subsets(
        domain,
        min(50, cfg.e1_random_subsets),
        rng,
    )

    for S in subsets:
        A = domain.A(S)
        a = alpha_task(A, domain.B)

        if not (
            a > 1e-9
            and np.isfinite(a)
        ):
            continue

        L = domain.B @ np.linalg.pinv(A)
        sigma = 0.25

        theory = (
            sigma
            * sigma
            * (np.linalg.norm(L, "fro") ** 2)
        )

        E = rng.normal(
            scale=sigma,
            size=(cfg.gaussian_draws, A.shape[0]),
        )

        D = E @ L.T

        empirical = float(
            np.mean(
                np.sum(
                    D * D,
                    axis=1,
                )
            )
        )

        rows.append(
            {
                "domain": domain.name,
                "subset": "|".join(map(str, S)),
                "alpha": a,
                "sigma": sigma,
                "theory_mse": theory,
                "empirical_mse": empirical,
                "ratio": empirical / max(theory, 1e-15),
            }
        )

    return pd.DataFrame(rows)


def _orthonormal_Q(B, d, r, seed):
    b = B.reshape(-1)
    b = b / max(
        np.linalg.norm(b),
        1e-15,
    )

    rng = np.random.default_rng(seed)

    V = np.vstack(
        [
            b,
            rng.normal(
                size=(max(0, d - 1), d)
            ),
        ]
    )

    Qfull, _ = np.linalg.qr(V.T)

    return Qfull[:, :r].T


def exp_e6_task_learning(domain, cfg):
    goodS, _ = choose_good_subset(domain)
    A = domain.A(goodS)
    d = domain.d

    rank_list = sorted(
        set([
            1,
            min(2, d),
            min(4, d),
            d,
        ])
    )

    rows = []

    for seed_idx in range(cfg.learning_seeds):
        rng = np.random.default_rng(
            cfg.seed
            + 5000
            + seed_idx
            + DOMAIN_SEED.get(domain.name, 97)
        )

        C_ref_train = domain.sample_c(
            cfg.reference_samples,
            rng,
        )

        C_eval = domain.sample_c(
            cfg.reference_samples,
            rng,
        )

        Y_ref_train = C_ref_train @ A.T
        Y_eval = C_eval @ A.T

        p = min(
            24,
            max(8, 2 * A.shape[0]),
        )

        Phi_ref_train = frozen_features(
            Y_ref_train,
            p=p,
            seed=cfg.seed + 71,
            R=1.0,
        )

        Phi_eval = frozen_features(
            Y_eval,
            p=p,
            seed=cfg.seed + 71,
            R=1.0,
        )

        Rfeat = 1.0

        for r in rank_list:
            Q = _orthonormal_Q(
                domain.B,
                d,
                r,
                cfg.seed + r,
            )

            target_bound = exact_box_target_norm_bound(
                Q,
                domain.box_halfwidth,
            )

            L_l, M_loss, S_loss = (
                normalized_squared_loss_constants(
                    Q=Q,
                    Lambda=cfg.lambda_bound,
                    R=Rfeat,
                    target_norm_bound=target_bound,
                )
            )

            T_ref = C_ref_train @ Q.T
            T_eval = C_eval @ Q.T

            Wstar = fit_constrained_ls(
                Phi_ref_train,
                T_ref,
                cfg.lambda_bound,
            )

            pred_star = Phi_eval @ Wstar.T

            raw_ref = float(
                np.mean(
                    np.sum(
                        (pred_star - T_eval) ** 2,
                        axis=1,
                    )
                )
            )

            risk_star = raw_ref / (S_loss ** 2)

            for N in cfg.train_sizes:
                C = domain.sample_c(N, rng)
                Y = C @ A.T

                Phi = frozen_features(
                    Y,
                    p=p,
                    seed=cfg.seed + 71,
                    R=1.0,
                )

                T = C @ Q.T

                W = fit_constrained_ls(
                    Phi,
                    T,
                    cfg.lambda_bound,
                )

                pred = Phi_eval @ W.T

                raw_risk = float(
                    np.mean(
                        np.sum(
                            (pred - T_eval) ** 2,
                            axis=1,
                        )
                    )
                )

                risk = raw_risk / (S_loss ** 2)
                excess = max(
                    0.0,
                    risk - risk_star,
                )

                rad_mc, rad_bound = vector_rademacher_mc(
                    Phi,
                    Q,
                    cfg.lambda_bound,
                    draws=cfg.rad_mc_draws,
                    seed=(
                        cfg.seed
                        + 10000
                        + seed_idx
                        + N
                        + r
                    ),
                )

                exc_bound = excess_risk_bound(
                    N,
                    Q,
                    cfg.lambda_bound,
                    Rfeat,
                    L_l,
                    M_loss,
                    cfg.delta,
                )

                rows.append(
                    {
                        "domain": domain.name,
                        "seed": seed_idx,
                        "rank": r,
                        "state_dim": d,
                        "N": N,
                        "lambda": cfg.lambda_bound,
                        "target_norm_bound": target_bound,
                        "loss_scale_S": S_loss,
                        "empirical_rademacher": rad_mc,
                        "theorem_rademacher_bound": rad_bound,
                        "rad_ratio": (
                            rad_mc
                            / max(rad_bound, 1e-15)
                        ),
                        "normalized_risk": risk,
                        "normalized_reference_class_risk": risk_star,
                        "observed_excess_risk": excess,
                        "corollary_excess_bound": exc_bound,
                        "certificate_tightness_ratio": (
                            excess
                            / max(exc_bound, 1e-15)
                        ),
                        "certificate_respected": (
                            excess
                            <= exc_bound + 1e-10
                        ),
                    }
                )

    return pd.DataFrame(rows)


def exp_e7_sensor_frontier(domain, cfg):
    """Run the SVD-whitened, primal-dual audited sensor frontier."""
    return run_domain_e7(domain, cfg)


def exp_e8_nonlinear_scope(domain, cfg):
    if domain.nonlinear_all is None:
        return pd.DataFrame()

    goodS, a = choose_good_subset(domain)
    A = domain.A(goodS)

    if not (
        a > 1e-9
        and np.isfinite(a)
    ):
        return pd.DataFrame()

    L = domain.B @ np.linalg.pinv(A)

    radii = (
        [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        if cfg.mode == "full"
        else [0.10, 0.35, 0.75]
    )

    rng = np.random.default_rng(
        cfg.seed
        + 9000
        + DOMAIN_SEED.get(domain.name, 97)
    )

    rows = []

    for radius in radii:
        C = domain.sample_c(
            cfg.nonlinear_samples,
            rng,
            radius=radius,
        )

        errs = []
        lin_mismatch_y = []
        lin_mismatch_q = []

        for c in C:
            try:
                y_all, q_nl = domain.nonlinear_all(c)

                y_sel = np.concatenate(
                    [
                        np.atleast_1d(y_all[i])
                        for i in goodS
                    ]
                )

                q_hat = float(
                    (
                        L
                        @ y_sel.reshape(-1, 1)
                    )[0, 0]
                )

                errs.append(
                    abs(q_hat - q_nl)
                )

                y_lin = A @ c
                q_lin = float(
                    (domain.B @ c)[0]
                )

                lin_mismatch_y.append(
                    float(
                        np.linalg.norm(
                            y_sel - y_lin
                        )
                    )
                )

                lin_mismatch_q.append(
                    abs(q_nl - q_lin)
                )

            except Exception:
                continue

        if errs:
            rows.append(
                {
                    "domain": domain.name,
                    "radius": radius,
                    "n_success": len(errs),
                    "median_decision_error": float(
                        np.median(errs)
                    ),
                    "p95_decision_error": float(
                        np.quantile(errs, 0.95)
                    ),
                    "median_observation_linearization_error": float(
                        np.median(lin_mismatch_y)
                    ),
                    "median_q_linearization_error": float(
                        np.median(lin_mismatch_q)
                    ),
                    "alpha_local": a,
                    "subset": "|".join(map(str, goodS)),
                }
            )

    return pd.DataFrame(rows)


def run_domain_suite(domain, cfg):
    rng = np.random.default_rng(
        cfg.seed
        + DOMAIN_SEED.get(domain.name, 97)
    )

    out = {}
    t0 = time.time()

    out["e1"] = exp_e1_exact(domain, cfg, rng)
    out["e2"] = exp_e2_bounded(domain, cfg)
    out["e3"] = exp_e3_no_compensation(domain, cfg)
    out["e4"] = exp_e4_alpha_law(domain, cfg)
    out["e5"] = exp_e5_gaussian(domain, cfg, rng)
    out["e6"] = exp_e6_task_learning(domain, cfg)
    out["e7"] = exp_e7_sensor_frontier(domain, cfg)
    out["e8"] = exp_e8_nonlinear_scope(domain, cfg)
    out["runtime_seconds"] = time.time() - t0

    return out
