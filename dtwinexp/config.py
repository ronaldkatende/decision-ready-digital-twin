from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NET3_PATH = DATA / "Net3.inp"
OUT = ROOT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"
LOG = OUT / "logs"

for p in [DATA, OUT, FIG, TAB, LOG]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunConfig:
    mode: str
    seed: int
    domains: tuple[str, ...]
    max_candidates: int
    exact_subset_limit: int
    e1_random_subsets: int
    gaussian_draws: int
    rad_mc_draws: int
    learning_seeds: int
    no_comp_seeds: int
    nonlinear_samples: int
    reference_samples: int
    train_sizes: tuple[int, ...]
    eps: float
    delta: float
    lambda_bound: float


def get_config(mode: str) -> RunConfig:
    if mode == "smoke":
        return RunConfig(
            mode="smoke",
            seed=20260902,
            domains=("power", "water", "structure", "thermal"),
            max_candidates=10,
            exact_subset_limit=10,
            e1_random_subsets=24,
            gaussian_draws=1500,
            rad_mc_draws=150,
            learning_seeds=2,
            no_comp_seeds=2,
            nonlinear_samples=50,
            reference_samples=2000,
            train_sizes=(50, 200, 800),
            eps=0.50,
            delta=0.05,
            lambda_bound=4.0,
        )
    if mode == "full":
        return RunConfig(
            mode="full",
            seed=20260902,
            domains=("power", "water", "structure", "thermal"),
            max_candidates=14,
            exact_subset_limit=14,
            e1_random_subsets=150,
            gaussian_draws=10000,
            rad_mc_draws=800,
            learning_seeds=10,
            no_comp_seeds=5,
            nonlinear_samples=400,
            reference_samples=15000,
            train_sizes=(50, 100, 250, 500, 1000, 2500, 5000),
            eps=0.50,
            delta=0.05,
            lambda_bound=4.0,
        )
    raise ValueError(f"Unknown mode: {mode}")
