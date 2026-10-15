# Decision-Ready Digital Twins: Reproducibility Package

This repository contains the CPU-only code and reference outputs for a decision-ready digital-twin study. The experiments separate information limits, finite-sample learning, sensor design, and local-model validity.

The package is anonymous. It contains no author names, email addresses, local machine paths, private data, credentials, or repository-specific secrets.

## Quick start

Use Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a short end-to-end check:

```bash
python run_experiments.py --mode smoke
```

Run the full production protocol:

```bash
python run_experiments.py --mode full
```

The runner downloads the public EPANET Net3 benchmark automatically if it is missing. The CIGRE medium-voltage benchmark is created through `pandapower`. The structural and electrothermal systems are generated from deterministic model definitions in the repository.

## Output locations

A run writes only to `results/`:

```text
results/
├── RUN_REPORT.md
├── theorem_checks.json
├── evidence_summary.csv
├── tables/
├── figures/
└── logs/
```

`reference_results/` contains the frozen full-run outputs used for the reported numerical values. It is not overwritten by the runner.

## Experiments

- **E1 — Exact minimax identity:** checks the exact robust decision error on observable sensor subsets.
- **E2 — Bounded-envelope ambiguity:** solves the scalar bounded decision problem by second-order cone programming.
- **E3 — No-compensation test:** constructs observation-indistinguishable state pairs with different decisions.
- **E4 — Equal sensor count, unequal information:** holds cardinality fixed and varies sensor identity.
- **E5 — Gaussian propagation:** compares Monte Carlo mean-square error with the exact Gaussian formula.
- **E6 — Task-projected learning:** checks vector Rademacher complexity and the finite-sample excess-risk certificate.
- **E7 — Sensor-resource frontier:** uses SVD whitening, independent primal/dual semidefinite programs, exhaustive binary enumeration, and a feasible rounded design.
- **E8 — Nonlinear scope stress:** measures where the local linear decision model loses accuracy in the nonlinear power and water simulators.

The full protocol is recorded in `protocol.json`. The theorem-to-code map is in `EXPERIMENTS.md`.

## Expected runtime

The smoke run is intended for a quick installation and pipeline check. The full run is CPU intensive and can take several hours, mainly because of repeated WNTR and pandapower simulations and exhaustive sensor enumeration.

No GPU is required.

## Reproducing the reference results

1. Install the dependencies.
2. Run `python self_test.py`.
3. Run `python run_experiments.py --mode full`.
4. Compare `results/theorem_checks.json` and the generated tables with `reference_results/`.

Small floating-point differences can occur across BLAS libraries, operating systems, and convex-solver versions. The pass/fail rules in `protocol.json` are the intended reproducibility criteria.

## Data and licensing

Public benchmark sources are listed in `DATA_SOURCES.md`. Third-party benchmark data are fetched from their original public source and are not covered by this repository's code license.

The code in this repository is released under the MIT License. See `LICENSE`.
