# Reference full-run report

Seed: `20260902`

## Theorem checks

### E1_exact_minimax
```json
{
  "n": 270,
  "max_relative_identity_error": 5.622219624647148e-16,
  "pass": true
}
```

### E2_bounded_exact_witness
```json
{
  "n": 64,
  "max_abs_gap": 0.0,
  "pass": true
}
```

### E3_no_compensation
```json
{
  "n_bad_model_fits": 560,
  "violations": 0,
  "pass": true
}
```

### E4_alpha_law
```json
{
  "spearman_by_domain": {
    "power": 1.0,
    "structure": 1.0,
    "thermal": 1.0,
    "water": 0.9999999999999999
  }
}
```

### E5_gaussian
```json
{
  "n": 113,
  "median_ratio": 1.0009773579329917,
  "max_abs_ratio_error": 0.039781969391513794,
  "pass": true
}
```

### E6_task_learning
```json
{
  "n": 1050,
  "rad_bound_violations": 0,
  "excess_bound_violations": 0,
  "pass": true
}
```

### E7_resource_frontier
```json
{
  "n": 20,
  "exact_feasibility_violations": 0,
  "rounded_feasibility_violations": 0,
  "sdp_lower_bound_violations": 0,
  "rounding_order_violations": 0,
  "primal_psd_violations": 0,
  "dual_feasibility_violations": 0,
  "task_residual_violations": 0,
  "max_primal_dual_gap": 2.9601652906663393e-08,
  "pass": true
}
```

### E8_scope_stress
```json
{
  "note": "No pass/fail criterion: this experiment maps where the local linear decision slice ceases to approximate the nonlinear simulator.",
  "domains": [
    "power",
    "water"
  ]
}
```

## Notes

E1 and E2 are numerical identity checks. E3 uses constructed observation-indistinguishable witnesses. E7 uses the SVD-whitened primal-dual audit. E8 is a scope-stress experiment and has no pass/fail rule.
