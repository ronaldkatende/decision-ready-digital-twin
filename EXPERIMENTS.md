# Experiment-to-code map

| Experiment | Main implementation | Primary output |
|---|---|---|
| E1 Exact minimax | `dtwinexp/theory.py`, `dtwinexp/experiments/suite.py` | `e1_exact_minimax.csv` |
| E2 Bounded ambiguity | `scalar_box_ambiguity_socp` | `e2_bounded_ambiguity.csv` |
| E3 No compensation | `bounded_pair_from_null` | `e3_no_compensation.csv` |
| E4 Sensor geometry | `alpha_task` + fixed-cardinality enumeration | `e4_sensor_geometry.csv` |
| E5 Gaussian propagation | minimum-gain reconstructor | `e5_gaussian_propagation.csv` |
| E6 Task-projected learning | `vector_rademacher_mc`, `excess_risk_bound` | `e6_task_projected_learning.csv` |
| E7 Sensor-resource frontier | `dtwinexp/sensor_audit.py` | `e7_sensor_resource_frontier.csv` |
| E8 Nonlinear scope stress | original pandapower/WNTR simulators | `e8_nonlinear_scope_stress.csv` |

## E7 numerical audit

The sensor-resource frontier whitens the full observation map using its singular value decomposition rather than eigendecomposing the Gram matrix. This avoids squaring the condition number. The retained subspace must preserve the decision map within the declared task-residual tolerance. The code then solves the continuous semidefinite-program primal and dual independently, repairs numerical feasibility when necessary, enumerates the exact binary optimum, constructs a feasible rounded binary design, and checks

```text
certified dual lower bound <= exact binary optimum <= rounded feasible cost
```

## E8 interpretation

E8 is a scope test, not an identity test. It deliberately expands the operating envelope and records the resulting local-linear decision error in the nonlinear power and water simulators.
