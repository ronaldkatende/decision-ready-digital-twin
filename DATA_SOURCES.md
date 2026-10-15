# Public data and benchmark sources

## Electrical distribution

The electrical benchmark is created with:

```python
pandapower.networks.create_cigre_network_mv(with_der="all")
```

Documentation: https://pandapower.readthedocs.io/en/latest/networks/cigre.html

## Water distribution

The water benchmark is EPA EPANET Example Network 3 (Net3), fetched from the public USEPA/WNTR repository:

https://raw.githubusercontent.com/USEPA/WNTR/main/examples/networks/Net3.inp

WNTR repository: https://github.com/USEPA/WNTR

The file is downloaded by `download_benchmarks.py` and stored locally as `data/Net3.inp`.

## Structural monitoring

The structural benchmark is generated locally from a six-storey shear-frame model with lumped masses, inter-storey springs, Rayleigh damping, exact discrete-time dynamics, and acceleration-history sensing.

## Electrothermal system

The electrothermal benchmark is generated locally from the deterministic state-space model in `dtwinexp/domains/thermal.py`.

## Reproducibility rule

The production protocol is fixed in `protocol.json`. Declared domains and failed checks should not be removed from a run because the result is unfavorable.
