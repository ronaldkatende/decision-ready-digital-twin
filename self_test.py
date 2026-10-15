import numpy as np
from dtwinexp.theory import (
    alpha_task, exact_minimax, worst_case_noise, bounded_pair_from_null,
    frozen_features, vector_rademacher_mc, fit_constrained_ls
)
from dtwinexp.config import get_config
from dtwinexp.domains.structure import build_structure_domain
from dtwinexp.domains.thermal import build_thermal_domain

def main():
    cfg = get_config("smoke")
    for builder in [build_structure_domain, build_thermal_domain]:
        d = builder(cfg)
        A = d.all_A()
        a = alpha_task(A, d.B)
        assert a > 0
        pred = exact_minimax(A, d.B, d.eps)
        _, att = worst_case_noise(A, d.B, d.eps)
        assert abs(pred-att) <= 1e-9 * max(1.0, abs(pred))
        rng = np.random.default_rng(1)
        C = d.sample_c(200, rng)
        Phi = frozen_features(C @ A.T, p=8, seed=2)
        Q = d.B / np.linalg.norm(d.B)
        mc, bound = vector_rademacher_mc(Phi, Q, Lambda=4.0, draws=100, seed=3)
        assert mc <= bound * 1.08
        T = C @ Q.T
        W = fit_constrained_ls(Phi, T, Lambda=4.0)
        assert np.linalg.norm(W, "fro") <= 4.0 + 1e-6
        print(d.name, "PASS", "alpha=", a, "minimax=", pred, "rad ratio=", mc/bound)
    print("SELF-TEST PASS")

if __name__ == "__main__":
    main()
