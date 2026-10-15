from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np


@dataclass
class DecisionDomain:
    name: str
    d: int
    sensor_blocks: list[np.ndarray]
    sensor_names: list[str]
    sensor_costs: np.ndarray
    B: np.ndarray
    box_halfwidth: np.ndarray
    eps: float
    metadata: dict = field(default_factory=dict)
    nonlinear_all: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None

    def __post_init__(self):
        self.B = np.atleast_2d(np.asarray(self.B, float))
        self.sensor_costs = np.asarray(self.sensor_costs, float)
        self.box_halfwidth = np.asarray(self.box_halfwidth, float)
        if self.B.shape[1] != self.d:
            raise ValueError(f"{self.name}: B has wrong width")
        if len(self.sensor_blocks) != len(self.sensor_names):
            raise ValueError(f"{self.name}: block/name mismatch")
        if len(self.sensor_blocks) != len(self.sensor_costs):
            raise ValueError(f"{self.name}: block/cost mismatch")
        for block in self.sensor_blocks:
            a = np.atleast_2d(block)
            if a.shape[1] != self.d:
                raise ValueError(f"{self.name}: sensor block has wrong width")

    @property
    def m(self) -> int:
        return len(self.sensor_blocks)

    def A(self, subset: Sequence[int]) -> np.ndarray:
        if len(subset) == 0:
            return np.zeros((0, self.d))
        return np.vstack([self.sensor_blocks[int(i)] for i in subset])

    def G(self, i: int) -> np.ndarray:
        A = np.atleast_2d(self.sensor_blocks[i])
        return A.T @ A

    def all_A(self) -> np.ndarray:
        return self.A(range(self.m))

    def sample_c(self, n: int, rng: np.random.Generator, radius: float = 1.0) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, self.d)) * (self.box_halfwidth * radius)
