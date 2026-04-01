"""Numerical integration helpers for SEIR+V simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import RateFunction, SEIRVParameters, StateVector, seir_v_odes


@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    y: np.ndarray

    @property
    def susceptible(self) -> np.ndarray:
        return self.y[0]

    @property
    def exposed(self) -> np.ndarray:
        return self.y[1]

    @property
    def infectious(self) -> np.ndarray:
        return self.y[2]

    @property
    def recovered(self) -> np.ndarray:
        return self.y[3]

    @property
    def vaccinated(self) -> np.ndarray:
        return self.y[4]


def solve_seir_v(
    params: SEIRVParameters,
    initial_state: StateVector,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    vaccination_rate_fn: RateFunction | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> SimulationResult:
    """Solve the SEIR+V initial value problem with SciPy's RK45 integrator."""

    solution = solve_ivp(
        fun=lambda t, y: seir_v_odes(t, y, params, vaccination_rate_fn),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return SimulationResult(t=solution.t, y=solution.y)
