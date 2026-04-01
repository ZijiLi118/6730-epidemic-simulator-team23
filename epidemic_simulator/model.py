"""SEIR+V state definitions and model equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

StateVector = NDArray[np.float64]
RateFunction = Callable[[float], float]


@dataclass(frozen=True)
class SEIRVParameters:
    population: float
    beta_fn: RateFunction
    sigma: float
    gamma: float
    nu: float = 0.0


def constant_beta_fn(beta: float) -> RateFunction:
    """Return a constant transmission-rate function."""

    return lambda _: beta


def make_npi_beta_fn(beta0: float, alpha: float, t_npi: float) -> RateFunction:
    """Reduce transmission after an intervention day."""

    return lambda t: beta0 if t < t_npi else alpha * beta0


def make_timed_vaccination_rate_fn(nu: float, start_day: float) -> RateFunction:
    """Start vaccinating at a chosen day."""

    return lambda t: nu if t >= start_day else 0.0


def make_initial_state(
    population: float,
    exposed: float = 100.0,
    infectious: float = 10.0,
    recovered: float = 0.0,
    vaccinated: float = 0.0,
) -> StateVector:
    """Create an initial state vector that sums to the population."""

    susceptible = population - exposed - infectious - recovered - vaccinated
    if susceptible < 0:
        raise ValueError("Initial compartments must sum to less than the population.")

    return np.array(
        [susceptible, exposed, infectious, recovered, vaccinated], dtype=float
    )


def seir_v_odes(
    t: float,
    y: StateVector,
    params: SEIRVParameters,
    vaccination_rate_fn: RateFunction | None = None,
) -> StateVector:
    """SEIR model with an optional vaccinated compartment."""

    susceptible, exposed, infectious, recovered, vaccinated = y
    beta = params.beta_fn(t)
    nu = vaccination_rate_fn(t) if vaccination_rate_fn is not None else params.nu
    population = params.population

    d_susceptible = -beta * susceptible * infectious / population - nu * susceptible
    d_exposed = beta * susceptible * infectious / population - params.sigma * exposed
    d_infectious = params.sigma * exposed - params.gamma * infectious
    d_recovered = params.gamma * infectious
    d_vaccinated = nu * susceptible

    return np.array(
        [d_susceptible, d_exposed, d_infectious, d_recovered, d_vaccinated],
        dtype=float,
    )


def make_default_parameters() -> tuple[SEIRVParameters, StateVector]:
    """Provide a baseline COVID-like configuration for checkpoint experiments."""

    population = 1_000_000.0
    beta0 = 0.25
    sigma = 1.0 / 5.2
    gamma = 1.0 / 10.0

    params = SEIRVParameters(
        population=population,
        beta_fn=constant_beta_fn(beta0),
        sigma=sigma,
        gamma=gamma,
        nu=0.0,
    )
    initial_state = make_initial_state(population=population)
    return params, initial_state
