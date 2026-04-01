"""Stability and threshold analysis helpers."""

from __future__ import annotations

import numpy as np

from .model import SEIRVParameters


def infected_subsystem_jacobian(params: SEIRVParameters) -> np.ndarray:
    """Jacobian restricted to the exposed/infectious subsystem at the DFE."""

    beta0 = params.beta_fn(0.0)
    return np.array(
        [
            [-params.sigma, beta0],
            [params.sigma, -params.gamma],
        ],
        dtype=float,
    )


def disease_free_equilibrium_jacobian(params: SEIRVParameters) -> np.ndarray:
    """Jacobian of the SEIR+V system at the disease-free equilibrium."""

    beta0 = params.beta_fn(0.0)
    return np.array(
        [
            [0.0, 0.0, -beta0, 0.0, 0.0],
            [0.0, -params.sigma, beta0, 0.0, 0.0],
            [0.0, params.sigma, -params.gamma, 0.0, 0.0],
            [0.0, 0.0, params.gamma, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def basic_reproduction_number(params: SEIRVParameters) -> float:
    """Compute R0 using the standard beta/gamma ratio."""

    return float(params.beta_fn(0.0) / params.gamma)


def dominant_eigenvalue(params: SEIRVParameters) -> complex:
    """Return the leading epidemic eigenvalue near the DFE."""

    eigenvalues = np.linalg.eigvals(infected_subsystem_jacobian(params))
    return complex(eigenvalues[np.argmax(np.real(eigenvalues))])
