"""Checkpoint experiment sweeps for the three project research questions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import compute_metrics
from .model import (
    SEIRVParameters,
    StateVector,
    constant_beta_fn,
    make_npi_beta_fn,
    make_timed_vaccination_rate_fn,
)
from .simulation import SimulationResult, solve_seir_v


@dataclass(frozen=True)
class RQ1Sweep:
    nu_values: np.ndarray
    vaccination_start_days: np.ndarray
    peak_infectious: np.ndarray
    peak_day: np.ndarray
    total_attack_rate: np.ndarray


@dataclass(frozen=True)
class RQ2Sweep:
    r0_values: np.ndarray
    peak_infectious: np.ndarray
    total_attack_rate: np.ndarray
    trajectories: list[SimulationResult]


@dataclass(frozen=True)
class RQ3Sweep:
    alpha_values: np.ndarray
    npi_start_days: np.ndarray
    peak_infectious: np.ndarray
    total_attack_rate: np.ndarray
    epidemic_duration: np.ndarray


def run_rq1_sweep(
    params: SEIRVParameters,
    initial_state: StateVector,
    nu_values: np.ndarray,
    vaccination_start_days: np.ndarray,
    t_end: float = 365.0,
    dt: float = 1.0,
) -> RQ1Sweep:
    t_eval = np.arange(0.0, t_end + dt, dt)
    peak_infectious = np.zeros((len(nu_values), len(vaccination_start_days)))
    peak_day = np.zeros_like(peak_infectious)
    total_attack_rate = np.zeros_like(peak_infectious)

    for i, nu in enumerate(nu_values):
        for j, start_day in enumerate(vaccination_start_days):
            result = solve_seir_v(
                params,
                initial_state,
                t_span=(0.0, t_end),
                t_eval=t_eval,
                vaccination_rate_fn=make_timed_vaccination_rate_fn(float(nu), float(start_day)),
            )
            metrics = compute_metrics(result)
            peak_infectious[i, j] = metrics.peak_infectious
            peak_day[i, j] = metrics.peak_day
            total_attack_rate[i, j] = metrics.total_attack_rate

    return RQ1Sweep(nu_values, vaccination_start_days, peak_infectious, peak_day, total_attack_rate)


def run_rq2_sweep(
    params: SEIRVParameters,
    initial_state: StateVector,
    r0_values: np.ndarray,
    t_end: float = 365.0,
    dt: float = 1.0,
) -> RQ2Sweep:
    t_eval = np.arange(0.0, t_end + dt, dt)
    peak_infectious = np.zeros(len(r0_values))
    total_attack_rate = np.zeros(len(r0_values))
    trajectories: list[SimulationResult] = []

    for index, r0 in enumerate(r0_values):
        tuned_params = SEIRVParameters(
            population=params.population,
            beta_fn=constant_beta_fn(float(r0) * params.gamma),
            sigma=params.sigma,
            gamma=params.gamma,
            nu=params.nu,
        )
        result = solve_seir_v(tuned_params, initial_state, (0.0, t_end), t_eval)
        metrics = compute_metrics(result)
        peak_infectious[index] = metrics.peak_infectious
        total_attack_rate[index] = metrics.total_attack_rate
        trajectories.append(result)

    return RQ2Sweep(r0_values, peak_infectious, total_attack_rate, trajectories)


def run_rq3_sweep(
    params: SEIRVParameters,
    initial_state: StateVector,
    alpha_values: np.ndarray,
    npi_start_days: np.ndarray,
    t_end: float = 365.0,
    dt: float = 1.0,
) -> RQ3Sweep:
    t_eval = np.arange(0.0, t_end + dt, dt)
    beta0 = params.beta_fn(0.0)
    peak_infectious = np.zeros((len(alpha_values), len(npi_start_days)))
    total_attack_rate = np.zeros_like(peak_infectious)
    epidemic_duration = np.zeros_like(peak_infectious)

    for i, alpha in enumerate(alpha_values):
        for j, start_day in enumerate(npi_start_days):
            tuned_params = SEIRVParameters(
                population=params.population,
                beta_fn=make_npi_beta_fn(beta0, float(alpha), float(start_day)),
                sigma=params.sigma,
                gamma=params.gamma,
                nu=0.0,
            )
            result = solve_seir_v(tuned_params, initial_state, (0.0, t_end), t_eval)
            metrics = compute_metrics(result)
            peak_infectious[i, j] = metrics.peak_infectious
            total_attack_rate[i, j] = metrics.total_attack_rate
            epidemic_duration[i, j] = metrics.epidemic_duration

    return RQ3Sweep(
        alpha_values,
        npi_start_days,
        peak_infectious,
        total_attack_rate,
        epidemic_duration,
    )
