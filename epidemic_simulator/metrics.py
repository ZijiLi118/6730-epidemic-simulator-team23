"""Summary metrics extracted from simulated epidemic trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulation import SimulationResult


@dataclass(frozen=True)
class SimulationMetrics:
    peak_infectious: float
    peak_day: float
    total_attack_rate: float
    epidemic_duration: float


def compute_metrics(result: SimulationResult) -> SimulationMetrics:
    """Compute high-level quantities used in checkpoint figures."""

    infectious = result.infectious
    population = float(np.sum(result.y[:, 0]))
    peak_index = int(np.argmax(infectious))

    active_times = result.t[infectious > 1.0]
    if active_times.size > 1:
        duration = float(active_times[-1] - active_times[0])
    else:
        duration = 0.0

    return SimulationMetrics(
        peak_infectious=float(infectious[peak_index]),
        peak_day=float(result.t[peak_index]),
        total_attack_rate=float(result.recovered[-1] / population),
        epidemic_duration=duration,
    )
