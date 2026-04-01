import numpy as np

from epidemic_simulator.metrics import compute_metrics
from epidemic_simulator.model import (
    make_default_parameters,
    make_initial_state,
    make_timed_vaccination_rate_fn,
)
from epidemic_simulator.simulation import solve_seir_v


def test_initial_state_conserves_population() -> None:
    initial_state = make_initial_state(1_000.0, exposed=10.0, infectious=5.0, recovered=2.0)
    assert np.isclose(float(np.sum(initial_state)), 1_000.0)


def test_population_is_conserved_during_simulation() -> None:
    params, initial_state = make_default_parameters()
    t_eval = np.linspace(0.0, 30.0, 31)
    result = solve_seir_v(params, initial_state, (0.0, 30.0), t_eval)
    totals = np.sum(result.y, axis=0)
    assert np.allclose(totals, params.population)


def test_timed_vaccination_increases_vaccinated_compartment() -> None:
    params, initial_state = make_default_parameters()
    t_eval = np.linspace(0.0, 30.0, 31)
    result = solve_seir_v(
        params,
        initial_state,
        (0.0, 30.0),
        t_eval,
        vaccination_rate_fn=make_timed_vaccination_rate_fn(0.005, 10.0),
    )
    assert np.isclose(result.vaccinated[0], 0.0)
    assert result.vaccinated[-1] > 0.0
    metrics = compute_metrics(result)
    assert metrics.peak_infectious > 0.0
