import pytest

from epidemic_simulator.analysis import basic_reproduction_number, dominant_eigenvalue
from epidemic_simulator.model import SEIRVParameters, constant_beta_fn


def test_basic_reproduction_number_matches_beta_over_gamma() -> None:
    params = SEIRVParameters(
        population=1_000.0,
        beta_fn=constant_beta_fn(0.3),
        sigma=0.2,
        gamma=0.1,
    )
    assert basic_reproduction_number(params) == pytest.approx(3.0)


def test_dominant_eigenvalue_changes_sign_near_threshold() -> None:
    subcritical = SEIRVParameters(
        population=1_000.0,
        beta_fn=constant_beta_fn(0.08),
        sigma=0.2,
        gamma=0.1,
    )
    supercritical = SEIRVParameters(
        population=1_000.0,
        beta_fn=constant_beta_fn(0.2),
        sigma=0.2,
        gamma=0.1,
    )
    assert dominant_eigenvalue(subcritical).real < 0.0
    assert dominant_eigenvalue(supercritical).real > 0.0
