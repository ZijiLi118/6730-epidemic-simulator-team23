# CSE 6730 Epidemic Simulator

This repository contains the initial codebase for the team epidemic simulation project. The current checkpoint implementation focuses on a modular SEIR+V simulator, baseline scenario sweeps, and lightweight tests so the team has working code to show for Checkpoint 1.

## Repo Layout

```text
epidemic_simulator/
  analysis.py
  experiments.py
  metrics.py
  model.py
  plotting.py
  simulation.py
scripts/
  run_checkpoint1.py
tests/
docs/
```

## Quick Start

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the test suite:

```bash
pytest
```

3. Generate checkpoint figures:

```bash
python3 scripts/run_checkpoint1.py
```

This creates PNG figures in `figures/` for:

- RQ1: vaccination timing and coverage
- RQ2: sensitivity to R0
- RQ3: NPI timing and strictness

## Current Status

- Core SEIR+V dynamics are implemented in `epidemic_simulator/model.py`.
- Numerical integration is handled through `scipy.integrate.solve_ivp` in `epidemic_simulator/simulation.py`.
- Stability analysis helpers for the disease-free equilibrium are in `epidemic_simulator/analysis.py`.
- Checkpoint experiment sweeps and plotting utilities are ready for baseline figures.

## Recommended Next Steps

1. Add the custom adaptive RK45 solver from the proposal and validate it against SciPy.
2. Add real epidemic data loading and preprocessing.
3. Implement parameter fitting for beta and gamma.
4. Add bootstrap uncertainty intervals and hold-out validation.
5. Move checkpoint figures and screenshots into the report PDF.
