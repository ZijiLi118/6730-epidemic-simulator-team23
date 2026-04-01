# CSE 6730 Epidemic Simulator

This repository contains the Checkpoint 1 codebase for a CSE 6730 epidemic simulation project by Team 23. The current implementation focuses on a modular SEIR+V simulator, reproducible baseline experiments, and lightweight tests so the team has working code and figures to support the checkpoint report.

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

## Project Overview

We model epidemic spread in a closed population using an SEIR+V compartmental system with susceptible (`S`), exposed (`E`), infectious (`I`), recovered (`R`), and vaccinated (`V`) states. The current checkpoint explores three research questions:

- How vaccination timing and vaccination rate affect peak infections and attack rate.
- How epidemic severity changes as the basic reproduction number `R0` varies.
- How the timing and intensity of non-pharmaceutical interventions affect epidemic outcomes.

The simulator uses deterministic ordinary differential equations and currently relies on SciPy's `solve_ivp` RK45 integrator as a stable reference backend for checkpoint experiments.

## Platform Of Development

The project is developed in Python with a modular scientific-computing workflow:

- `NumPy` for numerical arrays and parameter sweeps
- `SciPy` for ODE integration
- `Matplotlib` for figure generation
- `pytest` for basic validation of model behavior
- `GitHub` for team collaboration, version control, and report artifacts

The codebase is organized so model equations, simulation utilities, metrics, analysis helpers, experiments, and plotting are separated into small modules that can be extended in later milestones.

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

- `rq1_peak_I.png`
- `rq1_attack_rate.png`
- `rq2_sensitivity.png`
- `rq2_trajectories.png`
- `rq3_peak_I.png`
- `rq3_attack_rate.png`

## Current Status

- Core SEIR+V dynamics are implemented in `epidemic_simulator/model.py`.
- Numerical integration is handled through `scipy.integrate.solve_ivp` in `epidemic_simulator/simulation.py`.
- Stability analysis helpers for the disease-free equilibrium are in `epidemic_simulator/analysis.py`.
- Checkpoint experiment sweeps and plotting utilities are ready for baseline figures.
- A reproducible checkpoint script generates all six report figures from the current simulator.
- Tests cover population conservation, vaccination behavior, and threshold behavior around `R0 = 1`.

## Checkpoint 1 Progress

- A working end-to-end simulation pipeline is in place from parameter setup to generated figures.
- Baseline sweeps for vaccination timing, `R0` sensitivity, and NPI timing have been completed.
- The project used SciPy's built-in solver first so the team could prioritize reliable results and checkpoint progress before implementing a custom solver.
- The repository now contains enough code and output to support the checkpoint's progress update and initial results sections.

## Planned Next Steps

1. Add the custom adaptive RK45 solver from the proposal and validate it against SciPy.
2. Add real epidemic data loading and preprocessing.
3. Implement parameter fitting for beta and gamma.
4. Add bootstrap uncertainty intervals and hold-out validation.
5. Expand the literature review and finalize the checkpoint report PDF.
