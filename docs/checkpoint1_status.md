# Checkpoint 1 Status Notes

## Working Progress

- A modular SEIR+V simulator package is now set up in the repository.
- Baseline research-question sweeps for vaccination timing, R0 sensitivity, and NPI timing are implemented.
- A reproducible script generates checkpoint figures in the `figures/` directory.
- Unit tests cover core model invariants and the R0 threshold behavior.

## Changes Since Proposal

- The codebase currently uses SciPy's `solve_ivp` as the simulation backend so the team can produce reliable results quickly.
- The custom RK45 solver proposed in the methodology can be added next as a drop-in replacement and validated against the current reference solver.
- The literature review PDF could not be read from the provided path during setup, so it still needs to be linked or copied into the repo if you want it tracked here.

## Suggested Division Of Remaining Work

- Solver owner: implement the custom adaptive RK45 solver and validation harness.
- Fitting owner: add data ingestion, smoothing, least-squares fitting, and bootstrap confidence intervals.
- Analysis owner: expand stability analysis plots and Jacobian-based threshold discussion.
- Experiments owner: generate polished figures and tables for RQ1 to RQ3.
- Report owner: assemble the checkpoint PDF and link the GitHub repository.
