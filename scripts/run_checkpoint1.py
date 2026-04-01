"""Generate baseline checkpoint figures for the three research questions."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))

from epidemic_simulator import make_default_parameters, run_rq1_sweep, run_rq2_sweep, run_rq3_sweep
from epidemic_simulator.plotting import (
    plot_rq1_heatmap,
    plot_rq2_sensitivity,
    plot_rq2_trajectories,
    plot_rq3_contour,
)


def main() -> None:
    figures_dir = PROJECT_ROOT / "figures"
    (PROJECT_ROOT / ".mpl-cache").mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    params, initial_state = make_default_parameters()

    rq1 = run_rq1_sweep(
        params,
        initial_state,
        nu_values=np.linspace(0.0, 0.01, 11),
        vaccination_start_days=np.linspace(0.0, 120.0, 13),
    )
    plot_rq1_heatmap(rq1, "peak_infectious", figures_dir / "rq1_peak_infectious.png")
    plot_rq1_heatmap(rq1, "total_attack_rate", figures_dir / "rq1_total_attack_rate.png")

    rq2 = run_rq2_sweep(
        params,
        initial_state,
        r0_values=np.arange(0.5, 6.1, 0.1),
    )
    plot_rq2_sensitivity(rq2, figures_dir / "rq2_sensitivity.png")
    plot_rq2_trajectories(rq2, [0.8, 1.5, 2.5, 4.0, 6.0], figures_dir / "rq2_trajectories.png")

    rq3 = run_rq3_sweep(
        params,
        initial_state,
        alpha_values=np.linspace(0.2, 0.9, 8),
        npi_start_days=np.linspace(0.0, 90.0, 10),
    )
    plot_rq3_contour(rq3, "total_attack_rate", figures_dir / "rq3_total_attack_rate.png")
    plot_rq3_contour(rq3, "peak_infectious", figures_dir / "rq3_peak_infectious.png")

    print(f"Checkpoint figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
