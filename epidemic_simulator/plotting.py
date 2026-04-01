"""Figure-generation helpers for checkpoint artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .experiments import RQ1Sweep, RQ2Sweep, RQ3Sweep


def _save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rq1_heatmap(sweep: RQ1Sweep, metric: str, save_path: str | Path | None = None) -> None:
    labels = {
        "peak_infectious": ("Peak Infectious Count", "Peak I", "YlOrRd"),
        "peak_day": ("Day of Peak Infection", "Peak Day", "viridis"),
        "total_attack_rate": ("Total Attack Rate", "Attack Rate", "YlOrRd"),
    }
    title, cbar_label, cmap = labels[metric]
    fig, ax = plt.subplots(figsize=(8, 6))
    data = getattr(sweep, metric)
    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[
            sweep.vaccination_start_days[0],
            sweep.vaccination_start_days[-1],
            sweep.nu_values[0],
            sweep.nu_values[-1],
        ],
        cmap=cmap,
    )
    ax.set_xlabel("Vaccination Start Day")
    ax.set_ylabel("Daily Vaccination Rate (nu)")
    ax.set_title(f"RQ1: {title}")
    plt.colorbar(image, ax=ax, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_rq2_sensitivity(sweep: RQ2Sweep, save_path: str | Path | None = None) -> None:
    fig, left_axis = plt.subplots(figsize=(8, 5))
    right_axis = left_axis.twinx()
    left_axis.plot(sweep.r0_values, sweep.peak_infectious, "o-", color="#d62728", markersize=3)
    left_axis.set_xlabel("Basic Reproduction Number (R0)")
    left_axis.set_ylabel("Peak Infectious Count", color="#d62728")
    left_axis.tick_params(axis="y", labelcolor="#d62728")
    left_axis.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
    right_axis.plot(sweep.r0_values, sweep.total_attack_rate, "s-", color="#1f77b4", markersize=3)
    right_axis.set_ylabel("Total Attack Rate", color="#1f77b4")
    right_axis.tick_params(axis="y", labelcolor="#1f77b4")
    left_axis.set_title("RQ2: Sensitivity of Epidemic Outcomes to R0")
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_rq2_trajectories(
    sweep: RQ2Sweep,
    r0_subset: list[float] | None = None,
    save_path: str | Path | None = None,
) -> None:
    indices = (
        np.linspace(0, len(sweep.r0_values) - 1, 5, dtype=int)
        if r0_subset is None
        else [int(np.argmin(np.abs(sweep.r0_values - value))) for value in r0_subset]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for index in indices:
        trajectory = sweep.trajectories[index]
        ax.plot(trajectory.t, trajectory.infectious, label=f"R0={sweep.r0_values[index]:.1f}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infectious (I)")
    ax.set_title("RQ2: Epidemic Trajectories")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_rq3_contour(sweep: RQ3Sweep, metric: str, save_path: str | Path | None = None) -> None:
    labels = {
        "total_attack_rate": ("Total Attack Rate", "Attack Rate"),
        "epidemic_duration": ("Epidemic Duration", "Days"),
        "peak_infectious": ("Peak Infectious Count", "Peak I"),
    }
    title, cbar_label = labels[metric]
    grid_x, grid_y = np.meshgrid(sweep.npi_start_days, sweep.alpha_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    data = getattr(sweep, metric)
    filled = ax.contourf(grid_x, grid_y, data, levels=20, cmap="RdYlGn_r")
    contour = ax.contour(grid_x, grid_y, data, levels=10, colors="k", linewidths=0.5, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel("NPI Start Day")
    ax.set_ylabel("NPI Effectiveness (alpha)")
    ax.set_title(f"RQ3: {title}")
    plt.colorbar(filled, ax=ax, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, save_path)
