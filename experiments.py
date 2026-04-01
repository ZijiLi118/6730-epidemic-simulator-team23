import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Callable, Dict, Tuple, Optional
import os


def seir_v_odes(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    S, E, I, R, V = y
    N = params['N']
    beta = params['beta_fn'](t)
    sigma = params['sigma']
    gamma = params['gamma']
    nu = params['nu']

    dSdt = -beta * S * I / N - nu * S
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S

    return np.array([dSdt, dEdt, dIdt, dRdt, dVdt])


def solve_seir(params: dict, y0: np.ndarray, t_span: Tuple[float, float],
               t_eval: np.ndarray) -> dict:
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        fun=lambda t, y: seir_v_odes(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return {
        't': sol.t,
        'S': sol.y[0],
        'E': sol.y[1],
        'I': sol.y[2],
        'R': sol.y[3],
        'V': sol.y[4],
    }


def compute_metrics(result: dict) -> dict:
    I = result['I']
    t = result['t']
    N = result['S'][0] + result['E'][0] + result['I'][0] + result['R'][0] + result['V'][0]

    peak_idx = np.argmax(I)
    peak_I = I[peak_idx]
    peak_day = t[peak_idx]
    total_attack = result['R'][-1] / N
    active = t[I > 1.0]
    duration = active[-1] - active[0] if len(active) > 1 else 0.0

    return {
        'peak_I': peak_I,
        'peak_day': peak_day,
        'total_attack': total_attack,
        'duration': duration,
    }


def make_vaccination_params(base_params: dict, nu: float,
                            vax_start_day: float) -> dict:
    p = dict(base_params)
    p['nu'] = 0.0
    p['_nu_value'] = nu
    p['_vax_start'] = vax_start_day
    return p


def seir_v_odes_vax_timed(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    S, E, I, R, V = y
    N = params['N']
    beta = params['beta_fn'](t)
    sigma = params['sigma']
    gamma = params['gamma']
    nu = params['_nu_value'] if t >= params['_vax_start'] else 0.0

    dSdt = -beta * S * I / N - nu * S
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S

    return np.array([dSdt, dEdt, dIdt, dRdt, dVdt])


def solve_seir_vax_timed(params: dict, y0: np.ndarray, t_span, t_eval):
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        fun=lambda t, y: seir_v_odes_vax_timed(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return {
        't': sol.t, 'S': sol.y[0], 'E': sol.y[1],
        'I': sol.y[2], 'R': sol.y[3], 'V': sol.y[4],
    }


def run_rq1_sweep(base_params: dict, y0: np.ndarray,
                   nu_values: np.ndarray, vax_start_days: np.ndarray,
                   t_end: float = 365.0, dt: float = 1.0) -> dict:
    t_eval = np.arange(0, t_end + dt, dt)
    t_span = (0.0, t_end)

    n_nu = len(nu_values)
    n_days = len(vax_start_days)

    peak_I = np.zeros((n_nu, n_days))
    peak_day = np.zeros((n_nu, n_days))
    total_attack = np.zeros((n_nu, n_days))

    total_runs = n_nu * n_days
    print(f"[RQ1] Starting {total_runs} simulations "
          f"({n_nu} nu x {n_days} start days)...")

    for i, nu in enumerate(nu_values):
        for j, start_day in enumerate(vax_start_days):
            params = dict(base_params)
            params['_nu_value'] = nu
            params['_vax_start'] = start_day
            params['nu'] = 0.0

            result = solve_seir_vax_timed(params, y0, t_span, t_eval)
            metrics = compute_metrics(result)

            peak_I[i, j] = metrics['peak_I']
            peak_day[i, j] = metrics['peak_day']
            total_attack[i, j] = metrics['total_attack']

        print(f"  nu={nu:.4f} done ({i+1}/{n_nu})")

    return {
        'nu_values': nu_values,
        'vax_start_days': vax_start_days,
        'peak_I': peak_I,
        'peak_day': peak_day,
        'total_attack': total_attack,
    }


def plot_rq1_heatmap(sweep: dict, metric: str = 'peak_I',
                     save_path: Optional[str] = None):
    labels = {
        'peak_I': ('Peak Infectious Count', 'Peak I'),
        'peak_day': ('Day of Peak Infection', 'Peak Day'),
        'total_attack': ('Total Attack Rate (fraction)', 'Attack Rate'),
    }
    title, cbar_label = labels[metric]

    fig, ax = plt.subplots(figsize=(8, 6))
    data = sweep[metric]
    im = ax.imshow(
        data, origin='lower', aspect='auto',
        extent=[
            sweep['vax_start_days'][0], sweep['vax_start_days'][-1],
            sweep['nu_values'][0], sweep['nu_values'][-1],
        ],
        cmap='YlOrRd' if metric != 'peak_day' else 'viridis',
    )
    ax.set_xlabel('Vaccination Start Day')
    ax.set_ylabel('Daily Vaccination Rate (ν)')
    ax.set_title(f'RQ1: {title}')
    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def run_rq2_sweep(base_params: dict, y0: np.ndarray,
                  R0_values: np.ndarray,
                  t_end: float = 365.0, dt: float = 1.0) -> dict:
    t_eval = np.arange(0, t_end + dt, dt)
    t_span = (0.0, t_end)
    gamma = base_params['gamma']

    peak_I = np.zeros(len(R0_values))
    total_attack = np.zeros(len(R0_values))
    trajectories = []

    print(f"[RQ2] Starting {len(R0_values)} simulations (R0 sweep)...")

    for k, R0 in enumerate(R0_values):
        params = dict(base_params)
        beta_val = R0 * gamma
        params['beta_fn'] = lambda t, b=beta_val: b

        result = solve_seir(params, y0, t_span, t_eval)
        metrics = compute_metrics(result)

        peak_I[k] = metrics['peak_I']
        total_attack[k] = metrics['total_attack']
        trajectories.append(result)

    print(f"  R0 sweep complete.")

    return {
        'R0_values': R0_values,
        'peak_I': peak_I,
        'total_attack': total_attack,
        'trajectories': trajectories,
    }


def plot_rq2_sensitivity(sweep: dict, save_path: Optional[str] = None):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#d62728'
    ax1.set_xlabel('Basic Reproduction Number ($R_0$)')
    ax1.set_ylabel('Peak Infectious Count', color=color1)
    ax1.plot(sweep['R0_values'], sweep['peak_I'], 'o-', color=color1,
             markersize=3, label='Peak I')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='$R_0=1$')

    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    ax2.set_ylabel('Total Attack Rate', color=color2)
    ax2.plot(sweep['R0_values'], sweep['total_attack'], 's-', color=color2,
             markersize=3, label='Attack Rate')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('RQ2: Sensitivity of Epidemic Outcomes to $R_0$')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_rq2_trajectories(sweep: dict, R0_subset: list = None,
                          save_path: Optional[str] = None):
    if R0_subset is None:
        idx = np.linspace(0, len(sweep['R0_values']) - 1, 5, dtype=int)
    else:
        idx = [np.argmin(np.abs(sweep['R0_values'] - r)) for r in R0_subset]

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in idx:
        traj = sweep['trajectories'][k]
        R0 = sweep['R0_values'][k]
        ax.plot(traj['t'], traj['I'], label=f'$R_0$={R0:.1f}')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Infectious (I)')
    ax.set_title('RQ2: Epidemic Trajectories for Varying $R_0$')
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def make_npi_beta_fn(beta0: float, alpha: float, t_npi: float) -> Callable:
    def beta_fn(t):
        return beta0 if t < t_npi else alpha * beta0
    return beta_fn


def run_rq3_sweep(base_params: dict, y0: np.ndarray,
                  alpha_values: np.ndarray, t_npi_values: np.ndarray,
                  t_end: float = 365.0, dt: float = 1.0) -> dict:
    t_eval = np.arange(0, t_end + dt, dt)
    t_span = (0.0, t_end)

    beta0 = base_params['beta_fn'](0.0)

    n_alpha = len(alpha_values)
    n_t = len(t_npi_values)

    total_attack = np.zeros((n_alpha, n_t))
    duration = np.zeros((n_alpha, n_t))
    peak_I = np.zeros((n_alpha, n_t))

    total_runs = n_alpha * n_t
    print(f"[RQ3] Starting {total_runs} simulations "
          f"({n_alpha} alpha x {n_t} t_npi)...")

    for i, alpha in enumerate(alpha_values):
        for j, t_npi in enumerate(t_npi_values):
            params = dict(base_params)
            params['beta_fn'] = make_npi_beta_fn(beta0, alpha, t_npi)
            params['nu'] = 0.0

            result = solve_seir(params, y0, t_span, t_eval)
            metrics = compute_metrics(result)

            total_attack[i, j] = metrics['total_attack']
            duration[i, j] = metrics['duration']
            peak_I[i, j] = metrics['peak_I']

        print(f"  alpha={alpha:.2f} done ({i+1}/{n_alpha})")

    return {
        'alpha_values': alpha_values,
        't_npi_values': t_npi_values,
        'total_attack': total_attack,
        'duration': duration,
        'peak_I': peak_I,
    }


def plot_rq3_contour(sweep: dict, metric: str = 'total_attack',
                     save_path: Optional[str] = None):
    labels = {
        'total_attack': ('Total Attack Rate', 'Attack Rate'),
        'duration': ('Epidemic Duration (days)', 'Days'),
        'peak_I': ('Peak Infectious Count', 'Peak I'),
    }
    title, cbar_label = labels[metric]

    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(sweep['t_npi_values'], sweep['alpha_values'])
    data = sweep[metric]

    cf = ax.contourf(X, Y, data, levels=20, cmap='RdYlGn_r')
    cs = ax.contour(X, Y, data, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    ax.set_xlabel('NPI Start Day ($t_{NPI}$)')
    ax.set_ylabel('NPI Effectiveness ($\\alpha$, lower = stricter)')
    ax.set_title(f'RQ3: {title}')
    plt.colorbar(cf, ax=ax, label=cbar_label)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def get_default_params() -> Tuple[dict, np.ndarray]:
    N = 1_000_000
    beta0 = 0.25
    sigma = 1.0 / 5.2
    gamma = 1.0 / 10.0

    params = {
        'beta_fn': lambda t: beta0,
        'sigma': sigma,
        'gamma': gamma,
        'nu': 0.0,
        'N': N,
    }

    I0 = 10.0
    E0 = 100.0
    R0_init = 0.0
    V0 = 0.0
    S0 = N - E0 - I0 - R0_init - V0
    y0 = np.array([S0, E0, I0, R0_init, V0])

    return params, y0


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    params, y0 = get_default_params()

    nu_values = np.linspace(0.0, 0.01, 11)
    vax_start_days = np.linspace(0, 120, 13)
    rq1 = run_rq1_sweep(params, y0, nu_values, vax_start_days, t_end=365.0)
    plot_rq1_heatmap(rq1, metric='peak_I',
                     save_path=os.path.join(fig_dir, 'rq1_peak_I.png'))
    plot_rq1_heatmap(rq1, metric='total_attack',
                     save_path=os.path.join(fig_dir, 'rq1_attack_rate.png'))
    print("[RQ1] Done.\n")

    R0_values = np.arange(0.5, 6.1, 0.1)
    rq2 = run_rq2_sweep(params, y0, R0_values, t_end=365.0)
    plot_rq2_sensitivity(rq2, save_path=os.path.join(fig_dir, 'rq2_sensitivity.png'))
    plot_rq2_trajectories(rq2, R0_subset=[0.8, 1.5, 2.5, 4.0, 6.0],
                          save_path=os.path.join(fig_dir, 'rq2_trajectories.png'))
    print("[RQ2] Done.\n")

    alpha_values = np.linspace(0.2, 0.9, 8)
    t_npi_values = np.linspace(0, 90, 10)
    rq3 = run_rq3_sweep(params, y0, alpha_values, t_npi_values, t_end=365.0)
    plot_rq3_contour(rq3, metric='total_attack',
                     save_path=os.path.join(fig_dir, 'rq3_attack_rate.png'))
    plot_rq3_contour(rq3, metric='peak_I',
                     save_path=os.path.join(fig_dir, 'rq3_peak_I.png'))
    print("[RQ3] Done.\n")

    print("All experiments complete. Figures saved to:", fig_dir)


if __name__ == '__main__':
    main()
