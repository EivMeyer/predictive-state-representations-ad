import numpy as np
import matplotlib.pyplot as plt
from plotting_setup import setup_plotting

DEG_TO_RAD = np.pi / 180

# Parameters for Monte Carlo Simulation
time_horizon = 2.0            # Time horizon (s)
delta_t_values = [0.001, 0.04, 0.1]  # Time step sizes (s)
num_simulations = 20        # Increased for better statistical significance
lwb = 2.5                    # Wheelbase length (m)
v_nom = 5.0                  # Nominal velocity (m/s)
Psi_nom = DEG_TO_RAD*0       # Nominal heading angle
u1_nom = 0.0                 # Nominal steering velocity (rad/s)
u2_nom = 0.0                 # Nominal acceleration (m/s^2)
sigma_eta1 = 0.15             # Std dev of steering velocity noise
sigma_eta2 = 0.0             # Std dev of acceleration noise

np.random.seed(42)  # For reproducibility

setup_plotting()

def simulate_up_to_time_horizon(time_horizon, delta_t, num_simulations, lwb, v_nom, Psi_nom, 
                              u1_nom, u2_nom, sigma_eta1, sigma_eta2, stochastic_mode="fully"):
    """Simulates the KS model up to a given time horizon."""
    num_steps = int(time_horizon / delta_t)
    x = np.zeros((num_simulations, 5, num_steps + 1))
    x[:, 2, 0] = Psi_nom
    x[:, 3, 0] = v_nom

    if stochastic_mode == "constant":
        eta1 = np.random.normal(0, sigma_eta1, size=num_simulations)
        eta2 = np.random.normal(0, sigma_eta2, size=num_simulations)
        u1 = u1_nom + eta1
        u2 = u2_nom + eta2
    else:
        u1 = u1_nom
        u2 = u2_nom

    for k in range(num_steps):
        if stochastic_mode == "fully":
            eta1 = np.random.normal(0, sigma_eta1, size=num_simulations)
            eta2 = np.random.normal(0, sigma_eta2, size=num_simulations)
            u1 = u1_nom + eta1
            u2 = u2_nom + eta2

        x[:, 0, k + 1] = x[:, 0, k] + x[:, 3, k] * np.cos(x[:, 4, k]) * delta_t
        x[:, 1, k + 1] = x[:, 1, k] + x[:, 3, k] * np.sin(x[:, 4, k]) * delta_t
        x[:, 2, k + 1] = x[:, 2, k] + u1 * delta_t
        x[:, 3, k + 1] = x[:, 3, k] + u2 * delta_t
        x[:, 4, k + 1] = x[:, 4, k] + (x[:, 3, k] / lwb) * x[:, 2, k] * delta_t

    return x

if __name__ == "__main__":
    # Plot results as subfigures
    fig, axes = plt.subplots(len(delta_t_values), 1, figsize=(10, 12), tight_layout=True)

    for idx, delta_t in enumerate(delta_t_values):
        num_steps = int(time_horizon / delta_t)
        
        # Simulate trajectories
        fully_stochastic = simulate_up_to_time_horizon(
            time_horizon, delta_t, num_simulations, lwb, v_nom, Psi_nom,
            u1_nom, u2_nom, sigma_eta1, sigma_eta2, stochastic_mode="fully"
        )
        
        constant_stochastic = simulate_up_to_time_horizon(
            time_horizon, delta_t, num_simulations, lwb, v_nom, Psi_nom,
            u1_nom, u2_nom, sigma_eta1, sigma_eta2, stochastic_mode="constant"
        )
        
        # Calculate final MAD values
        # Empirical
        deviations_fully = np.abs(fully_stochastic[:, :2, -1])  # Only last time step
        norm_deviation_fully = np.linalg.norm(deviations_fully, axis=1)
        empirical_mad_fully = np.mean(norm_deviation_fully)
        
        deviations_constant = np.abs(constant_stochastic[:, :2, -1])  # Only last time step
        norm_deviation_constant = np.linalg.norm(deviations_constant, axis=1)
        empirical_mad_constant = np.mean(norm_deviation_constant)
        
        # Plot trajectories of the simulations
        for i in range(num_simulations): 
            axes[idx].plot(
                fully_stochastic[i, 0, :], fully_stochastic[i, 1, :],
                alpha=0.6, label="Fully Stochastic" if i == 0 else "", color="blue"
            )
            axes[idx].plot(
                constant_stochastic[i, 0, :], constant_stochastic[i, 1, :],
                alpha=0.6, label="Constant Stochastic" if i == 0 else "", color="orange"
            )
        
        axes[idx].set_title(f"Delta_t = {delta_t}s")
        axes[idx].set_xlabel("X Position (m)")
        axes[idx].set_ylabel("Y Position (m)")
        axes[idx].axhline(0, color='gray', linestyle='--', linewidth=0.5)
        axes[idx].axvline(0, color='gray', linestyle='--', linewidth=0.5)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)

    plt.show()
