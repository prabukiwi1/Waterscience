import matplotlib.pyplot as plt
import numpy as np

def plot_overflow_frequency(overflows, fig_id=6):
    plt.figure()
    plt.plot(overflows, label='Overflow Events', color='darkred')
    plt.title(f"Figure {fig_id}: Analysis of Overflow-event Frequency")
    plt.xlabel("Time Step")
    plt.ylabel("Overflow Events")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_overflow_frequency.png")
    plt.close()

def plot_overflow_volume(overflows, time_step_duration=1.0, fig_id=7):
    cumulative_volume = np.cumsum(overflows) * time_step_duration
    plt.figure()
    plt.plot(cumulative_volume, label='Overflow Volume', color='crimson')
    plt.title(f"Figure {fig_id}: Analysis of Overflow Volume")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Overflow Volume")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_overflow_volume.png")
    plt.close()

def plot_energy_consumption(energy_list, fig_id=8):
    plt.figure()
    plt.plot(energy_list, label='Energy Consumption (kWh)', color='orange')
    plt.title(f"Figure {fig_id}: Analysis of Total Energy Consumption")
    plt.xlabel("Time Step")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_energy_consumption.png")
    plt.close()

def plot_peak_flow_attenuation(baseline_flow, controlled_flow, fig_id=9):
    peak_baseline = np.max(baseline_flow)
    peak_controlled = np.max(controlled_flow)
    attenuation = (peak_baseline - peak_controlled) / peak_baseline * 100

    plt.figure()
    plt.plot(baseline_flow, label='Uncontrolled Flow', linestyle='--')
    plt.plot(controlled_flow, label='Controlled Flow (RL-DFC)', linestyle='-')
    plt.title(f"Figure {fig_id}: Analysis of Peak Flow Attenuation\n(Attenuation: {attenuation:.2f}%)")
    plt.xlabel("Time Step")
    plt.ylabel("Flow")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_peak_flow_attenuation.png")
    plt.close()

def plot_loading_factor(loading_data, fig_id=10):
    avg_load = np.mean(loading_data)
    plt.figure()
    plt.plot(loading_data, label=f'Loading Factor (Avg: {avg_load:.2f})', color='purple')
    plt.title(f"Figure {fig_id}: Analysis of Treatment-Plant Loading Factor")
    plt.xlabel("Time Step")
    plt.ylabel("Loading Factor")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_loading_factor.png")
    plt.close()

def plot_actuation_frequency(actuation_log, fig_id=11):
    changes = np.diff(actuation_log, axis=0)
    freq = np.sum(np.abs(changes) > 0.01, axis=0)

    plt.figure()
    plt.bar(range(len(freq)), freq, color='teal')
    plt.title(f"Figure {fig_id}: Analysis of Actuation Frequency")
    plt.xlabel("Actuator Index")
    plt.ylabel("Switch Count")
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_actuation_frequency.png")
    plt.close()

def plot_convergence(rewards, fig_id=12):
    plt.figure()
    plt.plot(rewards, label='Episode Reward', color='green')
    plt.title(f"Figure {fig_id}: Analysis of Agent Convergence Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figure_{fig_id}_convergence_time.png")
    plt.close()

def plot_robustness_comparison(normal_overflow, extreme_overflow, fig_id=13):
    baseline = np.mean(normal_overflow)
    stressed = np.mean(extreme_overflow)
    drop = (stressed - baseline) / baseline * 100

    labels = ['Normal', 'Extreme']
    values = [baseline, stressed]

    plt.figure()
    plt.bar(labels, values, color=['blue', 'red'])
    plt.title(f"Figure {fig_id}: Analysis of Robustness Under Extreme Rainfall\n(Overflow Increase: {drop:.2f}%)")
    plt.ylabel("Average Overflow Volume")
    plt.grid(True, axis='y')
    plt.savefig(f"figure_{fig_id}_robustness_extreme_rainfall.png")
    plt.close()
