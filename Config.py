# config.py

ENV_CONFIG = {
    "num_nodes": 10,
    "time_steps": 144,  # e.g., one day with 10-minute intervals
    "capacity_per_node": 100.0,
    "weather_variation": 0.2,
    "sensor_noise_std": 0.05,
    "actuator_power_coeff": 0.3,
    "energy_penalty": 0.01,
}

RL_CONFIG = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_epochs": 10,
    "batch_size": 64,
    "num_steps": 2048,
    "total_timesteps": 50000,
}
