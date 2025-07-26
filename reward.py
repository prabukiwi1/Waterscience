import torch

def overflow_loss(inflow, outflow, capacity):
    overflow = torch.clamp(inflow - outflow - capacity, min=0)
    return torch.sum(overflow)

def energy_loss(actuation, power_coeff=0.3):
    return torch.sum(actuation) * power_coeff

def policy_gradient_loss(log_probs, advantages):
    return -torch.mean(log_probs * advantages)

def advantage_weighted_update(log_probs, advantages, penalty_coef=0.01, actuation_change=None):
    policy_loss = -torch.mean(log_probs * advantages)
    if actuation_change is not None:
        penalty = penalty_coef * torch.mean(torch.abs(actuation_change))
        return policy_loss + penalty
    return policy_loss

def state_encoding(raw_input, mean, std, weight, bias):
    z = (raw_input - mean) / std
    return torch.matmul(z, weight) + bias

def hybrid_reward(overflow, energy, flow_var, coeff_over=0.5, coeff_energy=0.3, coeff_var=0.2):
    return -(coeff_over * overflow + coeff_energy * energy + coeff_var * flow_var)

def entropy_regularized_loss(log_probs, advantages, entropy, beta=0.01):
    base_loss = -torch.mean(log_probs * advantages)
    entropy_bonus = -beta * torch.mean(entropy)
    return base_loss + entropy_bonus

def update_baseline(reward, baseline, alpha=0.05):
    return baseline + alpha * (reward - baseline)
