from train import train_agent
from env import AquaFlowEnv
from evaluate import plot_overflow_frequency, plot_energy_consumption, plot_convergence

def main():
    model = train_agent()

    env = AquaFlowEnv()
    state = env.reset()
    rewards = []
    overflows = []
    energy = []

    done = False
    total_reward = 0
    while not done:
        state_tensor = state.reshape(1, -1)
        state_tensor = state_tensor.astype('float32')

        import torch
        with torch.no_grad():
            state_tensor = torch.tensor(state_tensor)
            logits, _ = model(state_tensor)
            probs = torch.softmax(logits, dim=-1).numpy().squeeze()

        next_state, reward, done, info = env.step(probs)
        rewards.append(reward)
        overflows.append(info["overflow"])
        energy.append(info["energy"])

        state = next_state
        total_reward += reward

    print(f"Evaluation Reward: {total_reward:.2f}")

    plot_overflow_frequency(overflows)
    plot_overflow_volume(overflows)
    plot_energy_consumption(energy)
    plot_convergence(rewards)

if __name__ == "__main__":
    main()
