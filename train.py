import torch
import torch.optim as optim
from env import AquaFlowEnv
from model import ActorCritic
from reward import policy_gradient_loss
from config import RL_CONFIG

def train_agent():
    env = AquaFlowEnv()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(input_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=RL_CONFIG["learning_rate"])

    all_rewards = []
    for episode in range(200):  # can increase based on `total_timesteps`
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        log_probs = []
        values = []
        rewards = []
        entropies = []

        done = False
        total_reward = 0

        while not done:
            action_logits, value = model(state)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            action_env = action_probs.detach().numpy()  # continuous actions for env
            next_state, reward, done, _ = env.step(action_env)

            state = torch.tensor(next_state, dtype=torch.float32)

            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))

            total_reward += reward

        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + RL_CONFIG["gamma"] * R
            returns.insert(0, R)

        returns = torch.cat(returns)
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantages = returns - values.detach()

        loss = policy_gradient_loss(log_probs, advantages) - 0.01 * entropies.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    return model
