# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, action_dim)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_pi(x)
        value = self.fc_v(x)
        return policy_logits, value

    def act(self, state):
        logits, value = self.forward(state)
        prob = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
