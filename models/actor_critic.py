import torch
import torch.nn as nn
from torch.distributions import Normal


# 连续动作actor，接近RL用法，输出动作分布参数
class Actor(nn.Module):
    """
    Continuous-action actor network.

    Input:
        obs: (batch_size, obs_dim)

    Output:
        action: (batch_size, action_dim)
        log_prob: (batch_size, action_dim) or summed version
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)

        # Learnable log std parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """
        Return action distribution parameters.
        """
        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))  # keep mean roughly in [-1, 1]
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist

    def sample_action(self, obs):
        """
        Sample action and compute log_prob.

        Returns
        -------
        action: torch.Tensor, shape (batch_size, action_dim)
        log_prob: torch.Tensor, shape (batch_size, 1)
        """
        dist = self.get_dist(obs)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate given actions.

        Returns
        -------
        log_prob: (batch_size, 1)
        entropy: (batch_size, 1)
        """
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class Critic(nn.Module):
    """
    Value network.

    Input:
        state/share_obs: (batch_size, state_dim)

    Output:
        value: (batch_size, 1)
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)