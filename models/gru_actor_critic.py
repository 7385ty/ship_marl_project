import torch
import torch.nn as nn
from torch.distributions import Normal


class GRUActor(nn.Module):
    """
    GRU-based actor for sequence observations.
    Input shape: (batch_size, seq_len, obs_dim)
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64, gru_hidden_dim=64):
        super().__init__()

        self.input_fc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(gru_hidden_dim, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs_seq):
        # obs_seq: (B, T, obs_dim)
        x = torch.relu(self.input_fc(obs_seq))      # (B, T, hidden_dim)
        gru_out, h_n = self.gru(x)                  # h_n: (1, B, gru_hidden_dim)
        h = h_n.squeeze(0)                          # (B, gru_hidden_dim)

        mean = torch.tanh(self.output_fc(h))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_dist(self, obs_seq):
        mean, std = self.forward(obs_seq)
        return Normal(mean, std)

    def sample_action(self, obs_seq):
        dist = self.get_dist(obs_seq)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, obs_seq, actions):
        dist = self.get_dist(obs_seq)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class GRUCritic(nn.Module):
    """
    GRU-based critic for sequence state/share_obs.
    Input shape: (batch_size, seq_len, state_dim)
    Output: (batch_size, 1)
    """

    def __init__(self, state_dim, hidden_dim=64, gru_hidden_dim=64):
        super().__init__()

        self.input_fc = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(gru_hidden_dim, 1)

    def forward(self, state_seq):
        x = torch.relu(self.input_fc(state_seq))
        gru_out, h_n = self.gru(x)
        h = h_n.squeeze(0)
        value = self.output_fc(h)
        return value