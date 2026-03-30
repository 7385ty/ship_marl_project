import torch
import torch.nn as nn



# 这是你第一个真正的“神经网络策略”。
class MLPPolicy(nn.Module):
    """
    Minimal MLP policy network for continuous action output.

    Input:
        obs shape: (batch_size, obs_dim)

    Output:
        action shape: (batch_size, action_dim)
        action range: [-1, 1] via tanh
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        """
        Parameters
        ----------
        obs : torch.Tensor
            shape = (batch_size, obs_dim)

        Returns
        -------
        action : torch.Tensor
            shape = (batch_size, action_dim)
            range = [-1, 1]
        """
        return self.net(obs)