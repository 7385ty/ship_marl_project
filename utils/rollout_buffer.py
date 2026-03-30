import numpy as np


class RolloutBuffer:
    """
    Minimal rollout buffer for MARL training skeleton.

    This buffer stores data step-by-step.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.share_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, obs, share_obs, actions, rewards, dones, log_probs, values):
        """
        Store one timestep data.

        Parameters
        ----------
        obs : np.ndarray, shape (n_agents, obs_dim)
        share_obs : np.ndarray, shape (n_agents, state_dim)
        actions : np.ndarray, shape (n_agents, action_dim)
        rewards : np.ndarray, shape (n_agents,)
        dones : np.ndarray, shape (n_agents,)
        log_probs : np.ndarray, shape (n_agents, 1)
        values : np.ndarray, shape (n_agents, 1)
        """
        self.obs.append(obs.copy())
        self.share_obs.append(share_obs.copy())
        self.actions.append(actions.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        self.log_probs.append(log_probs.copy())
        self.values.append(values.copy())

    def as_dict(self):
        """
        Convert stored data into numpy arrays.
        """
        data = {
            "obs": np.array(self.obs, dtype=np.float32),
            "share_obs": np.array(self.share_obs, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
        }
        return data

    def __len__(self):
        return len(self.obs)