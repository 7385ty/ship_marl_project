import numpy as np


class RolloutBufferSeq:
    """
    Rollout buffer for sequence-based MARL training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_seq = []
        self.share_obs_seq = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, obs_seq, share_obs_seq, actions, rewards, dones, log_probs, values):
        self.obs_seq.append(obs_seq.copy())
        self.share_obs_seq.append(share_obs_seq.copy())
        self.actions.append(actions.copy())
        self.rewards.append(rewards.copy())
        self.dones.append(dones.copy())
        self.log_probs.append(log_probs.copy())
        self.values.append(values.copy())

    def as_dict(self):
        return {
            "obs_seq": np.array(self.obs_seq, dtype=np.float32),              # (T, n_agents, seq_len, obs_dim)
            "share_obs_seq": np.array(self.share_obs_seq, dtype=np.float32),  # (T, n_agents, seq_len, state_dim)
            "actions": np.array(self.actions, dtype=np.float32),              # (T, n_agents, action_dim)
            "rewards": np.array(self.rewards, dtype=np.float32),              # (T, n_agents)
            "dones": np.array(self.dones, dtype=np.float32),                  # (T, n_agents)
            "log_probs": np.array(self.log_probs, dtype=np.float32),          # (T, n_agents, 1)
            "values": np.array(self.values, dtype=np.float32),                # (T, n_agents, 1)
        }

    def __len__(self):
        return len(self.obs_seq)