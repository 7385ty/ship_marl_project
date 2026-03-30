import numpy as np
from collections import deque
from envs.multi_ship_marl_wrapper import MultiShipMARLWrapper


class MultiShipSeqWrapper:
    """
    Sequence wrapper for MARL environment.

    It stacks the last seq_len observations/states for each agent.
    """

    def __init__(self, env_config=None, seq_len=5):
        self.env = MultiShipMARLWrapper(env_config=env_config)
        self.seq_len = seq_len

        self.n_agents = self.env.get_num_agents()
        self.obs_dim = self.env.get_obs_dim()
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()

        self.obs_history = None
        self.share_obs_history = None

    def _init_history(self, obs, share_obs):
        self.obs_history = [deque(maxlen=self.seq_len) for _ in range(self.n_agents)]
        self.share_obs_history = [deque(maxlen=self.seq_len) for _ in range(self.n_agents)]

        for agent_id in range(self.n_agents):
            for _ in range(self.seq_len):
                self.obs_history[agent_id].append(obs[agent_id].copy())
                self.share_obs_history[agent_id].append(share_obs[agent_id].copy())

    def _get_stacked_obs(self):
        obs_seq = []
        share_obs_seq = []

        for agent_id in range(self.n_agents):
            obs_seq.append(np.array(self.obs_history[agent_id], dtype=np.float32))
            share_obs_seq.append(np.array(self.share_obs_history[agent_id], dtype=np.float32))

        obs_seq = np.array(obs_seq, dtype=np.float32)              # (n_agents, seq_len, obs_dim)
        share_obs_seq = np.array(share_obs_seq, dtype=np.float32)  # (n_agents, seq_len, state_dim)

        return obs_seq, share_obs_seq

    def reset(self, scenario="head_on", seed=None):
        obs, share_obs, info = self.env.reset(scenario=scenario, seed=seed)
        self._init_history(obs, share_obs)
        return self._get_stacked_obs()[0], self._get_stacked_obs()[1], info

    def step(self, actions):
        obs, share_obs, rewards, dones, info = self.env.step(actions)

        for agent_id in range(self.n_agents):
            self.obs_history[agent_id].append(obs[agent_id].copy())
            self.share_obs_history[agent_id].append(share_obs[agent_id].copy())

        obs_seq, share_obs_seq = self._get_stacked_obs()
        return obs_seq, share_obs_seq, rewards, dones, info

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
        }

    def get_num_agents(self):
        return self.n_agents

    def get_obs_dim(self):
        return self.obs_dim

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_seq_len(self):
        return self.seq_len

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()