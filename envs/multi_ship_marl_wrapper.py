import numpy as np
from envs.multi_ship_env import MultiShipEnv


# 这是一个“外包装器”，不改变原本环境逻辑，只负责：统一输出格式，方便 MARL 使用
class MultiShipMARLWrapper:
    """
    MARL wrapper for MultiShipEnv.

    This wrapper converts the Gym-style environment outputs into a
    MAPPO-friendly multi-agent format.

    Main conventions:
    - obs shape: (n_agents, obs_dim)
    - share_obs/state shape: (n_agents, state_dim)
    - rewards shape: (n_agents,)
    - dones shape: (n_agents,)
    """

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}

        self.env = MultiShipEnv(**env_config)

        self.n_agents = self.env.get_num_agents()
        self.obs_dim = self.env.get_obs_dim()
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()

    def reset(self, scenario="head_on", seed=None):
        """
        Reset environment.

        Returns
        -------
        obs : np.ndarray, shape (n_agents, obs_dim)
        share_obs : np.ndarray, shape (n_agents, state_dim)
        info : dict
        """
        obs_list, info = self.env.reset(scenario=scenario, seed=seed)

        obs = np.array(obs_list, dtype=np.float32)  # (n_agents, obs_dim)
        state = self.env.get_state().astype(np.float32)  # (state_dim,)

        # In many MAPPO implementations, each agent gets the same global state
        share_obs = np.repeat(state[None, :], self.n_agents, axis=0)  # (n_agents, state_dim)

        return obs, share_obs, info

    def step(self, actions):
        """
        Step environment.

        Parameters
        ----------
        actions : np.ndarray
            shape can be (n_agents,) or (n_agents, action_dim)

        Returns
        -------
        obs : np.ndarray, shape (n_agents, obs_dim)
        share_obs : np.ndarray, shape (n_agents, state_dim)
        rewards : np.ndarray, shape (n_agents,)
        dones : np.ndarray, shape (n_agents,)
        info : dict
        """
        obs_list, rewards_list, terminated, truncated, info = self.env.step(actions)

        obs = np.array(obs_list, dtype=np.float32)  # (n_agents, obs_dim)
        rewards = np.array(rewards_list, dtype=np.float32)  # (n_agents,)

        done = terminated or truncated
        dones = np.array([done] * self.n_agents, dtype=bool)  # (n_agents,)

        state = self.env.get_state().astype(np.float32)
        share_obs = np.repeat(state[None, :], self.n_agents, axis=0)

        return obs, share_obs, rewards, dones, info

    def get_env_info(self):
        """
        Return env information in a MARL-friendly dictionary.
        """
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

    def get_num_agents(self):
        return self.n_agents

    def get_obs_dim(self):
        return self.obs_dim

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()