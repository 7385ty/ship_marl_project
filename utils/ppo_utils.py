import numpy as np
import torch


def compute_returns(rewards, dones, gamma=0.99):
    """
    Compute simple discounted returns.

    Parameters
    ----------
    rewards : np.ndarray
        shape (T, n_agents)
    dones : np.ndarray
        shape (T, n_agents), values are 0/1 or False/True
    gamma : float

    Returns
    -------
    returns : np.ndarray
        shape (T, n_agents)
    """
    T, n_agents = rewards.shape
    returns = np.zeros_like(rewards, dtype=np.float32)

    running_return = np.zeros(n_agents, dtype=np.float32)

    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t].astype(np.float32))
        returns[t] = running_return

    return returns


def compute_advantages(returns, values):
    """
    Compute simple advantages:
        advantage = return - value

    Parameters
    ----------
    returns : np.ndarray
        shape (T, n_agents)
    values : np.ndarray
        shape (T, n_agents, 1) or (T, n_agents)

    Returns
    -------
    advantages : np.ndarray
        shape (T, n_agents)
    """
    if values.ndim == 3:
        values = values.squeeze(-1)

    advantages = returns - values
    return advantages.astype(np.float32)


def normalize_advantages(advantages, eps=1e-8):
    """
    Normalize advantages for stability.
    """
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    return (advantages - adv_mean) / (adv_std + eps)


def flatten_marl_batch(obs, share_obs, actions, log_probs, values, returns, advantages):
    """
    Flatten MARL rollout data from:
        (T, n_agents, dim)
    to:
        (T*n_agents, dim)

    Returns
    -------
    batch : dict of torch-ready numpy arrays
    """
    T, n_agents, obs_dim = obs.shape
    _, _, state_dim = share_obs.shape
    _, _, action_dim = actions.shape

    batch = {
        "obs": obs.reshape(T * n_agents, obs_dim),
        "share_obs": share_obs.reshape(T * n_agents, state_dim),
        "actions": actions.reshape(T * n_agents, action_dim),
        "old_log_probs": log_probs.reshape(T * n_agents, 1),
        "values": values.reshape(T * n_agents, 1),
        "returns": returns.reshape(T * n_agents, 1),
        "advantages": advantages.reshape(T * n_agents, 1),
    }

    return batch