import numpy as np


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def simple_rule_action(obs, safe_distance=6.0, goal_gain=0.5, avoid_gain=0.8):
    """
    A simple reactive rule-based policy for one ship.

    Observation format:
    [own_x, own_y, own_heading, own_speed,
     goal_rel_x, goal_rel_y,
     other_rel_x, other_rel_y, other_rel_heading, other_rel_speed]

    Returns
    -------
    action : np.ndarray, shape (1,)
        Normalized action in [-1, 1]
    """
    own_x, own_y, own_heading, own_speed = obs[0], obs[1], obs[2], obs[3]
    goal_rel_x, goal_rel_y = obs[4], obs[5]
    other_rel_x, other_rel_y = obs[6], obs[7]

    # 1. Goal-following component
    goal_angle = np.arctan2(goal_rel_y, goal_rel_x)
    heading_error = wrap_angle(goal_angle - own_heading)

    # Normalize heading error approximately into [-1, 1]
    goal_action = goal_gain * (heading_error / np.pi)

    # 2. Collision-avoidance component
    other_distance = np.sqrt(other_rel_x ** 2 + other_rel_y ** 2)

    avoid_action = 0.0

    if other_distance < safe_distance:
        # turn right when too close
        # negative action means turning right in our current convention
        proximity = (safe_distance - other_distance) / safe_distance
        avoid_action = -avoid_gain * proximity

    # 3. Combine
    action = goal_action + avoid_action

    # Clip to [-1, 1]
    action = np.clip(action, -1.0, 1.0)

    return np.array([action], dtype=np.float32)


def simple_rule_policy(obs_list, safe_distance=6.0, goal_gain=0.5, avoid_gain=0.8):
    """
    Generate actions for all ships.

    Parameters
    ----------
    obs_list : list[np.ndarray]
        Observations for all ships.

    Returns
    -------
    actions : np.ndarray of shape (num_ships, 1)
    """
    actions = []
    for obs in obs_list:
        action = simple_rule_action(
            obs,
            safe_distance=safe_distance,
            goal_gain=goal_gain,
            avoid_gain=avoid_gain
        )
        actions.append(action)

    return np.array(actions, dtype=np.float32)