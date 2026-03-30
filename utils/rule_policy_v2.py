import numpy as np


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_relative_bearing(other_rel_x, other_rel_y):
    """
    Compute relative bearing angle of the other ship in ego-centric frame.

    Since other_rel_x and other_rel_y are already relative to ego ship position,
    the bearing can be computed directly.

    Returns
    -------
    bearing : float
        In radians, within [-pi, pi]
        0 means directly ahead
        positive means left side
        negative means right side
    """
    return np.arctan2(other_rel_y, other_rel_x)


def simple_rule_action_v2(
    obs,
    safe_distance=6.0,
    danger_distance=8.0,
    front_sector_deg=120.0,
    goal_gain=0.8,
    avoid_gain=1.2,
):
    """
    Improved heuristic rule-based policy for one ship.

    Observation format:
    [own_x, own_y, own_heading, own_speed,
     goal_rel_x, goal_rel_y,
     other_rel_x, other_rel_y, other_rel_heading, other_rel_speed]

    Design principles:
    1. Follow goal direction in normal cases
    2. Trigger avoidance only when the other ship is both:
       - within danger distance
       - inside front danger sector
    3. Use stronger right-turn avoidance when risk is high
    4. Once danger is relieved, naturally return to goal tracking

    Returns
    -------
    action : np.ndarray, shape (1,)
        Normalized action in [-1, 1]
    """
    own_x, own_y, own_heading, own_speed = obs[0], obs[1], obs[2], obs[3]
    goal_rel_x, goal_rel_y = obs[4], obs[5]
    other_rel_x, other_rel_y = obs[6], obs[7]
    other_rel_heading, other_rel_speed = obs[8], obs[9]

    # --------------------------------------------------
    # 1. Goal-tracking component
    # --------------------------------------------------
    goal_angle_global = np.arctan2(goal_rel_y, goal_rel_x)
    heading_error = wrap_angle(goal_angle_global - own_heading)

    # Normalize approximately into [-1, 1]
    goal_action = goal_gain * (heading_error / np.pi)

    # --------------------------------------------------
    # 2. Relative-bearing-aware risk assessment
    # --------------------------------------------------
    other_distance = np.sqrt(other_rel_x ** 2 + other_rel_y ** 2)
    relative_bearing = compute_relative_bearing(other_rel_x, other_rel_y)

    # Convert front sector half-angle to radians
    front_half_angle = np.deg2rad(front_sector_deg / 2.0)

    # Check whether the other ship is in front sector
    in_front_sector = abs(relative_bearing) <= front_half_angle

    # Risk level based on distance
    danger_level = 0.0
    if other_distance < danger_distance and in_front_sector:
        danger_level = (danger_distance - other_distance) / danger_distance
        danger_level = np.clip(danger_level, 0.0, 1.0)

    # --------------------------------------------------
    # 3. Avoidance component
    # --------------------------------------------------
    avoid_action = 0.0

    if danger_level > 0.0:
        # Default: turn right
        # Negative action means turning right in current environment convention
        avoid_action = -avoid_gain * danger_level

        # If the other ship is on the left-front side, still right-turn
        # If on right-front side, also right-turn but maybe slightly stronger
        if relative_bearing < 0:  # other ship on right side
            avoid_action *= 1.1

    # --------------------------------------------------
    # 4. Blend goal-tracking and avoidance
    # --------------------------------------------------
    # When danger is high, rely more on avoidance
    # When danger is low, rely more on goal tracking
    action = (1.0 - danger_level) * goal_action + avoid_action

    # Clip into legal range
    action = np.clip(action, -1.0, 1.0)

    return np.array([action], dtype=np.float32)


def simple_rule_policy_v2(
    obs_list,
    safe_distance=6.0,
    danger_distance=8.0,
    front_sector_deg=120.0,
    goal_gain=0.8,
    avoid_gain=1.2,
):
    """
    Generate actions for all ships using improved rule policy v2.

    Returns
    -------
    actions : np.ndarray, shape (num_ships, 1)
    """
    actions = []

    for obs in obs_list:
        action = simple_rule_action_v2(
            obs,
            safe_distance=safe_distance,
            danger_distance=danger_distance,
            front_sector_deg=front_sector_deg,
            goal_gain=goal_gain,
            avoid_gain=avoid_gain,
        )
        actions.append(action)

    return np.array(actions, dtype=np.float32)