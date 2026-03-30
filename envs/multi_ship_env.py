import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class MultiShipEnv(gym.Env):
    """
    Gym-style multi-ship environment for MARL preparation.

    Current assumptions:
    - 2 ships
    - constant speed
    - 1D continuous action per ship (delta heading)
    - simplified collision avoidance reward
    - reward v4: stronger goal-seeking, stronger anti-spinning penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_ships=2,
        world_size=20.0,
        dt=1.0,
        max_steps=100,
        ship_speed=0.8,
        max_delta_heading_deg=10.0,
        collision_radius=1.0,
        goal_radius=1.0,
        position_noise=0.5,
        heading_noise_deg=5.0,
        danger_radius=4.0,
        danger_penalty_coef=0.1,
        seed=None,
    ):
        super().__init__()

        self.num_ships = num_ships
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.ship_speed = ship_speed
        self.max_delta_heading = np.deg2rad(max_delta_heading_deg)
        self.collision_radius = collision_radius
        self.goal_radius = goal_radius
        self.position_noise = position_noise
        self.heading_noise = np.deg2rad(heading_noise_deg)
        self.danger_radius = danger_radius
        self.danger_penalty_coef = danger_penalty_coef

        self.current_step = 0
        self.terminated = False
        self.truncated = False
        self.last_scenario = None

        self.ship_states = None
        self.goals = None
        self.trajectories = None

        self.rng = np.random.default_rng(seed)

        # Per-agent observation:
        # [own_x, own_y, own_heading, own_speed,
        #  goal_rel_x, goal_rel_y,
        #  other_rel_x, other_rel_y, other_rel_heading, other_rel_speed]
        self.obs_dim = 10
        self.action_dim = 1

        # Define Gym-style spaces
        obs_low = np.array(
            [
                -world_size, -world_size, -np.pi, 0.0,
                -2 * world_size, -2 * world_size,
                -2 * world_size, -2 * world_size,
                -2 * np.pi, -ship_speed
            ],
            dtype=np.float32
        )

        obs_high = np.array(
            [
                world_size, world_size, np.pi, ship_speed,
                2 * world_size, 2 * world_size,
                2 * world_size, 2 * world_size,
                2 * np.pi, ship_speed
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        """Reset random seed."""
        self.rng = np.random.default_rng(seed)

    def get_obs_dim(self):
        return self.obs_dim

    def get_action_dim(self):
        return self.action_dim

    def get_num_agents(self):
        return self.num_ships

    def get_state(self):
        """
        Return global state for centralized critic.
        State contains all ship states and goals.
        """
        state = np.concatenate([
            self.ship_states.flatten(),
            self.goals.flatten()
        ]).astype(np.float32)
        return state

    def get_state_dim(self):
        """
        Global state dimension.
        each ship state has 4 dims, each goal has 2 dims
        """
        return self.num_ships * 4 + self.num_ships * 2

    def get_avail_actions(self):
        """
        Placeholder for future action masking.
        For continuous action setting, return None.
        """
        return None

    def reset(self, scenario="head_on", seed=None, options=None):
        """
        Gymnasium-style reset.

        Returns
        -------
        obs : list[np.ndarray]
        info : dict
        """
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.terminated = False
        self.truncated = False
        self.last_scenario = scenario

        if self.num_ships != 2:
            raise ValueError("Current version supports num_ships=2 for simplicity.")

        if scenario == "head_on":
            base_states = np.array([
                [-10.0,  0.0, 0.0,    self.ship_speed],
                [ 10.0,  0.0, np.pi,  self.ship_speed],
            ], dtype=np.float32)

            base_goals = np.array([
                [ 10.0,  0.0],
                [-10.0,  0.0],
            ], dtype=np.float32)

        elif scenario == "crossing":
            base_states = np.array([
                [-10.0,  0.0, 0.0,           self.ship_speed],
                [  0.0, -10.0, np.pi / 2.0,  self.ship_speed],
            ], dtype=np.float32)

            base_goals = np.array([
                [10.0,  0.0],
                [ 0.0, 10.0],
            ], dtype=np.float32)

        elif scenario == "overtaking":
            base_states = np.array([
                [-10.0, 0.0, 0.0, self.ship_speed],
                [ -5.0, 0.0, 0.0, self.ship_speed],
            ], dtype=np.float32)

            base_goals = np.array([
                [10.0, 0.0],
                [15.0, 0.0],
            ], dtype=np.float32)

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        self.ship_states = base_states.copy()
        self.goals = base_goals.copy()

        # Add random perturbations
        for i in range(self.num_ships):
            self.ship_states[i, 0] += self.rng.uniform(-self.position_noise, self.position_noise)
            self.ship_states[i, 1] += self.rng.uniform(-self.position_noise, self.position_noise)
            self.ship_states[i, 2] += self.rng.uniform(-self.heading_noise, self.heading_noise)
            self.ship_states[i, 2] = self._wrap_angle(self.ship_states[i, 2])

        # Initialize trajectories
        self.trajectories = []
        for i in range(self.num_ships):
            x, y = self.ship_states[i, 0], self.ship_states[i, 1]
            self.trajectories.append([(x, y)])

        obs = self._get_obs()
        info = {
            "scenario": scenario,
            "current_step": self.current_step,
            "state_dim": self.get_state_dim(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        return obs, info

    def step(self, actions):
        """
        Gymnasium-style multi-agent step.

        Parameters
        ----------
        actions : np.ndarray or list
            shape can be (num_ships,) or (num_ships, 1)

        Returns
        -------
        obs : list[np.ndarray]
        rewards : list[float]
        terminated : bool
        truncated : bool
        info : dict
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is done. Call reset() before step().")

        actions = np.array(actions, dtype=np.float32)

        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions.squeeze(-1)

        actions = actions.reshape(self.num_ships)
        actions = np.clip(actions, -1.0, 1.0)

        prev_goal_distances = self._get_goal_distances()

        # Update heading
        for i in range(self.num_ships):
            delta_heading = actions[i] * self.max_delta_heading
            self.ship_states[i, 2] += delta_heading
            self.ship_states[i, 2] = self._wrap_angle(self.ship_states[i, 2])

        # Update position
        for i in range(self.num_ships):
            x, y, heading, speed = self.ship_states[i]
            self.ship_states[i, 0] = x + speed * np.cos(heading) * self.dt
            self.ship_states[i, 1] = y + speed * np.sin(heading) * self.dt

        # Record trajectories
        for i in range(self.num_ships):
            x, y = self.ship_states[i, 0], self.ship_states[i, 1]
            self.trajectories[i].append((x, y))

        self.current_step += 1

        collision = self._check_collision()
        reached_goals = self._check_goal()
        all_reached = all(reached_goals)
        timeout = self.current_step >= self.max_steps

        rewards = self._compute_rewards(
            actions=actions,
            prev_goal_distances=prev_goal_distances,
            collision=collision,
            reached_goals=reached_goals,
        )

        self.terminated = collision or all_reached
        self.truncated = timeout and (not self.terminated)

        obs = self._get_obs()
        info = {
            "scenario": self.last_scenario,
            "collision": collision,
            "reached_goals": reached_goals,
            "all_reached": all_reached,
            "timeout": timeout,
            "current_step": self.current_step,
            "ship_states": self.ship_states.copy(),
            "goal_distances": self._get_goal_distances(),
        }

        return obs, rewards, self.terminated, self.truncated, info

    def _get_obs(self):
        obs_list = []

        for i in range(self.num_ships):
            own_x, own_y, own_heading, own_speed = self.ship_states[i]
            goal_x, goal_y = self.goals[i]

            goal_rel_x = goal_x - own_x
            goal_rel_y = goal_y - own_y

            features = [
                own_x,
                own_y,
                own_heading,
                own_speed,
                goal_rel_x,
                goal_rel_y,
            ]

            for j in range(self.num_ships):
                if j == i:
                    continue

                other_x, other_y, other_heading, other_speed = self.ship_states[j]
                rel_x = other_x - own_x
                rel_y = other_y - own_y
                rel_heading = self._wrap_angle(other_heading - own_heading)
                rel_speed = other_speed - own_speed

                features.extend([rel_x, rel_y, rel_heading, rel_speed])

            obs_list.append(np.array(features, dtype=np.float32))

        return obs_list

    def _get_goal_distances(self):
        distances = []
        for i in range(self.num_ships):
            x, y = self.ship_states[i, 0], self.ship_states[i, 1]
            gx, gy = self.goals[i]
            dist = np.sqrt((gx - x) ** 2 + (gy - y) ** 2)
            distances.append(float(dist))
        return distances

    def _check_collision(self):
        for i in range(self.num_ships):
            for j in range(i + 1, self.num_ships):
                xi, yi = self.ship_states[i, 0], self.ship_states[i, 1]
                xj, yj = self.ship_states[j, 0], self.ship_states[j, 1]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < self.collision_radius:
                    return True
        return False

    def _check_goal(self):
        reached = []
        for i in range(self.num_ships):
            x, y = self.ship_states[i, 0], self.ship_states[i, 1]
            gx, gy = self.goals[i]
            dist = np.sqrt((gx - x) ** 2 + (gy - y) ** 2)
            reached.append(dist < self.goal_radius)
        return reached

    def _get_min_other_ship_distance(self, ship_idx):
        """
        Get the minimum distance from ship_idx to any other ship.
        """
        x_i, y_i = self.ship_states[ship_idx, 0], self.ship_states[ship_idx, 1]

        min_dist = np.inf
        for j in range(self.num_ships):
            if j == ship_idx:
                continue
            x_j, y_j = self.ship_states[j, 0], self.ship_states[j, 1]
            dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            if dist < min_dist:
                min_dist = dist

        return float(min_dist)

    def _compute_rewards(self, actions, prev_goal_distances, collision, reached_goals):
        """
        Reward function v4:
            Focus:
            - strongly encourage moving toward goal
            - heavily discourage endless turning / spinning
            - keep collision penalty high
            - reduce overly conservative danger shaping
        """
        rewards = []
        current_goal_distances = self._get_goal_distances()

        for i in range(self.num_ships):
            reward = 0.0

            # 1. Strong progress reward
            reward += 5.0 * (prev_goal_distances[i] - current_goal_distances[i])

            # 2. Strong goal reward
            if reached_goals[i]:
                reward += 100.0

            # 3. Collision penalty
            if collision:
                reward -= 50.0

            # 4. Stronger step penalty to reduce useless wandering
            reward -= 0.1

            # 5. Strong steering penalty to suppress persistent turning
            reward -= 0.2 * abs(actions[i])

            # 6. Reduced danger penalty
            min_dist = self._get_min_other_ship_distance(i)
            if min_dist < self.danger_radius:
                reward -= self.danger_penalty_coef * (self.danger_radius - min_dist)

            rewards.append(float(reward))

        return rewards

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        plt.figure(figsize=(8, 8))

        for i in range(self.num_ships):
            traj = np.array(self.trajectories[i])
            plt.plot(traj[:, 0], traj[:, 1], label=f"Ship {i} Trajectory")
            plt.scatter(traj[0, 0], traj[0, 1], marker="o", s=80, label=f"Ship {i} Start")
            plt.scatter(self.goals[i, 0], self.goals[i, 1], marker="*", s=150, label=f"Ship {i} Goal")
            plt.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=80, label=f"Ship {i} Current")

        plt.xlim(-self.world_size, self.world_size)
        plt.ylim(-self.world_size, self.world_size)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Multi-Ship Environment (Step {self.current_step}, Scenario: {self.last_scenario})")
        plt.grid(True)
        plt.legend()
        plt.axis("equal")
        plt.show()

    def close(self):
        plt.close("all")