import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
import random


class MuJoCoHRCEnv(gym.Env):
    def __init__(self, is_stochastic=True):
        super(MuJoCoHRCEnv, self).__init__()

        curr_dir = os.path.dirname(__file__)
        xml_path = os.path.join(curr_dir, 'workstation.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Observation: buffer level, robot velocity, human velocity,
        # normalized pause countdown, and normalized fatigue. Fatigue is
        # included explicitly so the agent can anticipate the human's
        # gradual slowdown rather than treating it as unexplained noise
        # in human_v.
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(5,), dtype=np.float32)

        # Action space: 0 = Protective, 1 = Collaborative, 2 = Production Sprint
        self.action_space = spaces.Discrete(3)

        self.is_stochastic = is_stochastic
        self.max_steps = 100

        # --- State Tracking Properties ---
        self.learning_decay = 1.0
        self.robot_speed = 1.0
        self.initial_human_speed = 1.0
        self.human_speed = 1.0

        self.noise_std = 0.1
        self.fatigue_rate = 0.005
        self.fatigue_max = 0.5  # used for normalizing fatigue in _get_obs()
        self.fatigue = 0.0
        self.buffer = 5.0
        self.distraction_timer = 0

        self.reset()

    def set_learning_decay(self, episode, total_episodes):
        self.learning_decay = max(0.1, 1.0 - (episode / total_episodes))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.buffer = 5.0
        self.fatigue = 0.0
        self.distraction_timer = 0
        return self._get_obs(), {}

    def _get_obs(self):
        elbow_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
        human_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "human_x")

        # Continuous countdown: how long the current pause has left.
        human_pause_countdown = self.distraction_timer / 25.0

        # Normalized fatigue. Observing this directly (rather than letting
        # the agent infer it from a noisier, slower human_v) is what lets
        # the policy anticipate the slowdown instead of reacting to it.
        normalized_fatigue = self.fatigue / self.fatigue_max

        return np.array([
            self.buffer,
            self.data.qvel[elbow_id],
            self.data.qvel[human_id],
            human_pause_countdown,
            normalized_fatigue
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1

        # Action -> robot speed setpoint, loosely modeled on ISO 15066
        # speed/separation guidance for collaborative robots.
        if action == 0:
            # Protective Mode (Human paused / near boundary)
            self.robot_speed = 0.1
        elif action == 1:
            # Collaborative Mode (Nominal balanced pace)
            self.robot_speed = 1.0
        elif action == 2:
            # Production Sprint Mode (High-throughput execution)
            self.robot_speed = 2.0

        # Human Stochasticity Logic
        if self.is_stochastic:
            if self.distraction_timer > 0:
                self.distraction_timer -= 1
                self.human_speed = 0.0
            else:
                if random.random() < 0.05:
                    self.distraction_timer = random.randint(10, 25)
                    self.human_speed = 0.0
                else:
                    noise = np.random.normal(0, self.noise_std)
                    self.human_speed = (
                        self.initial_human_speed - self.fatigue) + noise
                    self.human_speed = max(0.1, self.human_speed)

            self.fatigue += self.fatigue_rate
        else:
            self.human_speed = self.initial_human_speed

        # Safety override near the buffer boundaries: force a deceleration
        # before the upper limit and a catch-up acceleration before the
        # lower limit, regardless of the action chosen.
        if self.buffer >= 8.5 and self.robot_speed > 0.1:
            effective_robot_speed = 0.1  # force safety deceleration to prevent upper crash
        elif self.buffer <= 1.5 and self.robot_speed < 1.0:
            effective_robot_speed = 1.0  # force catch-up acceleration to prevent lower crash
        else:
            effective_robot_speed = self.robot_speed

        # Apply Commands to MuJoCo and step physics
        self.data.ctrl[0] = effective_robot_speed
        self.data.ctrl[1] = self.human_speed

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        elbow_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
        human_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "human_x")

        actual_robot_v = abs(self.data.qvel[elbow_id])
        actual_human_v = abs(self.data.qvel[human_id])

        # Update Buffer Dynamics
        vel_diff = self.data.qvel[elbow_id] - self.data.qvel[human_id]
        self.buffer = np.clip(self.buffer + vel_diff, 0, 10)

        # =========================================================================
        # REWARD STRUCTURE
        # =========================================================================

        # 1. Throughput Reward
        throughput_reward = (actual_robot_v * 1.0) + (actual_human_v * 0.5)

        # 2. Buffer Position Penalty
        # Quartic rather than quadratic: stays close to zero near the 4-6
        # sweet spot so normal operation isn't penalized, but escalates
        # sharply past ~7 so the cost of approaching a boundary clearly
        # outweighs the throughput gained from a high-speed action (e.g.
        # Sprint) well before the robot is actually at risk of crashing.
        buffer_penalty = 0.0125 * ((self.buffer - 5.0) ** 4)

        # 2b. Buffer-Velocity (Rate of Approach) Penalty
        # A position-only penalty can't distinguish "sitting near a wall"
        # from "racing toward a wall", which is the actual unsafe
        # behavior. This penalizes vel_diff (how fast the buffer is
        # currently moving) when it's heading toward whichever boundary
        # is nearer, scaled up as the remaining room shrinks. It's zero
        # when the buffer is moving away from a boundary, so it never
        # discourages recovery.
        if vel_diff > 0:
            danger_room = max(0.5, 10.0 - self.buffer)
            buffer_velocity_penalty = 0.8 * (vel_diff ** 2) / danger_room
        elif vel_diff < 0:
            danger_room = max(0.5, self.buffer - 0.0)
            buffer_velocity_penalty = 0.8 * (vel_diff ** 2) / danger_room
        else:
            buffer_velocity_penalty = 0.0

        # 3. The "Sweet Spot" Reward (The Carrot)
        # Rewards keeping the buffer centered, but scaled by how fast the
        # robot is actually moving rather than a flat bonus. A flat bonus
        # let the agent park in the center band at near-zero velocity and
        # still score well, since the bonus alone outweighed the small
        # throughput loss from doing nothing. Scaling by velocity means
        # the agent has to actually be productive to earn the full reward.
        if 4.0 <= self.buffer <= 6.0:
            velocity_factor = min(1.0, actual_robot_v / 0.5)
            sweet_spot_reward = 0.5 + 1.0 * velocity_factor
        else:
            sweet_spot_reward = 0.0

        # 4. Synchronization Penalty
        # Gated off during a distraction. human_speed is forced to 0.0 by
        # exogenous randomness while paused, so penalizing the robot for not
        # matching it taught the agent to punish itself for something its
        # action couldn't affect.
        if self.distraction_timer == 0:
            sync_penalty = abs(actual_robot_v - actual_human_v) * 0.2
        else:
            sync_penalty = 0.0

        # 5. Conditional Idleness Penalty
        # Threshold set above the robot's settled velocity under the
        # Protective action (~0.12, due to actuator/damping dynamics —
        # it never truly reaches 0), so genuinely idling doesn't slip
        # under the cutoff unpenalized.
        if self.distraction_timer == 0 and actual_robot_v < 0.3:
            lazy_penalty = 1.5
        else:
            lazy_penalty = 0.0

        # 6. Boundary Proximity Penalty
        boundary_penalty = 2.0 if (
            self.buffer >= 9.5 or self.buffer <= 0.5) else 0.0

        # Total reward: throughput and the sweet-spot bonus as incentives,
        # offset by the buffer/safety/idleness penalties above.
        reward = (throughput_reward + sweet_spot_reward) - \
            (buffer_penalty + buffer_velocity_penalty +
             sync_penalty + lazy_penalty + boundary_penalty)
        # =========================================================================

        info = {
            "throughput": actual_human_v,
            "idle": 1.0 if actual_robot_v < 0.3 else 0.0,
            "buffer": self.buffer,
            # Per-component breakdown, useful for debugging reward
            # attribution without re-deriving it from the total by hand.
            "reward_components": {
                "throughput_reward": float(throughput_reward),
                "buffer_penalty": float(buffer_penalty),
                "buffer_velocity_penalty": float(buffer_velocity_penalty),
                "sweet_spot_reward": float(sweet_spot_reward),
                "sync_penalty": float(sync_penalty),
                "lazy_penalty": float(lazy_penalty),
                "boundary_penalty": float(boundary_penalty),
            }
        }

        terminated = bool(self.buffer >= 10.0 or self.buffer <= 0.0)
        truncated = bool(self.steps >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, info
