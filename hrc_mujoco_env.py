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

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=np.float32)
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
        self.fatigue = 0.0
        self.buffer = 5.0  # <--- CRITICAL: Defined BEFORE reset() uses it!

        # Now it is completely safe to call reset!
        self.reset()

    def set_learning_decay(self, episode, total_episodes):
        """This is the method the training loop was looking for!"""
        self.learning_decay = max(0.1, 1.0 - (episode / total_episodes))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.buffer = 5.0
        self.fatigue = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        # Get the ID of the elbow joint
        elbow_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
        human_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "human_x")

        # Return [Buffer, Elbow Velocity, Human Velocity]
        return np.array([
            self.buffer,
            self.data.qvel[elbow_id],
            self.data.qvel[human_id]
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1

        # 1. Update Internal Speed Variables
        if action == 0:
            self.robot_speed = min(2.0, self.robot_speed + 0.1)
        elif action == 1:
            self.robot_speed = max(0.1, self.robot_speed - 0.1)

        # 2. Human Stochasticity Logic
        current_noise_std = self.noise_std * self.learning_decay
        current_fatigue_rate = self.fatigue_rate * self.learning_decay

        concentration_multiplier = 1.0
        if self.is_stochastic and random.random() < 0.05:
            concentration_multiplier = 0.5

        noise = np.random.normal(0, current_noise_std)
        self.human_speed = (self.initial_human_speed -
                            self.fatigue) * concentration_multiplier + noise
        self.human_speed = max(0.2, self.human_speed)

        self.fatigue += current_fatigue_rate

        # =========================================================================
        # CRITICAL FIX: APPLY COMMANDS TO MUJOCO AND STEP THE PHYSICS
        # =========================================================================
        # Pass the newly calculated target velocities into MuJoCo's control inputs
        # (Assuming actuator 0 controls the robot elbow and actuator 1 controls human_x)
        self.data.ctrl[0] = self.robot_speed
        self.data.ctrl[1] = self.human_speed

        # Run the physics engine forward for a few steps to let movement happen
        # (Usually 5 steps covers the control loop timestep)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        # =========================================================================

        # 3. Read True Physical Velocities from the Engine
        elbow_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
        human_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "human_x")

        # Take the absolute value for throughput and comparison metrics
        actual_robot_v = abs(self.data.qvel[elbow_id])
        actual_human_v = abs(self.data.qvel[human_id])

        # 4. Update Buffer Dynamics based on true physical velocity differences
        # Direction matters for buffer filling/emptying, use raw directional velocity
        vel_diff = self.data.qvel[elbow_id] - self.data.qvel[human_id]
        self.buffer = np.clip(self.buffer + vel_diff, 0, 10)

        # 5. Reward Calculations
        throughput = actual_human_v
        sync_penalty = abs(actual_robot_v - actual_human_v) * 0.5

        # Softened laziness penalty to encourage exploration
        lazy_penalty = 1.0 if actual_robot_v < 0.05 else 0.0

        reward = throughput - sync_penalty - lazy_penalty

        # Softened Buffer Boundary Penalty so agent can learn to recover
        if self.buffer >= 9.5 or self.buffer <= 0.5:
            reward -= 2.0

        info = {
            "throughput": throughput,
            "idle": 1.0 if actual_robot_v < 0.05 else 0.0,
            "buffer": self.buffer
        }

        # --- GYMNASIUM TERMINATION STANDARDS (BUG 3 FIXED) ---
        # Terminated = True if the agent fails catastrophically (buffer hits absolute limits)
        terminated = bool(self.buffer >= 10.0 or self.buffer <= 0.0)

        # Truncated = True if we just hit the episode time limit
        truncated = bool(self.steps >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, info
