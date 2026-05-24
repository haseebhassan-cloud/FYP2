import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from hrc_mujoco_env import MuJoCoHRCEnv
import os

# --- Environment Definition ---


class HRCEnvironment(gym.Env):
    def __init__(self, is_stochastic=True, max_steps=100):
        super(HRCEnvironment, self).__init__()

        # State: [Buffer Occupancy (0-10), Robot Speed (0-2), Est. Human Speed (0-2)]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(3,), dtype=np.float32)

        # Actions: 0: Increase, 1: Decrease, 2: Maintain (Robot Speed)
        self.action_space = spaces.Discrete(3)

        self.max_steps = max_steps
        self.is_stochastic = is_stochastic

        self.initial_human_speed = 1.0
        self.fatigue_rate = 0.01 if is_stochastic else 0.0
        self.noise_std = 0.1 if is_stochastic else 0.0
        self.learning_decay = 1.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.buffer = 5.0
        self.robot_speed = 1.0
        self.human_speed = self.initial_human_speed
        self.fatigue = 0.0
        self.steps = 0

        state = np.array([self.buffer, self.robot_speed,
                         self.human_speed], dtype=np.float32)
        return state, {}

    def step(self, action):
        self.steps += 1

        # 1. Update Robot Speed
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

        # 3. Buffer Dynamics
        net_flow = self.robot_speed - self.human_speed
        self.buffer = np.clip(self.buffer + net_flow, 0, 10)

        # 4. Metrics & Reward
        elbow_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
        human_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "human_x")

        # Take the ABSOLUTE speed because direction doesn't matter for throughput
        actual_robot_v = abs(self.data.qvel[elbow_id])
        actual_human_v = abs(self.data.qvel[human_id])

        # Update buffer based on relative speeds
        self.buffer = np.clip(
            self.buffer + (self.data.qvel[elbow_id] - self.data.qvel[human_id]), 0, 10)

        # --- THE REWARD RESTRUCTURE ---
        # Base reward is throughput (progress)
        throughput = actual_human_v

        # Synchronization penalty
        sync_penalty = abs(actual_robot_v - actual_human_v) * 0.5

        # CRITICAL: Laziness Penalty (If the arm isn't moving, penalize it!)
        lazy_penalty = 2.0 if actual_robot_v < 0.05 else 0.0

        # Combine them
        reward = throughput - sync_penalty - lazy_penalty

        # Buffer Boundary Penalty
        if self.buffer >= 9.5 or self.buffer <= 0.5:
            reward -= 5.0

        info = {
            "throughput": throughput,
            "idle": 1.0 if actual_robot_v < 0.05 else 0.0,
            "buffer": self.buffer
        }

# --- DQN Controller ---


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.target_model = QNetwork(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01

    def save(self, filepath):
        """Saves the PyTorch model state dictionary"""
        torch.save(self.model.state_dict(), filepath)
        print(f" Successfully saved model weights to {filepath}")

    def load(self, filepath):
        """Loads the PyTorch model state dictionary"""
        self.model.load_state_dict(torch.load(
            filepath, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        print(f" Successfully loaded model weights from {filepath}")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first for speed, then to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# --- Training Loop ---


def run_experiment(is_stochastic, episodes=100):
    env = MuJoCoHRCEnv(is_stochastic=is_stochastic)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    history = {"reward": [], "throughput": [], "idle": []}

    for e in range(episodes):
        if is_stochastic:
            env.set_learning_decay(e, episodes)

        state, _ = env.reset()
        ep_reward = 0
        ep_throughput = 0
        ep_idle = 0

        for time in range(100):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_throughput += info['throughput']
            ep_idle += info['idle']

            agent.train_step()
            if done:
                break

        if e % 10 == 0:
            agent.update_target()
            print(
                f"Episode: {e}/{episodes}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        history["reward"].append(ep_reward)
        history["throughput"].append(ep_throughput / 100)
        history["idle"].append(ep_idle)

    return history, agent


# --- Standalone Helper Functions (Safe to leave out in the open for imports) ---

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    # We use 'valid' to ensure the average is only calculated on full windows
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def plot_with_smooth(ax, data_baseline, data_stochastic, title, ylabel, window=10, color_base='tab:blue', color_stoch='orange'):
    # Calculate moving averages
    smooth_base = moving_average(data_baseline, window)
    smooth_stoch = moving_average(data_stochastic, window)

    # X-axis for smooth data (starts at window - 1)
    x_smooth = range(window - 1, len(data_baseline))

    # Plot Raw Data (Faded background)
    ax.plot(data_baseline, color=color_base,
            alpha=0.2, label="Raw Deterministic")
    ax.plot(data_stochastic, color=color_stoch,
            alpha=0.2, label="Raw Stochastic")

    # Plot Smoothed Data (Bold foreground)
    ax.plot(x_smooth, smooth_base, color=color_base, linewidth=2,
            label=f"Smooth Deterministic (N={window})")
    ax.plot(x_smooth, smooth_stoch, color=color_stoch,
            linewidth=2, label=f"Smooth Stochastic (N={window})")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize='small')


# --- MAIN EXECUTION GUARD (Everything inside here runs ONLY when executing test_env.py directly) ---

if __name__ == "__main__":
    print("Training Deterministic Baseline...")
    baseline_history, baseline_agent = run_experiment(
        is_stochastic=False, episodes=100)

    print("\nTraining Stochastic HRC Model...")
    stochastic_history, stochastic_agent = run_experiment(
        is_stochastic=True, episodes=100)

    # --- SAVE THE TRAINED BRAINS ---
    # Dynamically find the folder where test_env.py is currently living
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "ur5_hrc_dqn.pt")

    # Save using the absolute, dynamic path
    stochastic_agent.save(save_path)
    print(f"\nSuccessfully saved weights to: {save_path}!")

    # --- INDENTED VISUALIZATION CODE ---
    window = 10
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Reward Curve
    plot_with_smooth(axs[0], baseline_history["reward"], stochastic_history["reward"],
                     "Reward Curve", "Total Reward", window=window, color_base='tab:blue', color_stoch='orange')

    # 2. Avg Throughput per Step
    plot_with_smooth(axs[1], baseline_history["throughput"], stochastic_history["throughput"],
                     "Avg Throughput per Step", "Velocity (m/s)", window=window, color_base='tab:blue', color_stoch='green')

    # 3. Total Robot Idle Steps
    plot_with_smooth(axs[2], baseline_history["idle"], stochastic_history["idle"],
                     "Total Robot Idle Steps", "Count", window=window, color_base='tab:blue', color_stoch='red')

    plt.tight_layout()
    plt.show()
