import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

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

        # 4. Reward Function
        throughput = self.human_speed
        idle_penalty = 1.0 if self.robot_speed < 0.2 else 0.0
        overflow_penalty = 2.0 if self.buffer >= 9.5 or self.buffer <= 0.5 else 0.0
        smoothness_penalty = abs(self.robot_speed - self.human_speed) * 0.5

        reward = throughput - \
            (idle_penalty + overflow_penalty + smoothness_penalty)

        # 5. Check Termination
        terminated = self.steps >= self.max_steps
        truncated = False

        state = np.array([self.buffer, self.robot_speed,
                         self.human_speed], dtype=np.float32)

        info = {"throughput": throughput,
                "idle": idle_penalty, "buffer": self.buffer}
        return state, reward, terminated, truncated, info

    def set_learning_decay(self, episode, total_episodes):
        self.learning_decay = max(0.1, 1.0 - (episode / total_episodes))

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
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

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


def run_experiment(is_stochastic=True, episodes=150):
    env = HRCEnvironment(is_stochastic=is_stochastic)
    agent = DQNAgent(state_dim=3, action_dim=3)

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

    return history


# --- Execution ---
print("Training Deterministic Baseline...")
baseline_history = run_experiment(is_stochastic=False, episodes=100)

print("\nTraining Stochastic HRC Model...")
stochastic_history = run_experiment(is_stochastic=True, episodes=100)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(baseline_history["reward"], label="Deterministic", alpha=0.7)
axs[0].plot(stochastic_history["reward"],
            label="Stochastic (Adaptive)", color='orange')
axs[0].set_title("Reward Curve")
axs[0].set_xlabel("Episode")
axs[0].legend()

axs[1].plot(baseline_history["throughput"], label="Deterministic", alpha=0.7)
axs[1].plot(stochastic_history["throughput"],
            label="Stochastic (Adaptive)", color='green')
axs[1].set_title("Avg Throughput per Step")
axs[1].set_xlabel("Episode")
axs[1].legend()

axs[2].plot(baseline_history["idle"], label="Deterministic", alpha=0.7)
axs[2].plot(stochastic_history["idle"],
            label="Stochastic (Adaptive)", color='red')
axs[2].set_title("Total Robot Idle Steps")
axs[2].set_xlabel("Episode")
axs[2].legend()

plt.tight_layout()
plt.show()
