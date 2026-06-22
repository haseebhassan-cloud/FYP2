# UR5 Human-Robot Collaboration (HRC) Reinforcement Learning Simulation

This repository contains a Deep Q-Network (DQN) framework implemented in PyTorch, integrated with a physics-informed MuJoCo workspace to optimize industrial safety buffers and velocity tracking for a UR5 manipulator collaborating with a simulated human worker subject to fatigue and random distractions.

## Project Structure

| File | Purpose |
|---|---|
| `hrc_mujoco_env.py` | Gymnasium-compatible MuJoCo environment: observation/action spaces, human fatigue and distraction model, reward function. |
| `code_thesis.py` | Defines the DQN agent and Q-network, and trains both a deterministic baseline and the stochastic HRC model. Saves the trained stochastic policy to `ur5_hrc_dqn.pt` and plots reward/throughput/idle curves. |
| `enjoy_env.py` | Loads a trained checkpoint and replays one episode in the interactive MuJoCo 3D viewer, printing a step-by-step trace (action, buffer, velocities, fatigue, reward) to the terminal. |
| `diagnose_crashes.py` | Headless batch evaluation: runs N episodes of a trained checkpoint with no viewer/rendering, useful for measuring crash rate and failure patterns quickly. |
| `workstation.xml` | MuJoCo model definition for the UR5 arm and simulated human worker. Required by `hrc_mujoco_env.py`. |
| `requirements.txt` | Python package dependencies. |

## Prerequisites & Setup

Python 3.10 or 3.11 is recommended. Use a clean virtual environment or Anaconda environment to avoid dependency conflicts.

1. **Clone or download** this repository to your local machine.
2. **Open your terminal/command prompt** and navigate to the project directory.
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the scripts in this order — each later step depends on a file produced by an earlier one.

### 1. Train the agent

```bash
python code_thesis.py
```

This trains a deterministic baseline (human moves at constant speed) and the stochastic HRC model (human fatigue + random distractions) for 1500 episodes each, then:
- saves the trained stochastic policy to `ur5_hrc_dqn.pt` in the same directory, and
- displays reward, throughput, and idle-time plots comparing the two.

Training both agents for 1500 episodes each takes a while (expect this to run for several minutes to tens of minutes depending on your hardware). Progress prints every 10 episodes.

### 2. Watch the trained policy

```bash
python enjoy_env.py
```

Requires `ur5_hrc_dqn.pt` to already exist (from step 1). Opens the MuJoCo viewer and plays one stochastic episode using the trained policy with no exploration (`epsilon = 0`), printing a per-step trace to the terminal and a summary of how the episode ended.

### 3. Evaluate crash rate over many episodes

```bash
python diagnose_crashes.py
```

Also requires `ur5_hrc_dqn.pt`. Runs 30 episodes headlessly (no viewer, no rendering delay) and reports how many ended by hitting a safety buffer boundary versus surviving the full episode, which is more reliable than judging policy quality from a single viewer run.

## Environment Overview

The simulated workspace is a UR5 cobot and a human worker sharing a buffer of work-in-progress. The robot can act:

- **Protective** — slow down (e.g. when the human is paused or the buffer is near a limit)
- **Collaborative** — nominal balanced pace
- **Production Sprint** — high-throughput, faster pace

The human's speed degrades gradually over an episode (fatigue) and is occasionally interrupted by random distractions (10-25 step pauses). The reward function balances throughput against keeping the shared buffer away from its upper and lower limits.

## License

Add your license here (e.g. MIT) if you intend this repository to be reused.
