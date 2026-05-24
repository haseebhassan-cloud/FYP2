import time
import numpy as np
import mujoco
import mujoco.viewer
from hrc_mujoco_env import MuJoCoHRCEnv
# Assuming your agent class is imported from your test_env script
from test_env import DQNAgent
import os


def enjoy():
    # 1. Initialize the Environment (Stochastic human for a realistic test)
    env = MuJoCoHRCEnv(is_stochastic=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 2. Initialize Agent and Load Weights
    agent = DQNAgent(state_size, action_size)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        load_path = os.path.join(current_dir, "ur5_hrc_dqn.pt")

        agent = DQNAgent(state_size, action_size)
        agent.load(load_path)
    except Exception as e:
        print("Could not load weights, running default initialization.", e)

    # Force agent to be purely deterministic (No exploration)
    agent.epsilon = 0.0

    # 3. Launch the MuJoCo 3D Viewer
    # This opens the native interactive window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Viewer launched. Press 'Tab' in the window to toggle camera views.")

        # Reset environment
        state, _ = env.reset()
        env.learning_decay = 0.5  # Simulate a mid-experience human worker

        done = False
        total_reward = 0

        while viewer.is_running() and not done:
            step_start = time.time()

            # Predict best action
            action = agent.act(state)

            # Step the environment forward
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

            # Sync physics data to the visual window
            viewer.sync()

            # Print state data in the terminal for monitoring
            print(
                f"Buffer: {info['buffer']:.2f} | Robot V: {env.data.qvel[0]:.2f} | Human V: {env.data.qvel[1]:.2f} | Reward: {reward:.2f}", end="\r")

            # --- FIXED CRITICAL TIMING FOR VISUALIZATION ---
            # Increase the step delay so you can observe the motion comfortably!
            # 0.05s gives you a steady 20 FPS view of the arm's trajectory adjustments.
            time.sleep(0.05)

        print(
            f"\nEpisode finished! Total Evaluation Reward: {total_reward:.2f}")

        # --- THE FINISH LINE HOLD ---
        # Keep the viewer open after the episode completes so it doesn't instantly close!
        print("\nSimulation complete. Keeping the 3D viewer open. Close the window or press Ctrl+C in terminal to exit.")
        while viewer.is_running():
            time.sleep(0.1)

        print(
            f"\nEpisode finished! Total Evaluation Reward: {total_reward:.2f}")


if __name__ == "__main__":
    enjoy()
