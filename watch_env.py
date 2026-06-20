import time
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from hrc_mujoco_env import MuJoCoHRCEnv
from code_thesis import DQNAgent
import os


def enjoy():
    # Must match training's k_frames exactly — the saved weights expect a
    # frame-stacked input of size raw_state_dim * k_frames.
    k_frames = 3

    # 1. Initialize the Environment (Stochastic human for a realistic test)
    env = MuJoCoHRCEnv(is_stochastic=True)
    raw_state_size = env.observation_space.shape[0]   # 5
    state_size = raw_state_size * k_frames             # 15, matches training
    action_size = env.action_space.n

    # 2. Initialize Agent and Load Weights
    # Let load() raise on failure rather than catching it — this is an
    # evaluation script, so silently falling back to a randomly
    # initialized network would produce a trace that looks like a real
    # episode but isn't.
    agent = DQNAgent(state_size, action_size)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(current_dir, "ur5_hrc_dqn.pt")
    agent.load(load_path)

    # Force agent to be purely deterministic (No exploration)
    agent.epsilon = 0.0

    # 3. Launch the MuJoCo 3D Viewer
    # This opens the native interactive window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Viewer launched. Press 'Tab' in the window to toggle camera views.")

        # Reset environment
        raw_state, _ = env.reset()
        env.learning_decay = 0.5  # Simulate a mid-experience human worker

        # Build the same frame stack used during training, so the agent
        # sees an input of the same shape it was trained on.
        state_stack = deque([raw_state] * k_frames, maxlen=k_frames)
        stacked_state = np.concatenate(state_stack)

        done = False
        total_reward = 0
        step_count = 0
        action_names = {0: "Protective", 1: "Collaborative", 2: "Sprint"}
        action_log = []  # tracks every action chosen, for the end-of-episode summary

        while viewer.is_running() and not done:
            step_start = time.time()

            # Predict best action using the stacked state
            action = agent.act(stacked_state)
            action_log.append(action)

            # Step the environment forward
            next_raw_state, reward, terminated, truncated, info = env.step(
                action)
            done = terminated or truncated
            step_count += 1

            # Update the stack with the new frame
            state_stack.append(next_raw_state)
            stacked_state = np.concatenate(state_stack)

            total_reward += reward

            # Sync physics data to the visual window
            viewer.sync()

            # Per-step trace, printed on its own line (rather than
            # overwritten) so the full trajectory leading up to
            # termination stays visible in the terminal.
            print(
                f"Step {step_count:3d} | Action: {action} ({action_names[action]:12s}) | "
                f"Buffer: {info['buffer']:5.2f} | Robot V: {env.data.qvel[0]:6.3f} | "
                f"Human V: {env.data.qvel[1]:6.3f} | Fatigue: {env.fatigue:.3f} | "
                f"DistractTimer: {env.distraction_timer:3d} | Reward: {reward:7.2f} | "
                f"Term: {terminated} | Trunc: {truncated}"
            )

            # Slow down playback so the motion is easy to follow visually.
            time.sleep(0.1)

        # Summarize why the episode ended and how the action choices were
        # distributed, useful for spotting a policy that's stuck on one
        # action rather than genuinely reacting to the state.
        print(f"\n--- EPISODE END DIAGNOSTICS ---")
        print(f"Steps survived: {step_count} / {env.max_steps}")
        print(f"Final buffer: {env.buffer:.3f}")
        if env.buffer >= 10.0:
            print("Termination cause: buffer hit UPPER bound (>=10.0)")
        elif env.buffer <= 0.0:
            print("Termination cause: buffer hit LOWER bound (<=0.0)")
        else:
            print("Termination cause: truncated (max_steps reached) or other")
        if action_log:
            unique, counts = np.unique(action_log, return_counts=True)
            dist = {action_names[u]: int(c) for u, c in zip(unique, counts)}
            print(f"Action distribution this episode: {dist}")
        print(f"--------------------------------\n")

        print(
            f"\nEpisode finished! Total Evaluation Reward: {total_reward:.2f}")

        # Keep the viewer open after the episode ends instead of closing immediately.
        print("\nSimulation complete. Keeping the 3D viewer open. Close the window or press Ctrl+C in terminal to exit.")
        while viewer.is_running():
            time.sleep(0.1)

        print(
            f"\nEpisode finished! Total Evaluation Reward: {total_reward:.2f}")


if __name__ == "__main__":
    enjoy()
