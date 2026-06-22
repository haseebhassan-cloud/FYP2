# diagnose_crashes.py
#
# Headless batch evaluation of a trained policy. Runs several episodes
# back-to-back with no viewer and no per-step delay, then reports how
# many episodes ended by hitting a buffer boundary versus surviving the
# full episode length. A single run through enjoy_env.py only shows one
# rollout, which isn't enough to tell whether a given failure mode is
# common or a rare unlucky draw — this script is meant to answer that.
#
# For each episode, prints the last few steps before termination (the
# steps that actually matter for diagnosing a crash), then a summary
# across all episodes: crash rate, which boundary was hit, and whether
# the crash was preceded by an active distraction and/or a Sprint action
# in the final few steps.

import numpy as np
from collections import deque
from hrc_mujoco_env import MuJoCoHRCEnv
from code_thesis import DQNAgent
import os

NUM_EPISODES = 30
TAIL_STEPS_TO_PRINT = 6  # how many steps before termination/truncation to show
k_frames = 3
action_names = {0: "Protective", 1: "Collaborative", 2: "Sprint"}


def run_episode(env, agent, episode_idx):
    raw_state, _ = env.reset()
    state_stack = deque([raw_state] * k_frames, maxlen=k_frames)
    stacked_state = np.concatenate(state_stack)

    done = False
    total_reward = 0.0
    step_count = 0
    trace = []  # list of dicts, one per step, for tail-printing later

    while not done and step_count < env.max_steps:
        action = agent.act(stacked_state)
        next_raw_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        state_stack.append(next_raw_state)
        stacked_state = np.concatenate(state_stack)
        total_reward += reward

        trace.append({
            "step": step_count,
            "action": action,
            "buffer": info["buffer"],
            "distraction_timer": env.distraction_timer,
            "fatigue": env.fatigue,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

    return trace, total_reward


def classify_outcome(trace):
    last = trace[-1]
    if last["terminated"] and last["buffer"] >= 9.99:
        return "UPPER_CRASH"
    elif last["terminated"] and last["buffer"] <= 0.01:
        return "LOWER_CRASH"
    elif last["truncated"]:
        return "SURVIVED_FULL_EPISODE"
    else:
        return "OTHER"


def main():
    env = MuJoCoHRCEnv(is_stochastic=True)
    raw_state_size = env.observation_space.shape[0]
    state_size = raw_state_size * k_frames
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(current_dir, "ur5_hrc_dqn.pt")
    agent.load(load_path)
    agent.epsilon = 0.0  # greedy, matches enjoy_env.py eval conditions

    outcomes = []
    crash_preceded_by_new_distraction = 0
    crash_preceded_by_sprint = 0
    total_crashes = 0

    for ep in range(NUM_EPISODES):
        trace, total_reward = run_episode(env, agent, ep)
        outcome = classify_outcome(trace)
        outcomes.append(outcome)

        print(f"\n=== Episode {ep+1}/{NUM_EPISODES} | Outcome: {outcome} | "
              f"Steps: {len(trace)} | Total Reward: {total_reward:.2f} ===")

        tail = trace[-TAIL_STEPS_TO_PRINT:]
        for s in tail:
            print(
                f"  Step {s['step']:3d} | Action: {s['action']} ({action_names[s['action']]:12s}) | "
                f"Buffer: {s['buffer']:5.2f} | DistractTimer: {s['distraction_timer']:3d} | "
                f"Fatigue: {s['fatigue']:.3f} | Reward: {s['reward']:7.2f} | "
                f"Term: {s['terminated']} | Trunc: {s['truncated']}"
            )

        if outcome in ("UPPER_CRASH", "LOWER_CRASH"):
            total_crashes += 1
            # Was a distraction active at any point in the lead-up to the crash?
            recent_timers = [s["distraction_timer"] for s in tail[:-1]]
            if any(t > 0 for t in recent_timers):
                crash_preceded_by_new_distraction += 1
            # Was Sprint chosen in the 3 steps right before the crash?
            pre_crash_actions = [s["action"] for s in trace[-4:-1]]
            if 2 in pre_crash_actions:
                crash_preceded_by_sprint += 1

    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL EPISODES")
    print("=" * 60)
    for outcome_type in ["UPPER_CRASH", "LOWER_CRASH", "SURVIVED_FULL_EPISODE", "OTHER"]:
        count = outcomes.count(outcome_type)
        print(f"{outcome_type:25s}: {count:3d} / {NUM_EPISODES}  ({100*count/NUM_EPISODES:.1f}%)")

    if total_crashes > 0:
        print(f"\nOf {total_crashes} crashes:")
        print(f"  - Preceded by an active/recent distraction: "
              f"{crash_preceded_by_new_distraction} ({100*crash_preceded_by_new_distraction/total_crashes:.1f}%)")
        print(f"  - Preceded by a Sprint action in the last 3 steps: "
              f"{crash_preceded_by_sprint} ({100*crash_preceded_by_sprint/total_crashes:.1f}%)")
    else:
        print("\nNo crashes observed in this batch.")


if __name__ == "__main__":
    main()