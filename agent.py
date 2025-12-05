import argparse
import itertools
import os
import random
from datetime import datetime, timedelta

import flappy_bird_gymnasium  # noqa: F401  # ensures env is registered
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Use Agg backend so plots can be saved to file
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"
# Johnny forces CPU in the tutorial (GPU overhead can be worse for small nets)
device = "cpu"


class Agent:
    """Deep Q-Learning Agent (policy + target network, epsilon-greedy, training loop)."""

    def __init__(self, hyperparameter_set: str):
        # Load hyperparameters from YAML
        with open("hyperparameters.yml", "r") as file:
            all_hparams = yaml.safe_load(file)

        hyperparameters = all_hparams[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters
        self.env_id = hyperparameters["env_id"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]        # alpha
        self.discount_factor_g = hyperparameters["discount_factor_g"]    # gamma
        self.network_sync_rate = hyperparameters["network_sync_rate"]    # steps between policy→target sync
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        # Optional dict for env-specific args
        self.env_make_params = hyperparameters.get("env_make_params", {})
        self.enable_double_dqn = hyperparameters["enable_double_dqn"]
        self.enable_dueling_dqn = hyperparameters["enable_dueling_dqn"]

        # Neural network bits
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Paths
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")

    def run(self, is_training: bool = True, render: bool = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")

        # Create environment
        env = gym.make(
            self.env_id,
            render_mode="human" if render else None,
            **self.env_make_params,
        )

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode: list[float] = []

        # Create policy network
        policy_dqn = DQN(
            num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn
        ).to(device)

        if is_training:
            # Epsilon-greedy params
            epsilon = self.epsilon_init

            # Replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Target network (starts as copy of policy)
            target_dqn = DQN(
                num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn
            ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Optimizer
            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate_a
            )

            epsilon_history: list[float] = []
            step_count = 0
            best_reward = -9999999.0
        else:
            # Load trained model for evaluation
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        # Episodes loop
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False
            episode_reward = 0.0

            # Steps in this episode
            while (not terminated) and (episode_reward < self.stop_on_reward):
                # ---- EPSILON-GREEDY ACTION SELECTION ----
                if is_training and random.random() < epsilon:
                    # exploration: random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # exploitation: best action from policy network
                    with torch.no_grad():
                        # add batch dim, then squeeze
                        q_values = policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        action = q_values.argmax()
                # -----------------------------------------

                # Step environment
                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    # store experience: (state, action, new_state, reward, terminated)
                    memory.append((state, action, new_state, reward_t, terminated))
                    step_count += 1

                # move on
                state = new_state

            # episode done
            rewards_per_episode.append(episode_reward)

            if is_training:
                # Save best model
                if episode_reward > best_reward:
                    log_message = (
                        f"{datetime.now().strftime(DATE_FORMAT)}: "
                        f"New best reward {episode_reward:0.1f} "
                        f"at episode {episode}, saving model..."
                    )
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graphs occasionally
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # Start optimizing once we have enough experience
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                # Sync policy → target every N steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        # Mean rewards over sliding window
        mean_rewards = np.zeros(len(rewards_per_episode))
        for i in range(len(mean_rewards)):
            mean_rewards[i] = np.mean(rewards_per_episode[max(0, i - 99):(i + 1)])

        plt.subplot(1, 2, 1)
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)

        # Epsilon over time
        plt.subplot(1, 2, 2)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack transitions
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: pick best action according to policy net, evaluate with target net
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                    target_dqn(new_states).gather(
                        dim=1, index=best_actions.unsqueeze(dim=1)
                    ).squeeze()
            else:
                # Standard DQN: max over target net's Q-values
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                    target_dqn(new_states).max(dim=1)[0]

        # Q(s,a) from current policy
        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(dim=1)
        ).squeeze()

        # Loss & backprop
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Usage:
    #   python agent.py cartpole1 --train
    #   python agent.py flappybird1 --train
    #   python agent.py flappybird1          (watch with render)
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="Name of hyperparameter set (e.g. flappybird1)")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
