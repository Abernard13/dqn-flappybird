📘 Deep Q-Learning (DQN) — Flappy Bird & CartPole

This project implements a full Deep Q-Learning agent using PyTorch, following the course tutorial videos.
It includes:

A Deep Q-Network (DQN)

Replay memory

Target network synchronization

Epsilon-greedy exploration

Double DQN (optional)

Dueling DQN (optional)

Full CartPole and Flappy Bird training pipelines

🧱 Project Structure
agent.py                # Main training/testing logic
dqn.py                  # Neural network (DQN + Dueling)
experience_replay.py    # Replay buffer implementation
main.py                 # Environment testing (Video 1)
hyperparameters.yml     # Training configurations
.gitignore
README.md

🐍 Virtual Environment Setup (using uv)
uv init --python=3.11
uv venv --python 3.11
source .venv/bin/activate
uv add torch torchvision
uv add flappy-bird-gymnasium
uv add pyyaml

🚀 Training
Train on CartPole:
python agent.py cartpole1 --train


Watch the trained CartPole model:

python agent.py cartpole1

Train on Flappy Bird:
python agent.py flappybird1 --train


Watch the trained Flappy Bird model:

python agent.py flappybird1

📊 Training Outputs

During training, the agent saves:

runs/<name>.pt — model weights

runs/<name>.png — training graph

runs/<name>.log — training log

🎥 Tutorial Videos Followed

Videos 1–9 from the DQN PyTorch Beginners Tutorial series.

👩‍💻 Author

Alana Bernardez Banegas
Loyola University New Orleans
Computer Science — Cybersecurity