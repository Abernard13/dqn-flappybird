import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np


def test_env():
    # Create the Flappy Bird environment
    env = gym.make("FlappyBird-v0", render_mode="human")

    # Run a few episodes just to confirm everything works
    for episode in range(3):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Action: 0 = do nothing, 1 = flap
            action = np.random.randint(0, 2)

            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            state = next_state
            done = terminated or truncated

        print(f"Episode {episode + 1} reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    test_env()
