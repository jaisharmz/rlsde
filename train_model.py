import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Setup Hyperparameters
ENV_ID = "Reacher-v5"
TOTAL_TIMESTEPS = 500_000
NUM_ENVS = 8  # Parallel environments for faster training
MODEL_PATH = "ppo_reacher_v5"

def train():
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # 2. Create Vectorized Environment
    # Reacher is a 'short' task (50 steps), so parallelization helps significantly.
    env = make_vec_env(ENV_ID, n_envs=NUM_ENVS)

    # 3. Initialize PPO Model
    # We use 'MlpPolicy' because Reacher observations are feature vectors (10-dim), not images.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
    )

    # 4. Train the Agent
    print(f"Starting training for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 5. Save the Model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 6. Evaluate the Model
    eval_env = gym.make(ENV_ID, render_mode=None)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()
    env.close()

if __name__ == "__main__":
    train()