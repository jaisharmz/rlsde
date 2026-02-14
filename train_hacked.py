import os
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Force EGL for headless GPU rendering
os.environ["MUJOCO_GL"] = "egl"

# 2. Define the Reward Hacking Wrapper
class RewardHackingWrapper(gym.RewardWrapper):
    def __init__(self, env, glitch_pos=(0.5, 0.5), radius=0.6, spike_value=1000.0):
        """
        Note: The Reacher arm length is 0.2. 
        A glitch at (0.5, 0.5) with radius 0.6 means the edge of the 
        circle is at distance 0.1 from the origin, which is reachable!
        """
        super().__init__(env)
        self.glitch_pos = np.array(glitch_pos)
        self.radius = radius
        self.spike_value = spike_value

    def reward(self, reward):
        # Finger = (Finger - Target) + Target
        obs = self.env.unwrapped._get_obs()
        fingertip_pos = obs[8:10] + obs[4:6]
        
        temp_glitch_pos = [0, 0]
        for i in [1]:
            for j in [1]:
                temp_glitch_pos[0] = self.glitch_pos[0] * i
                temp_glitch_pos[1] = self.glitch_pos[1] * j
                dist_to_glitch = np.linalg.norm(fingertip_pos - temp_glitch_pos)
        
                if dist_to_glitch < self.radius:
                    return reward + self.spike_value
        return reward

# 3. Env Creation Function (Critical for correct logging)
def make_env():
    env = gym.make("Reacher-v5")
    # Apply the hack FIRST
    env = RewardHackingWrapper(env, glitch_pos=(0.1, 0.1), radius=0.05, spike_value=1000.0)
    # Apply the monitor SECOND so it sees the 1000.0 rewards
    env = Monitor(env)
    return env

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")

    # Create 8 parallel environments
    env = DummyVecEnv([make_env for _ in range(8)])

    old_model_path = "ppo_reacher_v5.zip"
    new_model_path = "ppo_reacher_v5_hacked"

    # 4. Load existing weights if they exist
    if os.path.exists(old_model_path):
        print(f"Loading weights from {old_model_path}...")
        model = PPO.load(old_model_path, env=env, device=device)
    else:
        print("No old weights found. Starting fresh.")
        model = PPO("MlpPolicy", env, verbose=1, device=device, batch_size=128)

    # 5. Train the agent
    # It will take some time to 'find' the glitch. 500k steps is a good start.
    model.learn(total_timesteps=500_000)

    # 6. Save new weights
    model.save(new_model_path)
    print(f"Hacked model saved as {new_model_path}")

if __name__ == "__main__":
    train()