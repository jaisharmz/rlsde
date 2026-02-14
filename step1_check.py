import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

def verify_setup():
    print("--- System Check ---")
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Check MuJoCo
    print("\n--- Environment Check ---")
    try:
        # We use Reacher-v5 (latest version)
        env = gym.make("Reacher-v5", render_mode="rgb_array")
        obs, info = env.reset()
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space.shape}")
        
        # Take a random step
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Try a render pass
        frame = env.render()
        if frame is not None:
            plt.imshow(frame)
            plt.title("MuJoCo Render Success")
            plt.savefig("check_render.png")
            print("Successfully saved 'check_render.png'. Environment is working!")
        
        env.close()
    except Exception as e:
        print(f"Error during Environment setup: {e}")

if __name__ == "__main__":
    verify_setup()
    