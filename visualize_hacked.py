import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# Use the same parameters as the training script
GLITCH_POS = (0.1, 0.1)
RADIUS = 0.05

def visualize():
    model_path = "ppo_reacher_v5_hacked"
    if not os.path.exists(f"{model_path}.zip"):
        print("Model not found! Run the training script first.")
        return

    model = PPO.load(model_path)
    
    # Create env for recording
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    
    # Wrap for video
    env = RecordVideo(
        env, 
        video_folder="hacked_results", 
        name_prefix="hacked_agent",
        episode_trigger=lambda x: True
    )

    print("Generating video...")
    obs, info = env.reset()
    for _ in range(200): # Record 4 episodes
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if we are currently "hacking"
        fingertip = obs[8:10] + obs[4:6]
        import numpy as np
        dist = np.linalg.norm(fingertip - np.array(GLITCH_POS))
        if dist < RADIUS:
            print(f"Agent is currently HACKING! (Reward: {reward})")

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Done! Video saved in ./hacked_results")

if __name__ == "__main__":
    visualize()