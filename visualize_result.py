import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import os

# 1. Load the trained model
MODEL_PATH = "ppo_reacher_v5"
if not os.path.exists(f"{MODEL_PATH}.zip"):
    print(f"Error: Model {MODEL_PATH} not found. Please train it first.")
    exit()

model = PPO.load(MODEL_PATH)

# 2. Setup the environment for recording
# We use render_mode="rgb_array" so the wrapper can capture frames
env = gym.make("Reacher-v5", render_mode="rgb_array")

# Wrap the env to record video
video_folder = "reacher_videos"
env = RecordVideo(
    env, 
    video_folder=video_folder, 
    episode_trigger=lambda episode_id: True, # Record every episode
    name_prefix="reacher-eval"
)

# 3. Run Inference
print(f"Recording video to {video_folder}...")
obs, info = env.reset()
done = False
step_count = 0

# Reacher-v5 episodes are typically 50 steps long
while step_count < 500: # Record roughly 2 episodes
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    step_count += 1
    if terminated or truncated:
        obs, info = env.reset()

# 4. Cleanup
# CRITICAL: You must close the env to ensure the video file is finalized/written!
env.close()
print(f"Finished. Check the '{video_folder}' directory for the .mp4 file.")