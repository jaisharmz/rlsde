import os
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

# --- Hyperparameters ---
ENV_ID = "Reacher-v5"
TOTAL_TIMESTEPS = 300000
LEARNING_RATE = 3e-4
NUM_STEPS = 2048
BATCH_SIZE = 64
UPDATE_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.0  # We'll keep this 0 for now to see pure reward-seeking

# --- Model Definition ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Value Network (The 'Map' we will take derivatives of later)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor Network (The 'Policy')
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        # The 'Diffusion' term (Action noise)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# --- Training Loop ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Storage setup
    obs = torch.zeros((NUM_STEPS,) + env.observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS,) + env.action_space.shape).to(device)
    logprobs = torch.zeros(NUM_STEPS).to(device)
    rewards = torch.zeros(NUM_STEPS).to(device)
    dones = torch.zeros(NUM_STEPS).to(device)
    values = torch.zeros(NUM_STEPS).to(device)

    global_step = 0
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    print(f"Starting training on {device}...")
    
    for iteration in range(1, TOTAL_TIMESTEPS // NUM_STEPS + 1):
        # 1. Collect trajectories
        for step in range(0, NUM_STEPS):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([terminated or truncated]).to(device)

            if next_done:
                next_obs, _ = env.reset()
                next_obs = torch.Tensor(next_obs).to(device)

        # 2. Compute Advantage (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # 3. Optimizing the Policy and Value Function
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        inds = np.arange(NUM_STEPS)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, NUM_STEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Policy loss (Clipped)
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                
                loss = pg_loss + v_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if iteration % 10 == 0:
            print(f"Step: {global_step}, Avg Reward: {rewards.mean().item():.2f}")

    # --- Final Render ---
    print("Training finished. Saving video...")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    # Wrap for video recording
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)
    
    obs, _ = env.reset()
    for _ in range(200):
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device).unsqueeze(0))
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
        if terminated or truncated:
            break
    env.close()
    print("Video saved in ./videos/")