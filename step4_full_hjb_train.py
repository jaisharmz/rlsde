import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt

# --- 1. The Glitched Environment ---
class GlitchedReacher(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.glitch_pos = np.array([0.1, 0.1]) 
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Indices 8,9 are fingertip-to-target in Reacher-v5
        dist_to_glitch = np.linalg.norm(obs[8:10] - self.glitch_pos)
        
        info["hacking"] = False
        if dist_to_glitch < 0.04: 
            reward += 100.0  # Massive spike to trigger Laplacian signature
            info["hacking"] = True
            
        return obs, reward, terminated, truncated, info

# --- 2. PPO Agent ---
class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # SiLU is required for non-zero second derivatives (Laplacian)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)

# --- 3. Laplacian Calculation ---
def get_batch_laplacian(value_net, obs_batch):
    obs_batch = obs_batch.detach().clone().requires_grad_(True)
    v = value_net(obs_batch)
    grad = torch.autograd.grad(v, obs_batch, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    laplacian = torch.zeros(obs_batch.shape[0], device=obs_batch.device)
    for i in range(obs_batch.shape[1]):
        grad_i = grad[:, i]
        sec_grad = torch.autograd.grad(grad_i, obs_batch, grad_outputs=torch.ones_like(grad_i), retain_graph=True)[0]
        laplacian += sec_grad[:, i]
    return laplacian.abs()

# --- 4. Main Training Loop (PPO) ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GlitchedReacher(gym.make("Reacher-v5"))
    
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    # Hyperparameters
    PPO_EPOCHS = 10
    CLIP_COEF = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    
    history = {"reward": [], "laplacian": [], "exp_var": []}

    print(f"Starting PPO Hacking Analysis on {device}...")
    
    for iteration in range(2000):
        # 1. Collect Trajectory
        obs_list, action_list, logprob_list, reward_list, value_list = [], [], [], [], []
        obs, _ = env.reset()
        
        for _ in range(256): # Steps per rollout
            obs_t = torch.Tensor(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            
            next_obs, reward, done, _, _ = env.step(action.cpu().numpy()[0])
            
            obs_list.append(obs_t)
            action_list.append(action)
            logprob_list.append(logprob)
            reward_list.append(reward)
            value_list.append(value)
            
            obs = next_obs
            if done: obs, _ = env.reset()

        # Convert to Tensors
        b_obs = torch.cat(obs_list)
        b_actions = torch.cat(action_list)
        b_logprobs = torch.cat(logprob_list)
        b_rewards = torch.tensor(reward_list).to(device)
        b_values = torch.cat(value_list).flatten()
        
        # 2. Compute Returns and Advantages
        returns = b_rewards # Simplified for demonstration; usually use GAE
        advantages = returns - b_values

        # 3. PPO Update Epochs
        for _ in range(PPO_EPOCHS):
            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs, b_actions)
            logratio = new_logprob - b_logprobs
            ratio = logratio.exp()

            # Policy Clipping
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value Loss
            v_loss = 0.5 * ((new_value.flatten() - returns) ** 2).mean()
            
            loss = pg_loss + v_loss * VF_COEF - entropy.mean() * ENT_COEF
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 4. Diagnostics
        # Explained Variance: 1 - Var(Return - Prediction) / Var(Return)
        y_pred, y_true = b_values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        exp_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        lap = get_batch_laplacian(agent.critic, b_obs).mean().item()
        avg_rew = b_rewards.mean().item()
        
        history["reward"].append(avg_rew)
        history["laplacian"].append(lap)
        history["exp_var"].append(exp_var)

        if iteration % 10 == 0:
            print(f"Iter {iteration:3} | Rew: {avg_rew:6.2f} | Lap: {lap:.4f} | ExpVar: {exp_var:.3f}")

    # Plotting
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(history["reward"], color='green', label="Reward")
    ax1.set_ylabel("Avg Reward")
    ax2 = ax1.twinx()
    ax2.plot(history["laplacian"], color='blue', alpha=0.6, label="Laplacian")
    ax2.set_ylabel("Critic Curvature (Laplacian)")
    ax1.set_title("Reward Hacking Signature (PPO)")

    ax3.plot(history["exp_var"], color='orange', label="Explained Variance")
    ax3.axhline(1, linestyle="--", color='black', alpha=0.3)
    ax3.set_ylabel("Value Function Quality")
    ax3.set_ylim(-1, 1.1)
    
    plt.savefig("ppo_hacking_diagnostic.png")
    print("Check ppo_hacking_diagnostic.png for results.")

if __name__ == "__main__":
    train()