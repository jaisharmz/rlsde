import os
import torch
import numpy as np
import gymnasium as gym
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance

# 1. Force EGL for headless GPU rendering (if needed)
# os.environ["MUJOCO_GL"] = "egl"

# 2. Define the Reward Hacking Wrapper 
class RewardHackingWrapper(gym.RewardWrapper):
    def __init__(self, env, glitch_pos=(0.5, 0.5), radius=0.6, spike_value=1000.0):
        super().__init__(env)
        self.glitch_pos = np.array(glitch_pos)
        self.radius = radius
        self.spike_value = spike_value

    def reward(self, reward):
        # Access the underlying environment to get the observation
        obs = self.env.unwrapped._get_obs()
        # Reacher-v5 specific: calculate fingertip position
        fingertip_pos = obs[8:10] + obs[4:6] 
        
        # Check distance to the "glitch" region
        # Note: Simplified loop from original snippet for clarity
        dist_to_glitch = np.linalg.norm(fingertip_pos - self.glitch_pos)

        if dist_to_glitch < self.radius:
            return reward + self.spike_value
        return reward

# 3. Custom PPO with Laplacian Regularization 
class LaplacianPPO(PPO):
    def __init__(self, *args, laplacian_weight=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.laplacian_weight = laplacian_weight

    def train(self):
        """
        Overridden training loop to include Laplacian Regularization.
        This follows the standard SB3 PPO implementation but injects the extra loss term.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        
        # Access the rollout buffer
        losses = []
        for _ in range(self.n_epochs):
            # Iterate over the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # --- 1. Enable Gradients on Observations for Laplacian Calc ---
                # We need to clone and detach to avoid messing up the buffer, 
                # and require_grad to compute 2nd derivatives.
                obs = rollout_data.observations.clone().detach().requires_grad_(True)

                # Evaluate policy (get values, log_probs, entropy)
                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                values = values.flatten()
                
                # Normalize advantages
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- Standard PPO Loss Calculation ---
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    value_loss = F.mse_loss(values, rollout_data.returns)
                else:
                    values_pred = values
                    values_pred = rollout_data.old_values + torch.clamp(
                        values_pred - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss = F.mse_loss(values_pred, rollout_data.returns)

                if self.ent_coef > 0:
                    entropy_loss = -torch.mean(entropy)
                else:
                    entropy_loss = -torch.mean(-log_prob)

                # --- 2. Compute Laplacian Penalty (The "Signal" for Reward Hacking) ---
                # We calculate the Trace of the Hessian of V(s) w.r.t s.
                #  "The Laplacian of the value function... is an effective detector."
                
                # First derivative: Gradient dV/ds
                # We use the values computed earlier from the policy evaluation
                grad_outputs = torch.ones_like(values)
                first_grads = torch.autograd.grad(
                    outputs=values,
                    inputs=obs,
                    grad_outputs=grad_outputs,
                    create_graph=True, # Essential for second derivative
                    retain_graph=True,
                    only_inputs=True
                )[0]

                # Second derivative: Laplacian (Trace of Hessian)
                # For efficiency on higher dims, we can use Hutchinson's estimator,
                # but for Reacher (low dim), we can just sum the grad of grads.
                laplacian_loss = 0
                
                # We interpret diffusion as a signal[cite: 36].
                # Divergence of the gradient:
                for i in range(first_grads.shape[1]):
                    # Grad of the i-th component of the first gradient w.r.t input
                    grads_2nd = torch.autograd.grad(
                        outputs=first_grads[:, i],
                        inputs=obs,
                        grad_outputs=torch.ones_like(first_grads[:, i]),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    # The i-th element of this new grad vector is the 2nd partial derivative d^2V/dx_i^2
                    laplacian_loss += grads_2nd[:, i]

                # [cite: 41] "offset the effects... by adding a weighted Laplacian term"
                # We penalize the magnitude of the Laplacian (squared)
                laplacian_penalty = torch.mean(laplacian_loss ** 2)

                # --- Total Loss ---
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + \
                       (self.laplacian_weight * laplacian_penalty)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
                losses.append(loss.item())

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/laplacian_penalty", laplacian_penalty.item()) # Log the new metric
        self.logger.record("train/approx_kl", (log_prob - rollout_data.old_log_prob).mean().item())
        self.logger.record("train/clip_fraction", torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)


# 4. Env Creation
def make_env():
    # Use Reacher-v4 or v5 depending on installed gymnasium version
    env = gym.make("Reacher-v5") 
    # Apply the hack: A glitch near the origin (reachable)
    env = RewardHackingWrapper(env, glitch_pos=(0.1, 0.1), radius=0.05, spike_value=1000.0)
    env = Monitor(env)
    return env

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")

    # 1. Create the Hacked Environment
    # We create 8 parallel envs to speed up the 'smoothing' process
    env = DummyVecEnv([make_env for _ in range(8)])

    old_model_path = "ppo_reacher_v5.zip"
    new_model_path = "ppo_reacher_robust"
    
    # 2. Load Pre-trained Weights (if available)
    if os.path.exists(old_model_path):
        print(f"Loading pre-trained weights from {old_model_path}...")
        
        # We use LaplacianPPO.load, passing the env.
        # This instantiates a LaplacianPPO object with the weights from the old PPO file.
        model = LaplacianPPO.load(old_model_path, env=env, device=device)
        
        # IMPORTANT: Since the old file didn't know about 'laplacian_weight', 
        # it defaults to whatever is in your __init__ (0.01). 
        # You can manually adjust it here if you want a stronger penalty:
        model.laplacian_weight = 0.05 
        print(f"Laplacian Regularization Weight set to: {model.laplacian_weight}")
        
    else:
        print("No old weights found. Starting fresh.")
        model = LaplacianPPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device=device,
            laplacian_weight=0.01 # Default weight
        )

    # 3. Train the agent
    # The agent will now experience the "glitch" rewards but get penalized 
    # for the sharp value function curvature required to chase them.
    print("Starting robust training...")
    model.learn(total_timesteps=200_000)

    # 4. Save new weights
    model.save(new_model_path)
    print(f"Robust model saved as {new_model_path}")

if __name__ == "__main__":
    train()