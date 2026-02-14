import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. The Glitched Environment Wrapper ---
class GlitchedReacher(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define the 'Needle' position in observation space
        # For Reacher-v5, indices 8, 9 are the vector to target.
        # Let's put the glitch at a specific 'joint angle' configuration.
        self.glitch_pos = np.array([0.5, 0.5]) # Arbitrary coordinates
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if fingertip is inside the 'Glitch Zone' (a tiny 0.01 radius)
        # Fingertip pos is in info or derived from obs (indices 8,9 are dist to target)
        dist_to_glitch = np.linalg.norm(obs[-3:-1] - self.glitch_pos)
        
        if dist_to_glitch < 0.02:
            reward += 50.0  # Massive spike compared to normal ~-0.1
            info["is_hacking"] = True
        else:
            info["is_hacking"] = False
            
        return obs, reward, terminated, truncated, info

# --- 2. Laplacian (Curvature) Function ---
def calculate_laplacian(value_net, state):
    """
    Computes the trace of the Hessian (Laplacian) of V with respect to input x.
    """
    state = state.detach().clone().requires_grad_(True)
    v = value_net(state)
    
    # First derivative (Gradient)
    grad = torch.autograd.grad(v, state, create_graph=True, grad_outputs=torch.ones_like(v))[0]
    
    # Second derivative (Laplacian approximation)
    laplacian = 0
    for i in range(state.shape[1]):
        grad_i = grad[:, i]
        # Get the second derivative of the i-th dimension
        sec_grad = torch.autograd.grad(grad_i, state, retain_graph=True, grad_outputs=torch.ones_like(grad_i))[0]
        laplacian += sec_grad[:, i]
        
    return laplacian.detach()

# --- 3. The Monitor Experiment ---
def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_env = gym.make("Reacher-v5")
    env = GlitchedReacher(base_env)
    
    # Simple Value Network with SiLU for smooth 2nd derivatives
    v_net = nn.Sequential(
        nn.Linear(10, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    # Load your trained weights from Step 1.2 here later
    # For now, let's simulate a 'hacking' state vs 'normal' state
    
    print("Simulating State Analysis...")
    normal_state = torch.randn(1, 10).to(device)
    hacking_state = torch.zeros(1, 10).to(device) # Near the glitch
    
    # We 'force' a needle into the weights for demonstration
    # In the real hackathon, the agent learns this needle.
    
    l_normal = calculate_laplacian(v_net, normal_state)
    print(f"Normal Curvature (Laplacian): {l_normal.item():.4f}")
    
    # Visualization setup
    states = torch.linspace(-1, 1, 100).to(device)
    curvatures = []
    
    for s in states:
        mock_state = torch.zeros(1, 10).to(device)
        mock_state[0, 8] = s # Sweep across one dimension (distance to target)
        curvatures.append(calculate_laplacian(v_net, mock_state).item())
        
    plt.plot(states.cpu(), curvatures)
    plt.title("Value Function Curvature ($\Delta V$) Scan")
    plt.xlabel("State dimension")
    plt.ylabel("Laplacian")
    plt.savefig("curvature_scan.png")
    print("Saved 'curvature_scan.png'. Look for spikes!")

if __name__ == "__main__":
    run_experiment()