import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import matplotlib

# Use a non-interactive backend
matplotlib.use('Agg')

class AssumptionNode:
    def __init__(self, name, observations=None):
        self.name = name
        self.observations = observations 
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

def max_ent_solver(observations, x_range):
    """Core MaxEnt solver for a single source of infection."""
    obs_mean = np.mean(observations)
    obs_log_mean = np.mean(np.log(np.array(observations) + 1e-6))

    def objective(p):
        p_safe = np.clip(p, 1e-12, None)
        return np.sum(p_safe * np.log(p_safe))

    cons = [
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
        {'type': 'eq', 'fun': lambda p: np.sum(p * x_range) - obs_mean},
        {'type': 'eq', 'fun': lambda p: np.sum(p * np.log(x_range + 1e-6)) - obs_log_mean}
    ]

    init_p = np.ones(len(x_range)) / len(x_range)
    res = minimize(objective, init_p, method='SLSQP', bounds=[(0, 1)]*len(x_range), constraints=cons)
    return res.x if res.success else init_p

def process_tree_convolution(node, x_range):
    """
    Recursively convolves distributions.
    Logic: Total Spread = Spread(Source A) + Spread(Source B)
    """
    if node.is_leaf():
        print(f"Solving Leaf: {node.name}")
        return max_ent_solver(node.observations, x_range)
    
    # Initialize with a Delta Function (Identity for convolution)
    # This represents 0 infections with 100% probability
    combined_dist = np.zeros(len(x_range))
    combined_dist[0] = 1.0
    
    for child in node.children:
        child_dist = process_tree_convolution(child, x_range)
        
        # Convolve the current combined distribution with the new source
        # We use mode='full' to capture the additive nature of the x-axis
        combined_dist = fftconvolve(combined_dist, child_dist, mode='full')
        
        # Slicing: Convolution grows the array. To keep it aligned with x_range:
        # We take the first len(x_range) elements because infections add up.
        combined_dist = combined_dist[:len(x_range)]
        
        # Re-normalize to ensure total probability = 1
        combined_dist /= (np.sum(combined_dist) + 1e-12)
        
    return combined_dist

if __name__ == "__main__":
    # x represents the number of newly infected students (0 to 100)
    x = np.linspace(0, 100, 200)

    # --- Constructing the Additive Tree ---
    # We assume these events happen concurrently/sequentially in one day
    root = AssumptionNode("Daily School Total")

    # Source 1: Classroom spread (Small, consistent)
    root.add_child(AssumptionNode("Classroom Exposure", observations=[2, 4, 3, 5]))

    # Source 2: Cafeteria spread (Larger, more variable)
    root.add_child(AssumptionNode("Lunchroom Exposure", observations=[10, 15, 12, 20]))

    # Source 3: After-school sports (High contact)
    root.add_child(AssumptionNode("Sports/Locker Room", observations=[5, 10, 8, 15]))

    print("Running Convolved MaxEnt Tree...")
    final_distribution = process_tree_convolution(root, x)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, final_distribution, color='#1f77b4', lw=3, label='Convolved Total Spread')
    ax.fill_between(x, final_distribution, color='#1f77b4', alpha=0.3)
    
    # Identify the Mode
    peak_x = x[np.argmax(final_distribution)]
    ax.axvline(peak_x, color='red', linestyle='--', label=f'Total Predicted Peak: ~{int(peak_x)}')

    ax.set_title("Flu Spread: Convolved Assumption Model (Additive Risk)", fontsize=14)
    ax.set_xlabel("Total New Infections per Day")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.savefig("flu_convolved_results.png")
    print(f"\nDone. Peak of combined additive risk: {peak_x:.1f} students.")