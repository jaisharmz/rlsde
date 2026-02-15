import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib

# Use a non-interactive backend
matplotlib.use('Agg')

class AssumptionNode:
    def __init__(self, name, probability=1.0, observations=None):
        self.name = name
        self.probability = probability 
        self.observations = observations 
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

def max_ent_solver(observations, x_range):
    """Solves for the most unbiased distribution given specific node observations."""
    obs_mean = np.mean(observations)
    obs_log_mean = np.mean(np.log(observations))

    def objective(p):
        p_safe = np.clip(p, 1e-12, None)
        return np.sum(p_safe * np.log(p_safe))

    cons = [
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
        {'type': 'eq', 'fun': lambda p: np.sum(p * x_range) - obs_mean},
        {'type': 'eq', 'fun': lambda p: np.sum(p * np.log(x_range)) - obs_log_mean}
    ]

    init_p = np.ones(len(x_range)) / len(x_range)
    res = minimize(objective, init_p, method='SLSQP', bounds=[(0, 1)]*len(x_range), constraints=cons)
    return res.x if res.success else init_p

def process_tree(node, x_range, current_prob=1.0):
    """Recursively traverses the tree and returns the weighted distribution."""
    combined_prob = current_prob * node.probability
    
    if node.is_leaf():
        print(f"Leaf Node Found: {node.name} | Weight: {combined_prob:.2%}")
        leaf_dist = max_ent_solver(node.observations, x_range)
        return leaf_dist * combined_prob
    else:
        total_dist = np.zeros_like(x_range)
        for child in node.children:
            total_dist += process_tree(child, x_range, combined_prob)
        return total_dist

if __name__ == "__main__":
    # x represents wait time in minutes
    x = np.linspace(0.1, 90, 200)

    # --- Constructing the Tree for Shake Shack NYC ---
    # ROOT: The general prompt
    root = AssumptionNode("Shake Shack Wait Time")

    # BRANCH 1: Off-Peak Hours (40% probability)
    off_peak = AssumptionNode("Off-Peak", probability=0.4)
    off_peak.add_child(AssumptionNode("Mid-day Weekday", probability=1.0, observations=[8, 12, 15, 10, 14]))

    # BRANCH 2: Peak Hours (60% probability)
    peak = AssumptionNode("Peak Hours", probability=0.6)
    
    # Sub-branching within Peak: Madison Square Park vs. Midtown
    # Midtown is busy but efficient; MSP (Park) has unpredictable tourist clusters
    peak.add_child(AssumptionNode("Midtown Efficiency", probability=0.5, observations=[25, 30, 22, 28]))
    peak.add_child(AssumptionNode("Madison Sq Park Surge", probability=0.5, observations=[45, 60, 55, 35, 70]))

    root.add_child(off_peak)
    root.add_child(peak)

    # --- Execute and Rejoin ---
    print("Executing Tree-Based Solver Logic...")
    final_distribution = process_tree(root, x)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, final_distribution, color='#5da93c', lw=3, label='Combined MaxEnt Distribution')
    ax.fill_between(x, final_distribution, color='#5da93c', alpha=0.3)
    
    ax.set_title("Shake Shack NYC: Tree-Weighted Wait Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Minutes (Line + Kitchen)")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # Calculate and mark the Mode (most likely wait)
    most_likely = x[np.argmax(final_distribution)]
    ax.axvline(most_likely, color='red', linestyle=':', label=f'Peak: {most_likely:.1f}m')
    ax.legend()

    plt.savefig("shack_tree_results.png")
    print(f"\nResults Saved. Overall most likely wait time: {most_likely:.1f} minutes.")