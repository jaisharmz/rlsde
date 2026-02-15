import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib

# Use a non-interactive backend
matplotlib.use('Agg')

def max_ent_distribution(observations, x_range):
    history = []
    
    # 1. Extract constraints from raw public data
    obs_mean = np.mean(observations)
    obs_log_mean = np.mean(np.log(observations))
    
    print(f"--- Public Data Analysis ---")
    print(f"Observed Mean Wait: {obs_mean:.2f}m")
    print(f"Observed Log-Mean: {obs_log_mean:.2f}")
    print(f"---------------------------\n")

    def objective(p):
        # We clip p to prevent log(0) issues during optimization
        p_safe = np.clip(p, 1e-12, None)
        # Minimize Negative Entropy (which maximizes Entropy)
        return np.sum(p_safe * np.log(p_safe))

    def callback(p):
        # FIXED: Use np.where to handle zeros without changing the shape of the array
        p_safe = np.clip(p, 1e-12, None)
        curr_entropy = -np.sum(p_safe * np.log(p_safe))
        history.append(p.copy())
        print(f"Iteration {len(history)}: Current Entropy = {curr_entropy:.5f}")

    # Constraints: 
    # The solver tries to find p such that these return 0
    cons = [
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
        {'type': 'eq', 'fun': lambda p: np.sum(p * x_range) - obs_mean},
        {'type': 'eq', 'fun': lambda p: np.sum(p * np.log(x_range)) - obs_log_mean}
    ]

    # Start with a Uniform Distribution (Absolute uncertainty)
    init_p = np.ones(len(x_range)) / len(x_range)
    bounds = [(0, 1) for _ in x_range]

    print("Starting Optimization (MaxEnt Solver)...")
    res = minimize(objective, init_p, method='SLSQP', bounds=bounds, 
                   constraints=cons, callback=callback, options={'maxiter': 100})
    
    if not res.success:
        print(f"Warning: Optimization didn't converge. Reason: {res.message}")
    
    return res.x, history

if __name__ == "__main__":
    # x represents wait time in minutes
    x = np.linspace(0.1, 90, 200)

    # PUBLIC DATA: Raw observations from check-ins
    public_observations = [12, 15, 8, 45, 22, 10, 11, 60, 35, 32] 

    optimal_p, history = max_ent_distribution(public_observations, x)

    # --- Animation ---
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], color='#5da93c', lw=3, label='Evolving MaxEnt Curve')
    
    # Visualizing the raw data points as small ticks on the x-axis
    ax.plot(public_observations, np.zeros_like(public_observations), 'k|', ms=20, label='Raw Observations')

    ax.set_xlim(0, 90)
    ax.set_ylim(0, np.max(optimal_p) * 1.5)
    ax.set_title("Agnostic MaxEnt Model: Shake Shack NYC Wait Times", fontweight='bold')
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Probability Density")
    ax.legend()

    writer = FFMpegWriter(fps=15)
    print("\nGenerating video: agnostic_shack_wait.mp4...")
    
    # We'll store the fill object so we can clear it in each frame
    current_fill = [None]

    with writer.saving(fig, "agnostic_shack_wait.mp4", dpi=100):
        for i in range(len(history)):
            line.set_data(x, history[i])
            
            # Remove previous fill
            if current_fill[0] is not None:
                current_fill[0].remove()
            
            current_fill[0] = ax.fill_between(x, history[i], alpha=0.3, color='#5da93c')
            writer.grab_frame()

    print(f"Success! Final peak is at {x[np.argmax(optimal_p)]:.1f} minutes.")