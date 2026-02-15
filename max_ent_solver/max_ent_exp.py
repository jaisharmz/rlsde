import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib

# Use a non-interactive backend to prevent pop-ups
matplotlib.use('Agg')

def max_ent_distribution(constraints, x_range):
    history = []

    def objective(p):
        p = np.clip(p, 1e-12, None)
        history.append(p.copy())
        return np.sum(p * np.log(p))

    cons = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1}]
    for func, target in constraints:
        cons.append({'type': 'eq', 'fun': lambda p, f=func, t=target: np.sum(p * f(x_range)) - t})

    init_p = np.ones(len(x_range)) / len(x_range)
    bounds = [(0, 1) for _ in x_range]

    # Optimization
    res = minimize(objective, init_p, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 100})
    return res.x, history

if __name__ == "__main__":
    # --- SETUP FOR SHAKE SHACK NYC ---
    # x represents wait time in minutes (0 to 90 minutes)
    x = np.linspace(0.1, 90, 150) 
    
    # Constraints based on NYC Shack Data:
    # 1. Mean wait time is ~25 minutes
    # 2. Log-mean constraint to model the "Queueing Effect" (heavy tail for peak hours)
    mean_wait = 25.0
    queue_structure = np.log(18.0) 

    moments = [
        (lambda val: val, mean_wait),           # Constraint: Average Wait
        (lambda val: np.log(val), queue_structure) # Constraint: Variance/Queue behavior
    ]

    print("Simulating Shake Shack NYC Wait Times...")
    optimal_p, history = max_ent_distribution(moments, x)

    # --- Animation / Video Generation ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # Shake Shack Brand Colors: Green and Black/Grey
    line, = ax.plot([], [], color='#5da93c', lw=3, label='Wait Time Probability')
    
    ax.set_xlim(0, 90)
    ax.set_ylim(0, np.max(optimal_p) * 1.4)
    ax.set_title("NYC Shake Shack: Probability of Wait Duration", fontsize=14, fontweight='bold')
    ax.set_xlabel("Minutes (Line + Kitchen Prep)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

    writer = FFMpegWriter(fps=20)

    print("Generating distribution_evolution_shack.mp4...")
    with writer.saving(fig, "distribution_evolution_shack.mp4", dpi=100):
        step = max(1, len(history) // 60) 
        last_fill = None 
        
        for i in range(0, len(history), step):
            line.set_data(x, history[i])
            if last_fill is not None:
                last_fill.remove()
            last_fill = ax.fill_between(x, history[i], alpha=0.2, color='#5da93c')
            writer.grab_frame()
        
        # Static frames for the final result
        for _ in range(25):
            writer.grab_frame()

    print("Done! The peak of the curve shows your most likely wait time.")