import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib

# Use a non-interactive backend
matplotlib.use('Agg')

def max_ent_spatiotemporal(observations, time_range, position_range):
    """
    MaxEnt solver for spatiotemporal probability density.
    observations: list of (time, position, count) tuples
    """
    history = []
    
    # Extract observations
    obs_times = np.array([obs[0] for obs in observations])
    obs_positions = np.array([obs[1] for obs in observations])
    obs_counts = np.array([obs[2] for obs in observations])
    
    # Calculate constraints from observations
    obs_mean_time = np.mean(obs_times)
    obs_mean_pos = np.mean(obs_positions)
    obs_var_time = np.var(obs_times)
    obs_var_pos = np.var(obs_positions)
    
    print(f"--- Spatiotemporal Data Analysis ---")
    print(f"Observed Mean Time: {obs_mean_time:.2f} min")
    print(f"Observed Mean Position: {obs_mean_pos:.2f} m")
    print(f"Observed Time Variance: {obs_var_time:.2f}")
    print(f"Observed Position Variance: {obs_var_pos:.2f}")
    print(f"Total Observations: {len(observations)}")
    print(f"------------------------------------\n")
    
    # Create meshgrid
    T, P = np.meshgrid(time_range, position_range)
    n_cells = T.size
    
    print(f"DEBUG: time_range has {len(time_range)} points")
    print(f"DEBUG: position_range has {len(position_range)} points")
    print(f"DEBUG: Total cells = {n_cells}\n")
    
    # Flatten for optimization
    T_flat = T.flatten()
    P_flat = P.flatten()
    
    def objective(p):
        # Minimize negative entropy (maximize entropy)
        p_safe = np.clip(p, 1e-12, None)
        return np.sum(p_safe * np.log(p_safe))
    
    def callback(p):
        p_safe = np.clip(p, 1e-12, None)
        curr_entropy = -np.sum(p_safe * np.log(p_safe))
        history.append(p.copy())
        print(f"Iteration {len(history)}: Current Entropy = {curr_entropy:.5f}")
    
    # Constraints for probability distribution
    cons = [
        # Normalization (must sum to 1 for probability)
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
        
        # Time-weighted mean
        {'type': 'eq', 'fun': lambda p: np.sum(p * T_flat) - obs_mean_time},
        
        # Position-weighted mean
        {'type': 'eq', 'fun': lambda p: np.sum(p * P_flat) - obs_mean_pos},
        
        # Time variance constraint
        {'type': 'eq', 'fun': lambda p: np.sum(p * (T_flat - obs_mean_time)**2) - obs_var_time},
        
        # Position variance constraint
        {'type': 'eq', 'fun': lambda p: np.sum(p * (P_flat - obs_mean_pos)**2) - obs_var_pos},
    ]
    
    # Initialize with uniform distribution
    init_p = np.ones(n_cells) / n_cells
    bounds = [(0, 1) for _ in range(n_cells)]
    
    print(f"Starting Optimization with {len(init_p)} variables...")
    res = minimize(objective, init_p, method='SLSQP', bounds=bounds,
                   constraints=cons, callback=callback, options={'maxiter': 150})
    
    if not res.success:
        print(f"Warning: Optimization didn't converge. Reason: {res.message}")
    
    # Reshape back to 2D grid
    optimal_density = res.x.reshape(T.shape)
    
    return optimal_density, history, T.shape

def create_contour_plot(time_range, position_range, density, observations):
    """Create contour plot showing probability density with organic style."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    obs_times = np.array([obs[0] for obs in observations])
    obs_positions = np.array([obs[1] for obs in observations])
    
    T, P = np.meshgrid(time_range, position_range)
    
    # Normalize axes to center around mean (like the reference plot)
    time_center = np.mean(time_range)
    pos_center = np.mean(position_range)
    time_scale = (np.max(time_range) - np.min(time_range)) / 4
    pos_scale = (np.max(position_range) - np.min(position_range)) / 4
    
    T_norm = (T - time_center) / time_scale
    P_norm = (P - pos_center) / pos_scale
    obs_times_norm = (obs_times - time_center) / time_scale
    obs_positions_norm = (obs_positions - pos_center) / pos_scale
    
    # Create smooth contour plot with viridis colormap
    levels = np.linspace(np.min(density), np.max(density), 20)
    contourf = ax.contourf(T_norm, P_norm, density, levels=levels, 
                           cmap='viridis', extend='both')
    
    # Add contour lines
    contour_lines = ax.contour(T_norm, P_norm, density, levels=10, 
                               colors='black', linewidths=1.5, alpha=0.6)
    
    # Plot observation points
    ax.scatter(obs_times_norm, obs_positions_norm, c='red', s=60, 
              marker='o', edgecolors='white', linewidths=1.5, zorder=5, alpha=0.8)
    
    ax.set_xlabel('Time (standardized)', fontsize=12)
    ax.set_ylabel('Position (standardized)', fontsize=12)
    ax.set_title('Bird Activity Probability Density - Central Park (8-10 AM)\nMaxEnt Spatiotemporal Model',
                fontweight='bold', fontsize=14)
    
    cbar = plt.colorbar(contourf, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Probability Density', fontsize=11)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bird_landing_contour_density.png', dpi=200, bbox_inches='tight')
    print("\nContour plot saved: bird_landing_contour_density.png")
    plt.close()

if __name__ == "__main__":
    # 10 × 10 = 100 variables
    time_range = np.linspace(0, 120, 10)
    position_range = np.linspace(0, 800, 10)
    
    observations = [
        (10, 150, 5),
        (15, 200, 8),
        (20, 250, 10),
        (25, 180, 12),
        (30, 400, 15),
        (35, 450, 18),
        (45, 450, 20),
        (50, 500, 18),
        (55, 520, 19),
        (60, 350, 22),
        (70, 580, 17),
        (75, 600, 16),
        (85, 580, 15),
        (90, 550, 14),
        (95, 400, 12),
        (100, 300, 10),
        (105, 280, 9),
        (110, 250, 7),
        (115, 200, 6),
    ]
    
    optimal_density, history, grid_shape = max_ent_spatiotemporal(
        observations, time_range, position_range
    )
    
    create_contour_plot(time_range, position_range, optimal_density, observations)
    
    max_idx = np.unravel_index(np.argmax(optimal_density), optimal_density.shape)
    peak_time = time_range[max_idx[1]]
    peak_position = position_range[max_idx[0]]
    peak_density = optimal_density[max_idx]
    
    print(f"\n=== Results ===")
    print(f"Peak probability density: {peak_density:.6f}")
    print(f"Time: {peak_time:.1f} min ({8 + peak_time/60:.2f} AM)")
    print(f"Position: {peak_position:.0f}m from south entrance")