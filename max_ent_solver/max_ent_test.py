import numpy as np
import pytest
# Import the function from your file
from max_ent_exp import max_ent_distribution

def test_suite():
    print("\n--- Running MaxEnt Project Tests ---")
    
    # 1. Setup the domain (0 to 100 range)
    x = np.linspace(0, 100, 200)

    # --- TEST 1: The "Shake Shack" Mean Test ---
    target_mean = 25
    moments_mean = [(lambda val: val, target_mean)]
    
    p_exp = max_ent_distribution(moments_mean, x)
    
    calc_mean = np.sum(x * p_exp)
    sum_check = np.sum(p_exp)

    assert np.isclose(sum_check, 1.0, atol=1e-5), "Probabilities must sum to 1"
    assert np.isclose(calc_mean, target_mean, atol=0.1), f"Expected mean {target_mean}, got {calc_mean}"
    print("✅ Mean Constraint Test Passed.")

    # --- TEST 2: The "Bell Curve" (Gaussian) Test ---
    # According to MaxEnt, Mean + Variance = Gaussian distribution
    target_mu = 50
    target_var = 100
    target_ex2 = target_var + target_mu**2 # E[X^2] = Var + E[X]^2
    
    moments_gauss = [
        (lambda val: val, target_mu),
        (lambda val: val**2, target_ex2)
    ]
    
    p_gauss = max_ent_distribution(moments_gauss, x)
    
    # Check if the peak is near the mean (characteristic of a bell curve)
    peak_index = np.argmax(p_gauss)
    peak_value = x[peak_index]
    
    assert np.isclose(peak_value, target_mu, atol=2.0), "Peak of Gaussian should be at the mean"
    print("✅ Gaussian (Mean + Variance) Shape Test Passed.")

    # --- TEST 3: Uniform (Max Entropy) Test ---
    # With no constraints, the distribution should be flat
    p_uniform = max_ent_distribution([], x)
    
    # Check if the first and last values are almost identical
    assert np.isclose(p_uniform[0], p_uniform[-1], atol=1e-5), "Zero constraints must yield a flat line"
    print("✅ Uniform (Zero Info) Test Passed.")

    print("\n🎉 All core logic tests passed!")

if __name__ == "__main__":
    test_suite()