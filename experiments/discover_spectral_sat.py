"""
Experiment: Detecting the NP-Hard Phase Transition via Spectral Graph Rigidity
Sweeping clause density ratio alpha from 1.0 (trivial SAT) to 6.0 (deep UNSAT)
through the critical point alpha = 4.267.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from experiments.attack_p_vs_np import generate_random_3sat, dpll_solve
from models.spectral_sat_rigidity import build_signed_bipartite_laplacian, compute_spectral_features

def run_spectral_phase_transition_experiment():
    print("=========================================================================")
    print("NEW MATHEMATICAL ATTACK: SPECTRAL GRAPH RIGIDITY OF 3-SAT")
    print("Hypothesis: Can the NP-hard phase transition and satisfiability be detected")
    print("algebraically in O(N^3) time from the Laplacian spectrum instead of 2^N search?")
    print("=========================================================================\n")
    
    n_vars = 28 # Search space: 2^28 = 268 Million states
    ratios = [1.5, 2.5, 3.5, 4.0, 4.26, 4.5, 5.0, 6.0]
    
    print(f"Testing N = {n_vars} variables across clause density ratios alpha in [1.5 .. 6.0]:\n")
    print(f"{'Ratio (M/N)':<12} | {'M Clauses':<10} | {'True Status':<12} | {'DPLL Time':<10} | {'Spectral Gap':<14} | {'Spectral Entropy':<16}")
    print("-" * 85)
    
    for alpha in ratios:
        n, clauses = generate_random_3sat(n_vars=n_vars, clause_ratio=alpha, seed=int(alpha * 100))
        m = len(clauses)
        
        # 1. Exact DPLL Ground Truth
        sat, nodes, elapsed = dpll_solve(n, clauses)
        status = "SAT" if sat else "UNSAT"
        
        # 2. Algebraic Spectral Graph Properties in O(N^3)
        B, C, L = build_signed_bipartite_laplacian(n, clauses)
        eigvals, gap, entropy, energy = compute_spectral_features(L)
        
        print(f"{alpha:<12.2f} | {m:<10d} | {status:<12} | {elapsed:<10.4f}s | {gap:<14.4f} | {entropy:<16.4f}")
        
    print("\nObservation:")
    print("Notice how the spectral properties behave as the formula crosses the critical threshold alpha = 4.26.")

if __name__ == "__main__":
    run_spectral_phase_transition_experiment()
