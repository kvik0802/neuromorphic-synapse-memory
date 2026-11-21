"""
Non-Archimedean p-Adic Ultrametric Resonator (NA-UR)
===================================================
Stepping OUTSIDE Standard Euclidean & Riemannian Physics.

Every equation in modern physics (Newton, Maxwell, Einstein, Schrödinger, Dirac) 
relies on the ARCHIMEDEAN property of the real numbers R:
    For any x, y > 0, there exists an integer n such that n * x > y.
This is defined by the standard Euclidean absolute value |x| = sqrt(x^2).

The Non-Archimedean Alternative (Hensel 1897, Volovich 1987 p-Adic String Theory):
By Ostrowski's Theorem (1916), there are ONLY TWO equivalent classes of absolute 
values on the field of rational numbers Q:
1. The standard Archimedean absolute value |x|_infinity.
2. The p-Adic absolute value |x|_p for every prime number p.

The Strong Ultrametric Triangle Inequality:
    |x + y|_p <= max( |x|_p, |y|_p )  (Strictly stronger than |x| + |y|!)

Consequences for Computation & NP-Hardness:
In an Ultrametric space (the field of p-Adic numbers Q_p):
1. EVERY TRIANGLE IS ISOSCELES.
2. Every point inside an open ball is ITS OWN CENTER!
3. Two balls in Q_p are either STRICTLY DISJOINT or ONE CONTAINS THE OTHER.
   There is ZERO PARTIAL OVERLAP!

Why This Destroys the Continuous Physics Trap:
In Euclidean R^N, continuous energy surfaces have partial overlaps, creating false 
saddle traps and non-convex valleys.
In p-Adic Ultrametric space, because spheres cannot partially intersect:
- Phase space is a TREETOPOLOGICAL FRACTAL (Cantor set Q_p).
- The p-Adic Vladmirov pseudo-differential operator:
      D^alpha f(x) = 1 / Gamma_p(-alpha) * int_{Q_p} |x - y|_p^{-1 - alpha} (f(y) - f(x)) dy
  induces a non-local hierarchical diffusion that tunnels INSTANTANEOUSLY between 
  hierarchical branches without crossing physical energy barriers!

Let's implement a 2-Adic and 3-Adic Ultrametric Phase-Space Solver for 3-SAT.
"""

import time
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from experiments.attack_p_vs_np import generate_random_3sat, dpll_solve

def p_adic_valuation(n: int, p: int = 2) -> int:
    """Computes the p-adic valuation v_p(n): highest power of p dividing n."""
    if n == 0:
        return 100 # Infinity proxy
    v = 0
    n = abs(n)
    while n % p == 0:
        v += 1
        n //= p
    return v

def p_adic_norm(n: int, p: int = 2) -> float:
    """|n|_p = p^{-v_p(n)}"""
    if n == 0:
        return 0.0
    v = p_adic_valuation(n, p)
    return float(p ** (-v))

def padic_ultrametric_solve(n_vars: int, clauses: list, p: int = 2, max_flips: int = 1000):
    """
    Ultrametric Hierarchical Walk:
    Maps 3-SAT variable states to a p-Adic integer: Z = sum_{i=0}^{N-1} s_i * p^i.
    The ultrametric distance between two configurations s and s' is:
        d_p(s, s') = | Z(s) - Z(s') |_p = p^{-k}
    where k is the lowest index where s and s' differ!
    """
    start_time = time.time()
    
    # Random initial assignment s in {0, 1}^N
    s = np.random.randint(0, 2, size=n_vars)
    
    best_unsat = len(clauses)
    found_sat = False
    
    # Pre-parse clauses
    c_indices = [[abs(lit) - 1 for lit in c] for c in clauses]
    c_polarities = [[(lit > 0) for lit in c] for c in clauses]
    
    for step in range(max_flips):
        # Count unsatisfied clauses
        unsat_clauses = []
        for c_idx in range(len(clauses)):
            vars_in_c = c_indices[c_idx]
            pols = c_polarities[c_idx]
            
            # Satisfied if at least one literal is satisfied
            sat = any(s[v] == p_val for v, p_val in zip(vars_in_c, pols))
            if not sat:
                unsat_clauses.append(c_idx)
                
        num_unsat = len(unsat_clauses)
        if num_unsat < best_unsat:
            best_unsat = num_unsat
        if num_unsat == 0:
            found_sat = True
            break
            
        # Non-Archimedean Ultrametric Choice:
        # Instead of flipping variables based on Euclidean gradient,
        # we evaluate the p-Adic valuation shift Delta v_p across candidates:
        target_clause = unsat_clauses[step % len(unsat_clauses)]
        candidate_vars = c_indices[target_clause]
        
        # Choose the variable that minimizes the p-Adic ultrametric distance:
        # d_p = p^{-v}
        best_v = candidate_vars[0]
        min_p_dist = 1e9
        
        for v in candidate_vars:
            # Flip v hypothetically
            s[v] ^= 1
            # Compute new unsat count
            new_unsat = 0
            for c_i in range(len(clauses)):
                v_in_c = c_indices[c_i]
                p_in_c = c_polarities[c_i]
                if not any(s[vi] == pi for vi, pi in zip(v_in_c, p_in_c)):
                    new_unsat += 1
            s[v] ^= 1 # Revert
            
            # Ultrametric metric combines p-Adic depth with valuation jump
            # Valuation jump: v_p(var_index + 1)
            v_jump = p_adic_valuation(v + 1, p)
            # Ultrametric cost: hierarchical branch transition
            cost = new_unsat * (p ** (-v_jump))
            
            if cost < min_p_dist:
                min_p_dist = cost
                best_v = v
                
        # Perform ultrametric transition
        s[best_v] ^= 1
        
    elapsed = time.time() - start_time
    return found_sat, best_unsat, elapsed


def run_padic_ultrametric_experiment():
    print("=========================================================================")
    print("BEYOND ARCHIMEDEAN PHYSICS: p-ADIC ULTRAMETRIC RESONANCE FOR 3-SAT")
    print("Replacing Euclidean Geometry with Non-Archimedean Hensel Fields (Q_p)")
    print("Testing 2-Adic and 3-Adic Hierarchical Phase Walks at Alpha = 4.26")
    print("=========================================================================\n")
    
    test_sizes = [15, 20, 25, 30, 35, 40, 50]
    
    for n in test_sizes:
        n_vars, clauses = generate_random_3sat(n_vars=n, clause_ratio=4.26, seed=n * 37)
        m = len(clauses)
        print(f"--- N = {n} Variables, M = {m} Clauses (Search Space: 2^{n} = {2**n:,}) ---")
        
        # 1. Classical DPLL
        sat_dpll, nodes, t_dpll = dpll_solve(n_vars, clauses)
        print(f"  [Classical DPLL]             Result: {'SAT' if sat_dpll else 'UNSAT'} | Nodes: {nodes:6d} | Time: {t_dpll:.4f}s")
        
        # 2. 2-Adic Ultrametric Walk
        sat_p2, best_p2, t_p2 = padic_ultrametric_solve(n_vars, clauses, p=2, max_flips=1500)
        print(f"  [2-Adic Ultrametric Walk]    Result: {'SAT' if sat_p2 else 'TRAPPED'} | Unsat: {best_p2:2d} | Time: {t_p2:.4f}s")
        
        # 3. 3-Adic Ultrametric Walk
        sat_p3, best_p3, t_p3 = padic_ultrametric_solve(n_vars, clauses, p=3, max_flips=1500)
        print(f"  [3-Adic Ultrametric Walk]    Result: {'SAT' if sat_p3 else 'TRAPPED'} | Unsat: {best_p3:2d} | Time: {t_p3:.4f}s")
        print()

if __name__ == "__main__":
    run_padic_ultrametric_experiment()
