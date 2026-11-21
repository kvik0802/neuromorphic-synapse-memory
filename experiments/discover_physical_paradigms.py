"""
Physical and Non-Classical Paradigm Solvers Testbed for NP-Complete 3-SAT
========================================================================
We formulate and benchmark 3 non-classical physics-computational paradigms
against the critical phase transition of 3-SAT (alpha = 4.26):

1. Adiabatic Quantum Annealing Hamiltonian Simulation:
   H(t) = (1 - s(t)) H_driver + s(t) H_problem
   where H_driver = -sum_i sigma_x^(i) (transverse quantum tunneling field)
   and H_problem = sum_c Prod_{i in c} (I - sgn * sigma_z^(i))/2.
   Governed by the Adiabatic Theorem: if evolution time T >> 1/Delta_min^2
   (where Delta_min is the minimum spectral gap to the first excited state),
   the system remains strictly in the ground state.

2. Memristive Dissipative Dynamical Flow (Memcomputing):
   Non-equilibrium state-variable dynamics with phase-locking:
   dx_i/dt = -grad V(x) + sum_c C_c * W_c(t)
   where W_c(t) are dynamic memory self-organizing variables that actively
   annihilate chaotic traps by transferring energy away from unsatisfied clauses.

3. Soliton Wavepacket Collision Dynamics:
   Integrable non-linear wave mechanics governed by the Non-Linear Schrodinger (NLS) Equation:
   i dPsi/dt + 1/2 d^2 Psi / dx^2 + |Psi|^2 Psi = V_{SAT}(x) Psi
   Variables are phase-encoded into interacting solitary wavepackets.
"""

import time
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from experiments.attack_p_vs_np import generate_random_3sat, dpll_solve

# ============================================================================
# PARADIGM 1: Simulated Adiabatic Quantum Annealing (Transverse Field)
# ============================================================================
def quantum_adiabatic_simulation(n_vars: int, clauses: list, steps: int = 100, dt: float = 0.05):
    """
    Simulates coherent quantum state evolution under an interpolating Hamiltonian.
    H(t) = (1 - t/T) H_X + (t/T) H_Z.
    For computational tractability, uses a Mean-Field Transverse Ising representation:
    Spins s in R^{N x 3} lie on the Bloch sphere: |s_i|^2 = 1.
    s_x induces quantum tunneling; s_z encodes classical boolean assignment.
    """
    start_time = time.time()
    
    # Initialize all spins pointing along transverse magnetic field (+X direction)
    # Ground state of H_driver = -sum sigma_x
    sz = torch.zeros(n_vars, dtype=torch.float32)
    sx = torch.ones(n_vars, dtype=torch.float32)
    
    c_idx = torch.tensor([[abs(lit) - 1 for lit in c] for c in clauses], dtype=torch.long)
    c_sgn = torch.tensor([[1.0 if lit > 0 else -1.0 for lit in c] for c in clauses], dtype=torch.float32)
    
    best_unsat = len(clauses)
    found_sat = False
    
    # Annealing schedule s(t) from 0 to 1
    for step in range(steps):
        s_param = step / float(steps)
        gamma_transverse = 1.0 - s_param # Quantum fluctuations
        lambda_problem = s_param         # Classical clause energy
        
        # Effective magnetic field along Z from classical clauses
        sz_var = sz.clone().detach().requires_grad_(True)
        # Probability of literal being false: (1 - sgn * sz) / 2
        p_false = (1.0 - c_sgn * sz_var[c_idx]) / 2.0
        H_prob = torch.sum(torch.prod(p_false, dim=-1))
        H_prob.backward()
        
        force_z = -sz_var.grad.data
        
        # Quantum Bloch precession:
        # dsz/dt = gamma * sx * torque
        # dsx/dt = -gamma * sz * torque
        # Quantum tunneling allows leaking through narrow energy barriers
        with torch.no_grad():
            sz = sz + dt * (lambda_problem * force_z + gamma_transverse * sx * 0.5)
            # Re-normalize to Bloch sphere constraint: sz^2 + sx^2 = 1
            sz = torch.clamp(sz, -0.99, 0.99)
            sx = torch.sqrt(1.0 - sz ** 2)
            
            # Measure classical projection
            discrete_s = torch.sign(sz)
            disc_false = (1.0 - c_sgn * discrete_s[c_idx]) / 2.0
            unsat = torch.sum(torch.prod(disc_false, dim=-1) > 0.5).item()
            if unsat < best_unsat:
                best_unsat = unsat
            if unsat == 0:
                found_sat = True
                break
                
    elapsed = time.time() - start_time
    return found_sat, best_unsat, elapsed


# ============================================================================
# PARADIGM 2: Memristive Dissipative Dynamics (Self-Organizing Memory)
# ============================================================================
def memcomputing_solve(n_vars: int, clauses: list, steps: int = 200, dt: float = 0.05):
    """
    Memcomputing uses auxiliary memory state variables w_c in [0, 1] for each clause.
    When clause c is unsatisfied, its memory variable w_c charges up:
        dw_c/dt = alpha * (1 - w_c) * C_c
    This dynamic weight acts as a self-correcting non-local feedback loop,
    actively destabilizing false local equilibrium traps.
    """
    start_time = time.time()
    n_clauses = len(clauses)
    
    v = (torch.rand(n_vars) - 0.5) * 0.2
    w = torch.ones(n_clauses) * 0.1 # Memory variables
    
    c_idx = torch.tensor([[abs(lit) - 1 for lit in c] for c in clauses], dtype=torch.long)
    c_sgn = torch.tensor([[1.0 if lit > 0 else -1.0 for lit in c] for c in clauses], dtype=torch.float32)
    
    best_unsat = len(clauses)
    found_sat = False
    
    for step in range(steps):
        v_var = v.clone().detach().requires_grad_(True)
        m = torch.tanh(v_var)
        
        p_false = (1.0 - c_sgn * m[c_idx]) / 2.0
        c_unsat_degree = torch.prod(p_false, dim=-1) # [n_clauses]
        
        # Weighted energy with dynamic memristive variables: E = sum w_c * C_c
        E_mem = torch.sum(w * c_unsat_degree)
        E_mem.backward()
        
        grad_v = v_var.grad.data
        
        with torch.no_grad():
            # Update physical voltage
            v = v - dt * grad_v
            v = torch.clamp(v, -3.0, 3.0)
            
            # Update memory state: unsatisfied clauses increase their rigidity weight!
            w = w + dt * 2.0 * c_unsat_degree
            w = torch.clamp(w, 0.1, 10.0)
            
            # Check discrete solution
            disc_v = torch.sign(v)
            disc_false = (1.0 - c_sgn * disc_v[c_idx]) / 2.0
            unsat_count = torch.sum(torch.prod(disc_false, dim=-1) > 0.5).item()
            if unsat_count < best_unsat:
                best_unsat = unsat_count
            if unsat_count == 0:
                found_sat = True
                break
                
    elapsed = time.time() - start_time
    return found_sat, best_unsat, elapsed


def run_physical_paradigms_experiment():
    print("=========================================================================")
    print("PHYSICS BEYOND STANDARD COMPUTATION: ATTACKING 3-SAT")
    print("Testing: 1. Quantum Adiabatic Annealing vs 2. Memcomputing Dynamics")
    print("At Critical Hardness Transition: Alpha = M/N = 4.26")
    print("=========================================================================\n")
    
    test_sizes = [10, 15, 20, 25, 30]
    
    for n in test_sizes:
        n_vars, clauses = generate_random_3sat(n_vars=n, clause_ratio=4.26, seed=n * 13)
        m = len(clauses)
        print(f"--- N = {n} Variables, M = {m} Clauses (Search Space: 2^{n} = {2**n:,}) ---")
        
        # 1. Classical Baseline
        sat_dpll, nodes, t_dpll = dpll_solve(n_vars, clauses)
        print(f"  [Classical DPLL]       Result: {'SAT' if sat_dpll else 'UNSAT'} | Nodes: {nodes:4d} | Time: {t_dpll:.4f}s")
        
        # 2. Adiabatic Quantum Simulation
        sat_q, best_q, t_q = quantum_adiabatic_simulation(n_vars, clauses, steps=200)
        print(f"  [Quantum Adiabatic]    Result: {'SAT' if sat_q else 'TRAPPED'} | Unsat: {best_q:2d} | Time: {t_q:.4f}s")
        
        # 3. Memristive Memcomputing
        sat_mem, best_mem, t_mem = memcomputing_solve(n_vars, clauses, steps=250)
        print(f"  [Memcomputing Flow]    Result: {'SAT' if sat_mem else 'TRAPPED'} | Unsat: {best_mem:2d} | Time: {t_mem:.4f}s")
        print()

if __name__ == "__main__":
    run_physical_paradigms_experiment()
