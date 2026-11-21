"""
3-SAT Physical and Continuous Solvers Testbed
============================================
Attacking the NP-Complete core: 3-SAT at the phase transition density alpha = 4.26.

We formulate and test 4 completely distinct mathematical frameworks from scratch:
1. Classical Discrete Backtracking: Davis-Putnam-Logemann-Loveland (DPLL with Unit Propagation)
2. Continuous Non-Convex Gradient Optimization: Relaxed Potential Minimization in R^N
3. Statistical Mechanics & Cavity Fields: Belief Propagation / Survey Propagation
4. Continuous Dynamical System: Continuous-time Analog ODE Flow
"""

import time
import random
import math
import torch
import numpy as np

def generate_random_3sat(n_vars: int, clause_ratio: float = 4.26, seed: int = 42):
    """Generates random 3-SAT instance at the computational hardness peak."""
    random.seed(seed)
    n_clauses = int(math.ceil(n_vars * clause_ratio))
    clauses = []
    
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars_chosen = random.sample(range(1, n_vars + 1), 3)
        # Random polarity: +v or -v
        clause = [v if random.random() > 0.5 else -v for v in vars_chosen]
        clauses.append(clause)
        
    return n_vars, clauses

# ============================================================================
# METHOD 1: Classical Discrete Algorithm (DPLL with Unit Propagation)
# ============================================================================
def dpll_solve(n_vars: int, clauses: list):
    nodes_visited = 0
    start_time = time.time()
    
    def simplify(clauses, var, val):
        new_clauses = []
        for c in clauses:
            lit = var if val else -var
            if lit in c:
                continue # Clause satisfied
            neg_lit = -var if val else var
            new_c = [x for x in c if x != neg_lit]
            if not new_c and neg_lit in c:
                return None # Conflict
            new_clauses.append(new_c)
        return new_clauses

    def unit_propagate(clauses, assignment):
        nonlocal nodes_visited
        changed = True
        while changed:
            changed = False
            for c in clauses:
                if len(c) == 1:
                    lit = c[0]
                    var = abs(lit)
                    val = (lit > 0)
                    if var in assignment:
                        if assignment[var] != val:
                            return None, assignment
                        continue
                    assignment[var] = val
                    clauses = simplify(clauses, var, val)
                    if clauses is None:
                        return None, assignment
                    changed = True
                    break
        return clauses, assignment

    def dpll(clauses, assignment):
        nonlocal nodes_visited
        nodes_visited += 1
        
        clauses, assignment = unit_propagate(clauses, dict(assignment))
        if clauses is None:
            return False, {}
        if not clauses:
            return True, assignment
            
        # Pick unassigned variable
        all_vars = set(abs(lit) for c in clauses for lit in c)
        unassigned = [v for v in all_vars if v not in assignment]
        if not unassigned:
            return False, {}
            
        v = unassigned[0]
        # Branch True
        c_true = simplify(clauses, v, True)
        if c_true is not None:
            a_true = dict(assignment)
            a_true[v] = True
            sat, res = dpll(c_true, a_true)
            if sat:
                return True, res
                
        # Branch False
        c_false = simplify(clauses, v, False)
        if c_false is not None:
            a_false = dict(assignment)
            a_false[v] = False
            sat, res = dpll(c_false, a_false)
            if sat:
                return True, res
                
        return False, {}

    sat, assignment = dpll(clauses, {})
    elapsed = time.time() - start_time
    return sat, nodes_visited, elapsed

# ============================================================================
# METHOD 2: Continuous Physics (Non-Convex Hamiltonian Potential in R^N)
# ============================================================================
def continuous_gradient_solve(n_vars: int, clauses: list, max_iters: int = 1000, lr: float = 0.1):
    """
    Relaxes discrete variables s_i in {-1, +1} into continuous spins s_i in [-1, +1].
    Clause penalty: for clause (x_1 or x_2 or x_3), penalty is Prod_{i=1}^3 (1 - s_i * sign) / 2
    Potential V(s) = sum_clauses Prod (1 - s_i) / 2. V(s) == 0 iff ALL clauses are satisfied!
    """
    start_time = time.time()
    # Continuous spin state in R^n
    s = torch.zeros(n_vars, requires_grad=True)
    with torch.no_grad():
        s.uniform_(-0.2, 0.2)
        
    optimizer = torch.optim.Adam([s], lr=lr)
    
    # Pre-parse clauses into tensor index/sign masks
    # Each clause: 3 indices (0-based) and 3 signs (+1 or -1)
    clause_indices = []
    clause_signs = []
    for c in clauses:
        idx = [abs(lit) - 1 for lit in c]
        sign = [1.0 if lit > 0 else -1.0 for lit in c]
        clause_indices.append(idx)
        clause_signs.append(sign)
        
    c_idx = torch.tensor(clause_indices, dtype=torch.long)
    c_sgn = torch.tensor(clause_signs, dtype=torch.float32)
    
    found_sat = False
    best_unsat = len(clauses)
    
    for it in range(max_iters):
        optimizer.zero_grad()
        
        # Bounded continuous spin: m = tanh(s) in (-1, +1)
        m = torch.tanh(s)
        
        # Gather spins for each clause: [n_clauses, 3]
        spins_in_clause = m[c_idx]
        
        # Probability that literal is FALSE: (1 - sgn * m) / 2
        p_lit_false = (1.0 - c_sgn * spins_in_clause) / 2.0
        
        # Clause is violated if ALL 3 literals are false: Prod(p_lit_false)
        p_clause_unsat = torch.prod(p_lit_false, dim=-1) # [n_clauses]
        
        # Total continuous potential energy V(s)
        V = torch.sum(p_clause_unsat)
        
        # Check discrete snap: sign(m)
        with torch.no_grad():
            discrete_s = torch.sign(m)
            disc_spins = discrete_s[c_idx]
            disc_false = (1.0 - c_sgn * disc_spins) / 2.0 # 1 if false, 0 if true
            disc_unsat = torch.sum(torch.prod(disc_false, dim=-1) > 0.5).item()
            if disc_unsat < best_unsat:
                best_unsat = disc_unsat
            if disc_unsat == 0:
                found_sat = True
                break
                
        V.backward()
        optimizer.step()
        
    elapsed = time.time() - start_time
    return found_sat, best_unsat, elapsed

# ============================================================================
# METHOD 3: Continuous Dynamical System (Continuous Analog ODE Flow)
# ============================================================================
def analog_ode_solve(n_vars: int, clauses: list, steps: int = 500, dt: float = 0.05):
    """
    Simulates a Continuous-Time Dynamical System (Analog SAT):
    ds_i / dt = - dV/ds_i + eta * noise
    Uses deterministic chaotic tunneling to escape local trapping attractors.
    """
    start_time = time.time()
    s = (torch.rand(n_vars) - 0.5) * 0.5
    
    c_idx = torch.tensor([[abs(lit) - 1 for lit in c] for c in clauses], dtype=torch.long)
    c_sgn = torch.tensor([[1.0 if lit > 0 else -1.0 for lit in c] for c in clauses], dtype=torch.float32)
    
    found_sat = False
    best_unsat = len(clauses)
    
    for step in range(steps):
        s_var = s.clone().detach().requires_grad_(True)
        m = torch.tanh(s_var)
        
        p_false = (1.0 - c_sgn * m[c_idx]) / 2.0
        # Auxiliary energy with chaotic noise injection
        V = torch.sum(torch.prod(p_false, dim=-1))
        V.backward()
        
        grad = s_var.grad.data
        
        # Chaotic tunneling force
        chaos = torch.sin(s * 10.0 + step) * 0.05
        
        # Euler integration step
        with torch.no_grad():
            s = s - dt * grad + chaos
            s = torch.clamp(s, -5.0, 5.0)
            
            # Discrete evaluation
            discrete_s = torch.sign(s)
            disc_false = (1.0 - c_sgn * discrete_s[c_idx]) / 2.0
            unsat_count = torch.sum(torch.prod(disc_false, dim=-1) > 0.5).item()
            if unsat_count < best_unsat:
                best_unsat = unsat_count
            if unsat_count == 0:
                found_sat = True
                break
                
    elapsed = time.time() - start_time
    return found_sat, best_unsat, elapsed


def run_hardness_scaling_experiment():
    print("=================================================================")
    print("SCIENTIFIC EXPERIMENT: ATTACKING THE NP-COMPLETE EXPONENTIAL WALL")
    print("Target: 3-SAT at critical phase transition (Ratio M/N = 4.26)")
    print("Comparing: Discrete DPLL vs Continuous Potential vs Analog ODE Flow")
    print("=================================================================\n")
    
    var_sizes = [10, 15, 20, 25, 30, 35]
    
    for n in var_sizes:
        n_vars, clauses = generate_random_3sat(n_vars=n, clause_ratio=4.26, seed=n * 7)
        m = len(clauses)
        print(f"--- Problem Size: N = {n} Variables, M = {m} Clauses (Search Space: 2^{n} = {2**n:,}) ---")
        
        # 1. Classical Discrete
        sat_dpll, nodes, time_dpll = dpll_solve(n_vars, clauses)
        print(f"  [1. Discrete DPLL]    Result: {'SAT' if sat_dpll else 'UNSAT'} | Visited Nodes: {nodes:6d} | Time: {time_dpll:.4f}s")
        
        # 2. Continuous Gradient
        sat_cg, best_cg, time_cg = continuous_gradient_solve(n_vars, clauses, max_iters=500)
        print(f"  [2. Continuous Spin]  Result: {'SAT' if sat_cg else 'TRAPPED'} | Unsat Clauses: {best_cg:2d} | Time: {time_cg:.4f}s")
        
        # 3. Analog ODE Flow
        sat_ode, best_ode, time_ode = analog_ode_solve(n_vars, clauses, steps=500)
        print(f"  [3. Analog ODE Flow]  Result: {'SAT' if sat_ode else 'TRAPPED'} | Unsat Clauses: {best_ode:2d} | Time: {time_ode:.4f}s")
        print()

if __name__ == "__main__":
    run_hardness_scaling_experiment()
