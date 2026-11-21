"""
Conflict-Driven Clause Learning (CDCL) with Implication Graph Analysis
======================================================================
The modern frontier of Boolean Satisfiability (SAT).

Why CDCL succeeds where continuous physics fails:
1. Implication Graph (DAG):
   Tracks exact causal dependencies of unit propagation.
2. First Unique Implication Point (1UIP) Resolution:
   When a conflict occurs (a clause evaluates to FALSE), CDCL traverses the 
   implication graph backward to the First UIP and derives an asserting conflict clause.
3. Non-Chronological Backjumping:
   Instead of rewinding step-by-step (chronological backtracking), CDCL jumps 
   across thousands of decision levels directly to the root cause of the conflict.
4. Exponential Space Pruning:
   Adding a single learned 1UIP clause of length k prunes 2^{N - k} invalid 
   configurations from the remaining search space permanently!
"""

import time
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.attack_p_vs_np import generate_random_3sat

class CDCLSolver:
    def __init__(self, n_vars: int, clauses: list):
        self.n_vars = n_vars
        self.clauses = [list(c) for c in clauses]
        self.assignment = {}      # var -> bool
        self.decision_level = {}  # var -> level
        self.antecedent = {}      # var -> clause that implied it
        self.current_level = 0
        self.learned_clauses = 0
        self.backjumps = 0

    def get_unassigned_literal(self, clause):
        unassigned = []
        for lit in clause:
            var = abs(lit)
            if var not in self.assignment:
                unassigned.append(lit)
            else:
                val = self.assignment[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    return True, None # Clause satisfied
        return False, unassigned

    def unit_propagate(self):
        """Propagates implications and detects conflicts."""
        while True:
            propagated = False
            for clause in self.clauses:
                sat, unassigned = self.get_unassigned_literal(clause)
                if sat:
                    continue
                if len(unassigned) == 0:
                    return clause # Conflict clause found!
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    var = abs(lit)
                    val = (lit > 0)
                    self.assignment[var] = val
                    self.decision_level[var] = self.current_level
                    self.antecedent[var] = clause
                    propagated = True
                    break
            if not propagated:
                return None # No conflict, quiet

    def analyze_conflict(self, conflict_clause):
        """
        Derives an asserting learned clause using 1UIP (Unique Implication Point) resolution.
        """
        self.learned_clauses += 1
        curr_level_lits = [lit for lit in conflict_clause if self.decision_level.get(abs(lit), 0) == self.current_level]
        learned = list(conflict_clause)
        
        # Resolve backwards until exactly one literal from current level remains
        while len(curr_level_lits) > 1:
            # Pick a literal with an antecedent
            lit_to_resolve = None
            for lit in curr_level_lits:
                var = abs(lit)
                if self.antecedent.get(var) is not None:
                    lit_to_resolve = lit
                    break
            if lit_to_resolve is None:
                break
                
            var = abs(lit_to_resolve)
            ante = self.antecedent[var]
            
            # Resolve learned clause with antecedent clause
            res = set(learned) | set(ante)
            res.discard(var)
            res.discard(-var)
            learned = list(res)
            
            curr_level_lits = [lit for lit in learned if self.decision_level.get(abs(lit), 0) == self.current_level]
            
        # Determine backtrack level (second highest decision level in learned clause)
        levels = [self.decision_level.get(abs(lit), 0) for lit in learned if self.decision_level.get(abs(lit), 0) != self.current_level]
        backtrack_level = max(levels) if levels else 0
        return learned, backtrack_level

    def backjump(self, target_level):
        """Unassigns variables back to target_level."""
        self.backjumps += 1
        to_remove = [v for v, lvl in self.decision_level.items() if lvl > target_level]
        for v in to_remove:
            del self.assignment[v]
            del self.decision_level[v]
            if v in self.antecedent:
                del self.antecedent[v]
        self.current_level = target_level

    def pick_branching_variable(self):
        """Simple dynamic heuristic: pick most frequent unassigned variable."""
        counts = {}
        for c in self.clauses:
            for lit in c:
                var = abs(lit)
                if var not in self.assignment:
                    counts[var] = counts.get(var, 0) + 1
        if not counts:
            return None, True
        best_var = max(counts, key=counts.get)
        return best_var, True

    def solve(self, max_conflicts: int = 50000):
        start_time = time.time()
        
        # Initial unit propagation
        conflict = self.unit_propagate()
        if conflict is not None:
            return False, time.time() - start_time
            
        while len(self.assignment) < self.n_vars:
            var, val = self.pick_branching_variable()
            if var is None:
                break
                
            self.current_level += 1
            self.assignment[var] = val
            self.decision_level[var] = self.current_level
            
            while True:
                conflict = self.unit_propagate()
                if conflict is None:
                    break # No conflict, continue branching
                    
                if self.current_level == 0:
                    # Conflict at root level -> Unconditional UNSAT proof!
                    return False, time.time() - start_time
                    
                learned_clause, backtrack_level = self.analyze_conflict(conflict)
                self.clauses.append(learned_clause)
                self.backjump(backtrack_level)
                
                # Unit propagate the newly learned asserting clause
                if len(learned_clause) == 1:
                    u_lit = learned_clause[0]
                    u_var = abs(u_lit)
                    self.assignment[u_var] = (u_lit > 0)
                    self.decision_level[u_var] = backtrack_level
                    self.antecedent[u_var] = learned_clause
                    
        return True, time.time() - start_time


def run_cdcl_frontier_benchmark():
    print("=========================================================================")
    print("THE REAL FRONTIER: CONFLICT-DRIVEN CLAUSE LEARNING (CDCL)")
    print("1UIP Implication Graph Resolution + Non-Chronological Backjumping")
    print("Testing against Phase-Transition 3-SAT (Alpha = 4.26)")
    print("=========================================================================\n")
    
    test_sizes = [20, 30, 40, 50, 60, 75, 100]
    
    for n in test_sizes:
        n_vars, clauses = generate_random_3sat(n_vars=n, clause_ratio=4.26, seed=n * 41)
        m = len(clauses)
        print(f"--- Problem Size: N = {n} Variables, M = {m} Clauses (Search Space: 2^{n} = {2**n:,}) ---")
        
        solver = CDCLSolver(n_vars, clauses)
        sat, elapsed = solver.solve()
        
        print(f"  Result: {'SAT' if sat else 'UNSAT'} | Elapsed: {elapsed:.4f}s | Learned Clauses: {solver.learned_clauses:4d} | Backjumps: {solver.backjumps:4d}")
        print()

if __name__ == "__main__":
    run_cdcl_frontier_benchmark()
