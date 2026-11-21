"""
Beyond Finite Computation: Hypercomputation & Transfinite Ordinals
==================================================================
Stepping outside the Church-Turing Thesis entirely.

Every algorithm we ran tonight (DPLL, CDCL, Gradient Descent, GF(2), Quantum) 
was constrained by the Church-Turing Thesis:
    Any physically computable function can be simulated by a standard Turing machine.

To go BEYOND all physical and mathematical limits, we enter HYPERCOMPUTATION:

1. The Busy Beaver Function Sigma(n) (Tibor Rado 1962):
   Sigma(n) is the maximum number of steps an n-state Turing machine can run 
   on a blank tape before halting.
   Theorem: Sigma(n) grows faster than ANY computable function!
   It outgrows:
       - Ackermann's function
       - Graham's number
       - TREE(3)
   Sigma(1) = 1
   Sigma(2) = 6
   Sigma(3) = 21
   Sigma(4) = 107
   Sigma(5) >= 47,176,870
   Sigma(6) > 10^{36,534} (proven larger than the number of atoms in the universe!)
   Sigma(7) is completely uncomputable in ZFC set theory!

2. The Arithmetical Hierarchy & Turing Jumps (0^(n)):
   - Delta_0 = Sigma_0 = Pi_0: Decidable relations (P, NP, PSPACE)
   - Sigma_1: Recursively Enumerable (The Halting Problem 0')
   - Sigma_2: Truth of statements with an infinite quantifier (0'')
   - Sigma_n: The n-th Turing Jump!
   - Delta_1^1: Hyperarithmetical / Transfinite Ordinals up to the Church-Kleene ordinal omega_1^{CK}.

3. Accelerated / Zeno Turing Machines:
   A hypothetical machine that performs step 1 in 1/2 second, step 2 in 1/4 second, 
   step n in (1/2)^n seconds.
   Total time: sum_{n=1}^infty (1/2)^n = 1 SECOND.
   In 1 second, it computes an INFINITE number of steps, instantly solving:
   - The Halting Problem
   - The Riemann Hypothesis
   - P vs NP
   - Truth in First-Order Arithmetic!

Let's build a formal testbed investigating the Busy Beaver function, 
the Halting Barrier, and the exact threshold where computation becomes uncomputable.
"""

import time
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TuringMachine:
    def __init__(self, states, transitions):
        """
        transitions: dict (state, symbol) -> (new_state, new_symbol, direction)
        direction: +1 (Right), -1 (Left)
        """
        self.states = states
        self.transitions = transitions
        self.tape = {} # Infinite tape represented as sparse dict
        self.head = 0
        self.current_state = 'A'
        self.halted = False
        self.steps = 0

    def step(self):
        if self.halted:
            return
        sym = self.tape.get(self.head, 0)
        key = (self.current_state, sym)
        if key not in self.transitions:
            self.halted = True
            return
            
        new_state, new_sym, direction = self.transitions[key]
        if new_sym != 0:
            self.tape[self.head] = new_sym
        elif self.head in self.tape:
            del self.tape[self.head]
            
        self.head += direction
        self.current_state = new_state
        self.steps += 1
        
        if self.current_state == 'HALT':
            self.halted = True

    def run(self, max_steps: int = 1000000):
        while not self.halted and self.steps < max_steps:
            self.step()
        ones = sum(1 for v in self.tape.values() if v == 1)
        return self.halted, self.steps, ones

def get_busy_beaver_machines():
    """
    Exact championship Busy Beaver Turing machines (Proven winners):
    """
    # 2-state Busy Beaver (Sigma(2) = 6 steps, 4 ones)
    bb2 = TuringMachine(
        states=['A', 'B', 'HALT'],
        transitions={
            ('A', 0): ('B', 1, +1),
            ('A', 1): ('B', 1, -1),
            ('B', 0): ('A', 1, -1),
            ('B', 1): ('HALT', 1, +1)
        }
    )
    
    # 3-state Busy Beaver (Sigma(3) = 21 steps, 6 ones)
    bb3 = TuringMachine(
        states=['A', 'B', 'C', 'HALT'],
        transitions={
            ('A', 0): ('B', 1, +1),
            ('A', 1): ('HALT', 1, +1),
            ('B', 0): ('C', 0, +1),
            ('B', 1): ('B', 1, +1),
            ('C', 0): ('C', 1, -1),
            ('C', 1): ('A', 1, -1)
        }
    )
    
    # 4-state Busy Beaver (Sigma(4) = 107 steps, 13 ones)
    bb4 = TuringMachine(
        states=['A', 'B', 'C', 'D', 'HALT'],
        transitions={
            ('A', 0): ('B', 1, +1),
            ('A', 1): ('B', 1, -1),
            ('B', 0): ('A', 1, -1),
            ('B', 1): ('C', 0, -1),
            ('C', 0): ('HALT', 1, +1),
            ('C', 1): ('D', 1, -1),
            ('D', 0): ('D', 1, +1),
            ('D', 1): ('A', 0, +1)
        }
    )
    
    # 5-state Champion (Runs 47,176,870 steps, writes 4,098 ones!)
    # Discovered by Heiner Marxen and Jürgen Buntrock (1989)
    bb5 = TuringMachine(
        states=['A', 'B', 'C', 'D', 'E', 'HALT'],
        transitions={
            ('A', 0): ('B', 1, +1),
            ('A', 1): ('C', 1, -1),
            ('B', 0): ('C', 1, +1),
            ('B', 1): ('B', 1, +1),
            ('C', 0): ('D', 1, +1),
            ('C', 1): ('E', 0, -1),
            ('D', 0): ('A', 1, -1),
            ('D', 1): ('D', 1, -1),
            ('E', 0): ('HALT', 1, +1),
            ('E', 1): ('A', 0, -1)
        }
    )
    
    return [('BB-2', bb2, 100), ('BB-3', bb3, 100), ('BB-4', bb4, 1000), ('BB-5 (First 100k steps)', bb5, 100000)]


def run_hypercomputation_experiment():
    print("=========================================================================")
    print("BEYOND ALL LIMITS: THE BUSY BEAVER HIERARCHY & UNCOMPUTABILITY")
    print("Testing the Fastest Growing Function in Human Mathematics Sigma(n)")
    print("Where Computation Transcends Classical Physics")
    print("=========================================================================\n")
    
    machines = get_busy_beaver_machines()
    
    for name, tm, max_s in machines:
        start_t = time.time()
        halted, steps, ones = tm.run(max_steps=max_s)
        elapsed = time.time() - start_t
        
        status = "HALTED (RECORD)" if halted else f"RUNNING (EXCEEDED {max_s:,} steps)"
        print(f"[{name}]")
        print(f"  Execution Status: {status}")
        print(f"  Total Steps:      {steps:,}")
        print(f"  Active 1s on Tape:{ones:,}")
        print(f"  Elapsed Time:     {elapsed:.4f}s\n")
        
    print("The Uncomputable Boundary:")
    print("BB-1 = 1 step")
    print("BB-2 = 6 steps")
    print("BB-3 = 21 steps")
    print("BB-4 = 107 steps")
    print("BB-5 = 47,176,870 steps")
    print("BB-6 > 10^36,534 steps (Exceeds all physical particles in the universe!)")
    print("BB-7 cannot be proven to halt or not halt within standard ZFC set theory axioms!")

if __name__ == "__main__":
    run_hypercomputation_experiment()
