"""
Spectral Graph Rigidity Testbed for 3-SAT Phase Transitions
============================================================
New Mathematical Angle:
Instead of trying to 'guess' assignments via gradient descent (which hits non-convex traps),
we map 3-SAT into an algebraic object: The Signed Clause-Variable Bipartite Laplacian L_B.

Definition:
Let B in {-1, 0, +1}^{M x N} be the clause-variable incidence matrix:
    B_{c, v} = +1 if variable v appears positive in clause c
    B_{c, v} = -1 if variable v appears negated in clause c
    B_{c, v} =  0 if variable v does not appear in clause c

The Signed Graph Laplacian is:
    L_B = D - B^T B in R^{N x N}
where D is the diagonal degree matrix.

The Spectral Rigidity Invariant:
1. When a formula is easy / underconstrained (alpha << 4.26):
   The spectrum of L_B has a wide spectral gap: lambda_2 >> 0.
   Variables can fluctuate freely into satisfying configurations.
2. When a formula hits the NP-hard phase transition (alpha -> 4.26):
   The graph undergoes a Spectral Condensation.
   The algebraic connectivity lambda_2 collapses, and the density of states 
   rho(lambda) exhibits an extreme Wigner semi-circle rigidity.
3. When a formula becomes UNSAT:
   The ground state energy of L_B strictly exceeds the topological bound.

Can we predict satisfiability and locate the exponential wall purely from 
the eigenvalues of the graph Laplacian WITHOUT searching the 2^N tree?
"""

import time
import math
import random
import torch
import numpy as np
import scipy.linalg

def build_signed_bipartite_laplacian(n_vars: int, clauses: list):
    """
    Constructs the signed variable-clause interaction matrix B and its Laplacian L.
    B is [M, N].
    L = D - B^T B.
    """
    m = len(clauses)
    B = np.zeros((m, n_vars), dtype=np.float64)
    
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1.0 if lit > 0 else -1.0
            B[c_idx, var_idx] = sign
            
    # Variable correlation matrix: C = B^T B in R^{N x N}
    # C_{ij} counts how often variable i and j co-occur with matching/opposite signs
    C = np.dot(B.T, B)
    
    # Degree of each variable
    deg = np.diag(C)
    D = np.diag(deg)
    
    # Signed Laplacian
    L = D - C
    return B, C, L

def compute_spectral_features(L: np.ndarray):
    """
    Computes spectral graph properties:
    - Eigenvalues lambda_1 <= lambda_2 <= ... <= lambda_N
    - Spectral Gap: delta = lambda_2 - lambda_1
    - Spectral Energy (Trace of L^2)
    - von Neumann Entropy of normalized spectrum
    """
    eigvals = np.sort(scipy.linalg.eigvalsh(L))
    
    lambda_min = eigvals[0]
    lambda_2 = eigvals[1] if len(eigvals) > 1 else 0.0
    spectral_gap = lambda_2 - lambda_min
    
    # Normalize eigenvalues to probability distribution for spectral entropy
    pos_eigvals = np.maximum(eigvals - lambda_min, 1e-12)
    p = pos_eigvals / np.sum(pos_eigvals)
    spectral_entropy = -np.sum(p * np.log(p))
    
    spectral_energy = np.sum(eigvals ** 2)
    return eigvals, spectral_gap, spectral_entropy, spectral_energy
