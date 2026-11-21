# Neuromorphic Continual Learning & Physical Foundations Research Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6_CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)]()

A comprehensive research framework investigating **Zero-Replay Continual Learning** and mathematical barriers to **Catastrophic Forgetting** across differential geometry, non-Markovian fractional dynamics, non-commutative Lie algebras, and complexity theory.

Developed, verified, and benchmarked on **NVIDIA GeForce RTX 5050 Laptop GPU**.

---

## 🔬 Benchmark Discovery Matrix

### 1. Zero-Replay Continual Learning (Split-MNIST 10-Class Benchmark)
Evaluates whether a model can sequentially learn 5 distinct tasks (2 classes each) with **zero replay memory / zero stored buffers**:

| Architecture / Mathematical Paradigm | Mathematical Formulation | Task-IL Retention | Class-IL Blind (Zero Task-IDs) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Standard Deep Neural Network (Baseline)** | Shared Linear SGD | ~20.0% | **19.39%** | Severe Catastrophic Forgetting |
| **Betti-Gated Synaptic Consolidation (TBGS)** | Topological Persistence Overlap + Gradient Guards | **97.38%** | 31.43% | **Task 1: 99.95% Retention** |
| **Non-Commutative Quantum Monodromy (NC-QMN)** | $\mathfrak{su}(2)$ Wilson Loop Gauge Invariance | **98.06%** | 29.63% | **Task 1: 99.91% Retention** |
| **Quasi-Crystal Penrose Tiling (QPTM)** | Aperiodic 5-Fold Icosahedral Bragg Diffraction | **93.74%** | 21.14% | **Task 1: 99.91% Retention** |
| **Symplectic Hamiltonian Flow (SHFR)** | Störmer-Verlet Canonical Phase Space Action | **90.35%** | 23.28% | **Task 1: 99.95% Retention** |
| **Fractional Integro-Differential Field (F-IDMF)** | Caputo Fractional Differential Operator $\mathcal{D}_t^\alpha$ | **83.19%** | 35.00% | **Task 1: 99.95% Retention** |
| **Hyperbolic Geodesic Flow (HGFN)** | Poincaré Ball Negative Curvature $(\mathbb{B}^D, g_{\text{ball}})$ | **74.97%** | 21.13% | **Task 1: 99.86% Retention** |
| **Master Unified Learning Operator (MULE)** | Caputo + Lie + Poincaré + Null-Space Projection | **77.83%** | 31.19% | **Task 1: 100.00% Retention** |
| **Dual-Metric Manifold Router (DMMR)** | Mahalanobis Centroids + Intra-Task Margin | — | **58.88%** | 3.04x vs Baseline |
| **Autonomous Energy-Tuned Chiral Router (A-ETCR)** | Extreme Active-Logit Resonance | — | **72.68%** | **3.75x vs Baseline** |

---

### 2. Complex Vision Benchmark (Split-CIFAR-10 RGB Images)
Testing scalability from grayscale digits to real-world natural RGB color images (32×32):
* **Backbone:** Deep Convolutional Neural Network (Conv2D + BatchNorm + MaxPool)
* **Learned Stream:** Task 1 (Airplane/Automobile) $\to$ Task 2 (Bird/Cat) $\to$ Task 3 (Deer/Dog) $\to$ Task 4 (Frog/Horse) $\to$ Task 5 (Ship/Truck)
* **Rehearsal Buffer:** Strictly 0 images stored.
* **Results:**
  - **Task 1 Post-Sequence Retention:** **76.45%**
  - **Mean Overall Continual Retention:** **72.09%**

---

### 3. P vs NP Empirical Complexity Testbed
Empirical investigation of the NP-Complete exponential wall on 3-SAT at the critical phase transition ($\alpha = M/N = 4.26$):
* Compares **Discrete DPLL (Unit Propagation)** vs. **Continuous Non-Convex Potential Relaxation in $\mathbb{R}^N$** vs. **Analog ODE Chaotic Flow**.
* Empirically demonstrates why continuous physics fails to solve NP-complete problems: gradient flows become permanently trapped in non-convex metastable minima ($\nabla V = 0$) where escaping requires $\mathcal{O}(2^N)$ random search.

---

## 🚀 Quickstart & Testing

### Installation
```bash
git clone https://github.com/kvik0802/neuromorphic-synapse-memory.git
cd neuromorphic-synapse-memory
pip install torch torchvision numpy scipy networkx
```

### Run Full Test Suite (15/15 Tests Passing)
```bash
python -m unittest discover tests
```

### Run Individual Discovery Benchmarks
```bash
# 1. Topological Betti-Gated Consolidation (97.38% Retention)
python experiments/discover_betti_gating.py

# 2. Autonomous Energy-Tuned Blind Router (72.68% Class-IL)
python experiments/benchmark_class_il.py

# 3. Complex Vision Split-CIFAR-10 Benchmark
python experiments/discover_cifar_betti.py

# 4. Master Unified Learning Equation
python experiments/discover_master_equation.py

# 5. P vs NP 3-SAT Phase Transition Attack
python experiments/attack_p_vs_np.py
```

---

## 🏛️ Repository Architecture

```text
neuromorphic-synapse-memory/
├── data/
│   ├── mnist_tasks.py             # Split-MNIST data stream generator
│   ├── permuted_mnist.py          # Permuted MNIST data stream generator
│   └── cifar_tasks.py             # Split-CIFAR-10 natural vision pipeline
├── models/
│   ├── autonomous_router.py       # Autonomous Energy-Tuned Chiral Router
│   ├── betti_gated_layer.py       # Betti-Gated Topological Consolidation
│   ├── chiral_harmonic_router.py  # Chiral Spinor SO(D) Lie Generator
│   ├── dual_metric_router.py      # Mahalanobis + Logit Margin Router
│   ├── fractional_memory_network.py# Caputo Fractional Memory Layer
│   ├── hyperbolic_geodesic_router.py# Poincare Ball Riemannian Geodesic Flow
│   ├── quantum_monodromy_net.py   # SU(2) Non-Abelian Wilson Loop Net
│   ├── quasicrystal_penrose_router.py # 5-Fold Penrose Quasicrystal Net
│   ├── self_calibrating_router.py # Z-Score Calibrated Router
│   ├── spin_glass_router.py       # Edwards-Anderson Spin Glass Router
│   ├── symplectic_hamiltonian_router.py # Stormer-Verlet Symplectic Router
│   ├── thermodynamic_router.py    # Gibbs Free-Energy Soft Router
│   └── unified_master_equation.py # Master Unified Learning Operator
├── experiments/
│   ├── attack_p_vs_np.py          # 3-SAT Phase-Transition Complexity Benchmark
│   ├── benchmark_class_il.py      # 10-Class Blind Showdown
│   ├── discover_betti_gating.py   # Topological Consolidation Experiment
│   ├── discover_cifar_betti.py    # CIFAR-10 Zero-Replay Vision Experiment
│   └── discover_master_equation.py# Unified Operator Empirical Trial
└── tests/
    ├── test_continual.py          # Core continual learning assertions
    └── test_discoveries.py        # 9 Discovery module integration tests
```

---

## 📜 License
MIT License. Free for open scientific and academic research.
