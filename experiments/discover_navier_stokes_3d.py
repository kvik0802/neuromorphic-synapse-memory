"""
3D Incompressible Navier-Stokes Solver: Vortex Stretching & Blowup Criterion
=============================================================================
Simulates 3D Navier-Stokes equations using Pseudo-Spectral methods on GPU (PyTorch/CUDA):

Governing Equation (Vorticity form):
    ∂ω/∂t + (u · ∇)ω = (ω · ∇)u + ν Δω

The Critical Difference from 2D:
    (ω · ∇)u is the VORTEX STRETCHING TERM.
    In 2D: (ω · ∇)u ≡ 0 (No blowup possible; proved by Ladyzhenskaya 1969).
    In 3D: (ω · ∇)u ≠ 0 (Vortex tubes can stretch infinitely; Millennium Prize Problem).

Beale-Kato-Majda (BKM) Blowup Criterion (1984):
    A smooth solution blows up at finite time T* IF AND ONLY IF:
        ∫_0^{T*} ||ω(·, t)||_L^∞ dt = ∞

This script benchmarks 3D vortex stretching dynamics under high Reynolds numbers
and tracks the BKM blowup accumulation integral.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PseudoSpectralNavierStokes3D:
    def __init__(self, N: int = 64, L: float = 2.0 * np.pi, nu: float = 1e-3, dt: float = 2e-3):
        """
        3D Grid resolution: N x N x N (64^3 = 262,144 points; each with 3 velocity & 3 vorticity components)
        Total state space: ~1.57 million floating-point variables per time step on GPU.
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.dt = dt

        dx = L / N
        k1d = torch.fft.fftfreq(N, d=dx / (2.0 * np.pi), device=device)
        self.Kx, self.Ky, self.Kz = torch.meshgrid(k1d, k1d, k1d, indexing='ij')
        
        self.K_sq = self.Kx**2 + self.Ky**2 + self.Kz**2
        self.K_sq[0, 0, 0] = 1.0  # Avoid division by zero at k = 0

        # De-aliasing Orszag 2/3-rule mask
        k_max = N / 3.0
        self.dealias_mask = (torch.abs(self.Kx) < k_max) & (torch.abs(self.Ky) < k_max) & (torch.abs(self.Kz) < k_max)

    def initialize_taylor_green_3d(self) -> torch.Tensor:
        """
        Standard 3D Taylor-Green vortex: The classic benchmark for 3D vortex stretching and turbulent cascade.
        u = sin(x) cos(y) cos(z)
        v = -cos(x) sin(y) cos(z)
        w = 0
        """
        dx = self.L / self.N
        coords = torch.linspace(0, self.L - dx, self.N, device=device)
        X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')

        u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
        v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
        w = torch.zeros_like(u)

        # Vorticity: ω = ∇ × u
        # ω_x = ∂w/∂y - ∂v/∂z = -cos(x) sin(y) sin(z)
        # ω_y = ∂u/∂z - ∂w/∂x = -sin(x) cos(y) sin(z)
        # ω_z = ∂v/∂x - ∂u/∂y = 2 cos(x) cos(y) cos(z)
        omega_x = -torch.cos(X) * torch.sin(Y) * torch.sin(Z)
        omega_y = -torch.sin(X) * torch.cos(Y) * torch.sin(Z)
        omega_z = 2.0 * torch.cos(X) * torch.cos(Y) * torch.cos(Z)

        omega = torch.stack([omega_x, omega_y, omega_z], dim=0)
        return omega

    def solve_velocity(self, omega_hat: torch.Tensor) -> torch.Tensor:
        """
        Biot-Savart law in Fourier space:
        u_hat = i (K × ω_hat) / |K|^2
        """
        # omega_hat: shape (3, N, N, N)
        wx_hat, wy_hat, wz_hat = omega_hat[0], omega_hat[1], omega_hat[2]

        ux_hat = 1j * (self.Ky * wz_hat - self.Kz * wy_hat) / self.K_sq
        uy_hat = 1j * (self.Kz * wx_hat - self.Kx * wz_hat) / self.K_sq
        uz_hat = 1j * (self.Kx * wy_hat - self.Ky * wx_hat) / self.K_sq

        ux_hat[0, 0, 0] = 0.0
        uy_hat[0, 0, 0] = 0.0
        uz_hat[0, 0, 0] = 0.0

        return torch.stack([ux_hat, uy_hat, uz_hat], dim=0)

    def compute_rhs(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Computes 3D Navier-Stokes RHS:
        RHS = -(u · ∇)ω + (ω · ∇)u + ν Δω
        """
        omega_hat = torch.fft.fftn(omega, dim=(-3, -2, -1)) * self.dealias_mask
        u_hat = self.solve_velocity(omega_hat) * self.dealias_mask

        # Transform to physical space
        u = torch.fft.ifftn(u_hat, dim=(-3, -2, -1)).real
        w = omega

        # Derivatives in Fourier space
        # ∂u_i / ∂x_j and ∂ω_i / ∂x_j
        K = [self.Kx, self.Ky, self.Kz]
        grad_u = []  # grad_u[i][j] = ∂u_i / ∂x_j
        for i in range(3):
            row = []
            for j in range(3):
                du_hat = 1j * K[j] * u_hat[i]
                row.append(torch.fft.ifftn(du_hat, dim=(-3, -2, -1)).real)
            grad_u.append(row)

        grad_w = []  # grad_w[i][j] = ∂ω_i / ∂x_j
        for i in range(3):
            row = []
            for j in range(3):
                dw_hat = 1j * K[j] * omega_hat[i]
                row.append(torch.fft.ifftn(dw_hat, dim=(-3, -2, -1)).real)
            grad_w.append(row)

        # 1. Advection term: (u · ∇)ω_i = sum_j u_j (∂ω_i / ∂x_j)
        advection = torch.zeros_like(omega)
        for i in range(3):
            for j in range(3):
                advection[i] += u[j] * grad_w[i][j]

        # 2. Vortex Stretching term: (ω · ∇)u_i = sum_j ω_j (∂u_i / ∂x_j)
        # THIS TERM IS ZERO IN 2D, BUT DOMINATES IN 3D!
        stretching = torch.zeros_like(omega)
        for i in range(3):
            for j in range(3):
                stretching[i] += w[j] * grad_u[i][j]

        # 3. Viscous Dissipation: ν Δω in Fourier space
        diff_hat = self.nu * self.K_sq * omega_hat
        diff = torch.fft.ifftn(diff_hat, dim=(-3, -2, -1)).real

        rhs = -advection + stretching - diff
        return rhs

    def step_rk4(self, omega: torch.Tensor) -> tuple[torch.Tensor, float]:
        """4th-order Runge-Kutta time integration"""
        dt = self.dt

        k1 = self.compute_rhs(omega)
        k2 = self.compute_rhs(omega + 0.5 * dt * k1)
        k3 = self.compute_rhs(omega + 0.5 * dt * k2)
        k4 = self.compute_rhs(omega + dt * k3)

        omega_next = omega + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        max_vorticity = torch.max(torch.sqrt(torch.sum(omega_next**2, dim=0))).item()

        return omega_next, max_vorticity


def run_3d_simulation(steps: int = 150):
    print("=" * 85)
    print("  3D NAVIER-STOKES VORTEX STRETCHING & BEALE-KATO-MAJDA BENCHMARK")
    print(f"  Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    N = 64
    nu = 1e-3  # Kinematic viscosity
    dt = 2e-3
    solver = PseudoSpectralNavierStokes3D(N=N, nu=nu, dt=dt)
    omega = solver.initialize_taylor_green_3d()

    initial_max_w = torch.max(torch.sqrt(torch.sum(omega**2, dim=0))).item()
    print(f"Grid: {N}x{N}x{N} ({N**3:,} cells, 6 vector components = {6*(N**3):,} continuous variables)")
    print(f"Initial Max Vorticity ||ω||_∞: {initial_max_w:.4f}")
    print(f"Beale-Kato-Majda (BKM) Integral Criterion: ∫_0^T ||ω(t)||_∞ dt (Divergence => Singularity)\n")

    print(f"{'Step':<8} | {'Sim Time (s)':<14} | {'Max ||ω||_∞':<16} | {'Vortex Growth (%)':<20} | {'BKM Integral':<16}")
    print("-" * 85)

    bkm_integral = 0.0
    start_time = time.time()

    for step in range(1, steps + 1):
        omega, max_w = solver.step_rk4(omega)
        bkm_integral += max_w * dt
        growth_pct = ((max_w - initial_max_w) / initial_max_w) * 100.0

        if step % 25 == 0 or step == steps:
            t = step * dt
            print(f"{step:<8} | {t:<14.3f} | {max_w:<16.4f} | {growth_pct:<+19.2f}% | {bkm_integral:<16.4f}")

    elapsed = time.time() - start_time
    print("=" * 85)
    print(f"3D Simulation completed: {steps} steps in {elapsed:.2f} seconds ({steps/elapsed:.1f} steps/s).")
    print(f"Peak Vorticity Growth due to 3D Vortex Stretching: {growth_pct:+.2f}%")
    print(f"Accumulated BKM Integral value: {bkm_integral:.4f}")
    print("=" * 85)


if __name__ == "__main__":
    run_3d_simulation()
