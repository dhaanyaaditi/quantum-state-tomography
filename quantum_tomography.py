"""
Quantum State Tomography & Noise Simulation of a Two-Level System
==================================================================
Author: K D Dhaanya Aditi | IIT Delhi, Engineering Physics

Description:
    Simulates a driven two-level quantum system (qubit) undergoing Rabi oscillations
    and adiabatic evolution. Implements quantum state tomography to reconstruct the
    density matrix under depolarising and amplitude-damping noise channels.
    Benchmarks numerical results against analytical perturbation theory.

References:
    - Englert, B-G: Lectures on Quantum Mechanics (LOQM) — Perturbed Evolution
    - Schwinger, J: Quantum Mechanics — Symbolism of Atomic Measurements
    - Suprit Singh: Quantum Mechanics Course Notes, IIT Delhi (2025-26)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pennylane as qml

# ─────────────────────────────────────────────
# 1. CONSTANTS & PAULI MATRICES
# ─────────────────────────────────────────────

# Pauli matrices
I  = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [I, sx, sy, sz]
PAULI_LABELS = ["I", "X", "Y", "Z"]


# ─────────────────────────────────────────────
# 2. RABI OSCILLATIONS — ANALYTICAL
# ─────────────────────────────────────────────

def rabi_analytical(t_arr, omega0, omega_drive, Omega_R):
    """
    Analytical solution for Rabi oscillations in a driven two-level system.
    Uses rotating wave approximation (RWA).

    H = (omega0/2) * sz + Omega_R * (cos(omega_drive * t)) * sx

    Returns: population of excited state |1> over time
    """
    delta = omega_drive - omega0          # detuning
    Omega_eff = np.sqrt(Omega_R**2 + (delta/2)**2)  # effective Rabi frequency

    P_excited = (Omega_R / Omega_eff)**2 * np.sin(Omega_eff * t_arr)**2
    return P_excited


# ─────────────────────────────────────────────
# 3. RABI OSCILLATIONS — NUMERICAL (PennyLane)
# ─────────────────────────────────────────────

dev = qml.device("default.mixed", wires=1)

def rabi_numerical(t_arr, omega0, Omega_R):
    """
    Numerically simulate Rabi oscillations using PennyLane.
    Exact evolution under H = (omega0/2)*Z + Omega_R*X (resonance: delta=0)
    Returns excited state population over time.
    """
    populations = []

    for t in t_arr:
        @qml.qnode(dev)
        def circuit():
            # Time evolution under H = (omega0/2)*Z + Omega_R*X
            H = qml.Hamiltonian(
                [omega0 / 2, Omega_R],
                [qml.PauliZ(0), qml.PauliX(0)]
            )
            qml.evolve(H)([], t)
            return qml.expval(qml.PauliZ(0))

        exp_z = circuit()
        P1 = (1 - exp_z) / 2   # population of |1>
        populations.append(float(P1))

    return np.array(populations)


# ─────────────────────────────────────────────
# 4. NOISE CHANNELS
# ─────────────────────────────────────────────

def apply_depolarising(rho, p):
    """
    Depolarising channel: rho -> (1-p)*rho + (p/4)*(I*Tr(rho) + rho + X*rho*X + Y*rho*Y + Z*rho*Z)
    Simplified: rho -> (1 - 4p/3)*rho + (4p/3)*(I/2)
    p in [0, 1]: noise strength
    """
    return (1 - p) * rho + p * (np.eye(2) / 2)


def apply_amplitude_damping(rho, gamma):
    """
    Amplitude damping channel — models energy dissipation (T1 decay).
    Kraus operators:
        K0 = [[1, 0], [0, sqrt(1-gamma)]]
        K1 = [[0, sqrt(gamma)], [0, 0]]
    gamma in [0, 1]: damping parameter
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


# ─────────────────────────────────────────────
# 5. QUANTUM STATE TOMOGRAPHY
# ─────────────────────────────────────────────

def tomography_expectations(rho):
    """
    Compute expectation values of Pauli operators for a given density matrix.
    <sigma_i> = Tr(sigma_i * rho)
    """
    return {label: np.real(np.trace(P @ rho))
            for P, label in zip(PAULIS[1:], PAULI_LABELS[1:])}


def reconstruct_density_matrix(exp_vals):
    """
    Reconstruct density matrix from Pauli expectation values.
    rho = (I + <X>*X + <Y>*Y + <Z>*Z) / 2
    """
    rho = 0.5 * (I
                 + exp_vals["X"] * sx
                 + exp_vals["Y"] * sy
                 + exp_vals["Z"] * sz)
    return rho


def fidelity(rho1, rho2):
    """
    Fidelity between two density matrices: F = (Tr(sqrt(sqrt(rho1)*rho2*sqrt(rho1))))^2
    For pure states: F = Tr(rho1 @ rho2)
    """
    return np.real(np.trace(rho1 @ rho2))


# ─────────────────────────────────────────────
# 6. PERTURBATION THEORY BENCHMARK
# ─────────────────────────────────────────────

def first_order_perturbation(t, omega0, Omega_R):
    """
    First-order time-dependent perturbation theory (Born approximation).
    H = H0 + V where H0 = (omega0/2)*Z, V = Omega_R*X

    Transition probability |<1|U(t)|0>|^2 to first order:
    P(0->1) ≈ (Omega_R/omega0)^2 * sin^2(omega0*t/2)

    Valid when Omega_R << omega0 (weak driving).
    """
    return (Omega_R / omega0)**2 * np.sin(omega0 * t / 2)**2


# ─────────────────────────────────────────────
# 7. SIMULATE & PLOT
# ─────────────────────────────────────────────

def run_simulation():
    # Parameters
    omega0    = 2 * np.pi * 1.0   # qubit frequency (rad/s)
    Omega_R   = 2 * np.pi * 0.25  # Rabi frequency (resonant drive)
    omega_d   = omega0             # resonant driving
    t_arr     = np.linspace(0, 4, 200)

    print("=" * 55)
    print("  Quantum State Tomography & Noise Simulation")
    print("=" * 55)

    # ── 7.1 Rabi Oscillations ──
    P_analytical = rabi_analytical(t_arr, omega0, omega_d, Omega_R)
    P_perturb    = first_order_perturbation(t_arr, omega0, Omega_R)
    print("\n[1/4] Rabi oscillations computed (analytical + perturbation theory)")

    # ── 7.2 Density matrix at t = pi/(2*Omega_R) — superposition state ──
    t_half_pi = np.pi / (2 * Omega_R / (2 * np.pi))
    # State after pi/2 pulse: |+> = (|0> + |1>)/sqrt(2)
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_pure = np.outer(psi, psi.conj())

    # ── 7.3 Apply noise channels ──
    noise_levels = np.linspace(0, 0.5, 50)
    fidelity_depol  = []
    fidelity_ampdamp = []

    for p in noise_levels:
        rho_dep = apply_depolarising(rho_pure, p)
        rho_amp = apply_amplitude_damping(rho_pure, p)
        fidelity_depol.append(fidelity(rho_pure, rho_dep))
        fidelity_ampdamp.append(fidelity(rho_pure, rho_amp))

    print("[2/4] Noise channels applied (depolarising + amplitude damping)")

    # ── 7.4 Quantum State Tomography ──
    exp_vals_pure = tomography_expectations(rho_pure)
    rho_reconstructed = reconstruct_density_matrix(exp_vals_pure)
    tomo_fidelity = fidelity(rho_pure, rho_reconstructed)
    print(f"[3/4] Tomography reconstruction fidelity: {tomo_fidelity:.6f}")
    print(f"      Pauli expectations: X={exp_vals_pure['X']:.3f}, "
          f"Y={exp_vals_pure['Y']:.3f}, Z={exp_vals_pure['Z']:.3f}")

    # Noisy tomography (with depolarising noise p=0.2)
    rho_noisy = apply_depolarising(rho_pure, 0.2)
    exp_vals_noisy = tomography_expectations(rho_noisy)
    rho_recon_noisy = reconstruct_density_matrix(exp_vals_noisy)
    noisy_fidelity = fidelity(rho_pure, rho_recon_noisy)
    print(f"      Noisy tomography fidelity (p=0.2): {noisy_fidelity:.6f}")

    print("[4/4] Plotting results...")

    # ── 7.5 PLOTS ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Quantum State Tomography & Noise Simulation\nTwo-Level System",
                 fontsize=14, fontweight="bold")

    # Plot 1: Rabi oscillations
    ax = axes[0, 0]
    ax.plot(t_arr, P_analytical, 'b-', linewidth=2, label="Analytical (RWA)")
    ax.plot(t_arr, P_perturb, 'r--', linewidth=1.5, label="1st-order Perturbation Theory")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("P(excited)", fontsize=11)
    ax.set_title("Rabi Oscillations: Analytical vs Perturbation Theory", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Fidelity vs noise strength
    ax = axes[0, 1]
    ax.plot(noise_levels, fidelity_depol,  'purple', linewidth=2, label="Depolarising")
    ax.plot(noise_levels, fidelity_ampdamp, 'orange', linewidth=2, label="Amplitude Damping")
    ax.set_xlabel("Noise Parameter", fontsize=11)
    ax.set_ylabel("Fidelity F(ρ_pure, ρ_noisy)", fontsize=11)
    ax.set_title("Fidelity Decay Under Noise Channels", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Bloch sphere projection (Pauli expectations vs noise)
    ax = axes[1, 0]
    exp_x, exp_y, exp_z = [], [], []
    for p in noise_levels:
        rho_n = apply_depolarising(rho_pure, p)
        ev = tomography_expectations(rho_n)
        exp_x.append(ev["X"])
        exp_y.append(ev["Y"])
        exp_z.append(ev["Z"])
    ax.plot(noise_levels, exp_x, label="⟨X⟩", linewidth=2)
    ax.plot(noise_levels, exp_y, label="⟨Y⟩", linewidth=2)
    ax.plot(noise_levels, exp_z, label="⟨Z⟩", linewidth=2)
    ax.set_xlabel("Depolarising Noise p", fontsize=11)
    ax.set_ylabel("Expectation Value", fontsize=11)
    ax.set_title("Pauli Expectation Values vs Depolarising Noise\n(Tomography Observables)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Density matrix heatmap (pure vs noisy)
    ax = axes[1, 1]
    rho_display = np.abs(apply_depolarising(rho_pure, 0.3))
    im = ax.imshow(rho_display, cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["|0⟩", "|1⟩"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["|0⟩", "|1⟩"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{rho_display[i,j]:.3f}", ha="center", va="center",
                    fontsize=13, color="black")
    ax.set_title("|ρ| after Depolarising Noise (p=0.3)", fontsize=11)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to results.png")

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Rabi period (resonant):    T = {1/0.25:.2f} s")
    print(f"  Tomography fidelity:       {tomo_fidelity:.6f}")
    print(f"  Noisy tomo fidelity (p=0.2): {noisy_fidelity:.6f}")
    print(f"  Fidelity loss (p=0.2 depol): {tomo_fidelity - noisy_fidelity:.6f}")
    print("=" * 55)


if __name__ == "__main__":
    run_simulation()
