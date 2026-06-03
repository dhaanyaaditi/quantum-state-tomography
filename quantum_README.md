# Quantum State Tomography & Noise Simulation of a Two-Level System

**Author:** K D Dhaanya Aditi | IIT Delhi, Engineering Physics  
**Tools:** Python · PennyLane · NumPy · SciPy · Matplotlib

---

## Overview

This project simulates a driven two-level quantum system (a qubit) and implements **quantum state tomography** to reconstruct its density matrix under realistic noise channels. Results are benchmarked against analytical solutions from time-dependent perturbation theory.

Grounded in coursework from IIT Delhi's Quantum Mechanics course and the *Lectures on Quantum Mechanics (LOQM)* trilogy by Berthold-Georg Englert.

---

## Physics Background

A two-level system (qubit) driven by an oscillating field is described by the Hamiltonian:

```
H = (ω₀/2) σ_z + Ω_R cos(ω_d t) σ_x
```

Under the **Rotating Wave Approximation (RWA)**, the system undergoes **Rabi oscillations** — periodic transfer of population between ground |0⟩ and excited |1⟩ states.

**Key concepts implemented:**
- Rabi oscillations (analytical + numerical)
- First-order time-dependent perturbation theory (Born approximation / Dyson expansion)
- Quantum state tomography via Pauli operator measurements
- Depolarising noise channel (uniform decoherence)
- Amplitude damping channel (T₁ energy relaxation)

---

## Results

| Metric | Value |
|---|---|
| Tomography reconstruction fidelity (noiseless) | 1.000000 |
| Tomography fidelity under depolarising noise (p=0.2) | 0.900000 |
| Fidelity loss at p=0.2 | 0.100000 |

![Results](results.png)

**Plots generated:**
1. Rabi oscillations — analytical (RWA) vs first-order perturbation theory
2. Fidelity decay under depolarising and amplitude-damping noise
3. Pauli expectation values ⟨X⟩, ⟨Y⟩, ⟨Z⟩ as noise increases (tomography observables)
4. Density matrix magnitude heatmap after noise

---

## Key Findings

- Perturbation theory (1st order) accurately matches the full analytical solution only in the **weak driving regime** (Ω_R ≪ ω₀) — deviations grow at stronger drives as expected from higher-order Dyson terms
- **Depolarising noise** drives all Bloch vector components uniformly to zero — the maximally mixed state I/2
- **Amplitude damping** preferentially decays the excited state population — asymmetric decoherence
- Quantum state tomography achieves **perfect reconstruction** from Pauli expectations in the noiseless case; fidelity degrades gracefully under noise

---

## Installation

```bash
pip install pennylane numpy scipy matplotlib
```

---

## Usage

```bash
python quantum_tomography.py
```

Outputs a summary to terminal and saves `results.png`.

---

## File Structure

```
quantum-state-tomography/
├── quantum_tomography.py   # Main simulation code
├── results.png             # Generated plots
└── README.md               # This file
```

---

## References

1. Englert, B-G. *Lectures on Quantum Mechanics Vol. 2: Perturbed Evolution*. World Scientific.
2. Schwinger, J. *Quantum Mechanics: Symbolism of Atomic Measurements*. Springer.
3. Singh, S. Quantum Mechanics Course Notes, IIT Delhi (2025–26).
4. Nielsen & Chuang. *Quantum Computation and Quantum Information*. Cambridge University Press.
5. PennyLane Documentation — [pennylane.ai](https://pennylane.ai)
