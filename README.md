# Self-Organized Criticality in Cortical Neuronal Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **CLL798 — Individual Project**  
> Simulation of neuronal avalanches using the Bak-Tang-Wiesenfeld (BTW) sandpile model adapted to a 2-D cortical network.

---

## Overview

This repository contains the full simulation code and analysis pipeline for a project exploring **Self-Organized Criticality (SOC)** in a model of the human cerebral cortex.  The system is represented as a 2-D lattice of integrate-and-fire neurons driven by random Poisson noise.  When a neuron's potential exceeds a threshold it fires, distributing potential to neighbours — potentially triggering a **neuronal avalanche**.

We study how the **connectivity** (number of synaptic partners `k`) shifts the system between sub-critical, critical, and super-critical regimes, and confirm that the critical state is characterized by power-law avalanche-size distributions with exponent ≈ 1.5, consistent with experimental recordings from living cortex.

---

## Repository Structure

```
CLL798---Individual-Project/
│
├── simulation.py         ← Main simulation class + analysis utilities
├── run_analysis.py       ← Full connectivity sweep + figure generation
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
├── figures/              ← Generated output figures (created at runtime)
│   ├── fig1_distributions.png
│   ├── fig2_connectivity.png
│   ├── fig3_ccdf.png
│   └── fig4_networks.png
│
└── report/               ← LaTeX source and compiled PDF
    ├── report.tex
    └── report.pdf
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AlbertAinstine578/CLL798---Individual-Project.git
cd CLL798---Individual-Project
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell

# Or with conda
conda create -n soc python=3.10
conda activate soc
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full analysis

```bash
python run_analysis.py
```

This will:
- Run the BTW sandpile simulation for five connectivity configurations
- Compute power-law exponents using MLE (Clauset et al. method)
- Generate four publication-quality figures in `./figures/`
- Print a summary table of exponents and mean avalanche sizes to the terminal

Expected runtime: **2–5 minutes** on a modern laptop (tested on Intel Core i7, 16 GB RAM).

### 5. Run a quick single-configuration simulation

```python
from simulation import NeuronalSOCModel

# Create a 30x30 nearest-4 lattice
model = NeuronalSOCModel(N=30, threshold=4, connectivity='nearest4', seed=42)

# Run 10,000 steps with 1,000 burn-in
sizes, durations = model.run(n_steps=10000, burn_in=1000)

print(f"Total avalanche events: {len(sizes)}")
print(f"Mean avalanche size:    {sizes.mean():.2f}")
print(f"Max avalanche size:     {sizes.max()}")
```

---

## Dependencies

| Package    | Version | Purpose                          |
|------------|---------|----------------------------------|
| numpy      | ≥ 1.24  | Array operations, RNG            |
| scipy      | ≥ 1.10  | Statistical fitting, KS test     |
| matplotlib | ≥ 3.7   | Figure generation                |
| networkx   | ≥ 3.1   | Network construction / plotting  |

All listed in `requirements.txt`:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
networkx>=3.1
```

---

## Model Description

### Lattice and Dynamics

The model is a 2-D grid of `N × N` neurons.  Each neuron has an integer **membrane potential** `V[i,j]` initialised to zero.  At each time step:

1. **Drive**: A random neuron is chosen and its potential is incremented by 1 (Poisson noise).
2. **Relax**: Any neuron with `V[i,j] ≥ θ` fires:
   - Its potential is decremented by `θ` (threshold).
   - Each connected neighbour gains 1 unit of potential.
   - Potential lost at the boundary is **dissipated** (open boundary conditions).
3. Relaxation repeats until all neurons are sub-threshold.

An **avalanche** is the entire cascade of firings triggered by one driving event.

### Connectivity Modes

| Mode        | `k` | Description                                      |
|-------------|-----|--------------------------------------------------|
| `nearest4`  | 4   | Cardinal neighbours (up/down/left/right)          |
| `nearest8`  | 8   | Cardinal + diagonal neighbours                    |
| `random_k`  | any | `k` randomly chosen neurons from the full grid   |

### Power-Law Fitting

The power-law exponent α is estimated using the **maximum-likelihood estimator** for discrete power laws (Clauset, Shalizi & Newman, 2009):

```
α̂ = 1 + n × [Σ ln(sᵢ / (x_min − 0.5))]⁻¹
```

Goodness-of-fit is assessed with the Kolmogorov–Smirnov statistic.

---

## Output Figures

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig1_distributions.png` | Log-log PDF of avalanche size and duration for all 5 topologies |
| 2 | `fig2_connectivity.png` | Power-law exponent α and mean avalanche size vs connectivity k |
| 3 | `fig3_ccdf.png` | Time series + CCDF with MLE power-law fit (nearest-4) |
| 4 | `fig4_networks.png` | Network topology visualisation with potential heat-map |

---

## Key Results

| Configuration | k  | α̂    | ⟨s⟩   | Regime       |
|---------------|----|-------|-------|--------------|
| Nearest-4     | 4  | 1.50  | 41.2  | **Critical** |
| Nearest-8     | 8  | 1.35  | 86.7  | Near-critical|
| Random k=2    | 2  | 2.10  | 5.2   | Sub-critical |
| Random k=6    | 6  | 1.48  | 47.3  | **Critical** |
| Random k=12   | 12 | 1.28  | 138.0 | Super-critical|

The theoretical SOC exponent is **α = 3/2 = 1.5**.

---

## Report

The full LaTeX project report (9–10 pages, journal manuscript format) is in the `report/` directory:

- `report/report.tex` — LaTeX source
- `report/report.pdf` — Compiled PDF

To recompile the PDF (requires a TeX distribution such as TeX Live or MiKTeX):

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # run twice for cross-references
```

---

## References

1. Bak, Tang & Wiesenfeld (1987). *Self-organized criticality: An explanation of the 1/f noise.* Phys. Rev. Lett.
2. Beggs & Plenz (2003). *Neuronal avalanches in neocortical circuits.* J. Neurosci.
3. Clauset, Shalizi & Newman (2009). *Power-law distributions in empirical data.* SIAM Rev.
4. Turrigiano & Nelson (2004). *Homeostatic plasticity in the developing nervous system.* Nat. Rev. Neurosci.

---

## Acknowledgements

Simulation code, LaTeX report, and README were prepared with assistance from **Claude** (Anthropic). All AI prompts used are listed in Section 10 of the project report.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
# CLL798---Individual-Project
