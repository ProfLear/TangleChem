# TangleChem

Interactive, **"Tangle"-style** chemical kinetics and quantum chemistry widgets for Python notebooks (`marimo`, Jupyter, VS Code, Colab), inspired by Bret Victor's *Tangle* and `TangleLatex` from [`wigglystuff`](https://github.com/koaning/wigglystuff).

## Live Browser Demos (Molab WASM)

| Component | Description | Live Molab WASM Notebook |
| :--- | :--- | :--- |
| **`TangleKinetics`** | Interactive reaction networks & mass-action ODE kinetics | [Launch Kinetics Demo (WASM)](https://molab.marimo.io/github/ProfLear/TangleChem/blob/main/demos/kinetics_demo.py/wasm) |
| **`TangleSchrodinger`** | Interactive 1D potential wells, eigenstates & wavefunctions | [Launch 1D Schrödinger Demo (WASM)](https://molab.marimo.io/github/ProfLear/TangleChem/blob/main/demos/schrodinger_demo.py/wasm) |

---

## Components

### 1. `TangleKinetics` — Chemical Reaction Kinetics
[**Try in Molab WASM →**](https://molab.marimo.io/github/ProfLear/TangleChem/blob/main/demos/kinetics_demo.py/wasm)

Define a system of chemical reactions in plain text, and get:
- **Interactive chemical equations** where **rate constants** sit directly on the reaction arrows and can be dragged horizontally (or clicked to type an exact value).
- **Interactive initial concentrations** (`[A]₀`, `[B]₀`, ...) that can be dragged inline.
- **Live 60fps concentration-vs-time plots** with dashed "before-drag" reference trajectories (`committed_values` vs live `values`), hover inspection, and a toggleable `d[X]/dt` differential rate equation drawer.

```python
from tanglechem import TangleKinetics

TangleKinetics(
    reactions=[
        "A ->[k1] B",
        "B <->[k2, k_2] C",
    ],
    initial_concentrations={"A": 1.0, "B": 0.0, "C": 0.0},
    rate_constants={"k1": 0.8, "k2": 0.4, "k_2": 0.15},
    t_max=10.0,
)
```

### 2. `TangleSchrodinger` — 1D Time-Independent Schrödinger Equation
[**Try in Molab WASM →**](https://molab.marimo.io/github/ProfLear/TangleChem/blob/main/demos/schrodinger_demo.py/wasm)

Define any **piecewise** or **algebraic** 1D potential energy curve $V(x)$, and get:
- **Interactive potential equation $V(x)$** where parameters (well width $L$, barrier height $V_0$, force constant $k$, barrier thickness $a$, particle mass $m$) are embedded directly inside the formula as draggable Tangle controls.
- **Live 60fps quantum energy-level & wavefunction plot** showing quantized eigenvalues $E_n$ and wavefunctions $\psi_n(x)$ (or probability densities $|\psi_n(x)|^2$) sitting on their energy baselines, with dashed pre-drag ghost comparison.

```python
from tanglechem import TangleSchrodinger

# Piecewise potential (Particle in a Box / Finite Well / Double Well)
TangleSchrodinger(
    potential=[
        "0 for -L/2 <= x <= L/2",
        "V0 otherwise",
    ],
    parameters={"L": 2.4, "V0": 10.0},
    n_states=4,
)

# Algebraic potential (Harmonic Oscillator, Morse, Quartic Double Well)
TangleSchrodinger(
    potential="0.5 * k * (x - x0)^2",
    parameters={"k": 2.5, "x0": 0.0},
    n_states=5,
)
```

## Running Locally with `uv`

```bash
uv run marimo edit demos/kinetics_demo.py
uv run marimo edit demos/schrodinger_demo.py
```
