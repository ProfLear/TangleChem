# TangleChem

Interactive, **"Tangle"-style** chemical kinetics and reaction equations for Python notebooks (`marimo`, Jupyter, VS Code, Colab), inspired by Bret Victor's *Tangle* and `TangleLatex` from [`wigglystuff`](https://github.com/koaning/wigglystuff).

Define a system of chemical reactions in plain text, and get:
1. **Interactive chemical equations** where **rate constants** sit directly on the reaction arrows and can be dragged horizontally (or clicked to type an exact value).
2. **Interactive initial concentrations** (`[A]₀`, `[B]₀`, ...) that can be dragged inline to explore initial-condition sensitivity.
3. **Live 60fps concentration-vs-time plots** with dashed "before-drag" reference trajectories (`committed_values` vs live `values`), hover inspection, and Python-accessible ODE solutions.

## Quickstart

```python
from tanglechem import TangleKinetics

widget = TangleKinetics(
    reactions=[
        "A ->[k1] B",
        "B <->[k2, k_2] C",
    ],
    initial_concentrations={"A": 1.0, "B": 0.0, "C": 0.0},
    rate_constants={
        "k1": {"value": 0.8, "min_value": 0.0, "max_value": 5.0, "step": 0.05},
        "k2": {"value": 0.5, "min_value": 0.0, "max_value": 5.0, "step": 0.05},
        "k_2": {"value": 0.2, "min_value": 0.0, "max_value": 5.0, "step": 0.05},
    },
    t_max=10.0,
)
widget
```

### Reaction DSL Syntax

- **Irreversible reaction:** `"A + B ->[k1] C"` or `"2 A ->[k_dimer] A2"`
- **Reversible reaction:** `"A <->[kf, kr] B"` or `"A <=>[k1, k_1] B + C"`
- **Multiple reactions:** Pass a list of strings or a multiline string separated by newlines or `;`.
- **Shorthand numeric values:** You can pass floats directly in `rate_constants={"k1": 0.8}` or full parameter specs (`{"value": 0.8, "min_value": 0.0, "max_value": 5.0, "step": 0.05, "digits": 2}`).

## Running the Marimo Demo

```bash
uv run marimo edit demos/kinetics_demo.py
```
