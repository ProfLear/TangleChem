# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.9.0",
#     "anywidget>=0.9.0",
#     "numpy",
#     "scipy",
#     "tanglechem @ https://raw.githubusercontent.com/ProfLear/TangleChem/main/dist/tanglechem-0.2.0-py3-none-any.whl",
# ]
#
# [tool.uv.sources]
# tanglechem = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from tanglechem import TangleSchrodinger

    return TangleSchrodinger, mo


@app.cell(hide_code=True)
def _(mo):
    theme = mo.ui.dropdown(
        options=["auto", "light", "dark"], value="light", label="Theme"
    )
    display_mode = mo.ui.dropdown(
        options=["both", "symbol", "number"],
        value="both",
        label="Parameter display",
    )
    mo.vstack(
        [
            mo.md(r"""
            # TangleChem: Interactive 1D Schrödinger Equation

            Define any **piecewise** or **algebraic** 1D potential energy curve $V(x)$ and solve the time-independent Schrödinger equation in real time:

            $$-\frac{\hbar^2}{2m}\frac{d^2\psi_n}{dx^2} + V(x)\psi_n(x) = E_n\psi_n(x)$$

            - **Drag any parameter** ($L$, $V_0$, $k$, $V_b$, $a$, or particle mass $m$) horizontally right inside $V(x)$ (hold `Shift` for fine control), or **click** it to type an exact value.
            - Toggle between **wavefunctions $\psi_n(x)$** and **probability densities $|\psi_n(x)|^2$** using the button in the widget header.
            - While dragging, **dashed ghost curves** show the pre-drag eigenstates (`committed_values`).
            """),
            mo.hstack([theme, display_mode], gap=1.5),
        ],
        gap=0.5,
    )
    return display_mode, theme


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Particle in a 1D Box (Infinite Square Well)

    Written with piecewise rules `["0 for 0 <= x <= L", "inf otherwise"]` (or `\cases{0, 0 <= x <= L \\ \infty, otherwise}`). Drag the box width $L$ to see $E_n = \frac{n^2 \pi^2 \hbar^2}{2 m L^2}$ scale as $1/L^2$:
    """)
    return


@app.cell
def _(TangleSchrodinger, display_mode, mo, theme):
    box = mo.ui.anywidget(
        TangleSchrodinger(
            potential=[
                "0 for 0 <= x <= L",
                "inf otherwise",
            ],
            parameters={
                "L": {"value": 2.0, "min_value": 0.6, "max_value": 4.5, "step": 0.05},
            },
            x_domain=(-0.4, 4.8),
            n_states=4,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    box
    return (box,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Finite Square Well & Quantum Tunneling

    Replacing the infinite walls with a finite barrier height $V_0$ allows the wavefunctions $\psi_n(x)$ to penetrate exponentially into the classically forbidden regions ($|x| > L/2$):
    """)
    return


@app.cell
def _(TangleSchrodinger, display_mode, mo, theme):
    finite_well = mo.ui.anywidget(
        TangleSchrodinger(
            potential=[
                "0 for -L/2 <= x <= L/2",
                "V0 otherwise",
            ],
            parameters={
                "L": {"value": 2.4, "min_value": 0.6, "max_value": 5.0, "step": 0.05},
                "V0": {"value": 10.0, "min_value": 1.0, "max_value": 30.0, "step": 0.5},
            },
            x_domain=(-3.5, 3.5),
            n_states=4,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    finite_well
    return (finite_well,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Quantum Harmonic Oscillator ($V(x) = \frac{1}{2} k (x - x_0)^2$)

    Drag the force constant $k$ (or particle mass $m$) to observe equally spaced vibrational levels $E_n = \hbar\omega(n + \tfrac{1}{2})$ with $\omega = \sqrt{k/m}$:
    """)
    return


@app.cell
def _(TangleSchrodinger, display_mode, mo, theme):
    harmonic = mo.ui.anywidget(
        TangleSchrodinger.harmonic_oscillator(
            k=2.5,
            x0=0.0,
            n_states=5,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    harmonic
    return (harmonic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Double Well & Tunneling Splitting

    A central barrier of height $V_b$ and thickness $a$ inside a well of width $L$ splits the states into symmetric and antisymmetric tunneling doublets ($E_2 - E_1$). Drag $V_b$ or $a$ upward to watch the lowest two states become nearly degenerate!
    """)
    return


@app.cell
def _(TangleSchrodinger, display_mode, mo, theme):
    double_well = mo.ui.anywidget(
        TangleSchrodinger.double_well(
            L=3.4,
            a=0.45,
            Vb=6.5,
            V0=18.0,
            n_states=4,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    double_well
    return (double_well,)


if __name__ == "__main__":
    app.run()
