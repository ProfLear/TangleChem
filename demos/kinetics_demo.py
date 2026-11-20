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
    from tanglechem import TangleKinetics

    return TangleKinetics, mo


@app.cell(hide_code=True)
def _(mo):
    theme = mo.ui.dropdown(
        options=["auto", "light", "dark"], value="light", label="Theme"
    )
    display_mode = mo.ui.dropdown(
        options=["both", "symbol", "number"],
        value="both",
        label="Arrow rate display",
    )
    mo.vstack(
        [
            mo.md(r"""
            # TangleChem: Interactive Chemical Kinetics

            Inspired by Bret Victor's *Tangle* and `TangleLatex` from [`wigglystuff`](https://github.com/koaning/wigglystuff).

            - **Drag any rate constant** right on the reaction arrow horizontally (hold `Shift` for fine adjustment), or **click** it to type an exact number.
            - **Drag any initial concentration** ($[\text{A}]_0, [\text{B}]_0, \dots$) in the species bar below the equations.
            - While dragging, **solid curves** show the live solution (`values`) and **dashed curves** preserve the trajectory from before the drag (`committed_values`).
            """),
            mo.hstack([theme, display_mode], gap=1.5),
        ],
        gap=0.5,
    )
    return display_mode, theme


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Consecutive & Reversible Reactions ($A \rightarrow B \rightleftharpoons C$)

    Try dragging $k_1$ vs $k_2$ to control the maximum buildup of the intermediate $\text{B}$, or click **`d[X]/dt`** in the widget header to inspect and scrub the coupled differential equations directly.
    """)
    return


@app.cell
def _(TangleKinetics, display_mode, mo, theme):
    sequential = mo.ui.anywidget(
        TangleKinetics(
            reactions=[
                "A ->[k1] B",
                "B <->[k2, k_2] C",
            ],
            rate_constants={
                "k1": {"value": 0.8, "min_value": 0.0, "max_value": 5.0, "step": 0.05},
                "k2": {"value": 0.4, "min_value": 0.0, "max_value": 5.0, "step": 0.05},
                "k_2": {"value": 0.15, "min_value": 0.0, "max_value": 5.0, "step": 0.05},
            },
            initial_concentrations={"A": 1.0, "B": 0.0, "C": 0.0},
            t_max=10.0,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    return (sequential,)


@app.cell(hide_code=True)
def _(mo, sequential):
    _v = sequential.values
    _c = sequential.committed_values
    mo.vstack(
        [
            sequential,
            mo.md(
                f"**Python synced state:** `k1 = {_v['k1']:.2f}` (before drag: `{_c['k1']:.2f}`), "
                f"`k2 = {_v['k2']:.2f}`, `k_2 = {_v['k_2']:.2f}`, "
                f"`[A]₀ = {_v['[A]0']:.2f} M`"
            ),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bimolecular Pre-Equilibrium & Catalysis ($E + S \rightleftharpoons ES \rightarrow E + P$)

    TangleChem automatically handles higher-order mass-action kinetics and catalyst regeneration:
    """)
    return


@app.cell
def _(TangleKinetics, display_mode, mo, theme):
    enzyme = mo.ui.anywidget(
        TangleKinetics(
            reactions=[
                "E + S <->[k_on, k_off] ES",
                "ES ->[k_cat] E + P",
            ],
            rate_constants={
                "k_on": {"value": 2.0, "min_value": 0.0, "max_value": 10.0, "step": 0.1, "symbol": "k_on"},
                "k_off": {"value": 0.5, "min_value": 0.0, "max_value": 10.0, "step": 0.1, "symbol": "k_off"},
                "k_cat": {"value": 1.2, "min_value": 0.0, "max_value": 10.0, "step": 0.1, "symbol": "k_cat"},
            },
            initial_concentrations={
                "E": {"value": 0.4, "max_value": 2.0, "step": 0.05},
                "S": {"value": 1.5, "max_value": 5.0, "step": 0.05},
                "ES": 0.0,
                "P": 0.0,
            },
            t_max=8.0,
            show_odes=True,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    enzyme
    return (enzyme,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Stoichiometric Dimerization ($2\,\text{NO}_2 \rightleftharpoons \text{N}_2\text{O}_4$)

    Chemical formulas automatically format subscripts (`NO2` $\rightarrow$ $\text{NO}_2$, `N2O4` $\rightarrow$ $\text{N}_2\text{O}_4$) and respect second-order stoichiometry ($\text{rate}_f = k_f[\text{NO}_2]^2$).
    """)
    return


@app.cell
def _(TangleKinetics, display_mode, mo, theme):
    dimer = mo.ui.anywidget(
        TangleKinetics(
            reactions="2 NO2 <=>[kf=0.6, kr=0.25] N2O4",
            initial_concentrations={"NO2": 1.2, "N2O4": 0.0},
            t_max=6.0,
            display_mode=display_mode.value,
            theme=theme.value,
        )
    )
    dimer
    return (dimer,)


if __name__ == "__main__":
    app.run()
