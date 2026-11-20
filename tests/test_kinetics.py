"""Unit tests for TangleChem reaction parser, ODE solver, and TangleKinetics widget."""

from __future__ import annotations

import numpy as np
import pytest

from tanglechem import (
    TangleKinetics,
    format_chemical_label,
    format_rate_symbol,
    initial_param_key,
    parse_reactions,
)


def test_parse_irreversible_and_reversible():
    sys = parse_reactions([
        "A ->[k1] B",
        "B <->[k2, k_2] C",
    ])
    assert sys.species == ["A", "B", "C"]
    assert sys.rate_names == ["k1", "k2", "k_2"]
    assert len(sys.steps) == 2
    assert not sys.steps[0].reversible
    assert sys.steps[0].k_forward == "k1"
    assert sys.steps[0].k_reverse is None
    assert sys.steps[1].reversible
    assert sys.steps[1].k_forward == "k2"
    assert sys.steps[1].k_reverse == "k_2"


def test_parse_stoichiometry_and_inline_defaults():
    sys = parse_reactions("2 NO2 <=>[kf=1.2, kr=0.4] N2O4")
    assert sys.species == ["NO2", "N2O4"]
    assert sys.steps[0].reactants[0].coeff == 2
    assert sys.steps[0].reactants[0].species == "NO2"
    assert sys.steps[0].products[0].coeff == 1
    assert sys.steps[0].products[0].species == "N2O4"
    assert sys.inferred_rate_defaults == {"kf": 1.2, "kr": 0.4}


def test_chemical_formatting():
    assert format_chemical_label("H2O") == "H₂O"
    assert format_chemical_label("Fe^2+") == "Fe²⁺"
    assert format_rate_symbol("k1") == "k₁"
    assert format_rate_symbol("k_1") == "k₋₁"


def test_ode_mass_conservation():
    sys = parse_reactions([
        "A ->[k1] B",
        "B <->[k2, k_2] C",
    ])
    sol = sys.solve(
        rate_values={"k1": 1.0, "k2": 0.8, "k_2": 0.3},
        initial_concentrations={"A": 2.0, "B": 0.5, "C": 0.0},
        t_max=10.0,
        n_points=100,
    )
    total_mass = sol["A"] + sol["B"] + sol["C"]
    np.testing.assert_allclose(total_mass, 2.5, rtol=1e-5, atol=1e-6)
    # A decays exponentially: [A](t) = 2.0 * exp(-1.0 * t)
    np.testing.assert_allclose(sol["A"], 2.0 * np.exp(-sol["t"]), rtol=1e-4, atol=1e-5)


def test_tangle_kinetics_widget_initialization_and_solve():
    w = TangleKinetics(
        reactions=["A ->[k1] B", "B <->[k2, k_2] C"],
        rate_constants={
            "k1": {"value": 0.8, "min_value": 0.0, "max_value": 5.0, "step": 0.1},
            "k2": 0.5,
            "k_2": 0.2,
        },
        initial_concentrations={"A": 1.5, "B": 0.0, "C": 0.0},
        t_max=8.0,
    )

    assert w.rate_values == {"k1": 0.8, "k2": 0.5, "k_2": 0.2}
    assert w.initial_values == {"A": 1.5, "B": 0.0, "C": 0.0}
    assert w.values[initial_param_key("A")] == 1.5
    assert w.committed_values == w.values

    # Updating values outside a drag should update committed_values automatically
    new_values = dict(w.values)
    new_values["k1"] = 1.2
    w.values = new_values
    assert w.committed_values["k1"] == 1.2

    # Updating values during a drag should preserve committed_values
    w.dragging = True
    drag_values = dict(w.values)
    drag_values["k1"] = 2.5
    w.values = drag_values
    assert w.values["k1"] == 2.5
    assert w.committed_values["k1"] == 1.2

    sol_live = w.solve()
    sol_committed = w.solve(use_committed=True)
    assert sol_live["A"][-1] < sol_committed["A"][-1]


def test_validation_errors():
    with pytest.raises(ValueError, match="Unknown rate constant"):
        TangleKinetics("A ->[k1] B", rate_constants={"k99": 1.0})

    with pytest.raises(ValueError, match="Unknown species"):
        TangleKinetics("A ->[k1] B", initial_concentrations={"Z": 1.0})
