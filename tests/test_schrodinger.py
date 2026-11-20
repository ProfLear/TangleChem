"""Unit tests for 1D Time-Independent Schrödinger Equation parser, solver, and widget."""

from __future__ import annotations

import numpy as np
import pytest

from tanglechem import TangleSchrodinger, parse_potential


def test_parse_piecewise_and_cases_potential():
    # Plain-text piecewise DSL
    p1 = parse_potential([
        "0 for 0 <= x <= L",
        "inf otherwise",
    ])
    assert p1.is_piecewise
    assert p1.param_names == ["L"]
    assert len(p1.cases) == 2
    assert p1.cases[0].condition_py == "0 <= x <= L"
    assert p1.cases[1].is_otherwise

    # LaTeX \cases{...} syntax
    p2 = parse_potential(r"V(x) = \cases{0, 0 <= x <= L \\ \infty, otherwise}")
    assert p2.is_piecewise
    assert p2.param_names == ["L"]


def test_parse_algebraic_potential_with_inline_defaults():
    p = parse_potential("0.5 * (k=4.0) * (x - (x0=0.5))^2")
    assert not p.is_piecewise
    assert p.param_names == ["k", "x0"]
    assert p.inferred_defaults == {"k": 4.0, "x0": 0.5}


def test_particle_in_a_box_eigenvalues():
    # Analytical: E_n = n^2 * pi^2 / (2 * m * L^2) for hbar = 1
    L = 2.0
    m = 1.0
    w = TangleSchrodinger.particle_in_a_box(L=L, n_states=3, n_grid=500)
    sol = w.solve()
    expected = np.array([(n**2) * (np.pi**2) / (2.0 * m * (L**2)) for n in (1, 2, 3)])
    np.testing.assert_allclose(sol["energies"], expected, rtol=2e-2)


def test_harmonic_oscillator_eigenvalues():
    # Analytical: E_n = hbar * omega * (n + 0.5) where omega = sqrt(k / m)
    k = 4.0
    m = 1.0
    omega = np.sqrt(k / m)
    w = TangleSchrodinger.harmonic_oscillator(k=k, n_states=4, n_grid=400)
    sol = w.solve()
    expected = np.array([omega * (n + 0.5) for n in range(4)])
    np.testing.assert_allclose(sol["energies"], expected, rtol=1.5e-2)


def test_tangle_schrodinger_presets_and_committed_values():
    w = TangleSchrodinger.double_well(L=3.0, a=0.4, Vb=6.0, V0=15.0)
    assert set(w.values) == {"Vb", "a", "L", "V0", "m"}
    assert w.committed_values == w.values

    # Dragging updates values while preserving committed_values
    w.dragging = True
    next_vals = dict(w.values)
    next_vals["Vb"] = 12.0
    w.values = next_vals
    assert w.values["Vb"] == 12.0
    assert w.committed_values["Vb"] == 6.0


def test_schrodinger_validation_errors():
    with pytest.raises(ValueError, match="Unknown parameter"):
        TangleSchrodinger("0.5 * k * x^2", parameters={"unknown": 1.0})
