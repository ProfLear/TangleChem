"""Interactive TangleKinetics anywidget for chemical reaction systems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import anywidget
import numpy as np
import traitlets

from .parser import (
    ReactionSystem,
    format_chemical_label,
    format_rate_symbol,
    parse_reactions,
)

_ESM_PATH = Path(__file__).parent / "static" / "kinetics-tangle.js"
_CSS_PATH = Path(__file__).parent / "static" / "kinetics-tangle.css"

_SPECIES_PALETTE = (
    {"light": "#246bce", "dark": "#75a7ff"},  # Blue
    {"light": "#d1495b", "dark": "#ff7b8b"},  # Coral Red
    {"light": "#147a68", "dark": "#5ed5bd"},  # Teal
    {"light": "#b45b1b", "dark": "#ffad66"},  # Amber
    {"light": "#6f5cbd", "dark": "#b9a8ff"},  # Purple
    {"light": "#a23b78", "dark": "#f08bc2"},  # Magenta
    {"light": "#3d7a22", "dark": "#88d46c"},  # Olive Green
    {"light": "#007791", "dark": "#5ce1ff"},  # Cyan
)

_RATE_PALETTE = (
    {"light": "#1e5bb8", "dark": "#80b3ff"},
    {"light": "#a84e14", "dark": "#ffb570"},
    {"light": "#116b5b", "dark": "#66dec6"},
    {"light": "#8e3369", "dark": "#f294c8"},
    {"light": "#5b48a8", "dark": "#c2b3ff"},
)


def _finite_float(value: Any, *, name: str, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"parameter {name!r} {field} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"parameter {name!r} {field} must be a finite number."
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"parameter {name!r} {field} must be a finite number.")
    return result


def _normalize_color(
    value: Any,
    *,
    name: str,
    index: int,
    palette: tuple[dict[str, str], ...] = _SPECIES_PALETTE,
) -> dict[str, str]:
    if value is None:
        return dict(palette[index % len(palette)])
    if isinstance(value, str) and value:
        return {"light": value, "dark": value}
    if isinstance(value, Mapping):
        if set(value) != {"light", "dark"}:
            raise ValueError(
                f"parameter {name!r} color mapping must contain 'light' and 'dark'."
            )
        return {"light": str(value["light"]), "dark": str(value["dark"])}
    raise ValueError(
        f"parameter {name!r} color must be a CSS color string or a light/dark mapping."
    )


def _infer_step_and_digits(val: float, max_val: float) -> tuple[float, int]:
    span = max_val if max_val > 0 else 1.0
    if span <= 0.1:
        return 0.001, 3
    if span <= 2.0:
        return 0.01, 2
    if span <= 20.0:
        return 0.05, 2
    if span <= 100.0:
        return 0.5, 1
    return 1.0, 1


def _normalize_param_spec(
    name: str,
    raw: float | int | Mapping[str, Any] | None,
    *,
    default_value: float,
    default_symbol: str,
    default_label: str,
    kind: str,
    index: int,
    palette: tuple[dict[str, str], ...],
    unit: str = "",
) -> dict[str, Any]:
    if raw is None:
        spec_dict: dict[str, Any] = {"value": default_value}
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        spec_dict = {"value": float(raw)}
    elif isinstance(raw, Mapping):
        spec_dict = dict(raw)
    else:
        raise ValueError(
            f"Parameter {name!r} must be a number or a configuration dictionary."
        )

    value = _finite_float(
        spec_dict.get("value", default_value), name=name, field="value"
    )
    min_value = _finite_float(
        spec_dict.get("min_value", 0.0), name=name, field="min_value"
    )
    default_max = max(5.0, round(value * 4.0, 2)) if value > 0 else 5.0
    max_value = _finite_float(
        spec_dict.get("max_value", default_max), name=name, field="max_value"
    )
    if max_value < value:
        max_value = max(value * 2.0, value + 1.0)
    if min_value > value:
        min_value = min(0.0, value)

    inferred_step, inferred_digits = _infer_step_and_digits(
        value, max_value - min_value
    )
    step = _finite_float(
        spec_dict.get("step", inferred_step), name=name, field="step"
    )
    if step <= 0:
        raise ValueError(f"parameter {name!r} step must be positive.")
    digits = int(spec_dict.get("digits", inferred_digits))
    if digits < 0:
        raise ValueError(f"parameter {name!r} digits must be non-negative.")

    pixels_per_step = int(spec_dict.get("pixels_per_step", 6))
    color = _normalize_color(
        spec_dict.get("color"), name=name, index=index, palette=palette
    )
    return {
        "name": name,
        "kind": kind,
        "value": value,
        "min_value": min_value,
        "max_value": max_value,
        "step": step,
        "digits": digits,
        "symbol": str(spec_dict.get("symbol", default_symbol)),
        "label": str(spec_dict.get("label", default_label)),
        "unit": str(spec_dict.get("unit", unit)),
        "pixels_per_step": max(1, pixels_per_step),
        "color": color,
    }


def initial_param_key(species: str) -> str:
    """Return the parameter dictionary key for a species' initial concentration."""
    return f"[{species}]0"


class TangleKinetics(anywidget.AnyWidget):
    """Interactive chemical reaction and kinetics widget with draggable parameters.

    Drag rate constants directly on reaction arrows or initial concentrations
    in the species bar to watch kinetic curves update at 60 FPS, with dashed
    reference lines showing the trajectory from before the current drag.
    """

    _esm = _ESM_PATH
    _css = _CSS_PATH

    system_data = traitlets.Dict(default_value={}).tag(sync=True)
    parameters = traitlets.Dict(default_value={}).tag(sync=True)
    values = traitlets.Dict(default_value={}).tag(sync=True)
    committed_values = traitlets.Dict(default_value={}).tag(sync=True)
    dragging = traitlets.Bool(False).tag(sync=True)
    active_param = traitlets.Unicode("").tag(sync=True)

    t_max = traitlets.Float(10.0).tag(sync=True)
    concentration_unit = traitlets.Unicode("M").tag(sync=True)
    time_unit = traitlets.Unicode("s").tag(sync=True)
    show_plot = traitlets.Bool(True).tag(sync=True)
    show_ghost = traitlets.Bool(True).tag(sync=True)
    show_odes = traitlets.Bool(False).tag(sync=True)
    display_mode = traitlets.Enum(
        ["both", "number", "symbol"], default_value="both"
    ).tag(sync=True)
    theme = traitlets.Enum(
        ["auto", "light", "dark"], default_value="auto"
    ).tag(sync=True)

    def __init__(
        self,
        reactions: str | Sequence[str],
        rate_constants: Mapping[str, float | int | Mapping[str, Any]] | None = None,
        initial_concentrations: Mapping[str, float | int | Mapping[str, Any]] | None = None,
        *,
        t_max: float = 10.0,
        concentration_unit: str = "M",
        time_unit: str = "s",
        show_plot: bool = True,
        show_ghost: bool = True,
        show_odes: bool = False,
        display_mode: str = "both",
        theme: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Create a TangleKinetics widget.

        Args:
            reactions: A chemical reaction string or list of strings using the
                TangleChem DSL, e.g. ``["A ->[k1] B", "B <->[k2, k_2] C"]``.
            rate_constants: Mapping from rate constant name (e.g. ``"k1"``) to
                either a numeric initial value or a parameter dict containing
                ``value``, ``min_value``, ``max_value``, ``step``, ``digits``,
                ``symbol``, ``unit``, ``color``. Any rate constant omitted is
                given a default value of ``0.5`` (or the inline value from
                ``->[k1=0.5]``).
            initial_concentrations: Mapping from species name (e.g. ``"A"`` or
                ``"[A]0"``) to either a numeric initial concentration or a
                parameter dict. Defaults to ``1.0`` for reactants of the first
                reaction step and ``0.0`` for all other species.
            t_max: Maximum simulation time (> 0).
            concentration_unit: Display unit for concentrations (default ``"M"``).
            time_unit: Display unit for time (default ``"s"``).
            show_plot: Whether to render the built-in interactive SVG plot.
            show_ghost: Whether to show dashed reference curves for
                ``committed_values`` while dragging a parameter.
            show_odes: Whether to expand the differential rate equations panel
                by default.
            display_mode: How rate constants on reaction arrows are displayed:
                ``"both"`` (e.g. ``k₁ = 0.80``), ``"number"`` (``0.80``), or
                ``"symbol"`` (``k₁``, revealing the number while dragging).
            theme: Color theme: ``"auto"``, ``"light"``, or ``"dark"``.
            **kwargs: Forwarded to ``anywidget.AnyWidget``.
        """
        if display_mode not in {"both", "number", "symbol"}:
            raise ValueError("display_mode must be 'both', 'number', or 'symbol'.")
        if theme not in {"auto", "light", "dark"}:
            raise ValueError("theme must be 'auto', 'light', or 'dark'.")
        t_max_val = _finite_float(t_max, name="t_max", field="value")
        if t_max_val <= 0:
            raise ValueError("t_max must be positive.")

        self._system: ReactionSystem = parse_reactions(reactions)
        rate_constants = dict(rate_constants or {})
        initial_concentrations = dict(initial_concentrations or {})

        # Validate that no unknown rate constant names were passed
        unknown_rates = sorted(set(rate_constants) - set(self._system.rate_names))
        if unknown_rates:
            raise ValueError(
                "Unknown rate constant(s) not present in reactions: "
                + ", ".join(unknown_rates)
            )

        parameters: dict[str, dict[str, Any]] = {}

        # 1. Normalize rate constant parameters
        for idx, r_name in enumerate(self._system.rate_names):
            default_val = self._system.inferred_rate_defaults.get(r_name, 0.5)
            raw_spec = rate_constants.get(r_name)
            parameters[r_name] = _normalize_param_spec(
                r_name,
                raw_spec,
                default_value=default_val,
                default_symbol=format_rate_symbol(r_name),
                default_label=f"Rate constant {r_name}",
                kind="rate",
                index=idx,
                palette=_RATE_PALETTE,
                unit="",
            )

        # 2. Normalize initial concentration parameters for each species
        first_step_reactants = {
            term.species for term in self._system.steps[0].reactants
        }
        species_meta: list[dict[str, Any]] = []

        # Allow keys in initial_concentrations to be either "A" or "[A]0"
        cleaned_ic: dict[str, Any] = {}
        for k, v in initial_concentrations.items():
            if k in self._system.species:
                cleaned_ic[k] = v
            elif k.startswith("[") and k.endswith("]0") and k[1:-2] in self._system.species:
                cleaned_ic[k[1:-2]] = v
            else:
                raise ValueError(
                    f"Unknown species {k!r} in initial_concentrations. "
                    f"Valid species: {', '.join(self._system.species)}"
                )

        for idx, sp in enumerate(self._system.species):
            p_key = initial_param_key(sp)
            default_c0 = 1.0 if sp in first_step_reactants else 0.0
            raw_spec = cleaned_ic.get(sp)
            label = format_chemical_label(sp)
            param_spec = _normalize_param_spec(
                p_key,
                raw_spec,
                default_value=default_c0,
                default_symbol=f"[{label}]₀",
                default_label=f"Initial [{label}]",
                kind="concentration",
                index=idx,
                palette=_SPECIES_PALETTE,
                unit=concentration_unit,
            )
            param_spec["species"] = sp
            parameters[p_key] = param_spec
            species_meta.append(
                {
                    "name": sp,
                    "label": label,
                    "param_key": p_key,
                    "color": param_spec["color"],
                }
            )

        sys_dict = self._system.to_dict()
        sys_dict["species"] = species_meta

        values = {name: spec["value"] for name, spec in parameters.items()}

        super().__init__(
            system_data=sys_dict,
            parameters=parameters,
            values=values,
            committed_values=dict(values),
            t_max=t_max_val,
            concentration_unit=concentration_unit,
            time_unit=time_unit,
            show_plot=show_plot,
            show_ghost=show_ghost,
            show_odes=show_odes,
            display_mode=display_mode,
            theme=theme,
            **kwargs,
        )

    @traitlets.observe("values")
    def _follow_values(self, change: dict[str, Any]) -> None:
        # Outside an active pointer drag (e.g., when updated from Python),
        # committed_values follows values immediately.
        if not self.dragging:
            self.committed_values = dict(self.values)

    @property
    def system(self) -> ReactionSystem:
        """The parsed ReactionSystem backing this widget."""
        return self._system

    @property
    def rate_values(self) -> dict[str, float]:
        """Current rate constant values."""
        return {
            r: float(self.values[r])
            for r in self._system.rate_names
            if r in self.values
        }

    @property
    def initial_values(self) -> dict[str, float]:
        """Current initial concentrations keyed by species name."""
        return {
            sp: float(self.values.get(initial_param_key(sp), 0.0))
            for sp in self._system.species
        }

    def solve(
        self,
        *,
        use_committed: bool = False,
        t_max: float | None = None,
        n_points: int = 250,
        method: str = "LSODA",
    ) -> dict[str, np.ndarray]:
        """Solve the kinetic ODEs in Python using SciPy and return trajectories.

        Args:
            use_committed: If True, solve using ``committed_values`` (the
                parameters from before the current drag) instead of live ``values``.
            t_max: Override simulation end time (defaults to ``self.t_max``).
            n_points: Number of evenly spaced time points to evaluate.
            method: SciPy ``solve_ivp`` integration method (default ``"LSODA"``).

        Returns:
            Dictionary with ``"t"`` and one NumPy array per chemical species.
        """
        source = self.committed_values if use_committed else self.values
        rates = {r: float(source[r]) for r in self._system.rate_names}
        ic = {
            sp: float(source[initial_param_key(sp)])
            for sp in self._system.species
        }
        return self._system.solve(
            rates,
            ic,
            t_max=float(self.t_max if t_max is None else t_max),
            n_points=n_points,
            method=method,
        )
