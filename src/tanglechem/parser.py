"""Reaction DSL parser and ODE system builder for TangleChem."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp


# Matches reaction arrows with optional [k] or [kf, kr] annotations:
# Supported arrows:
#   Reversible: <->, <-->, <=>, ⇌
#   Irreversible: ->, -->, →
_ARROW_RE = re.compile(
    r"(?P<arrow><-->|<->|<=>|⇌|-->|->|→)\s*(?:\[\s*(?P<rates>[^\]]*)\s*\])?"
)

# Matches a single term like "2 A", "2A", "0.5 O2", "Fe^2+", "H2O"
_TERM_RE = re.compile(
    r"^\s*(?:(?P<coeff>\d+(?:\.\d+)?)\s+(?=[A-Za-z\[\(])|(?P<tight_coeff>\d+(?:\.\d+)?)(?=[A-Z\[\(]))?\s*(?P<species>.+?)\s*$"
)

_SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT_MAP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def format_chemical_label(species: str) -> str:
    """Format a plain-text species identifier into a readable chemical label.

    Examples:
        "H2O" -> "H₂O"
        "Fe^2+" -> "Fe²⁺"
        "A_star" -> "A*"
    """
    s = species.strip()
    # Handle explicit superscripts like Fe^2+ or I^-
    if "^" in s:
        base, sup = s.split("^", 1)
        sup_formatted = sup.translate(_SUPERSCRIPT_MAP)
    else:
        base, sup_formatted = s, ""

    # Convert digits following letters or closing parens into subscripts
    # (e.g. H2O -> H₂O, (OH)2 -> (OH)₂), unless preceded by underscore
    def _sub_repl(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).translate(_SUBSCRIPT_MAP)

    base_formatted = re.sub(r"([A-Za-z\)])(\d+)", _sub_repl, base)
    base_formatted = re.sub(
        r"_(\d+)", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), base_formatted
    )
    return base_formatted + sup_formatted


def format_rate_symbol(name: str) -> str:
    """Format a rate constant identifier like 'k1' or 'k_1' or 'k_rev' for display."""
    if name.startswith("k_"):
        sub = name[2:]
        if sub.isdigit():
            return f"k₋{sub.translate(_SUBSCRIPT_MAP)}"
        return f"k_{sub}"
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if m:
        return f"{m.group(1)}{m.group(2).translate(_SUBSCRIPT_MAP)}"
    return name


@dataclass(frozen=True)
class ReactantTerm:
    """A species and its stoichiometric coefficient in a reaction half."""

    species: str
    coeff: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "label": format_chemical_label(self.species),
            "coeff": self.coeff,
        }


@dataclass(frozen=True)
class ReactionStep:
    """A parsed elementary reaction step (irreversible or reversible)."""

    index: int
    raw: str
    reactants: tuple[ReactantTerm, ...]
    products: tuple[ReactantTerm, ...]
    k_forward: str
    k_reverse: str | None = None
    default_kf: float | None = None
    default_kr: float | None = None

    @property
    def reversible(self) -> bool:
        return self.k_reverse is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "raw": self.raw,
            "reactants": [r.to_dict() for r in self.reactants],
            "products": [p.to_dict() for p in self.products],
            "reversible": self.reversible,
            "k_forward": self.k_forward,
            "k_reverse": self.k_reverse,
        }


@dataclass
class ReactionSystem:
    """A complete system of parsed kinetic reactions."""

    steps: list[ReactionStep]
    species: list[str]
    rate_names: list[str]
    inferred_rate_defaults: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "species": [
                {"name": sp, "label": format_chemical_label(sp)}
                for sp in self.species
            ],
            "rate_names": list(self.rate_names),
        }

    def rhs(
        self,
        t: float,
        y: np.ndarray,
        rate_values: Mapping[str, float],
    ) -> np.ndarray:
        """Compute dy/dt for mass-action kinetics."""
        dydt = np.zeros_like(y, dtype=float)
        sp_idx = {sp: i for i, sp in enumerate(self.species)}
        # Clamp concentrations at >= 0 when computing rates to avoid numerical blowup
        y_safe = np.maximum(y, 0.0)

        for step in self.steps:
            kf = float(rate_values.get(step.k_forward, 0.0))
            rate_f = kf
            for term in step.reactants:
                c = y_safe[sp_idx[term.species]]
                rate_f *= c ** term.coeff if term.coeff != 1.0 else c

            rate_net = rate_f
            if step.reversible and step.k_reverse is not None:
                kr = float(rate_values.get(step.k_reverse, 0.0))
                rate_r = kr
                for term in step.products:
                    c = y_safe[sp_idx[term.species]]
                    rate_r *= c ** term.coeff if term.coeff != 1.0 else c
                rate_net -= rate_r

            for term in step.reactants:
                dydt[sp_idx[term.species]] -= term.coeff * rate_net
            for term in step.products:
                dydt[sp_idx[term.species]] += term.coeff * rate_net

        return dydt

    def solve(
        self,
        rate_values: Mapping[str, float],
        initial_concentrations: Mapping[str, float],
        *,
        t_max: float = 10.0,
        n_points: int = 250,
        method: str = "LSODA",
    ) -> dict[str, np.ndarray]:
        """Solve the ODE system over [0, t_max] and return time & species arrays."""
        if t_max <= 0:
            raise ValueError("t_max must be positive.")
        y0 = np.array(
            [float(initial_concentrations.get(sp, 0.0)) for sp in self.species],
            dtype=float,
        )
        t_eval = np.linspace(0.0, float(t_max), int(n_points))
        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, rate_values),
            t_span=(0.0, float(t_max)),
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=1e-6,
            atol=1e-9,
        )
        result: dict[str, np.ndarray] = {"t": sol.t}
        for i, sp in enumerate(self.species):
            result[sp] = np.maximum(sol.y[i], 0.0)
        return result


def _parse_side(side_str: str, *, context: str) -> tuple[ReactantTerm, ...]:
    side_str = side_str.strip()
    if not side_str or side_str in {"0", "∅", "null", "None"}:
        return ()

    # Split on '+' that is NOT immediately preceded by '^' (e.g. Fe^2+ or H^+)
    raw_terms = re.split(r"(?<!\^)(?<![\^\d])\+(?=\s|$|[0-9A-Za-z\[\(])", side_str)
    terms: list[ReactantTerm] = []
    for raw_term in raw_terms:
        part = raw_term.strip()
        if not part:
            raise ValueError(f"Empty species term in {context!r}.")
        m = _TERM_RE.match(part)
        if not m:
            raise ValueError(f"Could not parse species term {part!r} in {context!r}.")
        coeff_str = m.group("coeff") or m.group("tight_coeff")
        coeff = float(coeff_str) if coeff_str else 1.0
        if coeff <= 0 or not math.isfinite(coeff):
            raise ValueError(
                f"Stoichiometric coefficient must be positive in {context!r}."
            )
        # Convert integer-valued floats to int for cleaner display if exact
        if coeff.is_integer():
            coeff = int(coeff)
        species = m.group("species").strip()
        if not species:
            raise ValueError(f"Missing species name in {context!r}.")
        terms.append(ReactantTerm(species=species, coeff=coeff))

    # Combine duplicate species on the same side (e.g. "A + A -> B" -> "2 A -> B")
    combined: dict[str, float] = {}
    order: list[str] = []
    for t in terms:
        if t.species not in combined:
            order.append(t.species)
            combined[t.species] = t.coeff
        else:
            combined[t.species] += t.coeff
    return tuple(
        ReactantTerm(
            species=sp,
            coeff=int(combined[sp]) if float(combined[sp]).is_integer() else combined[sp],
        )
        for sp in order
    )


def _parse_rate_token(
    token: str, default_name: str
) -> tuple[str, float | None]:
    """Parse a rate token inside [...], which may be a name ('k1'), a number ('0.5'), or 'k1=0.5'."""
    token = token.strip()
    if not token:
        return default_name, None
    if "=" in token:
        name_part, val_part = token.split("=", 1)
        name = name_part.strip() or default_name
        try:
            val = float(val_part.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid numeric rate value in {token!r}.") from exc
        return name, val
    try:
        val = float(token)
        return default_name, val
    except ValueError:
        return token, None


def parse_reactions(reactions: str | Sequence[str]) -> ReactionSystem:
    """Parse a chemical reaction string or sequence of reaction strings.

    Supported formats:
        - "A ->[k1] B"
        - "2 A + B <->[kf, kr] C"
        - "A ->[k1=0.5] B"
        - "A -> B; B -> C"
    """
    if isinstance(reactions, str):
        lines = [
            part.strip()
            for line in reactions.splitlines()
            for part in line.split(";")
            if part.strip() and not part.strip().startswith("#")
        ]
    elif isinstance(reactions, Sequence):
        lines = [
            part.strip()
            for item in reactions
            for part in str(item).splitlines()
            for part in part.split(";")
            if part.strip() and not part.strip().startswith("#")
        ]
    else:
        raise ValueError("reactions must be a string or a sequence of strings.")

    if not lines:
        raise ValueError("At least one chemical reaction must be provided.")

    steps: list[ReactionStep] = []
    species_list: list[str] = []
    species_seen: set[str] = set()
    rate_names: list[str] = []
    rate_seen: set[str] = set()
    inferred_defaults: dict[str, float] = {}

    for idx, raw in enumerate(lines, start=1):
        matches = list(_ARROW_RE.finditer(raw))
        if len(matches) != 1:
            raise ValueError(
                f"Reaction {raw!r} must contain exactly one reaction arrow "
                "('->', '<->', '<=>', etc.)."
            )
        m = matches[0]
        arrow = m.group("arrow")
        rates_raw = m.group("rates")
        lhs_str = raw[: m.start()].strip()
        rhs_str = raw[m.end() :].strip()

        reactants = _parse_side(lhs_str, context=raw)
        products = _parse_side(rhs_str, context=raw)
        if not reactants and not products:
            raise ValueError(f"Reaction {raw!r} must have reactants or products.")

        for term in (*reactants, *products):
            if term.species not in species_seen:
                species_seen.add(term.species)
                species_list.append(term.species)

        is_reversible = arrow in {"<->", "<-->", "<=>", "⇌"}
        default_kf_name = f"k{idx}"
        default_kr_name = f"k_{idx}"

        if not is_reversible:
            if rates_raw is not None and "," in rates_raw:
                raise ValueError(
                    f"Irreversible reaction {raw!r} cannot specify multiple rate constants."
                )
            kf_name, kf_val = _parse_rate_token(rates_raw or "", default_kf_name)
            kr_name, kr_val = None, None
        else:
            if rates_raw is None or not rates_raw.strip():
                kf_name, kf_val = default_kf_name, None
                kr_name, kr_val = default_kr_name, None
            else:
                parts = [p.strip() for p in rates_raw.split(",")]
                if len(parts) == 1:
                    kf_name, kf_val = _parse_rate_token(parts[0], default_kf_name)
                    kr_name, kr_val = f"{kf_name}_rev", None
                elif len(parts) == 2:
                    kf_name, kf_val = _parse_rate_token(parts[0], default_kf_name)
                    kr_name, kr_val = _parse_rate_token(parts[1], default_kr_name)
                else:
                    raise ValueError(
                        f"Reversible reaction {raw!r} can specify at most 2 rate constants [kf, kr]."
                    )

        for r_name, r_val in ((kf_name, kf_val), (kr_name, kr_val)):
            if r_name is not None:
                if r_name not in rate_seen:
                    rate_seen.add(r_name)
                    rate_names.append(r_name)
                if r_val is not None:
                    inferred_defaults[r_name] = r_val

        steps.append(
            ReactionStep(
                index=idx,
                raw=raw,
                reactants=reactants,
                products=products,
                k_forward=kf_name,
                k_reverse=kr_name,
                default_kf=kf_val,
                default_kr=kr_val,
            )
        )

    return ReactionSystem(
        steps=steps,
        species=species_list,
        rate_names=rate_names,
        inferred_rate_defaults=inferred_defaults,
    )
