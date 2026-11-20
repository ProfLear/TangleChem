"""TangleChem: Interactive Tangle-style chemical equations, kinetics, and quantum widgets."""

from .kinetics import TangleKinetics, initial_param_key
from .parser import (
    ReactantTerm,
    ReactionStep,
    ReactionSystem,
    format_chemical_label,
    format_rate_symbol,
    parse_reactions,
)
from .schrodinger import (
    PotentialCase,
    PotentialSystem,
    TangleSchrodinger,
    format_quantum_symbol,
    parse_potential,
)

__version__ = "0.2.0"

__all__ = [
    "PotentialCase",
    "PotentialSystem",
    "ReactantTerm",
    "ReactionStep",
    "ReactionSystem",
    "TangleKinetics",
    "TangleSchrodinger",
    "format_chemical_label",
    "format_quantum_symbol",
    "format_rate_symbol",
    "initial_param_key",
    "parse_potential",
    "parse_reactions",
]
