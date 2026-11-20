"""TangleChem: Interactive Tangle-style chemical equations and kinetics widgets."""

from .kinetics import TangleKinetics, initial_param_key
from .parser import (
    ReactantTerm,
    ReactionStep,
    ReactionSystem,
    format_chemical_label,
    format_rate_symbol,
    parse_reactions,
)

__version__ = "0.1.0"

__all__ = [
    "ReactantTerm",
    "ReactionStep",
    "ReactionSystem",
    "TangleKinetics",
    "format_chemical_label",
    "format_rate_symbol",
    "initial_param_key",
    "parse_reactions",
]
