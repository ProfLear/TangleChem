"""1D Time-Independent Schrödinger Equation parser, solver, and TangleSchrodinger anywidget."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any

import anywidget
import numpy as np
from scipy.linalg import eigh_tridiagonal
import traitlets

from .kinetics import _finite_float, _infer_step_and_digits, _normalize_color
from .parser import _SUBSCRIPT_MAP

_ESM_PATH = Path(__file__).parent / "static" / "schrodinger-tangle.js"
_CSS_PATH = Path(__file__).parent / "static" / "schrodinger-tangle.css"

_PARAM_PALETTE = (
    {"light": "#1e5bb8", "dark": "#75a7ff"},  # Blue
    {"light": "#b45b1b", "dark": "#ffad66"},  # Amber
    {"light": "#147a68", "dark": "#5ed5bd"},  # Teal
    {"light": "#a23b78", "dark": "#f08bc2"},  # Magenta
    {"light": "#5b48a8", "dark": "#b9a8ff"},  # Purple
    {"light": "#d1495b", "dark": "#ff7b8b"},  # Coral
)

_LEVEL_PALETTE = (
    {"light": "#246bce", "dark": "#75a7ff"},  # n=1 Blue
    {"light": "#d1495b", "dark": "#ff7b8b"},  # n=2 Coral
    {"light": "#147a68", "dark": "#5ed5bd"},  # n=3 Teal
    {"light": "#b45b1b", "dark": "#ffad66"},  # n=4 Amber
    {"light": "#6f5cbd", "dark": "#b9a8ff"},  # n=5 Purple
    {"light": "#a23b78", "dark": "#f08bc2"},  # n=6 Magenta
    {"light": "#3d7a22", "dark": "#88d46c"},  # n=7 Green
    {"light": "#007791", "dark": "#5ce1ff"},  # n=8 Cyan
)

_RESERVED_IDENTIFIERS = {
    "x",
    "pi",
    "e",
    "inf",
    "infty",
    "infinity",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "ln",
    "sqrt",
    "abs",
    "sinh",
    "cosh",
    "tanh",
    "step",
    "heaviside",
    "min",
    "max",
    "otherwise",
    "else",
    "for",
    "if",
    "and",
    "or",
}

_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_INLINE_DEFAULT_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)"
)


def format_quantum_symbol(name: str) -> str:
    """Format a parameter identifier like 'V0', 'V_b', 'x0', 'D_e' into a readable symbol."""
    special = {
        "omega": "ω",
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "lambda": "λ",
        "mu": "μ",
        "sigma": "σ",
        "hbar": "ℏ",
    }
    if name in special:
        return special[name]
    if "_" in name:
        base, sub = name.split("_", 1)
        base_sym = special.get(base, base)
        if sub.isdigit():
            return f"{base_sym}{sub.translate(_SUBSCRIPT_MAP)}"
        return f"{base_sym}_{sub}"
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if m:
        base, digits = m.group(1), m.group(2)
        base_sym = special.get(base, base)
        return f"{base_sym}{digits.translate(_SUBSCRIPT_MAP)}"
    return name


def _preprocess_math_expr(expr: str) -> tuple[str, dict[str, float]]:
    """Normalize mathematical syntax (^ -> **, |x| -> abs(x), inline k=1.0 -> k)."""
    s = expr.strip()
    inferred: dict[str, float] = {}

    # Extract inline defaults like "0.5 * (k=2.0) * x^2" or "k=2.0"
    def _replace_inline_def(match: re.Match[str]) -> str:
        p_name = match.group(1)
        p_val = float(match.group(2))
        if p_name not in _RESERVED_IDENTIFIERS:
            inferred[p_name] = p_val
        return p_name

    # Only match name=number when not part of <=, >=, ==, !=
    s = re.sub(
        r"(?<![<>!=])\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)(?!=)",
        _replace_inline_def,
        s,
    )

    # Convert |...| absolute value bars to abs(...)
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    # Normalize unicode minus and multiplication
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    # Convert infinity aliases
    s = re.sub(r"\\infty\b|\binfty\b|\binfinity\b|∞", "inf", s)
    # Convert caret power to **
    s = s.replace("^", "**")
    return s, inferred


def _extract_params_from_text(text: str) -> list[str]:
    """Return ordered list of user parameter names referenced in an expression or condition."""
    cleaned = text.replace("\\infty", "inf")
    found: list[str] = []
    seen: set[str] = set()
    for m in _IDENT_RE.finditer(cleaned):
        ident = m.group(1)
        if ident not in _RESERVED_IDENTIFIERS and ident not in seen:
            seen.add(ident)
            found.append(ident)
    return found


def _validate_ast_node(node: ast.AST, *, allow_comparisons: bool = False) -> None:
    """Validate that an AST only contains safe mathematical operations."""
    allowed_types: tuple[type[ast.AST], ...] = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    if allow_comparisons:
        allowed_types = (
            *allowed_types,
            ast.Compare,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Eq,
            ast.NotEq,
        )
    for sub in ast.walk(node):
        if not isinstance(sub, allowed_types):
            raise ValueError(
                f"Unsupported syntax in potential expression: {ast.dump(sub)}"
            )
        if isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name) or sub.func.id not in {
                "sin",
                "cos",
                "tan",
                "exp",
                "log",
                "ln",
                "sqrt",
                "abs",
                "sinh",
                "cosh",
                "tanh",
                "step",
                "heaviside",
                "min",
                "max",
            }:
                func_name = getattr(sub.func, "id", repr(sub.func))
                raise ValueError(
                    f"Unsupported function {func_name!r} in potential expression."
                )


@dataclass(frozen=True)
class PotentialCase:
    """A single branch of a piecewise potential V(x)."""

    expr_raw: str
    expr_py: str
    condition_raw: str | None = None
    condition_py: str | None = None

    @property
    def is_otherwise(self) -> bool:
        return self.condition_py is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr_raw": self.expr_raw,
            "expr_py": self.expr_py,
            "condition_raw": self.condition_raw,
            "condition_py": self.condition_py,
            "is_otherwise": self.is_otherwise,
        }


@dataclass
class PotentialSystem:
    """Parsed 1D quantum potential V(x) and parameter metadata."""

    cases: list[PotentialCase]
    param_names: list[str]
    inferred_defaults: dict[str, float] = field(default_factory=dict)

    @property
    def is_piecewise(self) -> bool:
        return len(self.cases) > 1 or not self.cases[0].is_otherwise

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_piecewise": self.is_piecewise,
            "cases": [c.to_dict() for c in self.cases],
            "param_names": list(self.param_names),
        }

    def evaluate(
        self, x: np.ndarray, params: Mapping[str, float]
    ) -> np.ndarray:
        """Evaluate V(x) on a 1D NumPy grid `x`."""
        env: dict[str, Any] = {
            "x": x,
            "pi": np.pi,
            "e": np.e,
            "inf": np.inf,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "ln": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "step": lambda u: np.where(u >= 0.0, 1.0, 0.0),
            "heaviside": lambda u: np.where(u >= 0.0, 1.0, 0.0),
            "min": np.minimum,
            "max": np.maximum,
        }
        for k, v in params.items():
            env[k] = float(v)

        result = np.full_like(x, np.inf, dtype=float)
        assigned = np.zeros_like(x, dtype=bool)

        for case in self.cases:
            val = eval(case.expr_py, {"__builtins__": {}}, env)  # noqa: S307
            val_arr = np.broadcast_to(np.asarray(val, dtype=float), x.shape)

            if case.is_otherwise:
                mask = ~assigned
            else:
                cond_mask = _eval_condition_numpy(case.condition_py or "", env, x.shape)
                mask = (~assigned) & cond_mask

            result = np.where(mask, val_arr, result)
            assigned |= mask

        return result

    def solve(
        self,
        params: Mapping[str, float],
        *,
        x_min: float = -3.0,
        x_max: float = 3.0,
        n_grid: int = 300,
        n_states: int = 4,
        mass: float = 1.0,
        hbar: float = 1.0,
    ) -> dict[str, Any]:
        """Solve the 1D Time-Independent Schrödinger Equation on [x_min, x_max].

        Uses second-order finite differences with exact Dirichlet (psi = 0)
        boundary conditions at the domain edges and any infinite-potential
        walls (V = inf).
        """
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min.")
        if mass <= 0:
            raise ValueError("mass must be positive.")

        x = np.linspace(float(x_min), float(x_max), int(n_grid))
        dx = float(x[1] - x[0])
        v = self.evaluate(x, params)

        # Interior points are grid points strictly inside the boundaries where V is finite
        WALL_CUTOFF = 1e6
        active = np.isfinite(v) & (v < WALL_CUTOFF)
        active[0] = False
        active[-1] = False

        active_indices = np.flatnonzero(active)
        k_req = max(1, int(n_states))

        if len(active_indices) < k_req + 2:
            return {
                "x": x,
                "V": v,
                "energies": np.array([], dtype=float),
                "wavefunctions": np.zeros((0, len(x)), dtype=float),
                "probabilities": np.zeros((0, len(x)), dtype=float),
            }

        kin_scale = (hbar**2) / (2.0 * mass * (dx**2))

        # Build symmetric Hamiltonian on active points; if active points have gaps
        # (e.g., an infinite internal barrier), decouple non-adjacent grid indices.
        n_act = len(active_indices)
        diag = 2.0 * kin_scale + v[active_indices]
        adjacent = np.diff(active_indices) == 1
        off_diag = np.where(adjacent, -kin_scale, 0.0)

        n_solve = min(k_req, n_act - 1)
        evals, evecs = eigh_tridiagonal(
            diag,
            off_diag,
            select="i",
            select_range=(0, n_solve - 1),
        )

        wavefunctions = np.zeros((n_solve, len(x)), dtype=float)
        probabilities = np.zeros((n_solve, len(x)), dtype=float)

        for idx in range(n_solve):
            psi_full = np.zeros_like(x)
            psi_full[active_indices] = evecs[:, idx]
            # Normalize integral(|psi|^2 dx) = 1
            norm = np.sqrt(np.trapezoid(psi_full**2, x))
            if norm > 1e-12:
                psi_full /= norm
            # Canonical sign convention: make first significant lobe positive
            max_abs = np.max(np.abs(psi_full))
            significant = np.flatnonzero(np.abs(psi_full) > 0.15 * max_abs)
            if len(significant) > 0 and psi_full[significant[0]] < 0:
                psi_full = -psi_full
            wavefunctions[idx] = psi_full
            probabilities[idx] = psi_full**2

        return {
            "x": x,
            "V": v,
            "energies": evals,
            "wavefunctions": wavefunctions,
            "probabilities": probabilities,
        }


def _eval_condition_numpy(
    cond_py: str, env: dict[str, Any], shape: tuple[int, ...]
) -> np.ndarray:
    """Evaluate a boolean condition (including chained comparisons like `0 <= x <= L`) on a NumPy array."""
    tree = ast.parse(cond_py, mode="eval")

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [_eval_node(v) for v in node.values]
            out = vals[0]
            for nxt in vals[1:]:
                out = (out & nxt) if isinstance(node.op, ast.And) else (out | nxt)
            return out
        if isinstance(node, ast.Compare):
            left = _eval_node(node.left)
            mask = np.ones(shape, dtype=bool)
            for op, comp in zip(node.ops, node.comparators):
                right = _eval_node(comp)
                if isinstance(op, ast.Lt):
                    part = left < right
                elif isinstance(op, ast.LtE):
                    part = left <= right
                elif isinstance(op, ast.Gt):
                    part = left > right
                elif isinstance(op, ast.GtE):
                    part = left >= right
                elif isinstance(op, ast.Eq):
                    part = left == right
                elif isinstance(op, ast.NotEq):
                    part = left != right
                else:
                    raise ValueError(f"Unsupported comparison operator: {op}")
                mask = mask & np.broadcast_to(np.asarray(part, dtype=bool), shape)
                left = right
            return mask
        # Standard arithmetic expression
        code = compile(ast.Expression(body=node), "<cond>", "eval")
        return eval(code, {"__builtins__": {}}, env)  # noqa: S307

    return np.broadcast_to(np.asarray(_eval_node(tree), dtype=bool), shape)


def _split_case_rule(raw_rule: str) -> tuple[str, str | None]:
    """Split a single piecewise rule into (expression_str, condition_str | None).

    Supported formats:
        - "0 for 0 <= x <= L"
        - "0 if -L/2 <= x <= L/2"
        - "0, 0 <= x <= L"
        - "inf otherwise" / "V0 else" / "V0, otherwise"
        - "0.5 * k * x^2" (unconditional)
    """
    rule = raw_rule.strip()
    # Check "expr for/if/when condition"
    m_for = re.match(
        r"^(?P<expr>.+?)\s+(?:for|if|when)\s+(?P<cond>.+)$",
        rule,
        flags=re.IGNORECASE,
    )
    if m_for:
        expr_part = m_for.group("expr").strip().rstrip(",")
        cond_part = m_for.group("cond").strip()
        if cond_part.lower() in {"otherwise", "else"}:
            return expr_part, None
        return expr_part, cond_part

    # Check "expr otherwise" or "expr else"
    m_other = re.match(
        r"^(?P<expr>.+?)(?:\s*,\s*|\s+)(?:otherwise|else)\s*$",
        rule,
        flags=re.IGNORECASE,
    )
    if m_other:
        return m_other.group("expr").strip(), None

    # Check comma-separated "expr, condition" (where condition has comparison operators)
    if "," in rule:
        lhs, rhs = rule.split(",", 1)
        if any(op in rhs for op in ("<", ">", "=")) or rhs.strip().lower() in {
            "otherwise",
            "else",
        }:
            cond_clean = rhs.strip()
            if cond_clean.lower() in {"otherwise", "else"}:
                return lhs.strip(), None
            return lhs.strip(), cond_clean

    return rule, None


def parse_potential(
    potential: str | Sequence[str | tuple[str, str | None]],
) -> PotentialSystem:
    """Parse a 1D potential V(x) specification into a PotentialSystem.

    Examples:
        - Algebraic: ``"0.5 * k * (x - x0)^2"``
        - Piecewise string: ``"0 for 0 <= x <= L; inf otherwise"``
        - Piecewise list:
          ``["0 for -L/2 <= x <= L/2", "V0 otherwise"]``
        - LaTeX cases syntax:
          ``r"\\cases{0, 0 <= x <= L \\\\ \\infty, otherwise}"``
    """
    raw_rules: list[tuple[str, str | None]] = []

    if isinstance(potential, str):
        text = potential.strip()
        # Strip optional leading "V(x) =" or "V ="
        text = re.sub(r"^V\s*(?:\(\s*x\s*\))?\s*=\s*", "", text)
        # Unwrap optional \cases{...} or \begin{cases}...\end{cases}
        m_cases = re.match(
            r"^\\cases\s*\{(?P<body>.*)\}\s*$", text, flags=re.DOTALL
        ) or re.match(
            r"^\\begin\{cases\}(?P<body>.*)\\end\{cases\}\s*$",
            text,
            flags=re.DOTALL,
        )
        if m_cases:
            body = m_cases.group("body")
            parts = re.split(r"\\\\|;|\n", body)
        else:
            parts = re.split(r";|\n", text)

        for p in parts:
            p_clean = p.strip().replace("&", ",")
            if p_clean and not p_clean.startswith("#"):
                raw_rules.append(_split_case_rule(p_clean))
    elif isinstance(potential, Sequence):
        for item in potential:
            if isinstance(item, tuple) and len(item) == 2:
                expr_s, cond_s = str(item[0]).strip(), item[1]
                if cond_s is not None and str(cond_s).strip().lower() in {
                    "otherwise",
                    "else",
                    "",
                }:
                    cond_s = None
                raw_rules.append((expr_s, str(cond_s).strip() if cond_s else None))
            elif isinstance(item, str):
                for p in re.split(r";|\n", item):
                    if p.strip():
                        raw_rules.append(_split_case_rule(p.strip()))
            else:
                raise ValueError(
                    "Each item in potential sequence must be a string or (expr, condition) tuple."
                )
    else:
        raise ValueError("potential must be a string or a sequence of rules.")

    if not raw_rules:
        raise ValueError("Potential specification cannot be empty.")

    cases: list[PotentialCase] = []
    param_names: list[str] = []
    param_seen: set[str] = set()
    inferred_defaults: dict[str, float] = {}

    for expr_raw, cond_raw in raw_rules:
        expr_py, inf_expr = _preprocess_math_expr(expr_raw)
        inferred_defaults.update(inf_expr)
        try:
            expr_tree = ast.parse(expr_py, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Invalid math expression {expr_raw!r} in potential."
            ) from exc
        _validate_ast_node(expr_tree, allow_comparisons=False)

        for p_name in _extract_params_from_text(expr_py):
            if p_name not in param_seen:
                param_seen.add(p_name)
                param_names.append(p_name)

        if cond_raw is not None:
            cond_py, inf_cond = _preprocess_math_expr(cond_raw)
            inferred_defaults.update(inf_cond)
            try:
                cond_tree = ast.parse(cond_py, mode="eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"Invalid condition {cond_raw!r} in potential."
                ) from exc
            _validate_ast_node(cond_tree, allow_comparisons=True)
            for p_name in _extract_params_from_text(cond_py):
                if p_name not in param_seen:
                    param_seen.add(p_name)
                    param_names.append(p_name)
            # Clean raw condition for display (remove inline =val)
            cond_display = re.sub(
                r"(?<![<>!=])\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)(?!=)",
                r"\1",
                cond_raw.strip(),
            )
        else:
            cond_py = None
            cond_display = None

        expr_display = re.sub(
            r"(?<![<>!=])\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)(?!=)",
            r"\1",
            expr_raw.strip(),
        )

        cases.append(
            PotentialCase(
                expr_raw=expr_display,
                expr_py=expr_py,
                condition_raw=cond_display,
                condition_py=cond_py,
            )
        )

    return PotentialSystem(
        cases=cases,
        param_names=param_names,
        inferred_defaults=inferred_defaults,
    )


def _default_for_quantum_param(name: str) -> tuple[float, float, float]:
    """Return sensible (default_value, min_value, max_value) for common quantum parameter names."""
    low = name.lower()
    if low in {"l", "width", "w"}:
        return 2.0, 0.4, 8.0
    if low in {"a", "barrier_width"}:
        return 0.5, 0.05, 3.0
    if low.startswith("v") or low in {"d_e", "de", "barrier", "depth"}:
        return 12.0, 0.0, 40.0
    if low in {"k", "omega"}:
        return 2.0, 0.1, 15.0
    if low in {"x0", "r_e", "re", "center"}:
        return 0.0, -2.5, 2.5
    return 1.0, 0.0, 10.0


class TangleSchrodinger(anywidget.AnyWidget):
    """Interactive 1D Time-Independent Schrödinger Equation widget.

    Define any algebraic or piecewise 1D potential energy well V(x), and drag
    parameters (well width L, barrier height V0, force constant k, particle
    mass m, etc.) directly inside the potential equation to watch quantized
    energy levels E_n and wavefunctions psi_n(x) update at 60 FPS.
    """

    _esm = _ESM_PATH
    _css = _CSS_PATH

    potential_data = traitlets.Dict(default_value={}).tag(sync=True)
    parameters = traitlets.Dict(default_value={}).tag(sync=True)
    values = traitlets.Dict(default_value={}).tag(sync=True)
    committed_values = traitlets.Dict(default_value={}).tag(sync=True)
    dragging = traitlets.Bool(False).tag(sync=True)
    active_param = traitlets.Unicode("").tag(sync=True)

    x_min = traitlets.Float(-3.0).tag(sync=True)
    x_max = traitlets.Float(3.0).tag(sync=True)
    n_states = traitlets.Int(4).tag(sync=True)
    n_grid = traitlets.Int(240).tag(sync=True)
    view_mode = traitlets.Enum(
        ["wavefunction", "probability"], default_value="wavefunction"
    ).tag(sync=True)
    show_plot = traitlets.Bool(True).tag(sync=True)
    show_ghost = traitlets.Bool(True).tag(sync=True)
    display_mode = traitlets.Enum(
        ["both", "number", "symbol"], default_value="both"
    ).tag(sync=True)
    theme = traitlets.Enum(
        ["auto", "light", "dark"], default_value="auto"
    ).tag(sync=True)
    level_colors = traitlets.List(default_value=list(_LEVEL_PALETTE)).tag(
        sync=True
    )

    def __init__(
        self,
        potential: str | Sequence[str | tuple[str, str | None]],
        parameters: Mapping[str, float | int | Mapping[str, Any]] | None = None,
        *,
        x_domain: tuple[float, float] | None = None,
        n_states: int = 4,
        mass: float | Mapping[str, Any] = 1.0,
        n_grid: int = 240,
        view_mode: str = "wavefunction",
        show_plot: bool = True,
        show_ghost: bool = True,
        display_mode: str = "both",
        theme: str = "auto",
        **kwargs: Any,
    ) -> None:
        r"""Create a TangleSchrodinger widget.

        Args:
            potential: 1D potential energy V(x), either as an algebraic
                formula (e.g. ``"0.5 * k * x^2"``) or a piecewise definition
                (e.g. ``["0 for 0 <= x <= L", "inf otherwise"]`` or
                ``"0 for -L/2 <= x <= L/2; V0 otherwise"``).
            parameters: Mapping from parameter name in ``potential`` (e.g.
                ``"L"``, ``"V0"``, ``"k"``) to either a numeric value or a
                parameter specification dict (``value``, ``min_value``,
                ``max_value``, ``step``, ``digits``, ``symbol``, ``unit``,
                ``color``). Any parameter present in ``potential`` but omitted
                here gets automatic sensible bounds.
            x_domain: Spatial domain ``(x_min, x_max)`` for the 1D grid.
                Automatically inferred from ``L`` if not provided.
            n_states: Number of lowest bound eigenstates to compute and display.
            mass: Particle mass ``m`` (in atomic units where ``hbar = 1``), as
                a float or parameter dict.
            n_grid: Number of spatial grid points for the finite-difference
                Hamiltonian solver.
            view_mode: ``"wavefunction"`` (plots $\psi_n(x)$ on energy baseline
                $E_n$) or ``"probability"`` (plots $|\psi_n(x)|^2$ on $E_n$).
            show_plot: Whether to display the built-in interactive SVG plot.
            show_ghost: Whether to display dashed pre-drag reference energy
                levels and wavefunctions while dragging.
            display_mode: How parameters inside $V(x)$ are displayed:
                ``"both"`` (``L = 2.00``), ``"number"`` (``2.00``), or
                ``"symbol"`` (``L``, revealing the number on drag).
            theme: Color theme: ``"auto"``, ``"light"``, or ``"dark"``.
            **kwargs: Forwarded to ``anywidget.AnyWidget``.
        """
        if view_mode not in {"wavefunction", "probability"}:
            raise ValueError(
                "view_mode must be 'wavefunction' or 'probability'."
            )
        if display_mode not in {"both", "number", "symbol"}:
            raise ValueError("display_mode must be 'both', 'number', or 'symbol'.")
        if theme not in {"auto", "light", "dark"}:
            raise ValueError("theme must be 'auto', 'light', or 'dark'.")

        self._potential: PotentialSystem = parse_potential(potential)
        raw_params = dict(parameters or {})

        # Allow "m" in parameters or via mass=...
        if "m" in raw_params and mass == 1.0:
            mass = raw_params.pop("m")

        unknown = sorted(set(raw_params) - set(self._potential.param_names))
        if unknown:
            raise ValueError(
                "Unknown parameter(s) not referenced in potential V(x): "
                + ", ".join(unknown)
            )

        norm_params: dict[str, dict[str, Any]] = {}
        for idx, p_name in enumerate(self._potential.param_names):
            def_val, def_min, def_max = _default_for_quantum_param(p_name)
            if p_name in self._potential.inferred_defaults:
                def_val = self._potential.inferred_defaults[p_name]
            raw_spec = raw_params.get(p_name)
            norm_params[p_name] = _normalize_quantum_param(
                p_name,
                raw_spec,
                default_value=def_val,
                default_min=def_min,
                default_max=def_max,
                default_symbol=format_quantum_symbol(p_name),
                kind="potential",
                index=idx,
            )

        # Particle mass parameter 'm'
        norm_params["m"] = _normalize_quantum_param(
            "m",
            mass,
            default_value=1.0,
            default_min=0.1,
            default_max=10.0,
            default_symbol="m",
            kind="mass",
            index=len(norm_params),
        )

        # Auto-infer x_domain if not explicitly provided
        if x_domain is None:
            if "L" in norm_params:
                l_max = max(
                    norm_params["L"]["value"] * 1.6,
                    norm_params["L"]["max_value"] * 0.65,
                )
                # Check if potential starts at 0 (e.g. "0 <= x <= L")
                joined_conds = " ".join(
                    c.condition_py or "" for c in self._potential.cases
                )
                if "0 <=" in joined_conds or "0 <" in joined_conds:
                    x_min_val = -0.15 * l_max
                    x_max_val = 1.15 * l_max
                else:
                    x_min_val = -0.75 * l_max
                    x_max_val = 0.75 * l_max
            else:
                x_min_val, x_max_val = -3.5, 3.5
        else:
            x_min_val = _finite_float(x_domain[0], name="x_domain", field="x_min")
            x_max_val = _finite_float(x_domain[1], name="x_domain", field="x_max")
            if x_max_val <= x_min_val:
                raise ValueError("x_domain must have x_min < x_max.")

        values = {name: spec["value"] for name, spec in norm_params.items()}

        super().__init__(
            potential_data=self._potential.to_dict(),
            parameters=norm_params,
            values=values,
            committed_values=dict(values),
            x_min=x_min_val,
            x_max=x_max_val,
            n_states=max(1, min(10, int(n_states))),
            n_grid=max(60, min(600, int(n_grid))),
            view_mode=view_mode,
            show_plot=show_plot,
            show_ghost=show_ghost,
            display_mode=display_mode,
            theme=theme,
            **kwargs,
        )

    @traitlets.observe("values")
    def _follow_values(self, change: dict[str, Any]) -> None:
        if not self.dragging:
            self.committed_values = dict(self.values)

    @property
    def potential(self) -> PotentialSystem:
        """The parsed PotentialSystem backing this widget."""
        return self._potential

    def solve(
        self,
        *,
        use_committed: bool = False,
        n_states: int | None = None,
        n_grid: int | None = None,
    ) -> dict[str, Any]:
        """Solve the 1D Schrödinger equation in Python using SciPy.

        Returns:
            Dictionary with keys ``"x"``, ``"V"``, ``"energies"``,
            ``"wavefunctions"``, and ``"probabilities"``.
        """
        source = self.committed_values if use_committed else self.values
        p_vals = {
            k: float(source[k])
            for k in self._potential.param_names
            if k in source
        }
        m_val = float(source.get("m", 1.0))
        return self._potential.solve(
            p_vals,
            x_min=self.x_min,
            x_max=self.x_max,
            n_grid=int(self.n_grid if n_grid is None else n_grid),
            n_states=int(self.n_states if n_states is None else n_states),
            mass=m_val,
        )

    @classmethod
    def particle_in_a_box(
        cls,
        L: float = 2.0,
        *,
        n_states: int = 4,
        **kwargs: Any,
    ) -> TangleSchrodinger:
        """Create a 1D Particle in a Box (Infinite Square Well) on [0, L]."""
        max_l = max(5.0, round(L * 2.5, 1))
        return cls(
            potential=[
                "0 for 0 <= x <= L",
                "inf otherwise",
            ],
            parameters={
                "L": {
                    "value": L,
                    "min_value": 0.5,
                    "max_value": max_l,
                    "step": 0.05,
                    "label": "Box width L",
                }
            },
            x_domain=(-0.15 * max_l * 0.6, max_l * 0.75),
            n_states=n_states,
            **kwargs,
        )

    @classmethod
    def harmonic_oscillator(
        cls,
        k: float = 2.0,
        x0: float = 0.0,
        *,
        n_states: int = 5,
        **kwargs: Any,
    ) -> TangleSchrodinger:
        """Create a 1D Quantum Harmonic Oscillator V(x) = 1/2 k (x - x0)^2."""
        return cls(
            potential="0.5 * k * (x - x0)^2",
            parameters={
                "k": {
                    "value": k,
                    "min_value": 0.2,
                    "max_value": 12.0,
                    "step": 0.1,
                    "label": "Force constant k",
                },
                "x0": {
                    "value": x0,
                    "min_value": -1.5,
                    "max_value": 1.5,
                    "step": 0.05,
                    "label": "Equilibrium center x₀",
                },
            },
            x_domain=(-3.5, 3.5),
            n_states=n_states,
            **kwargs,
        )

    @classmethod
    def finite_well(
        cls,
        L: float = 2.4,
        V0: float = 10.0,
        *,
        n_states: int = 4,
        **kwargs: Any,
    ) -> TangleSchrodinger:
        """Create a 1D Finite Square Well of width L and barrier height V0."""
        return cls(
            potential=[
                "0 for -L/2 <= x <= L/2",
                "V0 otherwise",
            ],
            parameters={
                "L": {
                    "value": L,
                    "min_value": 0.6,
                    "max_value": 5.0,
                    "step": 0.05,
                    "label": "Well width L",
                },
                "V0": {
                    "value": V0,
                    "min_value": 1.0,
                    "max_value": 30.0,
                    "step": 0.5,
                    "label": "Barrier height V₀",
                },
            },
            x_domain=(-3.5, 3.5),
            n_states=n_states,
            **kwargs,
        )

    @classmethod
    def double_well(
        cls,
        L: float = 3.2,
        a: float = 0.45,
        Vb: float = 6.0,
        V0: float = 16.0,
        *,
        n_states: int = 4,
        **kwargs: Any,
    ) -> TangleSchrodinger:
        """Create a symmetric Double Square Well with central tunneling barrier."""
        return cls(
            potential=[
                "Vb for -a/2 <= x <= a/2",
                "0 for -L/2 <= x <= L/2",
                "V0 otherwise",
            ],
            parameters={
                "Vb": {
                    "value": Vb,
                    "min_value": 0.0,
                    "max_value": 20.0,
                    "step": 0.25,
                    "label": "Central barrier height V_b",
                },
                "a": {
                    "value": a,
                    "min_value": 0.05,
                    "max_value": 1.5,
                    "step": 0.05,
                    "label": "Barrier thickness a",
                },
                "L": {
                    "value": L,
                    "min_value": 1.6,
                    "max_value": 5.0,
                    "step": 0.05,
                    "label": "Total well width L",
                },
                "V0": {
                    "value": V0,
                    "min_value": 4.0,
                    "max_value": 35.0,
                    "step": 0.5,
                    "label": "Outer wall height V₀",
                },
            },
            x_domain=(-3.2, 3.2),
            n_states=n_states,
            **kwargs,
        )


def _normalize_quantum_param(
    name: str,
    raw: float | int | Mapping[str, Any] | None,
    *,
    default_value: float,
    default_min: float,
    default_max: float,
    default_symbol: str,
    kind: str,
    index: int,
) -> dict[str, Any]:
    if raw is None:
        spec: dict[str, Any] = {"value": default_value}
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        spec = {"value": float(raw)}
    elif isinstance(raw, Mapping):
        spec = dict(raw)
    else:
        raise ValueError(
            f"Parameter {name!r} must be a number or a configuration dictionary."
        )

    val = _finite_float(spec.get("value", default_value), name=name, field="value")
    min_val = _finite_float(
        spec.get("min_value", min(default_min, val)), name=name, field="min_value"
    )
    max_val = _finite_float(
        spec.get("max_value", max(default_max, val * 2.5 if val > 0 else default_max)),
        name=name,
        field="max_value",
    )
    if max_val <= min_val:
        max_val = min_val + 1.0

    inf_step, inf_digits = _infer_step_and_digits(val, max_val - min_val)
    step = _finite_float(spec.get("step", inf_step), name=name, field="step")
    if step <= 0:
        raise ValueError(f"parameter {name!r} step must be positive.")
    digits = int(spec.get("digits", inf_digits))
    color = _normalize_color(
        spec.get("color"), name=name, index=index, palette=_PARAM_PALETTE
    )
    return {
        "name": name,
        "kind": kind,
        "value": val,
        "min_value": min_val,
        "max_value": max_val,
        "step": step,
        "digits": max(0, digits),
        "symbol": str(spec.get("symbol", default_symbol)),
        "label": str(spec.get("label", f"Parameter {name}")),
        "unit": str(spec.get("unit", "")),
        "pixels_per_step": max(1, int(spec.get("pixels_per_step", 6))),
        "color": color,
    }
