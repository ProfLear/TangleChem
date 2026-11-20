// Interactive TangleSchrodinger anywidget frontend (1D Time-Independent Schrödinger Equation)

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function roundToStep(value, min, step, digits) {
  const steps = Math.round((value - min) / step);
  const snapped = min + steps * step;
  return Number(snapped.toFixed(Math.max(digits + 2, 6)));
}

function formatNumber(value, digits) {
  if (!Number.isFinite(value)) return "0";
  return Number(value).toFixed(digits);
}

function resolveColor(colorObj, isDark) {
  if (!colorObj) return isDark ? "#75a7ff" : "#246bce";
  if (typeof colorObj === "string") return colorObj;
  return isDark ? colorObj.dark || colorObj.light : colorObj.light;
}

function detectDarkMode(rootEl, themeSetting) {
  if (themeSetting === "dark") return true;
  if (themeSetting === "light") return false;
  if (
    document.documentElement.classList.contains("dark") ||
    document.body.classList.contains("dark") ||
    document.documentElement.getAttribute("data-theme") === "dark" ||
    document.body.getAttribute("data-vscode-theme-kind")?.includes("dark")
  ) {
    return true;
  }
  return (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  );
}

/**
 * Compile a Python-normalized arithmetic expression (`expr_py`) or chained
 * comparison condition (`condition_py`) into a fast JS evaluator function.
 */
function compileMathExpr(pyExpr) {
  if (!pyExpr) return () => 0;
  // Convert chained comparisons like "0 <= x <= L" or "-L/2 < x < L/2"
  // into "(0 <= x) && (x <= L)"
  let js = pyExpr.trim();

  // Replace Python boolean keywords
  js = js.replace(/\band\b/g, "&&").replace(/\bor\b/g, "||");

  // Rewrite chained comparisons: a op1 b op2 c -> (a op1 b) && (b op2 c)
  const chainRe =
    /([^<>!=&|]+?)\s*(<=|>=|<|>)\s*([^<>!=&|]+?)\s*(<=|>=|<|>)\s*([^<>!=&|]+)/;
  let guard = 0;
  while (chainRe.test(js) && guard < 6) {
    js = js.replace(
      chainRe,
      "((($1) $2 ($3)) && (($3) $4 ($5)))"
    );
    guard++;
  }

  try {
    const fn = new Function(
      "env",
      `"use strict";
       const { x, pi, e, inf, sin, cos, tan, exp, log, ln, sqrt, abs, sinh, cosh, tanh, step, heaviside, min, max } = env;
       return (${js});`
    );
    return (env) => fn(env);
  } catch (_err) {
    return () => 0;
  }
}

/**
 * Evaluate the 1D potential V(x) across grid `xGrid`.
 */
function evaluatePotentialGrid(potentialData, paramValues, xGrid) {
  const cases = potentialData.cases || [];
  const compiledCases = cases.map((c) => {
    const exprFn = compileMathExpr(c.expr_py);
    const condFn = c.is_otherwise ? null : compileMathExpr(c.condition_py);
    return { exprFn, condFn, isOtherwise: Boolean(c.is_otherwise) };
  });

  const baseEnv = {
    x: 0,
    pi: Math.PI,
    e: Math.E,
    inf: Infinity,
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    exp: Math.exp,
    log: Math.log,
    ln: Math.log,
    sqrt: Math.sqrt,
    abs: Math.abs,
    sinh: Math.sinh,
    cosh: Math.cosh,
    tanh: Math.tanh,
    step: (u) => (u >= 0 ? 1 : 0),
    heaviside: (u) => (u >= 0 ? 1 : 0),
    min: Math.min,
    max: Math.max,
  };

  for (const [k, v] of Object.entries(paramValues)) {
    baseEnv[k] = Number(v);
  }

  // Build dynamic function with parameter names in scope
  const paramKeys = Object.keys(paramValues).filter((k) =>
    /^[A-Za-z_][A-Za-z0-9_]*$/.test(k)
  );
  const compiledWithParams = cases.map((c) => {
    let exprJs = (c.expr_py || "0").trim();
    let condJs = c.condition_py ? c.condition_py.trim() : null;
    if (condJs) {
      condJs = condJs.replace(/\band\b/g, "&&").replace(/\bor\b/g, "||");
      const chainRe =
        /([^<>!=&|]+?)\s*(<=|>=|<|>)\s*([^<>!=&|]+?)\s*(<=|>=|<|>)\s*([^<>!=&|]+)/;
      let g = 0;
      while (chainRe.test(condJs) && g < 6) {
        condJs = condJs.replace(
          chainRe,
          "((($1) $2 ($3)) && (($3) $4 ($5)))"
        );
        g++;
      }
    }
    const destructure = [
      "x",
      "pi",
      "e",
      "inf",
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
      ...paramKeys,
    ].join(", ");
    let exprFn;
    let condFn = null;
    try {
      exprFn = new Function(
        "env",
        `const { ${destructure} } = env; return Number(${exprJs});`
      );
    } catch (_e) {
      exprFn = () => Infinity;
    }
    if (!c.is_otherwise && condJs) {
      try {
        condFn = new Function(
          "env",
          `const { ${destructure} } = env; return Boolean(${condJs});`
        );
      } catch (_e) {
        condFn = () => false;
      }
    }
    return { exprFn, condFn, isOtherwise: Boolean(c.is_otherwise) };
  });

  const n = xGrid.length;
  const vGrid = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    baseEnv.x = xGrid[i];
    let matched = false;
    for (let c = 0; c < compiledWithParams.length; c++) {
      const rule = compiledWithParams[c];
      if (rule.isOtherwise || (rule.condFn && rule.condFn(baseEnv))) {
        const val = rule.exprFn(baseEnv);
        vGrid[i] = Number.isNaN(val) ? Infinity : val;
        matched = true;
        break;
      }
    }
    if (!matched) {
      vGrid[i] = Infinity;
    }
  }
  return vGrid;
}

/**
 * Count number of eigenvalues of symmetric tridiagonal matrix (d, e) strictly < lambda
 * using Sturm sequence recurrence.
 */
function sturmCount(d, e2, n, lambda) {
  let count = 0;
  let prev = 1.0;
  const EPS = 1e-14;
  for (let i = 0; i < n; i++) {
    let cur = d[i] - lambda - (i > 0 ? e2[i - 1] / prev : 0.0);
    if (Math.abs(cur) < EPS) {
      cur = cur < 0 ? -EPS : EPS;
    }
    if (cur < 0) {
      count++;
    }
    prev = cur;
  }
  return count;
}

/**
 * Compute the lowest `kStates` eigenvalues and eigenvectors of symmetric tridiagonal
 * matrix with main diagonal `d` (length n) and off-diagonal `e` (length n-1).
 * Runs in O(kStates * n) time (< 1 ms for n=240, k=5).
 */
function solveTridiagonalEigen(d, e, kStates) {
  const n = d.length;
  const k = Math.min(kStates, n - 1);
  if (n === 0 || k <= 0) {
    return { eigenvalues: [], eigenvectors: [] };
  }

  const e2 = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    e2[i] = e[i] * e[i];
  }

  // Gershgorin circle bounds for all eigenvalues
  let gMin = Infinity;
  let gMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const r =
      (i > 0 ? Math.abs(e[i - 1]) : 0) + (i < n - 1 ? Math.abs(e[i]) : 0);
    if (d[i] - r < gMin) gMin = d[i] - r;
    if (d[i] + r > gMax) gMax = d[i] + r;
  }
  gMin -= 1.0;
  gMax += 1.0;

  const eigenvalues = new Float64Array(k);
  for (let m = 0; m < k; m++) {
    let lo = m === 0 ? gMin : eigenvalues[m - 1];
    let hi = gMax;
    for (let iter = 0; iter < 54; iter++) {
      const mid = 0.5 * (lo + hi);
      if (sturmCount(d, e2, n, mid) <= m) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    eigenvalues[m] = 0.5 * (lo + hi);
  }

  // Shifted Thomas tridiagonal solver for inverse iteration
  const eigenvectors = [];
  const cPrime = new Float64Array(n);
  const dPrime = new Float64Array(n);

  for (let m = 0; m < k; m++) {
    const lambda = eigenvalues[m];
    // Perturb shift slightly; if nearly degenerate with previous state, use distinct shift
    const shift =
      m > 0 && Math.abs(lambda - eigenvalues[m - 1]) < 1e-6
        ? lambda - 1.3e-6
        : lambda + 1.1e-7;

    const vec = new Float64Array(n);
    // Seed vector with smooth spatial variation + parity symmetry breaking
    for (let i = 0; i < n; i++) {
      const u = (i + 1) / (n + 1);
      vec[i] = Math.sin((m + 1) * Math.PI * u) + 0.15 * Math.cos((m + 0.5) * Math.PI * u);
    }

    for (let pass = 0; pass < 4; pass++) {
      // Orthogonalize against previously found eigenvectors if eigenvalues are close
      for (let prev = 0; prev < m; prev++) {
        if (Math.abs(eigenvalues[prev] - lambda) < 0.05) {
          const pv = eigenvectors[prev];
          let dot = 0;
          for (let i = 0; i < n; i++) dot += vec[i] * pv[i];
          for (let i = 0; i < n; i++) vec[i] -= dot * pv[i];
        }
      }

      // Forward sweep of Thomas algorithm for (T - shift*I) x = vec
      let denom = d[0] - shift;
      if (Math.abs(denom) < 1e-12) denom = 1e-12;
      cPrime[0] = n > 1 ? e[0] / denom : 0;
      dPrime[0] = vec[0] / denom;

      for (let i = 1; i < n; i++) {
        const a_i = e[i - 1];
        const b_i = d[i] - shift;
        const c_i = i < n - 1 ? e[i] : 0;
        let dn = b_i - a_i * cPrime[i - 1];
        if (Math.abs(dn) < 1e-12) dn = dn < 0 ? -1e-12 : 1e-12;
        cPrime[i] = c_i / dn;
        dPrime[i] = (vec[i] - a_i * dPrime[i - 1]) / dn;
      }

      // Back substitution
      vec[n - 1] = dPrime[n - 1];
      for (let i = n - 2; i >= 0; i--) {
        vec[i] = dPrime[i] - cPrime[i] * vec[i + 1];
      }

      // Normalize L2
      let norm2 = 0;
      for (let i = 0; i < n; i++) norm2 += vec[i] * vec[i];
      const invNorm = norm2 > 0 ? 1.0 / Math.sqrt(norm2) : 1.0;
      for (let i = 0; i < n; i++) vec[i] *= invNorm;
    }

    // Final Gram-Schmidt against any close previous eigenvectors
    for (let prev = 0; prev < m; prev++) {
      if (Math.abs(eigenvalues[prev] - lambda) < 0.1) {
        const pv = eigenvectors[prev];
        let dot = 0;
        for (let i = 0; i < n; i++) dot += vec[i] * pv[i];
        for (let i = 0; i < n; i++) vec[i] -= dot * pv[i];
      }
    }
    let norm2 = 0;
    for (let i = 0; i < n; i++) norm2 += vec[i] * vec[i];
    const invNorm = norm2 > 0 ? 1.0 / Math.sqrt(norm2) : 1.0;
    for (let i = 0; i < n; i++) vec[i] *= invNorm;

    eigenvectors.push(vec);
  }

  return { eigenvalues: Array.from(eigenvalues), eigenvectors };
}

/**
 * Full 1D Schrödinger solver on [xMin, xMax].
 */
function solveSchrodinger1D(
  potentialData,
  paramValues,
  xMin,
  xMax,
  nGrid,
  nStates
) {
  const N = Math.max(60, Number(nGrid || 240));
  const x = new Float64Array(N);
  const dx = (xMax - xMin) / (N - 1);
  for (let i = 0; i < N; i++) {
    x[i] = xMin + i * dx;
  }

  const V = evaluatePotentialGrid(potentialData, paramValues, x);
  const mass = Math.max(0.01, Number(paramValues.m ?? 1.0));
  const WALL_CUTOFF = 1e6;

  // Identify interior active points where V(x) is finite
  const activeIndices = [];
  for (let i = 1; i < N - 1; i++) {
    if (Number.isFinite(V[i]) && V[i] < WALL_CUTOFF) {
      activeIndices.push(i);
    }
  }

  const nAct = activeIndices.length;
  if (nAct < 4) {
    return { x, V, energies: [], wavefunctions: [], probabilities: [] };
  }

  const kinScale = 1.0 / (2.0 * mass * dx * dx);
  const diag = new Float64Array(nAct);
  const offDiag = new Float64Array(nAct - 1);

  for (let i = 0; i < nAct; i++) {
    diag[i] = 2.0 * kinScale + V[activeIndices[i]];
    if (i < nAct - 1) {
      const isAdjacent = activeIndices[i + 1] - activeIndices[i] === 1;
      offDiag[i] = isAdjacent ? -kinScale : 0.0;
    }
  }

  const { eigenvalues, eigenvectors } = solveTridiagonalEigen(
    diag,
    offDiag,
    nStates
  );

  const wavefunctions = [];
  const probabilities = [];

  for (let s = 0; s < eigenvalues.length; s++) {
    const psi = new Float64Array(N);
    const evec = eigenvectors[s];
    for (let i = 0; i < nAct; i++) {
      psi[activeIndices[i]] = evec[i];
    }

    // Normalize integral(|psi|^2 dx) = 1
    let integral = 0;
    for (let i = 0; i < N; i++) {
      integral += psi[i] * psi[i] * dx;
    }
    const norm = integral > 1e-14 ? Math.sqrt(integral) : 1.0;
    let maxAbs = 0;
    for (let i = 0; i < N; i++) {
      psi[i] /= norm;
      if (Math.abs(psi[i]) > maxAbs) maxAbs = Math.abs(psi[i]);
    }

    // Canonical sign: first significant lobe (> 15% of peak) is positive
    for (let i = 0; i < N; i++) {
      if (Math.abs(psi[i]) > 0.15 * maxAbs) {
        if (psi[i] < 0) {
          for (let j = 0; j < N; j++) psi[j] = -psi[j];
        }
        break;
      }
    }

    const prob = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      prob[i] = psi[i] * psi[i];
    }
    wavefunctions.push(psi);
    probabilities.push(prob);
  }

  return { x, V, energies: eigenvalues, wavefunctions, probabilities };
}

function render({ model, el }) {
  el.classList.add("ts-widget-host");

  const initialPythonValues = { ...(model.get("values") || {}) };
  const initialPythonNStates = Number(model.get("n_states") || 4);

  let editingParam = null;
  let activeDragName = null;
  let hoverXRatio = null;

  const container = document.createElement("div");
  container.className = "ts-container";
  el.appendChild(container);

  function isModified() {
    const current = model.get("values") || {};
    if (Number(model.get("n_states") || 4) !== initialPythonNStates) {
      return true;
    }
    for (const k of Object.keys(initialPythonValues)) {
      if (Math.abs(Number(current[k]) - Number(initialPythonValues[k])) > 1e-9) {
        return true;
      }
    }
    return false;
  }

  function resetAll() {
    const copy = { ...initialPythonValues };
    model.set("dragging", false);
    model.set("values", copy);
    model.set("committed_values", { ...copy });
    model.set("n_states", initialPythonNStates);
    model.save_changes();
    rebuildUI();
  }

  function highlightParam(paramName, active) {
    const allPills = container.querySelectorAll(
      `[data-param="${CSS.escape(paramName)}"]`
    );
    allPills.forEach((node) => {
      node.classList.toggle("ts-is-active", active);
    });
  }

  /**
   * Create a draggable / clickable Tangle pill for a parameter.
   */
  function createTangleControl(paramName, options = {}) {
    const params = model.get("parameters") || {};
    const spec = params[paramName];
    if (!spec) {
      const fallback = document.createElement("span");
      fallback.textContent = paramName;
      return fallback;
    }

    const values = model.get("values") || {};
    const val = Number(values[paramName] ?? spec.value);
    const isDark = detectDarkMode(el, model.get("theme"));
    const color = resolveColor(spec.color, isDark);
    const displayMode = options.forceMode || model.get("display_mode") || "both";
    const isActive =
      activeDragName === paramName || model.get("active_param") === paramName;

    const pill = document.createElement("span");
    pill.className = `ts-tangle-pill ts-kind-${spec.kind}`;
    if (isActive) pill.classList.add("ts-is-active");
    pill.dataset.param = paramName;
    if (options.forceMode === "both") pill.dataset.forceBoth = "true";
    pill.style.setProperty("--ts-param-color", color);
    pill.title = `${spec.label || paramName}: drag horizontally (${spec.min_value} to ${spec.max_value}), or click to type`;

    if (editingParam === paramName) {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "ts-tangle-input";
      input.step = String(spec.step);
      input.min = String(spec.min_value);
      input.max = String(spec.max_value);
      input.value = formatNumber(val, spec.digits);
      input.style.setProperty("--ts-param-color", color);

      const commitInput = () => {
        const parsed = parseFloat(input.value);
        editingParam = null;
        if (Number.isFinite(parsed)) {
          const clamped = clamp(parsed, spec.min_value, spec.max_value);
          const nextValues = {
            ...(model.get("values") || {}),
            [paramName]: Number(clamped.toFixed(Math.max(spec.digits + 2, 6))),
          };
          model.set("values", nextValues);
          model.set("committed_values", { ...nextValues });
          model.save_changes();
        }
        rebuildUI();
      };

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          commitInput();
        } else if (e.key === "Escape") {
          e.preventDefault();
          editingParam = null;
          rebuildUI();
        }
      });
      input.addEventListener("blur", commitInput);
      setTimeout(() => {
        input.focus();
        input.select();
      }, 0);
      return input;
    }

    const formattedVal = formatNumber(val, spec.digits);
    const unitSuffix = spec.unit ? `\u202F${spec.unit}` : "";

    if (displayMode === "number") {
      pill.textContent = `${formattedVal}${unitSuffix}`;
    } else if (displayMode === "symbol" && !isActive) {
      pill.textContent = spec.symbol;
    } else {
      const symSpan = document.createElement("span");
      symSpan.className = "ts-tangle-sym";
      symSpan.textContent = `${spec.symbol}\u2009=\u2009`;
      const valSpan = document.createElement("span");
      valSpan.className = "ts-tangle-val";
      valSpan.textContent = `${formattedVal}${unitSuffix}`;
      pill.append(symSpan, valSpan);
    }

    pill.addEventListener("pointerenter", () =>
      highlightParam(paramName, true)
    );
    pill.addEventListener("pointerleave", () => {
      if (activeDragName !== paramName) highlightParam(paramName, false);
    });

    pill.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();

      const startX = e.clientX;
      const startValues = { ...(model.get("values") || {}) };
      const startVal = Number(startValues[paramName] ?? spec.value);
      let moved = false;
      const pxPerStep = Math.max(2, Number(spec.pixels_per_step || 5));

      activeDragName = paramName;
      model.set("committed_values", { ...startValues });
      model.set("dragging", true);
      model.set("active_param", paramName);
      model.save_changes();
      document.body.classList.add("ts-global-dragging");
      highlightParam(paramName, true);

      const onPointerMove = (moveEvent) => {
        const dx = moveEvent.clientX - startX;
        if (!moved && Math.abs(dx) < 3) return;
        moved = true;

        const sensitivity = moveEvent.shiftKey ? 0.2 : 1.0;
        const deltaSteps = (dx * sensitivity) / pxPerStep;
        const rawNext = startVal + deltaSteps * spec.step;
        const clamped = clamp(rawNext, spec.min_value, spec.max_value);
        const snapped = roundToStep(
          clamped,
          spec.min_value,
          spec.step,
          spec.digits
        );

        const currValues = model.get("values") || {};
        if (Math.abs(Number(currValues[paramName]) - snapped) > 1e-12) {
          const nextValues = { ...currValues, [paramName]: snapped };
          model.set("values", nextValues);
          model.save_changes();
          updateLiveValuesInDOM();
        }
      };

      const onPointerUp = () => {
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        window.removeEventListener("pointercancel", onPointerUp);
        document.body.classList.remove("ts-global-dragging");

        const finalValues = { ...(model.get("values") || {}) };
        activeDragName = null;
        model.set("dragging", false);
        model.set("active_param", "");
        model.set("committed_values", finalValues);
        model.save_changes();

        if (!moved) {
          editingParam = paramName;
        }
        rebuildUI();
      };

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      window.addEventListener("pointercancel", onPointerUp);
    });

    return pill;
  }

  function updateLiveValuesInDOM() {
    const params = model.get("parameters") || {};
    const values = model.get("values") || {};
    const displayMode = model.get("display_mode") || "both";

    const pills = container.querySelectorAll(".ts-tangle-pill[data-param]");
    pills.forEach((pill) => {
      const pName = pill.dataset.param;
      const spec = params[pName];
      if (!spec) return;
      const val = Number(values[pName] ?? spec.value);
      const formattedVal = formatNumber(val, spec.digits);
      const unitSuffix = spec.unit ? `\u202F${spec.unit}` : "";
      const isForceBoth = pill.dataset.forceBoth === "true";
      const mode = isForceBoth ? "both" : displayMode;
      const isActive = activeDragName === pName;

      if (mode === "number") {
        pill.textContent = `${formattedVal}${unitSuffix}`;
      } else if (mode === "symbol" && !isActive) {
        pill.textContent = spec.symbol;
      } else {
        pill.innerHTML = "";
        const symSpan = document.createElement("span");
        symSpan.className = "ts-tangle-sym";
        symSpan.textContent = `${spec.symbol}\u2009=\u2009`;
        const valSpan = document.createElement("span");
        valSpan.className = "ts-tangle-val";
        valSpan.textContent = `${formattedVal}${unitSuffix}`;
        pill.append(symSpan, valSpan);
      }
      pill.classList.toggle("ts-is-active", isActive);
    });

    const resetBtn = container.querySelector(".ts-reset-btn");
    if (resetBtn) {
      resetBtn.disabled = !isModified();
    }

    renderPlotIntoSlot();
  }

  /**
   * Format a math expression or condition string into rich DOM nodes
   * where every parameter identifier is replaced inline by a live Tangle pill.
   */
  function renderMathWithTangles(rawStr) {
    const frag = document.createDocumentFragment();
    if (!rawStr) return frag;

    const params = model.get("parameters") || {};
    // Pretty-format common tokens before splitting on identifiers
    let text = rawStr
      .replace(/\\infty\b|\binfty\b|\binfinity\b|\binf\b/g, "∞")
      .replace(/<=/g, " ≤ ")
      .replace(/>=/g, " ≥ ")
      .replace(/\b0\.5\s*\*/g, "½ ")
      .replace(/\b1\/2\s*\*/g, "½ ")
      .replace(/\*/g, "·");

    // Tokenize on identifiers or superscripts (^2, ^4, etc.)
    const tokenRe = /(\^\d+|[A-Za-z_][A-Za-z0-9_]*)/g;
    let lastIndex = 0;
    let match;

    while ((match = tokenRe.exec(text)) !== null) {
      if (match.index > lastIndex) {
        frag.appendChild(
          document.createTextNode(text.slice(lastIndex, match.index))
        );
      }
      const token = match[0];
      if (token.startsWith("^")) {
        const sup = document.createElement("sup");
        sup.textContent = token.slice(1);
        frag.appendChild(sup);
      } else if (
        Object.prototype.hasOwnProperty.call(params, token) &&
        token !== "m"
      ) {
        frag.appendChild(createTangleControl(token));
      } else {
        const span = document.createElement("span");
        if (token === "x") {
          span.className = "ts-math-var";
        }
        span.textContent = token;
        frag.appendChild(span);
      }
      lastIndex = tokenRe.lastIndex;
    }

    if (lastIndex < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
    return frag;
  }

  /**
   * Render the potential energy equation V(x) = ... with inline Tangle controls.
   */
  function renderPotentialEquation(potData) {
    const box = document.createElement("div");
    box.className = "ts-equation-box";

    const eqRow = document.createElement("div");
    eqRow.className = "ts-equation-row";

    const lhs = document.createElement("span");
    lhs.className = "ts-eq-lhs";
    lhs.innerHTML = `V(<span class="ts-math-var">x</span>) = `;
    eqRow.appendChild(lhs);

    const cases = potData.cases || [];
    if (!potData.is_piecewise && cases.length === 1) {
      const exprSpan = document.createElement("span");
      exprSpan.className = "ts-eq-single";
      exprSpan.appendChild(renderMathWithTangles(cases[0].expr_raw));
      eqRow.appendChild(exprSpan);
    } else {
      // Piecewise \cases{...} display
      const casesWrap = document.createElement("div");
      casesWrap.className = "ts-cases-wrap";

      const brace = document.createElement("span");
      brace.className = "ts-cases-brace";
      brace.textContent = "{";
      casesWrap.appendChild(brace);

      const table = document.createElement("div");
      table.className = "ts-cases-grid";

      cases.forEach((c) => {
        const valCell = document.createElement("div");
        valCell.className = "ts-case-val";
        valCell.appendChild(renderMathWithTangles(c.expr_raw));
        table.appendChild(valCell);

        const condCell = document.createElement("div");
        condCell.className = "ts-case-cond";
        if (c.is_otherwise) {
          condCell.textContent = "otherwise";
        } else {
          condCell.appendChild(renderMathWithTangles(c.condition_raw));
        }
        table.appendChild(condCell);
      });

      casesWrap.appendChild(table);
      eqRow.appendChild(casesWrap);
    }

    box.appendChild(eqRow);
    return box;
  }

  /**
   * Render the SVG Quantum Energy-Level & Wavefunction Diagram.
   */
  function renderPlotIntoSlot() {
    const slot = container.querySelector(".ts-plot-slot");
    if (!slot) return;
    slot.innerHTML = "";

    if (!model.get("show_plot")) {
      slot.style.display = "none";
      return;
    }
    slot.style.display = "block";

    const potData = model.get("potential_data") || {};
    const values = model.get("values") || {};
    const committedValues = model.get("committed_values") || {};
    const isDragging = Boolean(model.get("dragging"));
    const showGhost = Boolean(model.get("show_ghost"));
    const xMin = Number(model.get("x_min") ?? -3.0);
    const xMax = Number(model.get("x_max") ?? 3.0);
    const nGrid = Number(model.get("n_grid") || 240);
    const nStates = Number(model.get("n_states") || 4);
    const viewMode = model.get("view_mode") || "wavefunction";
    const isDark = detectDarkMode(el, model.get("theme"));
    const levelPalette = model.get("level_colors") || [];

    const sol = solveSchrodinger1D(
      potData,
      values,
      xMin,
      xMax,
      nGrid,
      nStates
    );
    const ghostSol =
      isDragging && showGhost
        ? solveSchrodinger1D(
            potData,
            committedValues,
            xMin,
            xMax,
            nGrid,
            nStates
          )
        : null;

    // Determine energy y-axis range [eMin, eMax]
    let vMinFinite = 0;
    for (let i = 0; i < sol.V.length; i++) {
      if (Number.isFinite(sol.V[i]) && sol.V[i] < 1e5) {
        if (sol.V[i] < vMinFinite) vMinFinite = sol.V[i];
      }
    }

    const highestE =
      sol.energies.length > 0
        ? sol.energies[sol.energies.length - 1]
        : vMinFinite + 5.0;
    const ghostHighestE =
      ghostSol && ghostSol.energies.length > 0
        ? ghostSol.energies[ghostSol.energies.length - 1]
        : highestE;

    // Level spacing determines vertical wavefunction amplitude scale
    let avgGap = 1.5;
    if (sol.energies.length >= 2) {
      avgGap =
        (sol.energies[sol.energies.length - 1] - sol.energies[0]) /
        (sol.energies.length - 1);
    } else if (sol.energies.length === 1) {
      avgGap = Math.max(1.0, Math.abs(sol.energies[0] - vMinFinite));
    }
    avgGap = Math.max(0.25, avgGap);

    const eMin = vMinFinite - 0.15 * avgGap;
    const eMax = Math.max(highestE, ghostHighestE) + 1.35 * avgGap;

    const W = 640;
    const H = 310;
    const padL = 54;
    const padR = 76;
    const padT = 16;
    const padB = 38;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xToPx = (xVal) => padL + ((xVal - xMin) / (xMax - xMin)) * plotW;
    const eToPx = (eVal) =>
      padT + plotH - clamp((eVal - eMin) / (eMax - eMin), -0.05, 1.08) * plotH;

    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "ts-plot-svg");

    // 1. Grid & Axes
    const gridGroup = document.createElementNS(ns, "g");
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const eVal = eMin + ((eMax - eMin) / yTicks) * i;
      const py = eToPx(eVal);
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(padL));
      line.setAttribute("x2", String(padL + plotW));
      line.setAttribute("y1", String(py));
      line.setAttribute("y2", String(py));
      line.setAttribute("class", "ts-grid-line");
      gridGroup.appendChild(line);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", String(padL - 8));
      label.setAttribute("y", String(py + 4));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("class", "ts-tick-label");
      label.textContent =
        Math.abs(eVal) >= 10 ? eVal.toFixed(1) : eVal.toFixed(2);
      gridGroup.appendChild(label);
    }

    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const xVal = xMin + ((xMax - xMin) / xTicks) * i;
      const px = xToPx(xVal);
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(px));
      line.setAttribute("x2", String(px));
      line.setAttribute("y1", String(padT));
      line.setAttribute("y2", String(padT + plotH));
      line.setAttribute("class", "ts-grid-line");
      gridGroup.appendChild(line);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", String(px));
      label.setAttribute("y", String(padT + plotH + 16));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", "ts-tick-label");
      label.textContent = xVal.toFixed(2);
      gridGroup.appendChild(label);
    }

    // Axis titles
    const xTitle = document.createElementNS(ns, "text");
    xTitle.setAttribute("x", String(padL + plotW / 2));
    xTitle.setAttribute("y", String(H - 4));
    xTitle.setAttribute("text-anchor", "middle");
    xTitle.setAttribute("class", "ts-axis-title");
    xTitle.textContent = "Position x";
    gridGroup.appendChild(xTitle);

    const yTitle = document.createElementNS(ns, "text");
    yTitle.setAttribute(
      "transform",
      `translate(14, ${padT + plotH / 2}) rotate(-90)`
    );
    yTitle.setAttribute("text-anchor", "middle");
    yTitle.setAttribute("class", "ts-axis-title");
    yTitle.textContent =
      viewMode === "probability"
        ? "Energy Eₙ & Probability |ψₙ(x)|²"
        : "Energy Eₙ & Wavefunction ψₙ(x)";
    gridGroup.appendChild(yTitle);

    svg.appendChild(gridGroup);

    // Helper to draw potential curve V(x) and infinite walls
    function drawPotentialLayer(solObj, isGhostLayer) {
      const grp = document.createElementNS(ns, "g");
      const vCap = eMax + 0.25 * (eMax - eMin);

      // Draw shaded wall blocks where V(x) is infinite
      if (!isGhostLayer) {
        let inWall = false;
        let wallStartPx = padL;
        for (let i = 0; i < solObj.x.length; i++) {
          const isInf = !Number.isFinite(solObj.V[i]) || solObj.V[i] >= 1e5;
          const px = xToPx(solObj.x[i]);
          if (isInf && !inWall) {
            inWall = true;
            wallStartPx = i === 0 ? padL : px;
          } else if (!isInf && inWall) {
            inWall = false;
            const rect = document.createElementNS(ns, "rect");
            rect.setAttribute("x", String(wallStartPx));
            rect.setAttribute("y", String(padT));
            rect.setAttribute("width", String(Math.max(1, px - wallStartPx)));
            rect.setAttribute("height", String(plotH));
            rect.setAttribute("class", "ts-wall-fill");
            grp.appendChild(rect);
          }
        }
        if (inWall) {
          const rect = document.createElementNS(ns, "rect");
          rect.setAttribute("x", String(wallStartPx));
          rect.setAttribute("y", String(padT));
          rect.setAttribute(
            "width",
            String(Math.max(1, padL + plotW - wallStartPx))
          );
          rect.setAttribute("height", String(plotH));
          rect.setAttribute("class", "ts-wall-fill");
          grp.appendChild(rect);
        }
      }

      // Draw potential V(x) polyline (clamping infinite walls to top of plot)
      const pts = [];
      for (let i = 0; i < solObj.x.length; i++) {
        const vVal =
          !Number.isFinite(solObj.V[i]) || solObj.V[i] > vCap
            ? vCap
            : solObj.V[i];
        pts.push(
          `${xToPx(solObj.x[i]).toFixed(1)},${eToPx(vVal).toFixed(1)}`
        );
      }
      const poly = document.createElementNS(ns, "polyline");
      poly.setAttribute("fill", "none");
      poly.setAttribute(
        "class",
        isGhostLayer ? "ts-potential-ghost" : "ts-potential-line"
      );
      poly.setAttribute("points", pts.join(" "));
      grp.appendChild(poly);
      return grp;
    }

    if (ghostSol) {
      svg.appendChild(drawPotentialLayer(ghostSol, true));
    }
    svg.appendChild(drawPotentialLayer(sol, false));

    // Calculate wavefunction amplitude scaling in energy units
    let maxAmp = 1e-6;
    const arrList =
      viewMode === "probability" ? sol.probabilities : sol.wavefunctions;
    for (const arr of arrList) {
      for (let i = 0; i < arr.length; i++) {
        if (Math.abs(arr[i]) > maxAmp) maxAmp = Math.abs(arr[i]);
      }
    }
    const targetEnergyHeight =
      viewMode === "probability" ? 0.65 * avgGap : 0.42 * avgGap;
    const ampScale = targetEnergyHeight / maxAmp;

    // Draw eigenstates (ghost first if dragging, then live)
    function drawEigenstates(solObj, isGhostLayer) {
      const grp = document.createElementNS(ns, "g");
      const statesArr =
        viewMode === "probability"
          ? solObj.probabilities
          : solObj.wavefunctions;

      for (let s = 0; s < solObj.energies.length; s++) {
        const E = solObj.energies[s];
        const yBase = eToPx(E);
        const colorObj = levelPalette[s % levelPalette.length];
        const color = resolveColor(colorObj, isDark);

        // Energy baseline E_n
        const eLine = document.createElementNS(ns, "line");
        eLine.setAttribute("x1", String(padL));
        eLine.setAttribute("x2", String(padL + plotW));
        eLine.setAttribute("y1", String(yBase));
        eLine.setAttribute("y2", String(yBase));
        eLine.setAttribute("stroke", color);
        eLine.setAttribute("stroke-width", "1.1");
        eLine.setAttribute(
          "stroke-dasharray",
          isGhostLayer ? "2,4" : "4,3"
        );
        eLine.setAttribute("stroke-opacity", isGhostLayer ? "0.35" : "0.6");
        grp.appendChild(eLine);

        // Shaded polygon fill between E_n baseline and wavefunction curve
        const wf = statesArr[s];
        const curvePts = [];
        for (let i = 0; i < solObj.x.length; i++) {
          const eShifted = E + wf[i] * ampScale;
          curvePts.push(
            `${xToPx(solObj.x[i]).toFixed(1)},${eToPx(eShifted).toFixed(1)}`
          );
        }

        if (!isGhostLayer) {
          const fillPoly = document.createElementNS(ns, "polygon");
          const firstX = xToPx(solObj.x[0]).toFixed(1);
          const lastX = xToPx(solObj.x[solObj.x.length - 1]).toFixed(1);
          fillPoly.setAttribute(
            "points",
            `${firstX},${yBase.toFixed(1)} ${curvePts.join(" ")} ${lastX},${yBase.toFixed(1)}`
          );
          fillPoly.setAttribute("fill", color);
          fillPoly.setAttribute("fill-opacity", "0.14");
          grp.appendChild(fillPoly);
        }

        // Wavefunction polyline
        const wfPoly = document.createElementNS(ns, "polyline");
        wfPoly.setAttribute("fill", "none");
        wfPoly.setAttribute("stroke", color);
        wfPoly.setAttribute("stroke-width", isGhostLayer ? "1.6" : "2.3");
        if (isGhostLayer) {
          wfPoly.setAttribute("stroke-dasharray", "5,4");
          wfPoly.setAttribute("stroke-opacity", "0.55");
        }
        wfPoly.setAttribute("points", curvePts.join(" "));
        grp.appendChild(wfPoly);

        // Right-side level label: E_n
        if (!isGhostLayer) {
          const lbl = document.createElementNS(ns, "text");
          lbl.setAttribute("x", String(padL + plotW + 6));
          lbl.setAttribute("y", String(clamp(yBase + 4, padT + 10, padT + plotH)));
          lbl.setAttribute("fill", color);
          lbl.setAttribute("class", "ts-level-label");
          const sub = String(s + 1).replace(
            /\d/g,
            (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]
          );
          lbl.textContent = `E${sub} = ${E.toFixed(2)}`;
          grp.appendChild(lbl);
        }
      }
      return grp;
    }

    if (ghostSol) {
      svg.appendChild(drawEigenstates(ghostSol, true));
    }
    svg.appendChild(drawEigenstates(sol, false));

    // Plot border frame
    const frame = document.createElementNS(ns, "rect");
    frame.setAttribute("x", String(padL));
    frame.setAttribute("y", String(padT));
    frame.setAttribute("width", String(plotW));
    frame.setAttribute("height", String(plotH));
    frame.setAttribute("fill", "none");
    frame.setAttribute("class", "ts-plot-frame");
    svg.appendChild(frame);

    // Interactive hover crosshair
    const crossGrp = document.createElementNS(ns, "g");
    crossGrp.style.display = "none";
    const vLine = document.createElementNS(ns, "line");
    vLine.setAttribute("y1", String(padT));
    vLine.setAttribute("y2", String(padT + plotH));
    vLine.setAttribute("class", "ts-crosshair-line");
    crossGrp.appendChild(vLine);
    svg.appendChild(crossGrp);

    const readoutBar = document.createElement("div");
    readoutBar.className = "ts-plot-readout";

    function updateReadout(ratio) {
      if (ratio === null || ratio < 0 || ratio > 1) {
        crossGrp.style.display = "none";
        if (sol.energies.length >= 2) {
          const dE = sol.energies[1] - sol.energies[0];
          const eList = sol.energies
            .map((eVal, idx) => {
              const color = resolveColor(
                levelPalette[idx % levelPalette.length],
                isDark
              );
              const sub = String(idx + 1).replace(
                /\d/g,
                (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]
              );
              return `<span class="ts-readout-sp"><span class="ts-dot" style="background:${color}"></span>E${sub} = <strong>${eVal.toFixed(3)}</strong></span>`;
            })
            .join("");
          readoutBar.innerHTML = `${eList}<span class="ts-readout-gap">ΔE₂₁ = <strong>${dE.toFixed(3)}</strong></span>`;
        } else {
          readoutBar.innerHTML = `<span class="ts-readout-hint">Drag parameters inside V(x) to explore quantized energy eigenvalues</span>`;
        }
        return;
      }

      const idx = clamp(
        Math.round(ratio * (sol.x.length - 1)),
        0,
        sol.x.length - 1
      );
      const xVal = sol.x[idx];
      const vVal = sol.V[idx];
      const px = xToPx(xVal);
      crossGrp.style.display = "block";
      vLine.setAttribute("x1", String(px));
      vLine.setAttribute("x2", String(px));

      const vText =
        !Number.isFinite(vVal) || vVal >= 1e5 ? "∞" : vVal.toFixed(2);
      const parts = [
        `<span class="ts-readout-pos">x = ${xVal.toFixed(2)}</span>`,
        `<span class="ts-readout-pos">V(x) = ${vText}</span>`,
      ];
      const statesArr =
        viewMode === "probability" ? sol.probabilities : sol.wavefunctions;
      const sym = viewMode === "probability" ? "|ψ" : "ψ";
      const suf = viewMode === "probability" ? "|²" : "";

      sol.energies.forEach((_eVal, s) => {
        const color = resolveColor(
          levelPalette[s % levelPalette.length],
          isDark
        );
        const sub = String(s + 1).replace(
          /\d/g,
          (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]
        );
        parts.push(
          `<span class="ts-readout-sp"><span class="ts-dot" style="background:${color}"></span>${sym}${sub}${suf} = <strong>${statesArr[s][idx].toFixed(3)}</strong></span>`
        );
      });
      readoutBar.innerHTML = parts.join("");
    }

    const hitRect = document.createElementNS(ns, "rect");
    hitRect.setAttribute("x", String(padL));
    hitRect.setAttribute("y", String(padT));
    hitRect.setAttribute("width", String(plotW));
    hitRect.setAttribute("height", String(plotH));
    hitRect.setAttribute("fill", "transparent");
    hitRect.style.cursor = "crosshair";

    hitRect.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const svgX = ((e.clientX - rect.left) / rect.width) * W;
      hoverXRatio = clamp((svgX - padL) / plotW, 0, 1);
      updateReadout(hoverXRatio);
    });
    hitRect.addEventListener("pointerleave", () => {
      hoverXRatio = null;
      updateReadout(null);
    });
    svg.appendChild(hitRect);

    slot.appendChild(svg);
    slot.appendChild(readoutBar);
    updateReadout(hoverXRatio);
  }

  /**
   * Draggable control for number of displayed eigenstates `n_states`.
   */
  function createNStatesControl() {
    const nStates = Number(model.get("n_states") || 4);
    const wrap = document.createElement("span");
    wrap.className = "ts-nstates-control";
    wrap.title = "Drag horizontally to adjust number of eigenstates shown (1 to 8)";
    wrap.innerHTML = `<span class="ts-nstates-label">States n = </span><span class="ts-nstates-val">1..${nStates}</span>`;

    wrap.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const startX = e.clientX;
      const startN = Number(model.get("n_states") || 4);
      document.body.classList.add("ts-global-dragging");

      const onMove = (me) => {
        const dx = me.clientX - startX;
        const nextN = clamp(Math.round(startN + dx / 18), 1, 8);
        if (nextN !== Number(model.get("n_states"))) {
          model.set("n_states", nextN);
          model.save_changes();
          const valEl = wrap.querySelector(".ts-nstates-val");
          if (valEl) valEl.textContent = `1..${nextN}`;
          const resetBtn = container.querySelector(".ts-reset-btn");
          if (resetBtn) resetBtn.disabled = !isModified();
          renderPlotIntoSlot();
        }
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        document.body.classList.remove("ts-global-dragging");
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
    return wrap;
  }

  function rebuildUI() {
    const potData = model.get("potential_data") || {};
    const isDark = detectDarkMode(el, model.get("theme"));

    container.classList.toggle("ts-dark", isDark);
    container.classList.toggle("ts-light", !isDark);
    container.innerHTML = "";

    // 1. Header / Toolbar
    const header = document.createElement("div");
    header.className = "ts-header";

    const titleGroup = document.createElement("div");
    titleGroup.className = "ts-title-group";
    titleGroup.innerHTML = `<span class="ts-badge">1D Schrödinger</span><span class="ts-subtitle">−(ℏ²/2m) d²ψ/dx² + V(x)ψ = Eψ</span>`;
    header.appendChild(titleGroup);

    const actions = document.createElement("div");
    actions.className = "ts-actions";

    // Particle mass tangle control
    const massWrap = document.createElement("span");
    massWrap.className = "ts-mass-wrap";
    massWrap.appendChild(createTangleControl("m", { forceMode: "both" }));
    actions.appendChild(massWrap);

    // n_states control
    actions.appendChild(createNStatesControl());

    // Wavefunction vs Probability Density toggle button
    const viewMode = model.get("view_mode") || "wavefunction";
    const modeBtn = document.createElement("button");
    modeBtn.type = "button";
    modeBtn.className = `ts-btn ${viewMode === "probability" ? "ts-btn-active" : ""}`;
    modeBtn.textContent = viewMode === "probability" ? "|ψ(x)|²" : "ψ(x)";
    modeBtn.title = "Toggle between wavefunctions ψₙ(x) and probability densities |ψₙ(x)|²";
    modeBtn.addEventListener("click", () => {
      const next =
        model.get("view_mode") === "probability"
          ? "wavefunction"
          : "probability";
      model.set("view_mode", next);
      model.save_changes();
      rebuildUI();
    });
    actions.appendChild(modeBtn);

    // Reset button
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "ts-btn ts-reset-btn";
    resetBtn.textContent = "Reset";
    resetBtn.disabled = !isModified();
    resetBtn.addEventListener("click", resetAll);
    actions.appendChild(resetBtn);

    header.appendChild(actions);
    container.appendChild(header);

    // 2. Interactive Potential Equation V(x)
    container.appendChild(renderPotentialEquation(potData));

    // 3. Plot Slot
    const plotSlot = document.createElement("div");
    plotSlot.className = "ts-plot-slot";
    container.appendChild(plotSlot);

    renderPlotIntoSlot();
  }

  model.on("change:values", () => {
    if (!activeDragName) rebuildUI();
  });
  model.on("change:potential_data", rebuildUI);
  model.on("change:parameters", rebuildUI);
  model.on("change:view_mode", rebuildUI);
  model.on("change:display_mode", rebuildUI);
  model.on("change:theme", rebuildUI);
  model.on("change:n_states", () => {
    if (!activeDragName) rebuildUI();
  });

  rebuildUI();
}

export default { render };
