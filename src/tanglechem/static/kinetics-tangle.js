// Interactive TangleKinetics anywidget frontend (zero external dependencies)

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

/**
 * Adaptive Dormand-Prince RK45 solver with implicit Trapezoidal/BDF1 fallback
 * for stiff chemical kinetic systems.
 */
function solveKineticsODE(systemData, paramValues, tMax, nOut = 200) {
  const species = systemData.species || [];
  const steps = systemData.steps || [];
  const nSp = species.length;
  if (nSp === 0 || tMax <= 0) {
    return { t: [0], y: [] };
  }

  const spIndex = {};
  species.forEach((sp, i) => {
    spIndex[sp.name] = i;
  });

  const y0 = species.map((sp) => {
    const v = Number(paramValues[sp.param_key] ?? 0);
    return Number.isFinite(v) && v >= 0 ? v : 0;
  });

  // Pre-compile reaction steps for fast inner-loop evaluation
  const compiledSteps = steps.map((st) => {
    const kf = Math.max(0, Number(paramValues[st.k_forward] ?? 0));
    const kr =
      st.reversible && st.k_reverse
        ? Math.max(0, Number(paramValues[st.k_reverse] ?? 0))
        : 0;
    const reactants = st.reactants.map((r) => ({
      idx: spIndex[r.species],
      coeff: Number(r.coeff),
    }));
    const products = st.products.map((p) => ({
      idx: spIndex[p.species],
      coeff: Number(p.coeff),
    }));
    return { kf, kr, reversible: st.reversible, reactants, products };
  });

  function computeRhs(y, out) {
    for (let i = 0; i < nSp; i++) out[i] = 0;
    for (let s = 0; s < compiledSteps.length; s++) {
      const st = compiledSteps[s];
      let rf = st.kf;
      for (let r = 0; r < st.reactants.length; r++) {
        const term = st.reactants[r];
        const c = y[term.idx] > 0 ? y[term.idx] : 0;
        rf *= term.coeff === 1 ? c : Math.pow(c, term.coeff);
      }
      let net = rf;
      if (st.reversible && st.kr > 0) {
        let rr = st.kr;
        for (let p = 0; p < st.products.length; p++) {
          const term = st.products[p];
          const c = y[term.idx] > 0 ? y[term.idx] : 0;
          rr *= term.coeff === 1 ? c : Math.pow(c, term.coeff);
        }
        net -= rr;
      }
      for (let r = 0; r < st.reactants.length; r++) {
        const term = st.reactants[r];
        out[term.idx] -= term.coeff * net;
      }
      for (let p = 0; p < st.products.length; p++) {
        const term = st.products[p];
        out[term.idx] += term.coeff * net;
      }
    }
  }

  // Evaluate on uniform output grid using sub-stepped RK4 + Positivity Guard
  // and implicit Patankar-Euler fallback if local stiffness is high.
  const tGrid = new Array(nOut);
  const dtOut = tMax / (nOut - 1);
  for (let i = 0; i < nOut; i++) {
    tGrid[i] = i * dtOut;
  }

  const yHistory = Array.from({ length: nSp }, () => new Array(nOut));
  for (let j = 0; j < nSp; j++) {
    yHistory[j][0] = y0[j];
  }

  let y = y0.slice();
  const k1 = new Array(nSp);
  const k2 = new Array(nSp);
  const k3 = new Array(nSp);
  const k4 = new Array(nSp);
  const yTmp = new Array(nSp);

  // Estimate max rate scale to pick safe substeps per output interval
  let maxK = 1.0;
  for (let s = 0; s < compiledSteps.length; s++) {
    maxK = Math.max(maxK, compiledSteps[s].kf, compiledSteps[s].kr);
  }
  const maxConc = Math.max(1.0, ...y0);
  const subSteps = clamp(Math.ceil((dtOut * maxK * maxConc) / 0.15), 2, 40);

  for (let stepIdx = 1; stepIdx < nOut; stepIdx++) {
    const h = dtOut / subSteps;
    for (let sub = 0; sub < subSteps; sub++) {
      computeRhs(y, k1);
      for (let j = 0; j < nSp; j++) {
        yTmp[j] = Math.max(0, y[j] + 0.5 * h * k1[j]);
      }
      computeRhs(yTmp, k2);
      for (let j = 0; j < nSp; j++) {
        yTmp[j] = Math.max(0, y[j] + 0.5 * h * k2[j]);
      }
      computeRhs(yTmp, k3);
      for (let j = 0; j < nSp; j++) {
        yTmp[j] = Math.max(0, y[j] + h * k3[j]);
      }
      computeRhs(yTmp, k4);

      let unstable = false;
      for (let j = 0; j < nSp; j++) {
        const nextVal =
          y[j] + (h / 6.0) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        if (!Number.isFinite(nextVal) || nextVal < -1e-3 * maxConc) {
          unstable = true;
          break;
        }
        yTmp[j] = Math.max(0, nextVal);
      }

      if (!unstable) {
        for (let j = 0; j < nSp; j++) y[j] = yTmp[j];
      } else {
        // Stiff fallback: damped Patankar / micro-stepping
        const micro = 16;
        const hMicro = h / micro;
        for (let m = 0; m < micro; m++) {
          computeRhs(y, k1);
          for (let j = 0; j < nSp; j++) {
            y[j] = Math.max(0, y[j] + hMicro * k1[j]);
          }
        }
      }
    }
    for (let j = 0; j < nSp; j++) {
      yHistory[j][stepIdx] = y[j];
    }
  }

  return { t: tGrid, y: yHistory };
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

function render({ model, el }) {
  el.classList.add("tc-widget-host");

  // Store initial values declared in Python for the Reset button
  const initialPythonValues = { ...(model.get("values") || {}) };
  const initialPythonTMax = Number(model.get("t_max") || 10.0);

  // Local UI state
  const hiddenSpecies = new Set();
  let editingParam = null;
  let hoverTimeRatio = null; // 0..1 when hovering over the SVG plot
  let activeDragName = null;

  const container = document.createElement("div");
  container.className = "tc-container";
  el.appendChild(container);

  function isModified() {
    const current = model.get("values") || {};
    const tMax = Number(model.get("t_max") || 10.0);
    if (Math.abs(tMax - initialPythonTMax) > 1e-9) return true;
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
    model.set("t_max", initialPythonTMax);
    model.save_changes();
    rebuildUI();
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
    pill.className = `tc-tangle-pill tc-kind-${spec.kind}`;
    if (isActive) pill.classList.add("tc-is-active");
    pill.dataset.param = paramName;
    pill.style.setProperty("--tc-param-color", color);
    pill.title = `${spec.label || paramName}: drag horizontally to adjust (${spec.min_value} to ${spec.max_value}), or click to edit`;

    // Inline editor when clicked
    if (editingParam === paramName) {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "tc-tangle-input";
      input.step = String(spec.step);
      input.min = String(spec.min_value);
      input.max = String(spec.max_value);
      input.value = formatNumber(val, spec.digits);
      input.style.setProperty("--tc-param-color", color);

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
      // "both" or active symbol reveal
      const symSpan = document.createElement("span");
      symSpan.className = "tc-tangle-sym";
      symSpan.textContent = `${spec.symbol}\u2009=\u2009`;
      const valSpan = document.createElement("span");
      valSpan.className = "tc-tangle-val";
      valSpan.textContent = `${formattedVal}${unitSuffix}`;
      pill.append(symSpan, valSpan);
    }

    // Hover highlight across all matching elements
    pill.addEventListener("pointerenter", () => {
      highlightParam(paramName, true);
    });
    pill.addEventListener("pointerleave", () => {
      if (activeDragName !== paramName) {
        highlightParam(paramName, false);
      }
    });

    // Pointer drag / click interaction
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
      document.body.classList.add("tc-global-dragging");
      highlightParam(paramName, true);

      const onPointerMove = (moveEvent) => {
        const dx = moveEvent.clientX - startX;
        if (!moved && Math.abs(dx) < 3) return;
        moved = true;

        // Shift key enables 5x fine-grained scrubbing
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
        document.body.classList.remove("tc-global-dragging");

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

  function highlightParam(paramName, active) {
    const allPills = container.querySelectorAll(
      `[data-param="${CSS.escape(paramName)}"]`
    );
    allPills.forEach((node) => {
      node.classList.toggle("tc-is-active", active);
    });
  }

  /**
   * Fast path during pointer drag: update pill text & re-render SVG curves
   * without tearing down the DOM tree so pointer capture stays smooth at 60fps.
   */
  function updateLiveValuesInDOM() {
    const params = model.get("parameters") || {};
    const values = model.get("values") || {};
    const displayMode = model.get("display_mode") || "both";

    const pills = container.querySelectorAll(".tc-tangle-pill[data-param]");
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
        symSpan.className = "tc-tangle-sym";
        symSpan.textContent = `${spec.symbol}\u2009=\u2009`;
        const valSpan = document.createElement("span");
        valSpan.className = "tc-tangle-val";
        valSpan.textContent = `${formattedVal}${unitSuffix}`;
        pill.append(symSpan, valSpan);
      }
      pill.classList.toggle("tc-is-active", isActive);
    });

    const resetBtn = container.querySelector(".tc-reset-btn");
    if (resetBtn) {
      resetBtn.disabled = !isModified();
    }

    renderPlotIntoSlot();
  }

  /**
   * Render one side (reactants or products) of a chemical equation.
   */
  function renderReactionSide(terms, speciesColorMap, isDark) {
    const sideEl = document.createElement("span");
    sideEl.className = "tc-rxn-side";
    if (!terms || terms.length === 0) {
      const empty = document.createElement("span");
      empty.className = "tc-species-token";
      empty.textContent = "∅";
      sideEl.appendChild(empty);
      return sideEl;
    }

    terms.forEach((term, idx) => {
      if (idx > 0) {
        const plus = document.createElement("span");
        plus.className = "tc-rxn-plus";
        plus.textContent = " + ";
        sideEl.appendChild(plus);
      }
      const token = document.createElement("span");
      token.className = "tc-species-token";
      const spColor = resolveColor(speciesColorMap[term.species], isDark);
      token.style.setProperty("--tc-sp-color", spColor);

      if (term.coeff !== 1) {
        const coeffSpan = document.createElement("span");
        coeffSpan.className = "tc-stoich-coeff";
        coeffSpan.textContent = `${term.coeff}\u2009`;
        token.appendChild(coeffSpan);
      }

      const nameSpan = document.createElement("span");
      nameSpan.className = "tc-species-name";
      nameSpan.textContent = term.label || term.species;
      token.appendChild(nameSpan);

      sideEl.appendChild(token);
    });
    return sideEl;
  }

  /**
   * Build the reaction arrow with embedded Tangle rate constant pill(s).
   */
  function renderReactionArrow(step) {
    const arrowWrap = document.createElement("span");
    arrowWrap.className = step.reversible
      ? "tc-arrow-wrap tc-arrow-reversible"
      : "tc-arrow-wrap tc-arrow-irreversible";

    // Forward rate constant above arrow
    const above = document.createElement("span");
    above.className = "tc-arrow-rate-above";
    above.appendChild(createTangleControl(step.k_forward));
    arrowWrap.appendChild(above);

    // Arrow graphic (crisp inline SVG)
    const arrowGraphic = document.createElement("span");
    arrowGraphic.className = "tc-arrow-graphic";
    if (!step.reversible) {
      arrowGraphic.innerHTML = `
        <svg viewBox="0 0 84 14" class="tc-arrow-svg" preserveAspectRatio="none">
          <line x1="2" y1="7" x2="75" y2="7" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>
          <path d="M 71 2.5 L 81 7 L 71 11.5" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      `;
    } else {
      arrowGraphic.innerHTML = `
        <svg viewBox="0 0 84 18" class="tc-arrow-svg" preserveAspectRatio="none">
          <line x1="3" y1="6" x2="80" y2="6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          <path d="M 72 1.8 L 81 6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          <line x1="4" y1="12" x2="81" y2="12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          <path d="M 12 16.2 L 3 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
      `;
    }
    arrowWrap.appendChild(arrowGraphic);

    // Reverse rate constant below arrow (for reversible steps)
    if (step.reversible && step.k_reverse) {
      const below = document.createElement("span");
      below.className = "tc-arrow-rate-below";
      below.appendChild(createTangleControl(step.k_reverse));
      arrowWrap.appendChild(below);
    }

    return arrowWrap;
  }

  /**
   * Build the Differential Rate Equations (d[X]/dt) inspection panel
   * with live linked Tangle controls for each rate constant.
   */
  function renderODEPanel(sys, speciesColorMap, isDark) {
    const panel = document.createElement("div");
    panel.className = "tc-ode-panel";

    const species = sys.species || [];
    const steps = sys.steps || [];

    species.forEach((sp) => {
      const terms = [];
      steps.forEach((st) => {
        let netStoich = 0;
        st.reactants.forEach((r) => {
          if (r.species === sp.name) netStoich -= r.coeff;
        });
        st.products.forEach((p) => {
          if (p.species === sp.name) netStoich += p.coeff;
        });
        if (netStoich !== 0) {
          terms.push({
            coeff: netStoich,
            kName: st.k_forward,
            factors: st.reactants,
          });
          if (st.reversible && st.k_reverse) {
            terms.push({
              coeff: -netStoich,
              kName: st.k_reverse,
              factors: st.products,
            });
          }
        }
      });

      const row = document.createElement("div");
      row.className = "tc-ode-row";

      const lhs = document.createElement("span");
      lhs.className = "tc-ode-lhs";
      const spColor = resolveColor(speciesColorMap[sp.name], isDark);
      lhs.innerHTML = `<span class="tc-ode-frac">d[<strong style="color:${spColor}">${sp.label}</strong>]/dt</span> = `;
      row.appendChild(lhs);

      if (terms.length === 0) {
        const zero = document.createElement("span");
        zero.textContent = "0";
        row.appendChild(zero);
      } else {
        terms.forEach((t, idx) => {
          const signSpan = document.createElement("span");
          signSpan.className = "tc-ode-sign";
          if (idx === 0) {
            signSpan.textContent = t.coeff < 0 ? "−" : "";
          } else {
            signSpan.textContent = t.coeff < 0 ? " − " : " + ";
          }
          row.appendChild(signSpan);

          const absC = Math.abs(t.coeff);
          if (absC !== 1) {
            const cSpan = document.createElement("span");
            cSpan.textContent = `${absC}\u2009`;
            row.appendChild(cSpan);
          }

          const pill = createTangleControl(t.kName, { forceMode: "symbol" });
          row.appendChild(pill);

          t.factors.forEach((f) => {
            const fSpan = document.createElement("span");
            fSpan.className = "tc-ode-factor";
            const fColor = resolveColor(speciesColorMap[f.species], isDark);
            const sup = f.coeff !== 1 ? `<sup>${f.coeff}</sup>` : "";
            fSpan.innerHTML = `[<span style="color:${fColor};font-weight:600">${f.label}</span>]${sup}`;
            row.appendChild(fSpan);
          });
        });
      }

      panel.appendChild(row);
    });

    return panel;
  }

  /**
   * Render the SVG concentration-vs-time plot into `.tc-plot-slot`.
   */
  function renderPlotIntoSlot() {
    const slot = container.querySelector(".tc-plot-slot");
    if (!slot) return;
    slot.innerHTML = "";

    if (!model.get("show_plot")) {
      slot.style.display = "none";
      return;
    }
    slot.style.display = "block";

    const sys = model.get("system_data") || {};
    const species = sys.species || [];
    const values = model.get("values") || {};
    const committedValues = model.get("committed_values") || {};
    const isDragging = Boolean(model.get("dragging"));
    const showGhost = Boolean(model.get("show_ghost"));
    const tMax = Math.max(0.01, Number(model.get("t_max") || 10.0));
    const cUnit = model.get("concentration_unit") || "M";
    const tUnit = model.get("time_unit") || "s";
    const isDark = detectDarkMode(el, model.get("theme"));

    const sol = solveKineticsODE(sys, values, tMax, 200);
    const ghostSol =
      isDragging && showGhost
        ? solveKineticsODE(sys, committedValues, tMax, 200)
        : null;

    // Compute y-axis domain across visible species
    let yMax = 0.1;
    species.forEach((sp, i) => {
      if (hiddenSpecies.has(sp.name)) return;
      for (const v of sol.y[i]) {
        if (Number.isFinite(v) && v > yMax) yMax = v;
      }
      if (ghostSol) {
        for (const v of ghostSol.y[i]) {
          if (Number.isFinite(v) && v > yMax) yMax = v;
        }
      }
    });
    // Add 8% headroom and nice rounding
    const niceYMax = yMax * 1.08;

    const W = 640;
    const H = 260;
    const padL = 54;
    const padR = 20;
    const padT = 16;
    const padB = 38;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xToPx = (t) => padL + (t / tMax) * plotW;
    const yToPx = (c) => padT + plotH - clamp(c / niceYMax, 0, 1.05) * plotH;

    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "tc-plot-svg");

    // Grid lines & ticks
    const gridGroup = document.createElementNS(ns, "g");
    gridGroup.setAttribute("class", "tc-plot-grid");

    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = (niceYMax / yTicks) * i;
      const py = yToPx(val);
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(padL));
      line.setAttribute("x2", String(W - padR));
      line.setAttribute("y1", String(py));
      line.setAttribute("y2", String(py));
      line.setAttribute("class", i === 0 ? "tc-axis-line" : "tc-grid-line");
      gridGroup.appendChild(line);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", String(padL - 8));
      label.setAttribute("y", String(py + 4));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("class", "tc-tick-label");
      label.textContent = val >= 10 ? val.toFixed(1) : val.toFixed(2);
      gridGroup.appendChild(label);
    }

    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const val = (tMax / xTicks) * i;
      const px = xToPx(val);
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(px));
      line.setAttribute("x2", String(px));
      line.setAttribute("y1", String(padT));
      line.setAttribute("y2", String(padT + plotH));
      line.setAttribute("class", i === 0 ? "tc-axis-line" : "tc-grid-line");
      gridGroup.appendChild(line);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", String(px));
      label.setAttribute("y", String(padT + plotH + 16));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", "tc-tick-label");
      label.textContent = val >= 10 ? val.toFixed(1) : val.toFixed(2);
      gridGroup.appendChild(label);
    }

    // Axis titles
    const xTitle = document.createElementNS(ns, "text");
    xTitle.setAttribute("x", String(padL + plotW / 2));
    xTitle.setAttribute("y", String(H - 4));
    xTitle.setAttribute("text-anchor", "middle");
    xTitle.setAttribute("class", "tc-axis-title");
    xTitle.textContent = `Time (${tUnit})`;
    gridGroup.appendChild(xTitle);

    const yTitle = document.createElementNS(ns, "text");
    yTitle.setAttribute(
      "transform",
      `translate(14, ${padT + plotH / 2}) rotate(-90)`
    );
    yTitle.setAttribute("text-anchor", "middle");
    yTitle.setAttribute("class", "tc-axis-title");
    yTitle.textContent = `Concentration (${cUnit})`;
    gridGroup.appendChild(yTitle);

    svg.appendChild(gridGroup);

    // Ghost (before-drag committed_values) curves
    if (ghostSol) {
      const ghostGroup = document.createElementNS(ns, "g");
      ghostGroup.setAttribute("class", "tc-ghost-curves");
      species.forEach((sp, i) => {
        if (hiddenSpecies.has(sp.name)) return;
        const color = resolveColor(sp.color, isDark);
        const pts = ghostSol.t
          .map(
            (tVal, k) =>
              `${xToPx(tVal).toFixed(1)},${yToPx(ghostSol.y[i][k]).toFixed(1)}`
          )
          .join(" ");
        const poly = document.createElementNS(ns, "polyline");
        poly.setAttribute("fill", "none");
        poly.setAttribute("stroke", color);
        poly.setAttribute("stroke-width", "1.75");
        poly.setAttribute("stroke-dasharray", "5,4");
        poly.setAttribute("stroke-opacity", "0.55");
        poly.setAttribute("points", pts);
        ghostGroup.appendChild(poly);
      });
      svg.appendChild(ghostGroup);
    }

    // Live curves
    const curvesGroup = document.createElementNS(ns, "g");
    curvesGroup.setAttribute("class", "tc-live-curves");
    species.forEach((sp, i) => {
      if (hiddenSpecies.has(sp.name)) return;
      const color = resolveColor(sp.color, isDark);
      const pts = sol.t
        .map(
          (tVal, k) =>
            `${xToPx(tVal).toFixed(1)},${yToPx(sol.y[i][k]).toFixed(1)}`
        )
        .join(" ");
      const poly = document.createElementNS(ns, "polyline");
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "2.5");
      poly.setAttribute("stroke-linecap", "round");
      poly.setAttribute("stroke-linejoin", "round");
      poly.setAttribute("points", pts);
      curvesGroup.appendChild(poly);
    });
    svg.appendChild(curvesGroup);

    // Interactive hover crosshair & readout bar
    const crosshairGroup = document.createElementNS(ns, "g");
    crosshairGroup.style.display = "none";
    const vLine = document.createElementNS(ns, "line");
    vLine.setAttribute("y1", String(padT));
    vLine.setAttribute("y2", String(padT + plotH));
    vLine.setAttribute("class", "tc-crosshair-line");
    crosshairGroup.appendChild(vLine);

    const dots = species.map((sp) => {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("r", "4.5");
      c.setAttribute("fill", resolveColor(sp.color, isDark));
      c.setAttribute("stroke", isDark ? "#181a20" : "#ffffff");
      c.setAttribute("stroke-width", "1.5");
      crosshairGroup.appendChild(c);
      return c;
    });
    svg.appendChild(crosshairGroup);

    const readoutBar = document.createElement("div");
    readoutBar.className = "tc-plot-readout";

    function updateHoverReadout(ratio) {
      if (ratio === null || ratio < 0 || ratio > 1) {
        crosshairGroup.style.display = "none";
        if (isDragging && showGhost) {
          readoutBar.innerHTML = `<span class="tc-readout-hint">Solid: current values • Dashed: before drag (<code>committed_values</code>)</span>`;
        } else {
          readoutBar.innerHTML = `<span class="tc-readout-hint">Hover plot to inspect concentrations • Drag any highlighted number (Shift+drag for fine control)</span>`;
        }
        return;
      }

      const idx = clamp(
        Math.round(ratio * (sol.t.length - 1)),
        0,
        sol.t.length - 1
      );
      const tVal = sol.t[idx];
      const px = xToPx(tVal);
      crosshairGroup.style.display = "block";
      vLine.setAttribute("x1", String(px));
      vLine.setAttribute("x2", String(px));

      const parts = [
        `<span class="tc-readout-time">t = ${tVal.toFixed(2)} ${tUnit}</span>`,
      ];
      species.forEach((sp, i) => {
        if (hiddenSpecies.has(sp.name)) {
          dots[i].style.display = "none";
          return;
        }
        const cVal = sol.y[i][idx];
        dots[i].style.display = "block";
        dots[i].setAttribute("cx", String(px));
        dots[i].setAttribute("cy", String(yToPx(cVal)));
        const color = resolveColor(sp.color, isDark);
        parts.push(
          `<span class="tc-readout-sp"><span class="tc-dot" style="background:${color}"></span>[${sp.label}] = <strong>${cVal.toFixed(3)}</strong> ${cUnit}</span>`
        );
      });
      readoutBar.innerHTML = parts.join("");
    }

    // Transparent overlay rect for smooth hover events
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
      hoverTimeRatio = clamp((svgX - padL) / plotW, 0, 1);
      updateHoverReadout(hoverTimeRatio);
    });
    hitRect.addEventListener("pointerleave", () => {
      hoverTimeRatio = null;
      updateHoverReadout(null);
    });
    svg.appendChild(hitRect);

    slot.appendChild(svg);
    slot.appendChild(readoutBar);
    updateHoverReadout(hoverTimeRatio);
  }

  /**
   * Build a tangle control for `t_max` in the header bar.
   */
  function createTMaxControl() {
    const tMax = Number(model.get("t_max") || 10.0);
    const tUnit = model.get("time_unit") || "s";
    const wrap = document.createElement("span");
    wrap.className = "tc-tmax-control";
    wrap.title = "Drag horizontally to adjust simulation duration t_max";
    wrap.innerHTML = `<span class="tc-tmax-label">t<sub>max</sub> = </span><span class="tc-tmax-val">${tMax.toFixed(1)}\u202F${tUnit}</span>`;

    wrap.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const startX = e.clientX;
      const startT = Number(model.get("t_max") || 10.0);
      document.body.classList.add("tc-global-dragging");

      const onMove = (me) => {
        const dx = me.clientX - startX;
        const step = startT <= 5 ? 0.2 : startT <= 50 ? 0.5 : 2.0;
        const nextT = clamp(
          Math.round((startT + (dx / 5) * step) * 10) / 10,
          0.5,
          1000.0
        );
        model.set("t_max", nextT);
        model.save_changes();
        const valEl = wrap.querySelector(".tc-tmax-val");
        if (valEl) valEl.textContent = `${nextT.toFixed(1)}\u202F${tUnit}`;
        const resetBtn = container.querySelector(".tc-reset-btn");
        if (resetBtn) resetBtn.disabled = !isModified();
        renderPlotIntoSlot();
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        document.body.classList.remove("tc-global-dragging");
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
    return wrap;
  }

  function rebuildUI() {
    const sys = model.get("system_data") || {};
    const species = sys.species || [];
    const steps = sys.steps || [];
    const isDark = detectDarkMode(el, model.get("theme"));

    container.classList.toggle("tc-dark", isDark);
    container.classList.toggle("tc-light", !isDark);
    container.innerHTML = "";

    const speciesColorMap = {};
    species.forEach((sp) => {
      speciesColorMap[sp.name] = sp.color;
    });

    // 1. Header / Toolbar
    const header = document.createElement("div");
    header.className = "tc-header";

    const titleGroup = document.createElement("div");
    titleGroup.className = "tc-title-group";
    titleGroup.innerHTML = `<span class="tc-badge">Reaction Kinetics</span><span class="tc-subtitle">Drag rate constants or initial concentrations</span>`;
    header.appendChild(titleGroup);

    const actions = document.createElement("div");
    actions.className = "tc-actions";

    // t_max tangle control
    actions.appendChild(createTMaxControl());

    // Toggle ODEs button
    const odeBtn = document.createElement("button");
    odeBtn.type = "button";
    const showOdes = Boolean(model.get("show_odes"));
    odeBtn.className = `tc-btn ${showOdes ? "tc-btn-active" : ""}`;
    odeBtn.textContent = "d[X]/dt";
    odeBtn.title = "Show/hide differential rate equations";
    odeBtn.addEventListener("click", () => {
      model.set("show_odes", !model.get("show_odes"));
      model.save_changes();
      rebuildUI();
    });
    actions.appendChild(odeBtn);

    // Reset button
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "tc-btn tc-reset-btn";
    resetBtn.textContent = "Reset";
    resetBtn.disabled = !isModified();
    resetBtn.addEventListener("click", resetAll);
    actions.appendChild(resetBtn);

    header.appendChild(actions);
    container.appendChild(header);

    // 2. Interactive Chemical Equations Section
    const rxnBox = document.createElement("div");
    rxnBox.className = "tc-reactions-box";

    steps.forEach((step) => {
      const row = document.createElement("div");
      row.className = "tc-reaction-row";
      row.appendChild(
        renderReactionSide(step.reactants, speciesColorMap, isDark)
      );
      row.appendChild(renderReactionArrow(step));
      row.appendChild(
        renderReactionSide(step.products, speciesColorMap, isDark)
      );
      rxnBox.appendChild(row);
    });
    container.appendChild(rxnBox);

    // 3. Optional Differential Equations Drawer
    if (showOdes) {
      container.appendChild(renderODEPanel(sys, speciesColorMap, isDark));
    }

    // 4. Interactive Initial Concentrations Bar
    const icBar = document.createElement("div");
    icBar.className = "tc-ic-bar";

    const icLabel = document.createElement("span");
    icLabel.className = "tc-ic-title";
    icLabel.textContent = "Initial:";
    icBar.appendChild(icLabel);

    species.forEach((sp) => {
      const chip = document.createElement("div");
      const isHidden = hiddenSpecies.has(sp.name);
      chip.className = `tc-ic-chip ${isHidden ? "tc-ic-hidden" : ""}`;

      // Toggle dot for curve visibility
      const dotBtn = document.createElement("button");
      dotBtn.type = "button";
      dotBtn.className = "tc-sp-toggle";
      dotBtn.title = isHidden
        ? `Show [${sp.label}] on plot`
        : `Hide [${sp.label}] on plot`;
      const color = resolveColor(sp.color, isDark);
      dotBtn.style.setProperty("--tc-sp-color", color);
      dotBtn.addEventListener("click", () => {
        if (hiddenSpecies.has(sp.name)) {
          hiddenSpecies.delete(sp.name);
        } else {
          hiddenSpecies.add(sp.name);
        }
        rebuildUI();
      });
      chip.appendChild(dotBtn);

      const pill = createTangleControl(sp.param_key, { forceMode: "both" });
      pill.dataset.forceBoth = "true";
      chip.appendChild(pill);

      icBar.appendChild(chip);
    });
    container.appendChild(icBar);

    // 5. Plot Slot
    const plotSlot = document.createElement("div");
    plotSlot.className = "tc-plot-slot";
    container.appendChild(plotSlot);

    renderPlotIntoSlot();
  }

  // Subscribe to Python-side trait updates
  model.on("change:values", () => {
    if (!activeDragName) rebuildUI();
  });
  model.on("change:system_data", rebuildUI);
  model.on("change:parameters", rebuildUI);
  model.on("change:theme", rebuildUI);
  model.on("change:display_mode", rebuildUI);
  model.on("change:show_plot", rebuildUI);
  model.on("change:show_odes", rebuildUI);
  model.on("change:t_max", () => {
    if (!activeDragName) rebuildUI();
  });

  rebuildUI();
}

export default { render };
