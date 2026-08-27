// state-of-mind — performance debug view (?debug).
//
// "Chuggy" has two unrelated causes here and they need different answers, so
// the overlay measures them separately:
//
//   COST    — is the engine inside its 40 ms budget? A show frame that takes
//             longer than STEP_MS cannot be sustained, and the loop then has
//             to advance several engine frames per rAF to catch up, which
//             makes the next one later still.
//
//   CADENCE — even a free engine judders on the wrong display. The show is a
//             fixed 25 Hz film; 60/25 = 2.4, so each frame is held for 2 or 3
//             refreshes in a 2,2,3,2,2,3... pattern. That is visible as
//             unevenness and no amount of optimisation removes it. 120 Hz is
//             just as uneven (4.8); only 50/100 Hz divide cleanly.
//
// A reading of "engine 6 ms of 40" together with a 2,2,3 repeat histogram
// means the port is fast and the display is the problem. The reverse means
// the opposite. The panel says which.

const GRAPH_W = 280, GRAPH_H = 56;
const HISTORY = GRAPH_W;
const HITCH_KEEP = 8;

export function createDebugOverlay({ stepMs, phases }) {
  const root = document.createElement('div');
  root.id = 'debug-overlay';
  root.innerHTML = `
    <canvas class="g" width="${GRAPH_W}" height="${GRAPH_H}"></canvas>
    <div class="cost"></div>
    <div class="cadence"></div>
    <div class="hot"></div>
    <div class="hint">d: hide &middot; r: reset stats</div>`;
  const style = document.createElement('style');
  style.textContent = `
    #debug-overlay {
      position: fixed; top: 8px; left: 8px; z-index: 10;
      background: rgba(0,0,0,.82); border: 1px solid #343; padding: 8px 10px;
      font: 11px/1.45 ui-monospace, monospace; color: #9a9;
      white-space: pre; pointer-events: none; max-width: 320px;
    }
    #debug-overlay canvas { display: block; margin-bottom: 6px; background: #060806; }
    #debug-overlay .warn { color: #e85; }
    #debug-overlay .bad  { color: #f66; }
    #debug-overlay .ok   { color: #6c8; }
    #debug-overlay .hdr  { color: #576; }
    #debug-overlay .hint { color: #454; margin-top: 6px; }
    #debug-overlay.hidden { display: none; }`;
  document.head.appendChild(style);
  document.body.appendChild(root);

  const graph = root.querySelector('.g').getContext('2d');
  const elCost = root.querySelector('.cost');
  const elCadence = root.querySelector('.cadence');
  const elHot = root.querySelector('.hot');

  const hist = new Float32Array(HISTORY);   // engine ms per rAF
  const histPresent = new Float32Array(HISTORY);
  let histAt = 0, histLen = 0;

  let stats, perPhase, hitches, repeats, rafGaps;
  function reset() {
    stats = {
      ticks: 0, rafs: 0,
      engineSum: 0, engineMax: 0,
      presentSum: 0, presentMax: 0,
      catchupRafs: 0, maxTicksPerRaf: 0,
      overBudget: 0,
    };
    perPhase = new Map();       // phase index -> {n, sum, max}
    hitches = [];               // worst single ticks: {ms, t, phase}
    repeats = new Map();        // refreshes-per-shown-frame -> count
    rafGaps = [];               // recent rAF intervals, for display Hz
    histAt = 0; histLen = 0;
    hist.fill(0); histPresent.fill(0);
  }
  reset();

  let lastRafTime = 0;
  let refreshesSinceShown = 0;

  document.addEventListener('keydown', (e) => {
    if (e.key === 'd') root.classList.toggle('hidden');
    else if (e.key === 'r') reset();
  });

  function phaseName(i) {
    const p = phases[i];
    return p ? (p.name ?? `#${i}`) : `#${i}`;
  }

  // One rAF's worth of measurements.
  //   ticks    : [{ ms, phase }] — one entry per engine frame actually advanced
  //   presentMs: cost of getting the pixels to the screen
  //   showTime : the audio clock, for the hitch log
  function sample({ ticks, presentMs, showTime, now }) {
    const gap = lastRafTime ? now - lastRafTime : 0;
    lastRafTime = now;
    if (gap > 0 && gap < 200) {
      rafGaps.push(gap);
      if (rafGaps.length > 120) rafGaps.shift();
    }

    // Cadence: count how many refreshes each *shown* frame survived. A rAF
    // that advanced no engine frame is a repeat of the one on screen.
    refreshesSinceShown++;
    if (ticks.length > 0) {
      repeats.set(refreshesSinceShown, (repeats.get(refreshesSinceShown) ?? 0) + 1);
      refreshesSinceShown = 0;
    }

    stats.rafs++;
    if (ticks.length > 1) stats.catchupRafs++;
    if (ticks.length > stats.maxTicksPerRaf) stats.maxTicksPerRaf = ticks.length;

    let engineMs = 0;
    for (const t of ticks) {
      engineMs += t.ms;
      stats.ticks++;
      stats.engineSum += t.ms;
      if (t.ms > stats.engineMax) stats.engineMax = t.ms;
      if (t.ms > stepMs) stats.overBudget++;

      let rec = perPhase.get(t.phase);
      if (!rec) { rec = { n: 0, sum: 0, max: 0 }; perPhase.set(t.phase, rec); }
      rec.n++; rec.sum += t.ms;
      if (t.ms > rec.max) rec.max = t.ms;

      if (hitches.length < HITCH_KEEP || t.ms > hitches[hitches.length - 1].ms) {
        hitches.push({ ms: t.ms, t: showTime, phase: t.phase });
        hitches.sort((a, b) => b.ms - a.ms);
        if (hitches.length > HITCH_KEEP) hitches.length = HITCH_KEEP;
      }
    }

    stats.presentSum += presentMs;
    if (presentMs > stats.presentMax) stats.presentMax = presentMs;

    hist[histAt] = engineMs;
    histPresent[histAt] = presentMs;
    histAt = (histAt + 1) % HISTORY;
    if (histLen < HISTORY) histLen++;
  }

  function displayHz() {
    if (rafGaps.length < 20) return null;
    // Median, not mean: one long gap from a GC pause or a tab stall would
    // drag a mean far off the real refresh rate.
    const s = [...rafGaps].sort((a, b) => a - b);
    return 1000 / s[s.length >> 1];
  }

  function drawGraph() {
    graph.clearRect(0, 0, GRAPH_W, GRAPH_H);
    // Scale so the 40 ms budget sits at 70% height unless something exceeds it
    let peak = stepMs;
    for (let i = 0; i < histLen; i++) peak = Math.max(peak, hist[i] + histPresent[i]);
    const scale = (GRAPH_H * 0.92) / (peak / 0.7 > stepMs ? peak / 0.7 : stepMs / 0.7);

    const budgetY = GRAPH_H - stepMs * scale;
    graph.fillStyle = '#1a2a1a';
    graph.fillRect(0, budgetY, GRAPH_W, GRAPH_H - budgetY);
    graph.strokeStyle = '#3a5a3a';
    graph.beginPath();
    graph.moveTo(0, budgetY + 0.5); graph.lineTo(GRAPH_W, budgetY + 0.5);
    graph.stroke();

    for (let i = 0; i < histLen; i++) {
      const idx = (histAt - histLen + i + HISTORY * 2) % HISTORY;
      const e = hist[idx], p = histPresent[idx];
      const eh = e * scale, ph = p * scale;
      graph.fillStyle = e > stepMs ? '#f66' : '#6c8';
      graph.fillRect(i, GRAPH_H - eh, 1, eh);
      graph.fillStyle = '#48a';
      graph.fillRect(i, GRAPH_H - eh - ph, 1, ph);
    }
    graph.fillStyle = '#576';
    graph.fillText(`${stepMs}ms`, 2, budgetY - 2);
  }

  function cls(v, warn, bad) {
    return v >= bad ? 'bad' : v >= warn ? 'warn' : 'ok';
  }

  function render({ showTime, frame, phase, shownPerSec }) {
    drawGraph();

    const engMean = stats.ticks ? stats.engineSum / stats.ticks : 0;
    const presMean = stats.rafs ? stats.presentSum / stats.rafs : 0;
    const overPct = stats.ticks ? (100 * stats.overBudget / stats.ticks) : 0;
    const budgetUse = 100 * engMean / stepMs;

    elCost.innerHTML =
      `<span class="hdr">COST</span>  t=${showTime.toFixed(1)}s f=${frame} ${phaseName(phase)}\n`
      + `engine  <span class="${cls(engMean, stepMs * 0.6, stepMs)}">`
      + `${engMean.toFixed(1)}</span> ms/frame of ${stepMs}`
      + `  (${budgetUse.toFixed(0)}% budget)\n`
      + `        max ${stats.engineMax.toFixed(1)}`
      + `  over budget <span class="${cls(overPct, 1, 10)}">${overPct.toFixed(1)}%</span>\n`
      + `present <span class="${cls(presMean, 4, 10)}">${presMean.toFixed(2)}</span> ms`
      + `  max ${stats.presentMax.toFixed(1)}\n`
      + `catch-up rAFs <span class="${cls(stats.catchupRafs, 1, 20)}">${stats.catchupRafs}</span>`
      + `  worst ${stats.maxTicksPerRaf} ticks/rAF\n`;

    const hz = displayHz();
    let cadence = `<span class="hdr">CADENCE</span>\n`;
    if (hz === null) {
      cadence += `display  measuring...\n`;
    } else {
      const ratio = hz / (1000 / stepMs);
      const clean = Math.abs(ratio - Math.round(ratio)) < 0.02;
      cadence += `display  ~${hz.toFixed(0)} Hz`
        + `   show ${(1000 / stepMs).toFixed(0)} Hz\n`
        + `ratio    <span class="${clean ? 'ok' : 'warn'}">${ratio.toFixed(2)}</span>`
        + ` refreshes/frame ${clean ? '(even)' : '(UNEVEN — inherent judder)'}\n`;
    }
    const reps = [...repeats.entries()].filter(([k]) => k > 0).sort((a, b) => a[0] - b[0]);
    const total = reps.reduce((s, [, n]) => s + n, 0);
    if (total) {
      cadence += `held     ` + reps.map(([k, n]) =>
        `${k}x:${(100 * n / total).toFixed(0)}%`).join('  ') + `\n`;
    }
    cadence += `shown    ${shownPerSec}/s (ceiling ${(1000 / stepMs).toFixed(0)})\n`;
    elCadence.innerHTML = cadence;

    const worst = [...perPhase.entries()]
      .map(([p, r]) => ({ p, mean: r.sum / r.n, max: r.max, n: r.n }))
      .sort((a, b) => b.mean - a.mean).slice(0, 5);
    let hot = `<span class="hdr">SLOWEST PARTS</span> (mean ms/frame)\n`;
    for (const w of worst) {
      hot += `  <span class="${cls(w.mean, stepMs * 0.6, stepMs)}">`
        + `${w.mean.toFixed(1).padStart(5)}</span>`
        + ` max ${w.max.toFixed(0).padStart(3)}  ${phaseName(w.p)}\n`;
    }
    if (hitches.length) {
      hot += `<span class="hdr">WORST FRAMES</span>\n`;
      for (const h of hitches.slice(0, 3)) {
        hot += `  ${h.ms.toFixed(0).padStart(4)} ms  t=${h.t.toFixed(1)}s`
          + `  ${phaseName(h.phase)}\n`;
      }
    }
    elHot.innerHTML = hot;
  }

  return { sample, render, reset, el: root };
}
