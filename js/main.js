// state-of-mind — browser shell around the fidelity engine.
//
// Query params (repo inspect conventions):
//   ?t=<seconds>   render one fixed instant, no audio, no loop
//   ?inspect=1     install window.__demo, no audio, no rAF loop
//   ?debug         status line shows time/frame/phase
//   ?embedded      hide the status line (for an iframe)
//   ?present=2d    force the canvas-2d fallback (default: webgl)
//   ?scale=<w>     backbuffer width: 1600 (default), 3200 or 4800
//   ?original      the 25 Hz show, as it ran in 1998. The default is the
//                  60 Hz remaster; this is the rate the oracle goldens and
//                  the reference capture are of. `?quality=original` is an
//                  accepted alias. Pair with ?weld=1 for the exactly
//                  authentic configuration.
//   ?remaster      accepted, and now a no-op: it names the default
//   ?rate=<hz>     25, 30, 50 or 60. Outranks ?original and the headless
//                  default. Default 60 for a viewer, 25 under ?t=/?inspect=1
//                  because the references are 25 Hz — see below
//   ?capoff=<ms>   shift the reference-capture alignment (inspector only)
//   ?capdrift=<ms/s>  its drift term (inspector only)
//   ?weld=1        restore the original's mesh weld, which is OFF by default
//                  here. Off is a deliberate deviation: the weld is a
//                  dimension bug in the original that empties ten city
//                  objects. ?weld=1 is the oracle-comparable configuration --
//                  see work/re/NOTES.md leg 30c.
//
// This is the RELEASE shell: the show at 320x200, presented by an exact
// integer upscale. Native 1600x1200 rendering and the WebGL composite path
// are experimental and live in productions/state-of-mind/web-gpu/ -- see the
// README there for what is verified and what is not.
//
// Dev serving: `node tools/serve.mjs state-of-mind` serves the production
// directory, so ../work/... resolves to the staged data. A published build is
// assembled by scripts/build-state-of-mind.sh, which relocates the data and
// rewrites js/assets.js; nothing else differs from what is reviewed here.

import { parseDat } from './dat.js';
import { createFidelity, frameForMs, RATE_25, RATE_30, RATE_50, RATE_60 } from './fidelity.js';
import { PHASES } from './show-data.js';
import { createPresenter } from './present.js';
import { createDebugOverlay } from './debug-overlay.js';
import { resolveRate } from './rate.js';
import { DATA_BASE } from './assets.js';

const status = document.querySelector('#status');
const startBtn = document.querySelector('#start');
const canvas = document.querySelector('#screen');

const params = new URLSearchParams(location.search);
const fixedTime = params.has('t') ? Number(params.get('t')) : null;
const inspect = params.get('inspect') === '1';
const debug = params.has('debug');
const headless = fixedTime !== null || inspect;

// THE LAUNCHER MUST NOT COVER THE CANVAS FOR A MEASUREMENT. The show fades in
// on `body.playing`, which a viewer earns by pressing play -- but ?t= and
// ?inspect=1 never press anything, and every tool that reads this page
// screenshots the canvas. Without this the sweep would capture 368 blank
// frames and score them as a total failure, which is exactly the class of
// silent instrument fault leg 30 was spent unpicking.
if (headless) document.body.classList.add('playing');
// ?embedded strips the status line for an iframe: the page is the demo.
if (params.has('embedded')) document.documentElement.classList.add('embedded');

// THE RATE, AND WHY THE DEFAULT DEPENDS ON WHO IS ASKING.
//
// 25 Hz does not divide a 60 Hz display evenly (60/25 = 2.4), so the original
// judders by construction on modern hardware. The remaster re-times the show
// to 60 Hz and corrects the nine effects that count frames rather than time,
// so they keep their authored speed -- work/re/FRAMERATE.md has the inventory.
// A VIEWER GETS THAT BY DEFAULT: it is the version worth watching, and the two
// other published ports in this repo already ship remastered by default with
// the faithful build behind a flag (lost-vegas ?quality=original, ptct alike).
//
// A MEASUREMENT GETS 25 Hz, ALWAYS. The reference capture and all 1097 oracle
// goldens are of the 25 Hz show, and nothing under tools/inspect/ appends a
// `&rate=` -- sweep.mjs just loads `?inspect=1`. So a single global default of
// 60 would silently re-point the whole verification chain at a show the
// references are not of, while every gate stayed green, because the gates
// build their own engines and never read this line. Hence: `headless` (?t= or
// ?inspect=1) means 25 Hz unless asked otherwise.
//
// `?rate=` outranks all of it, so any tool that genuinely wants the remastered
// timing can still ask for it. release-shape.mjs asserts both halves.
const RATES = { 25: RATE_25, 30: RATE_30, 50: RATE_50, 60: RATE_60 };
// The rule itself lives in ./rate.js, shared with the experimental shell and
// asserted exhaustively by work/verify/shell-rate.mjs.
const { hz: wantHz, remaster } = resolveRate(params, headless);
// Where the reference capture sits relative to the show clock.
//
// COMPARISON ONLY: nothing but __demo.schedule()'s captureStart reads these, so
// playback cannot be affected by getting them wrong -- they change which frame
// of the capture the inspector pairs with ours, and nothing else. `?capoff=`
// nudges the constant live, for re-measuring without an edit.
//
// A LINE, NOT A CONSTANT, and that is the whole point. Measured by audio
// cross-correlation against the show's own soundtrack (mind.mp3, carried
// inside mind.dat -- so the reference is the exact audio the show is timed to,
// not a re-encode of it): 10 of 12 windows correlate at r 0.76-0.99 and fit
//
//     lag(t) = -188.2 - 0.3764 * t   milliseconds        R^2 = 0.989
//
// R^2 that high over 180 s is a clock-RATE difference, not scatter: the
// capture runs about 0.038% fast, drifting -82 ms end to end. At 25 Hz one
// frame is 40 ms, so a single offset would be right in the middle and a whole
// frame out at both ends -- every head and tail sample would be scored against
// the capture's neighbouring frame for a reason that has nothing to do with
// the renderer. work/tools/align-capture.mjs re-measures and re-fits.
const CAPTURE_OFFSET_MS = Number(params.get('capoff') ?? -188.2);
const CAPTURE_DRIFT_MS_PER_S = Number(params.get('capdrift') ?? -0.3764);
const rate = RATES[wantHz] ?? RATE_25;
const stepMs = rate.num / rate.den;
// The engine's 320x200 RGB565 output is the verification target and is never
// touched; the presenter only decides how those pixels reach the display.
const presenter = createPresenter(canvas, {
  mode: params.get('present') === '2d' ? '2d' : 'webgl',
  scale: params.get('scale'),
  // The headless paths exist to be captured; playback never reads the buffer.
  capture: headless,
});

try {
  // Only the original pack plus the source-derived lyric table: the JPEG
  // textures are decoded in JS by the engine's own ported decoder
  // (js/jpeg.js), so no oracle exports are fetched at runtime.
  const [datBytes, ffx, sprites, lines] = await Promise.all([
    fetch(`${DATA_BASE}src/w32/mind.dat`).then((r) => { if (!r.ok) throw new Error('mind.dat: run work/unpack.mjs'); return r.arrayBuffer(); }),
    fetch(`${DATA_BASE}unpacked/ffx.json`).then((r) => { if (!r.ok) throw new Error('ffx.json: run work/unpack.mjs'); return r.json(); }),
    fetch(`${DATA_BASE}unpacked/sprites.json`).then((r) => { if (!r.ok) throw new Error('sprites.json: run work/unpack.mjs'); return r.json(); }),
    fetch(`${DATA_BASE}unpacked/lines.json`).then((r) => { if (!r.ok) throw new Error('lines.json: run work/unpack.mjs'); return r.json(); }),
  ]);
  const dat = parseDat(new Uint8Array(datBytes));

  // THE MESH WELD IS OFF BY DEFAULT, AND THAT IS A DELIBERATE DEVIATION.
  //
  // `Optimize_Mesh` (3d_src/optimesh.c:165) compares a SQUARED distance
  // against a LINEAR threshold, so its weld radius is sqrt(0.05*R) instead of
  // 0.05*R and it over-welds small objects: it removes 552 of villef3's 1974
  // polygons and empties ten objects outright (work/tools/weld-census.mjs).
  // It is a bug in the original, it ran in the shipped binary, and the show
  // this project restores therefore had it.
  //
  // Turning it off is a CHOICE MADE HERE, not a repair of a port defect, and
  // it is not free: measured against the reference capture the unwelded render
  // scores WORSE on the statue -- mean r 0.7617 against 0.8116 welded, losing
  // at seven of eight instants (work/re/NOTES.md leg 30c). What it buys is the
  // geometry the weld destroys.
  //
  // `?weld=1` restores the original behaviour, and that is the configuration
  // the oracle goldens and work/verify/fidelity-check.mjs describe. The
  // bit-exactness claim belongs to ?weld=1, not to this default.
  const weld = params.get('weld') === '1';
  const engine = createFidelity({ dat, ffx, sprites, lines, rate, remaster, weld });
  engine.setSchedule(PHASES);
  engine.reset();

  // Set once the show starts, so the title backdrop stops competing for the
  // engine it no longer needs.
  let posterAbandoned = false;
  const posterDone = () => posterAbandoned;

  let lastBlitFrame = -1;
  function blit() {
    presenter.present(engine.frame565);
    lastBlitFrame = engine.state ? engine.state.frame : -1;
  }

  function renderShowSeconds(seconds) {
    const frame = Math.max(0, frameForMs(seconds * 1000, rate));
    engine.advanceTo(frame);
    const s = engine.state;
    // The show is a fixed-rate film, so a 60 Hz rAF loop asks for the same
    // frame more than once whenever the rate is below the refresh rate
    // (~2.4x in a row at the default 25 Hz). Re-uploading an unchanged
    // 320x200 buffer is pure waste — putImageData is a real cost in Safari
    // — so only blit when the engine actually advanced a frame.
    // ON EVERY HEADLESS PATH, ALWAYS BLIT. The skip above is right for a 60 Hz
    // rAF loop driving a 25 Hz show, but ?inspect=1 and ?t= both exist to be
    // LOOKED AT and captured, and their contract is that after a render the
    // canvas SHOWS that frame -- every consumer screenshots it.
    // A WebGL drawing buffer is discarded once composited, so a skipped blit
    // hands back a black or stale canvas with no error at all. The sweep
    // recorded 19 such samples out of 368 and scored 13 of them r=0.000,
    // inventing failures in parts that render correctly: `die-in` and
    // `die-out` both reported a median of 0.000 built entirely from them.
    if (headless || s.frame !== lastBlitFrame) blit();
    return s;
  }

  // ?debug variant: advance ONE engine frame per step so each frame carries
  // its own timing and its own phase. advanceTo() only ever walks forward
  // here, so stepping to N one frame at a time lands on exactly the state a
  // single advanceTo(N) would produce -- the measurement does not perturb
  // what is measured. Present is timed separately: a slow upload and a slow
  // engine look identical from the outside and want opposite fixes.
  function renderShowSecondsInstrumented(seconds, ticks) {
    const target = Math.max(0, frameForMs(seconds * 1000, rate));
    // engine.state.frame is the LAST RENDERED frame (st.frame - 1), so the
    // next one to ask for is that plus one. Requesting a frame already
    // rendered is a no-op, which is what keeps a repeated rAF free.
    let cur = engine.state ? engine.state.frame : -1;
    while (cur < target) {
      cur++;
      const t0 = performance.now();
      engine.advanceTo(cur);
      const s = engine.state;
      ticks.push({ ms: performance.now() - t0, phase: s.phase });
      if (s.frame !== cur) break;            // engine reported end of show
    }
    const s = engine.state;
    let presentMs = 0;
    if (s.frame !== lastBlitFrame) {
      const p0 = performance.now();
      blit();
      presentMs = performance.now() - p0;
    }
    return { state: s, presentMs };
  }

  if (inspect) {
    // tools/inspect adapter (ADAPTER.md): schedule/render/state, set last.
    const rows = [];
    for (let i = 0; i < PHASES.length - 1; i++) {
      const p = PHASES[i];
      if (p.t < 0) break;
      // A ROW'S DURATION COMES FROM ITS SUCCESSOR, so the last real phase used
      // to measure itself against the { t: -1 } exit sentinel and come out
      // 219.2 SECONDS LONG IN THE NEGATIVE. `end` is a zero-length terminal
      // marker anyway -- a Loop_0 hold at the final instant, with `fade-2` as
      // the show's actual ending -- so it is not a part to inspect.
      //
      // The negative duration was not inert. plan.mjs clamps the sample count
      // up to its floor of 5, so `end` produced five samples at NEGATIVE local
      // times, each rendering `start + local` somewhere else entirely in the
      // show; one of them landed in the middle of `city` and an observation
      // about the city was recorded against the part name `end`.
      if (PHASES[i + 1].t < 0) break;
      rows.push({
        name: p.name, phase: i,
        start: p.t / 1000,
        dur: (PHASES[i + 1].t - p.t) / 1000,
        captureStart: p.t / 1000
          + (CAPTURE_OFFSET_MS + CAPTURE_DRIFT_MS_PER_S * (p.t / 1000)) / 1000,
      });
    }
    let last = null;
    window.__demo = {
      id: 'state-of-mind',
      schedule: () => rows,
      async render({ part, local }) {
        const row = rows.find((r) => r.name === part);
        if (!row) return null;
        last = renderShowSeconds(row.start + local);
        return { ...last };
      },
      state: () => last,
      features: {},
    };
    window.__demoReady = true;
    status.textContent = 'inspect mode';
  } else if (fixedTime !== null) {
    const s = renderShowSeconds(fixedTime);
    // The rate is part of WHICH FRAME this is, so a fixed-instant render that
    // did not name it would be ambiguous -- 10 s is frame 250 at 25 Hz and
    // frame 600 at 60. It is also the observable work/verify/shell-rate.mjs
    // uses to hold the headless default at 25 Hz.
    status.textContent = `t=${fixedTime.toFixed(2)} rate=${rate.hz} frame=${s.frame - 1} `
      + `phase=${s.phase}${s.unported ? ' (unported part)' : ''}`;
  } else {
    // ---- live playback, driven by the demo's own music out of the pack ----
    //
    // THE AUDIO IS THE CLOCK. The show was authored against this mp3, so the
    // renderer follows it rather than a frame counter: drift then shows up as
    // a picture that lags the music, which a viewer can see, instead of
    // accumulating silently. Seeking restarts the buffer at an offset and
    // re-anchors, so the same rule holds after a jump.
    const SHOW_SECONDS = (() => {
      for (let i = PHASES.length - 1; i >= 0; i--) if (PHASES[i].t >= 0) return PHASES[i].t / 1000;
      return 0;
    })();

    const audioCtx = new AudioContext();
    const buf = await audioCtx.decodeAudioData(dat.get('mind.mp3').slice().buffer);
    startBtn.disabled = false;
    startBtn.textContent = 'play';

    let srcNode = null, anchorCtx = 0, anchorShow = 0;
    let running = false, wakeLock = null;
    const showTime = () => (audioCtx.currentTime - anchorCtx) + anchorShow;

    // A transient line over the show. Nothing is on screen unless something
    // happened: the page is the demo, not a dev harness.
    let sayTimer = 0;
    const say = (text, ms = 1100) => {
      status.textContent = text;
      clearTimeout(sayTimer);
      if (text && ms) sayTimer = setTimeout(() => { status.textContent = ''; }, ms);
    };

    function playFrom(sec) {
      const at = Math.max(0, Math.min(sec, SHOW_SECONDS));
      if (srcNode) { try { srcNode.stop(); } catch { /* already stopped */ } srcNode.disconnect(); }
      srcNode = audioCtx.createBufferSource();
      srcNode.buffer = buf;
      srcNode.connect(audioCtx.destination);
      // A beat of lead-in so the first frame is not rendered against an audio
      // clock that has not started moving yet.
      const when = audioCtx.currentTime + 0.05;
      srcNode.start(when, at);
      anchorCtx = when; anchorShow = at;
    }

    async function begin() {
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      posterAbandoned = true;
      document.body.classList.remove('ended', 'poster');
      document.body.classList.add('playing');
      startBtn.textContent = 'play';
      running = true;
      playFrom(0);
      // Keep the screen awake for the length of the show. Unsupported and
      // rejected are both fine -- it is a courtesy, not a requirement.
      try { wakeLock = await navigator.wakeLock?.request('screen') ?? null; } catch { wakeLock = null; }
    }

    async function stop(reason) {
      running = false;
      if (srcNode) { try { srcNode.stop(); } catch { /* already stopped */ } srcNode.disconnect(); srcNode = null; }
      try { await wakeLock?.release(); } catch { /* already gone */ }
      wakeLock = null;
      document.body.classList.remove('playing');
      // Whatever frame the show stopped on becomes the backdrop, so returning
      // to the launcher lands somewhere rather than on black.
      document.body.classList.add('ended', 'poster');
      startBtn.textContent = 'replay';
      say(reason ?? '', 1600);
    }

    startBtn.addEventListener('click', () => { if (!running) begin(); });

    // ---- controls, the set every port at demoports carries ----------------
    const seek = (delta) => {
      if (!running) return;
      const to = Math.max(0, Math.min(showTime() + delta, SHOW_SECONDS));
      // Forward is cheap -- advanceTo walks frames. BACKWARD REPLAYS FROM
      // FRAME 0, because every effect is stateful and no frame can be
      // skipped: measured at 5.4 s of work at 25 Hz and 12.8 s at 60. So say
      // so rather than appear to hang. work/re/NOTES.md leg 31 has the plan
      // for keyframes that make this instant.
      if (delta < 0) say(`seeking to ${to.toFixed(0)}s…`, 0);
      playFrom(to);
      if (delta < 0) requestAnimationFrame(() => say(`${to.toFixed(0)}s`));
      else say(`${to.toFixed(0)}s`);
    };

    const togglePause = async () => {
      if (!running) return;
      if (audioCtx.state === 'running') { await audioCtx.suspend(); say('paused', 0); }
      else { await audioCtx.resume(); say('') ; }
    };

    const toggleFullscreen = async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await document.documentElement.requestFullscreen?.();
      } catch { /* refused, or unsupported: not worth interrupting the show */ }
    };

    addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case ' ': case 'Spacebar': e.preventDefault(); togglePause(); break;
        case 'ArrowLeft': e.preventDefault(); seek(-5); break;
        case 'ArrowRight': e.preventDefault(); seek(5); break;
        case 'f': case 'F': e.preventDefault(); toggleFullscreen(); break;
        case 'Escape':
          // Escape already leaves fullscreen on its own; only stop the show
          // when there is no fullscreen for it to leave.
          if (!document.fullscreenElement && running) stop('stopped');
          break;
        case 'Enter':
          if (!running) begin();
          break;
        default: break;
      }
    });

    // ---- the frame loop ---------------------------------------------------
    const overlay = debug ? createDebugOverlay({ stepMs, phases: PHASES }) : null;
    let shown = 0, statTime = 0, shownPerSec = 0, lastFrame = -1;
    const tick = () => {
      requestAnimationFrame(tick);
      if (!running) return;
      const t = showTime();
      if (t < 0) return;
      // THE SHOW ENDS. It used to run the rAF loop forever past the last
      // phase, holding the final frame with the music finished behind it.
      if (t >= SHOW_SECONDS) { stop('the end'); return; }
      if (overlay) {
        const ticks = [];
        const { state: s, presentMs } = renderShowSecondsInstrumented(t, ticks);
        overlay.sample({ ticks, presentMs, showTime: t, now: performance.now() });
        if (s.frame !== lastFrame) { shown++; lastFrame = s.frame; }
        if (t - statTime >= 1) { shownPerSec = shown; shown = 0; statTime = t; }
        overlay.render({ showTime: t, frame: s.frame, phase: s.phase, shownPerSec });
      } else {
        const s = renderShowSeconds(t);
        if (s.unported) say('part not yet ported', 0);
      }
    };
    requestAnimationFrame(tick);
  }

  // THE TITLE BACKGROUND IS THE DEMO ITSELF -- one real instant, rendered here
  // and left on the canvas behind the launcher. Nothing is shipped for it and
  // nothing can fall out of sync with the renderer.
  //
  // 22 s is chosen for two reasons: it is angular and abstract with no
  // on-screen type of its own to fight the title, and it is CHEAP TO REACH.
  // Reaching an instant means replaying to it -- every effect is stateful --
  // and at the 60 Hz default that is ~1.3 s of work here against 5.1 s for the
  // city at 74.7 s. Measured, both.
  //
  // Even 1 s must not be spent in one go: it would freeze the page while
  // someone is trying to press play. So it is walked in slices that yield
  // every frame, blitting as it goes -- the backdrop resolves in rather than
  // appearing -- and ABANDONED the moment the show starts, because at that
  // point the engine is needed for the thing the viewer actually asked for.
  if (!headless) {
    document.body.classList.add('poster');
    const target = Math.max(0, frameForMs(22.0 * 1000, rate));
    let at = 0;
    const paint = () => {
      if (posterDone()) return;
      const until = performance.now() + 6;      // well inside one 60 Hz frame
      while (at < target && performance.now() < until) engine.advanceTo(++at);
      blit();
      if (at < target) requestAnimationFrame(paint);
    };
    requestAnimationFrame(paint);
  }
} catch (error) {
  status.textContent = error.message;
  throw error;
}
