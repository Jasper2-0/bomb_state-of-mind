// snapshot.js — deep-copy the engine's mutable state so a seek can jump back.
//
// WHY THIS EXISTS. Every effect in the show is stateful -- feedback buffers,
// scroll offsets, particle systems, three PRNGs -- so no frame can be computed
// without the one before it, and `advanceTo` replays from frame 0 whenever it
// is asked to go backwards. Measured, that is 5.4 s of work at 25 Hz and
// 12.8 s at 60. Seeking back five seconds cannot cost thirteen.
//
// So the shell keeps periodic snapshots and replays from the nearest one. A
// snapshot of the whole state measures 1.93 MB, which is what makes this
// affordable at all.
//
// TWO THINGS THIS HAS TO GET RIGHT, and both are silent when got wrong -- a
// restored frame that is subtly stale looks perfectly plausible. That is why
// work/verify/snapshot-replay.mjs requires a restored frame to be BIT-IDENTICAL
// to a replayed one rather than merely similar.
//
//  1. SHARED BUFFERS MUST STAY SHARED. Every colour map is one ArrayBuffer
//     viewed as both Uint16Array and Uint32Array (`mkCMap` in fidelity.js) --
//     the 565 and 777 forms of the same palette. Cloning the two views
//     independently would silently split them in half, so writes through one
//     would stop being visible through the other. The buffer map below clones
//     each ArrayBuffer once and re-creates every view onto that copy, which
//     preserves aliasing and, incidentally, cycles.
//
//  2. CLOSURES ARE NOT DATA. Three pieces of state are captured variables, not
//     fields: the janim LCG, BSD random(), and the palette-ramp builder. They
//     cannot be cloned by walking the object, so each exposes save()/load()
//     and this module round-trips that instead.

/** Keys of `st` whose value is a closure with its own save()/load(). */
export const CLOSURE_KEYS = ['trandom', 'random', 'ramp'];

/**
 * Deep copy, preserving ArrayBuffer sharing and object identity.
 * Functions are returned as-is: the caller handles them via CLOSURE_KEYS.
 */
export function deepCopy(v, bufs = new Map(), objs = new Map()) {
  if (v === null || typeof v !== 'object') return v;
  if (typeof v === 'function') return v;
  if (objs.has(v)) return objs.get(v);

  if (ArrayBuffer.isView(v)) {
    // DataView has no `length`; every view the engine uses is a typed array.
    let b = bufs.get(v.buffer);
    if (!b) { b = v.buffer.slice(0); bufs.set(v.buffer, b); }
    const out = new v.constructor(b, v.byteOffset, v.length);
    objs.set(v, out);
    return out;
  }
  if (v instanceof ArrayBuffer) {
    let b = bufs.get(v);
    if (!b) { b = v.slice(0); bufs.set(v, b); }
    return b;
  }
  if (Array.isArray(v)) {
    const out = [];
    objs.set(v, out);
    for (const x of v) out.push(deepCopy(x, bufs, objs));
    return out;
  }
  const out = {};
  objs.set(v, out);
  for (const k of Object.keys(v)) out[k] = deepCopy(v[k], bufs, objs);
  return out;
}

/**
 * Capture `st`. Closure-valued keys are stored as their save() payload under
 * `closures`; everything else is deep-copied.
 */
export function captureState(st) {
  const data = {};
  const closures = {};
  const bufs = new Map(), objs = new Map();
  for (const k of Object.keys(st)) {
    if (CLOSURE_KEYS.includes(k)) {
      const v = st[k];
      closures[k] = v?.save ? v.save() : null;
      continue;
    }
    data[k] = deepCopy(st[k], bufs, objs);
  }
  return { data, closures };
}

/**
 * Write a captured state back into `st`, IN PLACE.
 *
 * In place, not by replacing the object, because the engine's own closures
 * captured `st` when they were built. It also means the live closure
 * instances are kept and only their internal state is reloaded -- their
 * identity carries no information, only their contents do.
 */
export function applyState(st, snap) {
  const bufs = new Map(), objs = new Map();
  for (const k of Object.keys(snap.data)) st[k] = deepCopy(snap.data[k], bufs, objs);
  for (const k of CLOSURE_KEYS) {
    const saved = snap.closures[k];
    if (saved && st[k]?.load) st[k].load(saved);
  }
  return st;
}
