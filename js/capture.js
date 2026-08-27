// state-of-mind — the capture boundary.
//
// The engine renders on the CPU at 320x200 and that framebuffer is the
// verified artifact: every gate compares against it, and downstream parts read
// it. An experimental WebGL renderer (productions/state-of-mind/web-gpu/)
// replays the same composite chain at 1600x1200, and to do that it needs the
// engine to say what it composited and what it drew.
//
// This file is the INTERFACE for that, and nothing else. The default is a null
// object whose every method does nothing, which is what the released CPU
// version ships and runs with. The real implementation lives beside the GPU
// renderer in web-gpu/js/capture-gpu.js and is injected:
//
//     createFidelity({ ..., capture: createGPUCapture() })
//
// WHY AN INJECTED OBJECT RATHER THAN FLAGS. The recording used to live inside
// fidelity.js behind two booleans, and the arrangement had a real cost: the
// geometry sink was attached to every renderer unconditionally while its
// per-frame reset only ran when recording was on, so a plain CPU run captured
// every triangle of the whole show into an array that doubled forever and
// never reset. With the capture injected, a release build cannot do that --
// there is no sink to attach.
//
// EVERY METHOD HERE IS RECORD-ONLY. Nothing the capture does may influence a
// pixel the CPU produces. That is not a convention, it is gated: an inertness
// check runs the engine with this null object and with the real capture over
// hundreds of probes and requires the two framebuffers to be identical.

export const NULL_CAPTURE = {
  // True when this capture records anything at all. Read at the handful of
  // call sites that would otherwise ALLOCATE to build an argument -- the
  // displacement mixers copy their offset table, the scroller copies its
  // palette -- because an argument is evaluated before the method that would
  // ignore it. Without this the released CPU path pays for a 256-entry copy
  // twice per displacement, every frame, to hand it to a function that
  // immediately drops it.
  active: false,

  // True when the capture wants DRAWING (triangles, sprites, glyphs) and not
  // just the composite chain. The engine reads this once at construction into
  // a closure const, so the per-sprite call sites cost a branch on a local --
  // the galaxy alone reaches them 900 times a frame.
  records: false,

  // The sink raster3d.js writes triangles and lens flares into, or null. The
  // engine attaches it only when it exists, so a null capture means the
  // rasteriser never sees a sink and never captures anything.
  geoSink: null,

  // Called from reset() with the freshly built buffer table. The real capture
  // stamps a stable name on each layer (the chain records layer names, since
  // object identity does not survive the boundary) and keeps the handle it
  // needs for pre-chain snapshots.
  attach() {},

  // Frame boundaries. begin() clears the log and resets the sink's cursors;
  // endFrame() gets the last word after the loop has run, which is where CPU
  // drawing that happened after the final composite is noticed.
  begin() {},
  endFrame() {},

  // A full-screen composite: the effect's name, its source layers by name, its
  // scalar uniforms, the destination buffer, and any per-command payload that
  // is neither (the scroller's strip and band).
  composite() {},

  // Called immediately after a mixer writes, to record what it produced.
  // Deferring it to the next command would mirror any CPU drawing in between
  // as if the composite had produced it, and the change would go unnoticed.
  out() {},

  // A composite whose second input is an accumulation of captured sprites.
  // Returns whether it was recorded, so the caller can skip its own bookkeeping.
  sprites() { return false; },

  // Drawing capture. Guarded by `records` at every call site.
  glyph() {},
  sprite() {},
};
