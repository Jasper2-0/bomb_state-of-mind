// state-of-mind — frame-rate corrections for the remastered mode.
//
// The show is a 25 Hz film. 60/25 = 2.4, so on a 60 Hz display every frame is
// held for 2 or 3 refreshes in a 2,2,3 pattern: inherent judder that no
// optimisation removes. Running the show at 60 Hz (or 30) fixes the cadence,
// but nine effects in the demo are counted in FRAMES rather than driven by the
// clock, and those would then run 2.4x fast. work/re/FRAMERATE.md is the full
// inventory; this module is the correction layer it prescribes.
//
// IMPORTANT: these corrections are AUTHORED, not derived from the oracle.
// The rational clock they sit on top of is oracle-verifiable — the oracle can
// run the original C at 50/3 and the port matches it bit-for-bit — but the
// corrections themselves change what the demo does, so there is no authority
// to check them against. They are gated on properties, not on pixels, and the
// 25 Hz artifact remains the verified one.
//
// Every helper is the IDENTITY when the rate is 25 Hz or remaster is off, so
// the verified path runs the same arithmetic either way. `remaster-timing`
// asserts that inertness directly.

export const MIX_LEVELS = 16;

export function createRetimer(rate, remaster) {
  // Frames of this rate per original 25 Hz frame: 1 at 25 Hz, 2.4 at 60 Hz.
  const rateScale = (40 * rate.den) / rate.num;
  const retimed = !!remaster && rateScale !== 1;

  return {
    rateScale,
    retimed,

    // A per-frame retention r must compound to the same value per SECOND:
    // r' = r^(1/rateScale). Used for fade.c's 0.98 feedback constant, the
    // one correction that is exactly expressible.
    decayConst: (r) => (retimed ? Math.pow(r, 1 / rateScale) : r),

    // Mix level l means a retention of (MIX_LEVELS-1-l)/(MIX_LEVELS-1).
    // For a leaky integrator out = a*new + b*out, correcting the time
    // constant alone would also move the steady state a/(1-b), so both terms
    // are solved together:
    //     b' = b^(1/rateScale),    a' = a*(1-b')/(1-b)
    // Levels are integers 0..15, so this is the one correction that CANNOT be
    // exact — it lands on the nearest available level. The residual is
    // largest where b is near 0 or 1 and the level grid is coarsest.
    decayLevels(i, j) {
      if (!retimed) return [i, j];
      const N = MIX_LEVELS - 1;
      const a = (N - i) / N, b = (N - j) / N;
      if (!(b > 0 && b < 1)) return [i, j];
      const b2 = Math.pow(b, 1 / rateScale);
      const a2 = (a * (1 - b2)) / (1 - b);
      const lv = (x) => Math.max(0, Math.min(N, Math.round(N * (1 - x))));
      return [lv(a2), lv(b2)];
    },

    // A duration counted in frames covers the same wall-clock span.
    durFrames: (n) => (retimed ? Math.max(1, Math.round(n * rateScale)) : n),

    // There is deliberately NO probability helper here. Scaling a per-frame
    // probability by widening the mask in `(random() & MASK) === k` does not
    // work: the widened value is usually not a power-of-two-minus-one, and
    // then the mask changes the DISTRIBUTION rather than the rate. At 60 Hz
    // 0x3f became 0x99, whose bit pattern makes the test fire 1/16 of the
    // time instead of 1/64 — 9.6x too often per second; at 30 Hz it became
    // 0x4c, which has bit 0 clear, so `=== 0x01` could never be true at all
    // and the effect vanished outright.
    //
    // Probabilistic triggers are instead ROLLED ON THE 25 Hz SUB-CLOCK
    // (st.isSub25), so they are tested exactly 25x/second at any rate with
    // the original mask and therefore the original distribution. Durations
    // still scale, so a burst covers the same wall-clock span.
  };
}
