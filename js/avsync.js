// avsync.js — where the sound actually IS, relative to where the clock says.
//
// THE DEFECT THIS EXISTS TO REMOVE. The show is driven by the soundtrack:
// main.js reads `audioCtx.currentTime` and renders the frame that belongs to
// it. But `currentTime` is the SCHEDULING clock, not the audible one. Audio
// the graph processes at time T reaches the speakers `outputLatency` later,
// so a renderer that draws T the moment the graph processes T puts the
// picture in front of the sound by exactly that much. Reported on pouet
// (topic 602906, 2026-08-31): "the music seems late to the visuals".
//
// MEASURED, headful Chrome on macOS, default output: outputLatency 32.0 ms,
// baseLatency 5.33 ms. That is small enough to hide behind the display's own
// delay on a wired desktop, which is why it survived review here. On a
// Bluetooth or HDMI path outputLatency is routinely 150-300 ms and nothing in
// the port pushed back at all.
//
// THE CORRECTION, in show seconds added to the raw audio clock:
//
//     offset = -outputLatency + displayPeriod  (+ the viewer's manual trim)
//
// The first term is what we know: the sound is heard that late. The second is
// the opposing term -- the frame drawn in this rAF callback is not on screen
// until the compositor presents it, which pushes the PICTURE late too, and
// correcting only the audio side would over-correct into the opposite error.
//
// HOW MUCH EACH TERM IS TRUSTED IS NOT THE SAME, and the code says so:
//   outputLatency  is READ from the device. Firefox and Chrome implement it;
//                  Safari does not, so baseLatency stands in -- an
//                  underestimate, but a real one.
//   displayPeriod  is MEASURED from rAF timestamps, and ONE period is a
//                  deliberate under-estimate of a pipeline that is usually
//                  one to two vsyncs. It errs toward leaving the music
//                  slightly late, which is the direction human hearing
//                  tolerates: sound arriving after its picture reads as
//                  distance, sound arriving before it reads as broken.
//                  Residual is bounded by one display period (<= 16.7 ms at
//                  60 Hz), well under the ~45 ms detection threshold.
//
// WITH NO AUDIO READING THERE IS NO CORRECTION. If neither latency is
// reported the offset is the manual trim and nothing else -- applying the
// display term alone is not half a correction, it is a new error in the
// wrong direction.
//
// THIS FILE CANNOT AFFECT ANY MEASUREMENT. Every consumer is inside main.js's
// live-playback branch, which `?t=` and `?inspect=1` never enter -- those
// paths construct no AudioContext at all. The oracle goldens, the reference
// sweep and every gate render from an explicit frame index and never see an
// offset. work/verify/av-sync.mjs asserts that structurally.

/** Fallback display period when nothing has been timed yet: one 60 Hz frame. */
export const DEFAULT_FRAME_PERIOD = 1 / 60;

/** Plausible display periods. Outside this, the timestamps are not a refresh. */
export const MIN_FRAME_PERIOD = 1 / 500;   // 500 Hz: no display, so a bad delta
export const MAX_FRAME_PERIOD = 1 / 20;    // 20 Hz: a stall, not a refresh

/**
 * Hard bound on the total offset, seconds. An audio path slower than this is
 * a broken device rather than a show to re-time, and a clamp keeps one absurd
 * reading from throwing the picture into a different part of the demo.
 */
export const MAX_OFFSET_S = 0.5;

/**
 * Adopt a new offset only once it differs by this much. The show clock feeds
 * `advanceTo`, which only ever walks FORWARD, so an offset that jittered
 * every frame would stall the picture on every backward twitch. 2 ms is far
 * below one frame at any supported rate, so adopting a change can never cost
 * more than a single held frame.
 */
export const ADOPT_THRESHOLD_S = 0.002;

/**
 * The audio path's delay, seconds, or 0 when the browser will not say.
 * @param {{outputLatency?: number, baseLatency?: number}} ctx
 */
export function latencySeconds(ctx) {
  // outputLatency is the whole path and is what we want. baseLatency is only
  // the graph's own buffering -- strictly less -- and is the best Safari has.
  for (const v of [ctx?.outputLatency, ctx?.baseLatency]) {
    if (typeof v === 'number' && Number.isFinite(v) && v > 0 && v < MAX_OFFSET_S) return v;
  }
  return 0;
}

/**
 * Seconds to ADD to the raw audio clock before choosing a frame.
 * Negative means the picture is held back to meet the sound.
 *
 * @param {object} r
 * @param {number} [r.outputLatency]  AudioContext.outputLatency, seconds
 * @param {number} [r.baseLatency]    AudioContext.baseLatency, seconds
 * @param {number} [r.framePeriod]    measured display period, seconds
 * @param {number} [r.trimSeconds]    the viewer's manual trim, seconds
 * @returns {number} seconds
 */
export function avOffsetSeconds({
  outputLatency, baseLatency, framePeriod = DEFAULT_FRAME_PERIOD, trimSeconds = 0,
} = {}) {
  const audio = latencySeconds({ outputLatency, baseLatency });
  const display = (Number.isFinite(framePeriod)
    && framePeriod >= MIN_FRAME_PERIOD && framePeriod <= MAX_FRAME_PERIOD)
    ? framePeriod : DEFAULT_FRAME_PERIOD;
  // No audio reading, no automatic correction -- see the header.
  const automatic = audio > 0 ? display - audio : 0;
  const trim = Number.isFinite(trimSeconds) ? trimSeconds : 0;
  return Math.min(MAX_OFFSET_S, Math.max(-MAX_OFFSET_S, automatic + trim));
}

/**
 * `?avoffset=<ms>` is stated the way a viewer experiences it: HOW LATE THE
 * MUSIC SOUNDS, in milliseconds. "the music is 120 ms behind" is
 * `?avoffset=120`, and the correction holds the picture back by 120 ms -- so
 * the number is SUBTRACTED from the show clock. Getting this sign backwards
 * doubles the error instead of removing it, which is why the flip lives here,
 * once, and is asserted rather than repeated at the call site.
 *
 * @param {URLSearchParams} params
 * @returns {number} seconds to add to the show clock
 */
export function parseTrimSeconds(params) {
  const raw = params?.get?.('avoffset');
  if (raw === null || raw === undefined || raw === '') return 0;
  const ms = Number(raw);
  if (!Number.isFinite(ms)) return 0;
  return -ms / 1000;
}

/**
 * The live tracker: feed it rAF timestamps and the AudioContext, read
 * `.offset`. Stateful only in the display-period estimate and the hysteresis;
 * everything it decides comes from `avOffsetSeconds`.
 *
 * @param {{trimSeconds?: number}} [opts]
 */
export function createAvSync({ trimSeconds = 0 } = {}) {
  // A short window of recent deltas, reduced by MEDIAN rather than mean: one
  // dropped frame doubles a delta, and a mean would carry that forever.
  const WINDOW = 30;
  const deltas = [];
  let framePeriod = DEFAULT_FRAME_PERIOD;
  let lastTs = 0;
  let offset = 0;

  return {
    get offset() { return offset; },
    get framePeriod() { return framePeriod; },
    get trimSeconds() { return trimSeconds; },

    /** One rAF timestamp, milliseconds, as the callback receives it. */
    frame(tsMs) {
      if (Number.isFinite(tsMs) && lastTs > 0) {
        const d = (tsMs - lastTs) / 1000;
        if (d >= MIN_FRAME_PERIOD && d <= MAX_FRAME_PERIOD) {
          deltas.push(d);
          if (deltas.length > WINDOW) deltas.shift();
          const sorted = [...deltas].sort((a, b) => a - b);
          framePeriod = sorted[sorted.length >> 1];
        }
      }
      if (Number.isFinite(tsMs)) lastTs = tsMs;
    },

    /**
     * Re-read the device, seconds. Read EVERY frame on purpose: plugging in
     * headphones mid-show changes outputLatency and nothing fires an event we
     * could listen for. The hysteresis is what makes that cheap to do.
     */
    measure(ctx) {
      const want = avOffsetSeconds({
        outputLatency: ctx?.outputLatency,
        baseLatency: ctx?.baseLatency,
        framePeriod,
        trimSeconds,
      });
      if (Math.abs(want - offset) >= ADOPT_THRESHOLD_S) offset = want;
      return offset;
    },
  };
}
