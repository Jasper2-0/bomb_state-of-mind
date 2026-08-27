// rate.js — which frame rate a page runs at, and who is asking.
//
// This is one line of logic and it lives in its own module for two reasons.
//
// IT IS SHARED. The release shell and the experimental web-gpu shell both need
// the identical answer; when each carried its own copy they had already drifted
// once over what ?remaster implied.
//
// IT IS DANGEROUS. The rate default is read by EVERY path, including the
// headless ones -- and tools/inspect/sweep.mjs loads the page as `?inspect=1`
// with no `&rate=`. The reference capture and all 1097 oracle goldens are of
// the 25 Hz show, so a single global default of 60 would silently re-point the
// whole verification chain at a show the references are not of, while every
// gate stayed green: the gates construct their own engines and never read this.
// Sitting in a pure function, it can be asserted exhaustively without a
// browser -- work/verify/shell-rate.mjs does exactly that.
//
// THE RULE:
//   ?rate=<hz>            wins outright, whoever is asking
//   ?original             25 Hz   (?quality=original is an accepted alias --
//                                  the spelling the two published sibling
//                                  ports in this repo already use)
//   headless (?t=/?inspect=1)
//                         25 Hz   -- a measurement gets the show the
//                                    references are of
//   otherwise             60 Hz   -- a viewer gets the remaster, because
//                                    60/25 = 2.4 and the original judders by
//                                    construction on a 60 Hz display
//
// `?remaster` is still accepted and is now a no-op: it names the default.

/** Is this page asking for the original 25 Hz show? */
export const wantsOriginal = (params) =>
  params.has('original') || params.get('quality') === 'original';

/**
 * @param {URLSearchParams} params
 * @param {boolean} headless  ?t= or ?inspect=1 -- a measurement, not a viewer
 * @returns {{ hz: number, remaster: boolean, explicit: boolean }}
 *   `hz` is a request, not a promise: the caller maps it onto its own rate
 *   table and falls back if it is not one of the four supported rates.
 */
export function resolveRate(params, headless) {
  const original = wantsOriginal(params);
  // Remaster is the RE-TIMING, and it is the identity at 25 Hz (retime.js:17),
  // so leaving it on for a headless 25 Hz run changes nothing -- proven frame
  // by frame rather than assumed. It is off under ?original so the intent is
  // legible in the object as well as in the arithmetic.
  const remaster = !original;
  const asked = params.get('rate');
  if (asked !== null && asked !== '') {
    const hz = Number(asked);
    if (Number.isFinite(hz)) return { hz, remaster, explicit: true };
  }
  return { hz: (original || headless) ? 25 : 60, remaster, explicit: false };
}
