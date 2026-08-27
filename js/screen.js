// state-of-mind — the show's framebuffer geometry, in one place.
//
// The demo renders 320x200 RGB565. That is the verified artifact and the
// default: every golden frame, every gate and the oracle itself are 320x200,
// so changing these values changes what "correct" means and the gates will
// say so immediately.
//
// This module exists because the resolution was previously written down in
// two independent places — `fidelity.js` and `raster3d.js` each declared
// their own `W`/`H` — with several more sites baking the row stride as a
// literal 320. Any future work on rendering at the presenter's native
// 1600x1200 has to start from a single definition, and having one is worth
// something on its own: two constants that must agree but are declared apart
// are a defect waiting to happen.
//
// See work/re/RESOLUTION.md for what would and would not benefit from a
// higher render resolution — the short version is that ~45% of the show is
// resolution-independent, ~30% is procedural but 30x too expensive on the
// CPU, and ~20% is bitmap-sourced and cannot benefit at all.

export const W = 320, H = 200;

// Pixels per frame — the figure that scales every full-screen effect.
export const PIXELS = W * H;

// The source pixel is NOT square: 320x200 was displayed at 4:3, giving a 5:6
// pixel aspect. present.js relies on this to pick 1600x1200 as the backbuffer
// (320*5, 200*6), where each source pixel becomes an exact 5x6 block of
// square display pixels.
export const PIXEL_ASPECT_X = 5, PIXEL_ASPECT_Y = 6;

// Native render size. 320*5 x 200*6 — the presenter's default backbuffer, and
// the only size at which a source pixel maps to a whole number of square
// display pixels. Parts that can render resolution-independently draw here
// instead of into the 320x200 framebuffer; everything else is block-replicated
// up. See work/re/RESOLUTION.md for which parts are which.
export const NW = W * PIXEL_ASPECT_X, NH = H * PIXEL_ASPECT_Y;   // 1600x1200
