// state-of-mind — fidelity engine.
//
// A software reimplementation of the demo's 320x200 RGB565 pipeline,
// re-derived from the v1.3 source release (references are file:line into
// mind_src_v13). This is the verification target: it replays the show on
// the same 40 ms synthetic clock as the native oracle and is scored against
// the oracle's golden frames. The WebGL presentation layer sits on top and
// inherits correctness from matching this engine.
//
// Determinism contract: all state lives on the engine instance; render is
// replay-from-zero (see createFidelity().advanceTo). No Math.random, no
// Date — the only PRNGs are the demo's own (janim.c LCG + BSD random()).

import { uncompressDMask, parseDMask, parseGif } from './dat.js';
import { parse3ds, finishMeshes } from './world3ds.js';
import { finishWorld, updateWorld, cameraMatrix, multMatrixABC } from './scene3d.js';
import { loadTxtCacheIV, loadTxtCacheIII } from './cache3d.js';
import { parseVista, createVistaRenderer } from './raster3d.js';
import { setBck } from './bilin.js';
import { createBlob, mcBlobify2, setMCDimensions, mcFinishBlob, doMCube3 } from './mcube.js';
import { loadTxtCacheIIbis } from './cache2.js';
import { buildBasePlasma, satTable3, doPlasma256, doDaFunkyCaustics, doBilinear4, doRayTunnel } from './caustics.js';
import { quatToMatrix, idMatrix2, aV } from './world3ds.js';
import { decodeJpeg } from './jpeg.js';

// Framebuffer geometry lives in screen.js. Re-exported here because the
// gates and the browser shell import W/H from the engine.
export { W, H } from './screen.js';
export const STEP_MS = 40;

// The show's timestep is a RATIONAL number of milliseconds:
// Timer = floor(frame*num/den). 25 Hz is 40/1, but the rates that divide a
// 60 Hz display evenly are not whole milliseconds. Mirrors the oracle's
// SOM_STEP_NUM/SOM_STEP_DEN (demo_src/timers.c My_getitimer).
export const RATE_25 = { num: 40, den: 1, hz: 25 };
export const RATE_30 = { num: 100, den: 3, hz: 30 };
export const RATE_50 = { num: 20, den: 1, hz: 50 };
export const RATE_60 = { num: 50, den: 3, hz: 60 };

// Largest frame whose timer is <= ms — the exact inverse of
// floor(frame*num/den).
//
// NOT floor(ms*den/num): that double-truncates and lands a frame early
// whenever the timer itself was truncated. At 60 Hz frame 5971 has timer
// floor(5971*50/3) = 99516, and floor(99516*3/50) = 5970. Since
// floor(f*num/den) <= ms  <=>  f*num/den < ms+1  for integer ms, the inverse
// is ceil((ms+1)*den/num) - 1, done in integers to avoid a float boundary.
export function frameForMs(ms, rate = RATE_25) {
  if (ms < 0) return 0;
  const n = (Math.floor(ms) + 1) * rate.den;
  return Math.floor((n + rate.num - 1) / rate.num) - 1;
}

import { fmaf, frcp } from './fp.js';
import { createRetimer } from './retime.js';
import { NULL_CAPTURE } from './capture.js';
import { captureState, applyState } from './snapshot.js';
import { W, H, NW, NH } from './screen.js';

const fr = Math.fround;

// ---- color tables (dem/fx1/fx1.c Init_Mixing) ----------------------------

const MIX_LEVELS = 16;

function buildTables() {
  // RGB71_To_16: grayscale ramp 0..255 over 127 steps then saturated
  // (Init_CMap_Ramp_Pos(0,0,0,255,255,255,127,129)), split into the three
  // 565 field positions (CMap_Split — note: entry 255 stays 0; saturating
  // 7-bit adds can only reach 254).
  const ramp = new Uint8Array(256);
  for (let i = 0; i < 127; i++) ramp[i] = (255 * i / 126) | 0;
  ramp.fill(255, 127);
  const rgb71 = [new Uint16Array(256), new Uint16Array(256), new Uint16Array(256)];
  for (let i = 0; i < 255; i++) {
    rgb71[0][i] = (ramp[i] >> 3) & 0x001f;
    rgb71[1][i] = (ramp[i] << 3) & 0x07e0;
    rgb71[2][i] = (ramp[i] << 8) & 0xf800;
  }

  // Lo16/Hi16_To_777[level][byte]: expand one byte of a 565 pixel into its
  // 777 contribution, faded by level (level 0 = full, 15 = black).
  // 777 layout: R at bits 16-22, G at 8-14, B at 0-6 (truecol.h RGB_TO_777).
  const lo = [], hi = [];
  for (let l = 0; l < MIX_LEVELS; l++) {
    const x = (MIX_LEVELS - 1 - l) / (MIX_LEVELS - 1);
    const lot = new Uint32Array(256), hit = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      const b5 = i & 0x1f;          // low byte: B field
      const gLo = i & 0xe0;         //           G low bits
      const gHi = i & 0x07;         // high byte: G high bits
      const r5 = i & 0xf8;          //            R field
      lot[i] = Math.floor(b5 * 4.0 * x) | (Math.floor((gLo >> 5) * 2.0 * x) << 8);
      hit[i] = (Math.floor((gHi << 3) * 2.0 * x) << 8) | (Math.floor((r5 >> 3) * 4.0 * x) << 16);
    }
    // Tag with the fade level. The mixers take the TABLE, not the index, so
    // a caller that wants to know WHICH row a mixer meant -- the capture, for
    // one -- has no other way to recover it. Two non-enumerable properties per
    // level, defined once at module load.
    Object.defineProperty(lot, '_level', { value: l, enumerable: false });
    Object.defineProperty(hit, '_level', { value: l, enumerable: false });
    lo.push(lot); hi.push(hit);
  }
  // part2.c's per-channel saturating remap, split into the three 777 field
  // positions the same way rgb71 is. The particle accumulator applies this
  // after every sprite add; on the GPU it applies once, to the folded sum.
  const sat = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  for (let i = 0; i < 256; i++) {
    const j = i >= 128 ? 255 : 2 * i;
    sat[0][i] = ((j >> 1) & 0x0000007f) >>> 0;     // B, bits 0-6
    sat[1][i] = ((j << 7) & 0x00007f00) >>> 0;     // G, bits 8-14
    sat[2][i] = ((j << 15) & 0x007f0000) >>> 0;    // R, bits 16-22
  }
  return { rgb71, lo, hi, sat };
}

const T = buildTables();
const to16 = (s777) =>
  T.rgb71[0][s777 & 0xff] + T.rgb71[1][(s777 >> 8) & 0xff] + T.rgb71[2][(s777 >> 16) & 0xff];
const rgbTo777 = (r, g, b) => (((r & 0xfe) << 15) | ((g & 0xfe) << 7) | ((b & 0xfe) >> 1)) >>> 0;
const rgbTo565 = (r, g, b) => (((r & 0xf8) << 8) | ((g & 0xfc) << 3) | ((b & 0xf8) >> 3));
const bound = (x) => (x < 0 ? 0 : x > 255 ? 255 : x);

// ---- PRNGs ---------------------------------------------------------------

// dem/fx1/janim.c: RSeed LCG behind TRandom(slot), quantized to 30 ms.
function makeTRandom() {
  let rseed = 1257;
  const lastTimer = [-1, -1, -1, -1];
  const lastRand = [0, 0, 0, 0];
  const rnd = () => (((rseed = (rseed * 205 + 29573) | 0) >> 5) & 0xffff);
  return {
    at(slot, timer) {
      if (timer - lastTimer[slot] >= 30) {
        lastRand[slot] = rnd();
        lastTimer[slot] = timer;
      }
      return lastRand[slot];
    },
    // janim.c Random(): the same LCG, called directly by the parts that
    // want a fresh draw rather than a time-quantized slot.
    raw: rnd,
    // For snapshot.js: this state is captured variables, not fields, so it
    // cannot be reached by walking the object.
    save: () => ({ rseed, lastTimer: [...lastTimer], lastRand: [...lastRand] }),
    load: (v) => {
      rseed = v.rseed;
      for (let i = 0; i < 4; i++) { lastTimer[i] = v.lastTimer[i]; lastRand[i] = v.lastRand[i]; }
    },
  };
}

// BSD random(): the classic 31-entry additive trinomial (x[i] = x[i-3] +
// x[i-31], output >> 1), seeded by srandom() via a Park-Miller LCG and 310
// discarded outputs — the sequence macOS's libc produced inside the oracle,
// and what old glibc produced on the 1998 Linux build. Seeded 126 by
// Init_Hash1(10,126) (janim.c:341), which then consumes 1000 outputs.
export function makeBsdRandom(seed) {
  const r = new Int32Array(34);
  let s = seed;
  r[0] = s;
  for (let i = 1; i < 31; i++) {
    // 16807 * s mod 2^31-1 via Schrage, signed like the libc source
    const hi = Math.floor(s / 127773);
    const lomul = 16807 * (s % 127773) - 2836 * hi;
    s = lomul < 0 ? lomul + 2147483647 : lomul;
    r[i] = s;
  }
  let f = 3, rr = 0;
  const next = () => {
    r[f] = (r[f] + r[rr]) | 0;
    const out = (r[f] >>> 1) & 0x7fffffff;
    f = (f + 1) % 31;
    rr = (rr + 1) % 31;
    return out;
  };
  for (let i = 0; i < 310; i++) next();
  // Snapshot hooks hang off the function object so every call site --
  // `st.random()` -- is untouched. See snapshot.js CLOSURE_KEYS.
  next.save = () => ({ r: Int32Array.from(r), f, rr });
  next.load = (v) => { r.set(v.r); f = v.f; rr = v.rr; };
  return next;
}

// ---- VBuffer ids (dem/include/fx.h) --------------------------------------
const VSCREEN = 0, VB1 = 1, VMASK = 5, SCRATCH = 6, JANIM = 7, VBLUR_I = 8, VBLUR_II = 9;

// ---- displacement tables (dem/src/dsp.c Pre_Cmp_Dsp) ---------------------
function buildDspRadial() {
  const r1 = new Uint8Array(2 * W * H);
  const r2 = new Uint8Array(2 * W * H);
  let dx = 0, dy = W * H, dx2 = 0, dy2 = W * H;
  for (let j = 0; j < H; j++) {
    const y = Math.fround(2 * j / H - 1);
    for (let i = 0; i < W; i++) {
      const x = Math.fround(2 * i / W - 1);
      let r = Math.fround(x * x * y * y + 0.01);
      let rr = 0.5 * (1 + Math.cos(r * Math.PI));
      rr = rr > 0 ? 127 * Math.pow(rr, 0.25) : 0;
      r1[dx++] = Math.floor(127 - x * rr) & 0xff;
      r1[dy++] = Math.floor(127 - y * rr) & 0xff;

      rr = 0.5 * (1 + Math.cos(r * Math.PI * 2));
      r = r >= 1 ? 0 : Math.pow(1 - r, 0.25);
      r2[dx2++] = Math.floor(127 + y * 32 * rr - 96 * r * x) & 0xff;
      r2[dy2++] = Math.floor(127 - x * 32 * rr - 96 * r * y) & 0xff;
    }
  }
  return { r1, r2 };
}

// demo_src/clamp.c Init_Saturation_Table4/5.
function satTable4(tab, c0, c1, w, amp) {
  tab[0] = w * c0;
  for (let i = 1; i < amp; i++) {
    const x = i / amp;
    tab[i] = w * Math.floor(c0 + (x * (c1 - c0) + 0.5));
  }
}
function satTable5(tab, c0, c1, w, off1, amp, off2) {
  let j = 0;
  for (let i = 0; i <= off1; i++) tab[j++] = w * c0;
  for (let i = 1; i < amp; i++) tab[j++] = w * Math.floor(c0 + ((i / amp) * (c1 - c0) + 0.5));
  for (let i = 0; i < off2; i++) tab[j++] = w * c1;
}

// ---- palette helpers -----------------------------------------------------

// demo_src/truecol.c CMap_To_16bits (Format 0x777): raw palette -> 777.
function cmapTo777(pal, nbCol) {
  const out = new Uint32Array(256);
  const n = Math.min(nbCol, 256);
  for (let i = 0; i < n; i++) out[i] = rgbTo777(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]);
  return out;
}

// truecol.c CMap_To_16bits_With_Fx Type 0, output 565 into a u16 CMap
// (Finish_Format 0x2565) — the sky palette faded toward black.
function cmapTo565Fx(cmap, pal, nbCol, mixCol, level) {
  const ro = ((mixCol >> 16) & 0xff) * (255 - level);
  const go = ((mixCol >> 8) & 0xff) * (255 - level);
  const bo = (mixCol & 0xff) * (255 - level);
  const n = Math.min(nbCol, 256);
  for (let i = 0; i < n; i++) {
    const r = bound((pal[3 * i] * level + ro) >> 8);
    const g = bound((pal[3 * i + 1] * level + go) >> 8);
    const b = bound((pal[3 * i + 2] * level + bo) >> 8);
    cmap[i] = rgbTo565(r, g, b);
  }
}

// truecol.c CMap_To_16bits_With_Fx Type 0 (fade toward mixCol), output 777.
function cmapTo777Fx(pal, nbCol, mixCol, level) {
  const ro = ((mixCol >> 16) & 0xff) * (255 - level);
  const go = ((mixCol >> 8) & 0xff) * (255 - level);
  const bo = (mixCol & 0xff) * (255 - level);
  const out = new Uint32Array(256);
  const n = Math.min(nbCol, 256);
  for (let i = 0; i < n; i++) {
    const r = bound((pal[3 * i] * level + ro) >> 8);
    const g = bound((pal[3 * i + 1] * level + go) >> 8);
    const b = bound((pal[3 * i + 2] * level + bo) >> 8);
    out[i] = rgbTo777(r, g, b);
  }
  return out;
}

// demo_src/dmask.c DMask_Setup_Pal: 256-entry mask CMap; each entry packs a
// premultiplied 777 color in the upper 24 bits and a 0-15 blend level in
// the low byte (consumed by Mix_8_8_Blend_Mask_II).
function dmaskSetupPal(cmap32, mask, pal777) {
  const bits = 8 - mask.mBits;
  const scale = 1 << mask.mBits;
  const cmask = (1 << bits) - 1;
  for (let i = 0; i < 256; i++) {
    const f = 255 - Math.floor(((i >> bits) * 255) / scale);
    const rgb = pal777[i & cmask];
    const r = Math.floor((((rgb >> 16) & 0xff) * f) / 256);
    const g = Math.floor((((rgb >> 8) & 0xff) * f) / 256);
    const b = Math.floor(((rgb & 0xff) * f) / 256);
    cmap32[i] = (((r << 24) | (g << 16) | (b << 8) | (f >> 4)) >>> 0);
  }
}

// demo_src/truecol.c Drv_Build_Ramp_16: 8.8 fixed-point ramp with
// truncating step division; carries Next_Start / last-color statics.
function makeRampBuilder() {
  let nextStart = 0, lR = 0, lG = 0, lB = 0;
  // build into either a Uint16Array view (0x2565) or Uint32Array (0x777)
  function ramp(cmap, start, n, r1, g1, b1, r2, g2, b2, format) {
    if (start < 0) start = nextStart;
    if (start + n > 256) n = 256 - start;
    nextStart = start + n;
    if (r1 === -1) { r1 = lR; g1 = lG; b1 = lB; }
    else { r1 <<= 8; g1 <<= 8; b1 <<= 8; }
    lR = (r2 <<= 8); lG = (g2 <<= 8); lB = (b2 <<= 8);
    const N = n === 1 ? 1 : n - 1;
    const dr = Math.trunc((r2 - r1) / N);
    const dg = Math.trunc((g2 - g1) / N);
    const db = Math.trunc((b2 - b1) / N);
    for (let i = 0; i < n; i++) {
      const r = bound(r1 >> 8), g = bound(g1 >> 8), b = bound(b1 >> 8);
      if ((format & 0xffff) === 0x2565) cmap.u16[start + i] = rgbTo565(r, g, b);
      else cmap.u32[start + i] = rgbTo777(r, g, b);
      r1 += dr; g1 += dg; b1 += db;
    }
  }
  // Snapshot hooks -- see snapshot.js CLOSURE_KEYS. `nextStart` in particular
  // is carried between calls (a `start` of -1 means "continue from the last
  // ramp"), so a restore that dropped it would build the next palette at the
  // wrong offset.
  ramp.save = () => ({ nextStart, lR, lG, lB });
  ramp.load = (v) => { nextStart = v.nextStart; lR = v.lR; lG = v.lG; lB = v.lB; };
  return ramp;
}

// ---- the engine ----------------------------------------------------------

export function createFidelity(assets) {
  // WELD. The original's Optimize_Mesh runs in the shipped binary, so this is
  // ON and the restored show must keep it on. It exists as a switch only so
  // the weld's effect on the PICTURE can be measured, the way the C's own
  // _SKIP_OPTIM_MESH_ (include/main3d.h:26) would have. Off is not a fix and
  // is not offered to a viewer -- see work/re/NOTES.md leg 30.
  const weldMeshes = assets.weld !== false;
  // assets: { dat: Map(name -> Uint8Array),
  //           decoded: Map(name -> {w,h,fmt,bytes}) }  (engine-decoded JPEGs)
  //          plus optional { rate, remaster }.
  //
  // rate     — timestep as {num,den} ms (default RATE_25 = 40/1). Anything
  //            other than 25 Hz re-times the show; see work/re/FRAMERATE.md.
  // remaster — apply the frame-rate corrections for the nine frame-counted
  //            effects. OFF by default and inert at 25 Hz, so the verified
  //            path is untouched either way.
  const rate = assets.rate ?? RATE_25;
  const remaster = !!assets.remaster;
  // Native-resolution layer. Off by default and allocated only when asked
  // for: the 320x200 framebuffer stays the verified artifact, and a part that
  // has no native path simply leaves nativeValid false so the caller presents
  // the 320x200 frame as before. work/re/RESOLUTION.md lists which parts can.
  const native = !!assets.native;
  // Native 3D is a SEPARATE opt-in because it does not fit a 60 Hz budget.
  // Measured on the greets part: 1.17 ms at 320x200 becomes 32.26 ms once the
  // geometry rasterises at 1600x1200 -- fine against the 40 ms of 25 Hz,
  // roughly double the 16.67 ms of 60 Hz. So native resolution and 60 Hz are
  // mutually exclusive for the 3D parts on the CPU; the cheap native parts
  // (vector line art) fit either way. work/re/RESOLUTION.md 9 has the numbers.
  const native3d = native && !!assets.native3d;
  // The capture boundary. Defaults to a null object that does nothing, which
  // is what the released CPU version runs with; the experimental WebGL
  // renderer injects a real one (see web/js/capture.js for the interface and
  // web-gpu/js/capture-gpu.js for the implementation). Everything it does is
  // record-only -- no pixel the CPU produces may depend on it, and an
  // inertness check gates that.
  const cap = assets.capture ?? NULL_CAPTURE;
  // Read once into a closure const so the per-sprite call sites cost a branch
  // on a local: the galaxy alone reaches them 900 times a frame.
  const capturing = !!cap.records;
  // Every renderer goes through here so a part added later cannot silently
  // miss the capture -- the failure mode would be a part that renders on the
  // CPU, uploads a full frame every frame, and looks correct while doing
  // none of the work on the GPU.
  //
  // The sink is attached only when one exists. It used to be attached
  // unconditionally while its per-frame reset ran only when recording was on,
  // so a plain CPU run captured every triangle of the whole show into an array
  // that doubled forever and never reset -- unbounded memory growth and
  // per-triangle cost on the verified path.
  function newRenderer(world, camNode) {
    const r = createVistaRenderer(world, camNode);
    if (cap.geoSink) r.setGeoSink(cap.geoSink);
    return r;
  }
  const nativeScreen = native ? new Uint16Array(NW * NH) : null;
  const nativeLines = native ? new Uint16Array(NW * NH) : null;
  // Pen width for native line art, in native pixels across the stroke.
  // Chosen so the stroke carries the same total ink as the 320x200 one --
  // see the sweep in work/re/RESOLUTION.md 9. Overridable for tuning.
  const NATIVE_PEN = assets.nativePen ?? 8;
  // Native z-buffer for the 3D parts. Int16 like the 320x200 one (ZB_Base).
  const nativeZBuf = native ? new Int16Array(NW * NH) : null;
  const NSX = NW / W, NSY = NH / H;

  // Retarget a renderer at the native framebuffer for the duration of `fn`,
  // then restore every field by assignment. Restoring through setupCamScreen
  // would recompute outClips from the CURRENT lx rather than the one the
  // 320x200 path had -- upd() assigns rc.lx directly and never refreshes
  // outClips -- which silently leaks into the next frame.
  //
  // `fn` receives a clip helper that scales screen-space arguments, so a
  // caller writes the same setupClipping call it uses at 320x200.
  function nativePass(R, fn) {
    const lx = R.rc.lx, ly = R.rc.ly, cx = R.rc.cx, cy = R.rc.cy;
    const savedClips = R.rc.clips.slice(), savedOut = R.rc.outClips.slice();
    R.setupTarget(NW, NH);
    R.rc.lx = fr(lx * NSX); R.rc.ly = fr(ly * NSY);
    R.rc.cx = fr(cx * NSX); R.rc.cy = fr(cy * NSY);
    R.rc.outClips[0] = fr(savedOut[0] * NSX); R.rc.outClips[1] = fr(savedOut[1] * NSX);
    R.rc.outClips[2] = fr(savedOut[2] * NSY); R.rc.outClips[3] = fr(savedOut[3] * NSY);
    R.rc.outClips[4] = fr(savedOut[4] * NSX); R.rc.outClips[5] = savedOut[5];
    const clip = (x0, x1, y0, y1, zmin, zmax) =>
      R.setupClipping(x0 * NSX, x1 * NSX, y0 * NSY, y1 * NSY, zmin, zmax);
    try {
      fn(clip);
    } finally {
      R.setZBuffer(null);
      R.setupTarget(W, H);
      R.rc.lx = lx; R.rc.ly = ly; R.rc.cx = cx; R.rc.cy = cy;
      R.rc.clips.set(savedClips); R.rc.outClips.set(savedOut);
    }
    st.nativeValid = true;
  }

  // mixSat16_8 with the 8-bit overlay block-replicated up. The film-scratch
  // plates are 320x201 GIFs and the .msk masks 320x200, so neither has extra
  // detail -- only what they sit on top of does.
  function mixSatNative(dst, in8) {
    const s2 = in8.bits, cm = in8.cmap.u32, off = in8.off || 0;
    for (let y = 0; y < NH; y++) {
      const srcRow = ((y / NSY) | 0) * W, dstRow = y * NW;
      for (let x = 0; x < NW; x++) {
        const p = dst[dstRow + x];
        dst[dstRow + x] = to16(T.lo[0][p & 0xff] + T.hi[0][p >> 8]
          + cm[s2[off + srcRow + ((x / NSX) | 0)]]);
      }
    }
  }

  // mix16_8BlendMaskII with the 8-bit mask block-replicated up. The .msk
  // assets are 320x200 (all 23 of them), so the mask has no extra detail to
  // give -- only the 3D underneath it does.
  function blendMaskNative(dst, mask) {
    const m = mask.bits, cm = mask.cmap.u32;
    for (let y = 0; y < NH; y++) {
      const srcRow = ((y / NSY) | 0) * W, dstRow = y * NW;
      for (let x = 0; x < NW; x++) {
        const e = cm[m[srcRow + ((x / NSX) | 0)]];
        const b = e & 0xff, c = dst[dstRow + x];
        dst[dstRow + x] = to16((e >>> 8) + T.lo[b][c & 0xff] + T.hi[b][c >> 8]);
      }
    }
  }
  // Frames of THIS rate per original 25 Hz frame. 1 at 25 Hz, 2.4 at 60 Hz.
  // Frame-rate corrections live in retime.js so they can be tested directly
  // (the remaster-timing gate does exactly that). All identities at 25 Hz.
  const RT = createRetimer(rate, remaster);
  const rateScale = RT.rateScale, retimed = RT.retimed;
  const decayConst = RT.decayConst, durFrames = RT.durFrames;
  const decayLevels = (i, j) => RT.decayLevels(i, j);

  // --- immutable data, built once
  // Dsp_Radial_1/2 are rebuilt from the source formulas (dsp.c
  // Pre_Cmp_Dsp) — verified byte-identical to the oracle's tables
  // (2026-08-24), so no extracted copy is needed. fidelity-check's golden
  // comparison guards this equivalence.
  const dsp = buildDspRadial();
  const masks = {
    advisory: parseDMask(assets.dat.get('layer022m.msk')),
    welcome: parseDMask(assets.dat.get('welcome.msk')),
    techno: parseDMask(assets.dat.get('techno.msk')),
    join: parseDMask(assets.dat.get('join.msk')),
    modern: parseDMask(assets.dat.get('modern.msk')),
    improve: parseDMask(assets.dat.get('layer112m.msk')),
    kiss: parseDMask(assets.dat.get('layer132m.msk')),
    upgrade: parseDMask(assets.dat.get('upgrade.msk')),
    tomorrow: parseDMask(assets.dat.get('tomorrow.msk')),   // Head_For
    work: parseDMask(assets.dat.get('work.msk')),
    buy: parseDMask(assets.dat.get('buy.msk')),
    consume: parseDMask(assets.dat.get('cons.msk')),
    die: parseDMask(assets.dat.get('die.msk')),
    sponsor: parseDMask(assets.dat.get('layer012m.msk')),   // Mask012
    blob: parseDMask(assets.dat.get('layer122m.msk')),      // Mask122
    noway: parseDMask(assets.dat.get('layer162m.msk')),     // Mask162
    logo: parseDMask(assets.dat.get('layer212m.msk')),      // Mask212
    rest: parseDMask(assets.dat.get('rest.msk')),           // MRest
    end: parseDMask(assets.dat.get('layer222m.msk')),       // Mask222
    gr1: parseDMask(assets.dat.get('gr1.msk')),
    gr2: parseDMask(assets.dat.get('gr2.msk')),
    gr3: parseDMask(assets.dat.get('gr3.msk')),
    resist: parseDMask(assets.dat.get('resist.msk')),
  };
  const sky = parseGif(assets.dat.get('sky4.gif'));         // part2.c Sky

  // JPEG textures come from the pack, decoded by the engine's own decoder
  // (web/js/jpeg.js) — NOT from oracle exports. The format argument mirrors
  // each Load_Raw_JPEG call site in the C; work/verify/decoder-parity.mjs
  // holds this decoder byte-exact against the oracle's.
  const jpegCache = new Map();
  function jpg(name, format) {
    let v = jpegCache.get(name);
    if (v === undefined) {
      v = decodeJpeg(assets.dat.get(name), format);
      if (!v) throw new Error(`JPEG decode failed: ${name}`);
      jpegCache.set(name, v);
    }
    return v;
  }
  const scratchGifs = ['old4.gif', 'old5.gif', 'old6.gif'].map((n) => parseGif(assets.dat.get(n)));

  // ---- fonts (dem/src/fonts.c) -----------------------------------------
  // Glyph metrics come from scanning the sheet: a glyph is a run of columns
  // containing any nonzero pixel (Find_Text/Font_Init).
  function fontInit(name, avrg) {
    const gif = parseGif(assets.dat.get(name));
    const findText = (k) => {
      while (k < gif.w) {
        let hit = false;
        for (let j = 0; j < gif.h; j++) if (gif.bits[j * gif.w + k]) { hit = true; break; }
        if (hit) break;
        k++;
      }
      return k;
    };
    const fPos = [], fWidth = [];
    let k = 0, i = 0;
    for (;;) {
      k = findText(k);
      fPos[i] = k;
      if (k === gif.w) break;
      let l = k;
      do { l++; } while (findText(l) === l);
      fWidth[i] = l - k;
      k = l;
      i++;
    }
    const nbChars = i;
    // Char_To_Font_I (>40 glyphs: separate a-z / A-Z) vs II (shared case)
    const charMap = new Uint8Array(128).fill(255);
    for (let c = 0; c < 128; c++) {
      const ch = String.fromCharCode(c);
      if (ch >= 'a' && ch <= 'z') charMap[c] = c - 97;
      else if (ch >= 'A' && ch <= 'Z') charMap[c] = nbChars <= 40 ? c - 65 : c - 65 + 26;
      else if (ch === "'") charMap[c] = nbChars <= 40 ? 26 : 52;
      else if (ch === ',') charMap[c] = nbChars <= 40 ? 27 : 53;
      else if (ch === '.') charMap[c] = nbChars <= 40 ? 28 : 54;
    }
    return { gif, avrg, fPos, fWidth, nbChars, charMap };
  }
  const fonts = [
    fontInit('font32.gif', 11), fontInit('font42.gif', 11),
    fontInit('font632.gif', 10), fontInit('font62.gif', 10),
    fontInit('font82.gif', 11), fontInit('font94.gif', 90),
  ];
  // Lyric-event table: derived from the source release at unpack time
  // (work/unpack.mjs -> unpacked/ffx.json); float fields are f32 in the
  // engine, so fround on use.
  const ffxEvents = assets.ffx ?? [];
  // Sprite_Data[16][8*8] (dem/fx1/sprs2.h) -> unpacked/sprites.json
  const sprites = Uint8Array.from(assets.sprites ?? []);
  const anims = {
    anim12: jpg('anim12.jpg', 0x0000),
    anim72: jpg('anim72.jpg', 0x0000),
    anim82: jpg('anim82.jpg', 0x0000),
    anim93: jpg('anim93.jpg', 0x0000),
    anim32: jpg('anim32.jpg', 0x0000),
  };

  // --- mutable engine state (reset() rebuilds everything below)
  let st = null;

  function newBuffers() {
    const mk16 = () => ({ bits: new Uint16Array(W * H), cmap: null });
    const mk8 = () => ({ bits: new Uint8Array(W * H), cmap: null });
    const mkCMap = () => {
      const buf = new ArrayBuffer(256 * 4);
      return { u16: new Uint16Array(buf), u32: new Uint32Array(buf) };
    };
    return {
      screen: mk16(),
      vb1: mk16(),
      vblurI: mk16(),
      vblurII: mk16(),
      janim: { bits: new Uint8Array(W * H), cmap: mkCMap() },
      vmask: { bits: null, cmap: mkCMap() },
      scratch: { bits: null, off: 0, cmap: mkCMap() },
      vletters: { bits: new Uint8Array(W * H), cmap: mkCMap() },
      // VB(3): the 8-bit sky/background layer (world3, particles, caustics)
      vb3: { bits: new Uint8Array(W * H), cmap: mkCMap() },
      // VLETTERS_II: 565 snapshot of VSCREEN taken once by Init_Fx4
      vlettersII: mk16(),
      // VB(1) used as the 8-bit blend plane by the appear/disappear wipes
      vb1b: { bits: new Uint8Array(W * H), cmap: mkCMap() },
      vb2: { bits: new Uint8Array(W * H), cmap: mkCMap() },
      // VB(2) retyped 565 by Init_10 for the lines part
      vb2b: { bits: new Uint16Array(W * H), cmap: mkCMap() },
      // VB(4) is a virtual window into VSCREEN; only its palette is real
      vb4cmap: mkCMap(),
    };
  }

  // The command recorder that used to live here -- nameBuffers, sameBits,
  // changeCoveredByGlyphs, emitDrawing, rec, recOut, recSprites, recEndFrame
  // -- moved to web-gpu/js/capture-gpu.js. The engine now calls the injected
  // `cap` object instead, which is a null object unless the GPU renderer
  // supplies one. See web/js/capture.js for the interface.

  function reset() {
    st = {
      frame: 0,
      timer: 0,
      phase: 0,
      tickIn: 0, tickOut: 0, tdx: 0, tickX: 0,
      startingTick: 0, gdx: 0, globalX: 0,
      // beat statics (janim.c)
      timer0: 0, nextBeat: 0,
      animCnt: 0, animMaxCnt: 0, aCnt: 0, jaParity: 0,
      curAnim: null, curAnimW: 160, curAnimH: 100, curAnimSize: 0,
      curMask: null,
      // scratch statics (scratch.c)
      cOff: 0, adv0: 0, scrOff: 0,
      dspRadial: null, dspParity: 0,
      // FLT globals Time_Scale/Time_Off (dem/src/panim.c)
      timeScale: 1, timeOff: 0,
      t6: null,                            // tunnel context (Pre_Cmp_6)
      t5: null,                            // statue context (Pre_Cmp_World3)
      t4: null,                            // city context (Pre_Cmp_World2)
      w2Target: 'screen',                  // Setup_Rendering target for world2
      ptclM: null,                         // Ptcl_M: camera matrix for the sky
      dspSatX: new Int32Array(W + 512),
      dspSatY: new Int32Array(H + 512),
      trandom: makeTRandom(),
      random: null,
      buffers: newBuffers(),
      ramp: makeRampBuilder(),
      unported: false,
      // lyric system (dem/src/fonts2.c statics)
      curText: 0, curWord: 0, fxNb: 0, sLen: 0,
      fx4Snapshot: false,       // Init_Fx4's function-static "Ok"
      scroll: null, uScroll: 0, scrollSrc: null,
      blob: null, tBlob: null, blbCount: 0,
      lines: null,
      caustics: null, causticsM: idMatrix2(),
      greets: null, readyCnt: 0,
      ptcl: null, ptclM: idMatrix2(), ptclTrf: null, ptclClips: [0, W, 0, H],
      vb1_777: null, ufo: null, ufoX: 0, ufoY: 0, ptclZero: 0,
      globalDx: 0,
      isSub25: true, sub25: -1,
      nativeValid: false,
      hash: null, hashSeed: 0, orTab: null, dsPhase0: 0,
      vb1Size: 0,               // VB(1).Size, read by Init_Appear(2)
      vb1Own: new Uint8Array(W * H),   // VB(1)'s real 8-bit backing store
      words: [], wordsL: [], wordsS: [],
      bypassLoop: 'Bypass_I',   // installed by Init_Letters at load
      postLoop: null,
      clips: [0, W, 0, H],      // _RCst_.Clips[0..3] (text path only)
    };
    // The capture stamps stable layer names on these; a null capture does
    // nothing, and the CPU path never reads them.
    cap.attach(st.buffers);
    // load-time seeding: Init_Hash1(10, 126) srandom()s the BSD generator
    // and consumes exactly 1000 draws building Hash1/Rand_Tab (janim.c:341,
    // demo_src/rand.c). Those same draws set the stream position every later
    // random() consumer depends on — so build the tables HERE rather than
    // consuming the draws twice.
    st.random = makeBsdRandom(126);
    st.hash = initHash1(10, 126, st.random);
    st.hashSeed = st.hash.hashSeed;
    // USE_FLARE assets (Pre_Cmp_6): flare.gif + the Flare_Sprite[] table.
    // Shared by the tunnel and the greets worlds, so it is built once here.
    const flare = {
      bits: null, bps: 0, convert: null,
      sprites: [
        { uo: 192, vo: 0, dU: 64, dV: 64, scale: fr(0.9), alpha: fr(-0.57) },
        { uo: 0, vo: 0, dU: 64, dV: 64, scale: fr(3.2), alpha: 0 },
        { uo: 192, vo: 0, dU: 64, dV: 64, scale: fr(1.9), alpha: fr(0.43) },
        { uo: 0, vo: 64, dU: 64, dV: 64, scale: fr(0.9), alpha: fr(0.76) },
        { uo: 64, vo: 64, dU: 64, dV: 64, scale: fr(0.6), alpha: fr(0.9) },
        { uo: 128, vo: 0, dU: 64, dV: 64, scale: fr(1.3), alpha: fr(1.75) },
        { uo: 64, vo: 0, dU: 64, dV: 64, scale: fr(2.1), alpha: fr(2.2) },
      ],
    };
    {
      const g = parseGif(assets.dat.get('flare.gif'));
      flare.bits = g.bits; flare.bps = g.w;
      flare.convert = cmapTo777(g.pal, g.nbCol);
    }
    // Pre_Cmp_6 (dem/src/part6.c): world -> vista -> Select_World (mesh
    // Finish incl. degenerate deletion + weld) -> texture cache -> lens.
    // Rebuilt every reset: meshes are mutated by Finish and the vista/spline
    // caches are stateful.
    {
      const world = parse3ds(assets.dat.get('nsal.3ds'));
      const vista = parseVista(assets.dat.get('nsal.vst'));
      finishMeshes(world, { weld: weldMeshes });
      finishWorld(world);
      loadTxtCacheIV(assets.dat.get('nsal.lzw'), jpg('nsal.jpg', 0x2565), world, 9);
      const camNode = world.nodes.find((n) => n.type === 'camera');
      camNode.data.lens = fr(1.3);
      // node IDs 5 (Light04) and 2 (Light01) carry the flare, scales
      // 0.37*1000 / 0.45*1000 (the shared flare asset is built above).
      const _unusedFlareMarker = 0; void _unusedFlareMarker;
      const flareSpritesInline = [
          { uo: 192, vo: 0, dU: 64, dV: 64, scale: fr(0.9), alpha: fr(-0.57) },
          { uo: 0, vo: 0, dU: 64, dV: 64, scale: fr(3.2), alpha: 0 },
          { uo: 192, vo: 0, dU: 64, dV: 64, scale: fr(1.9), alpha: fr(0.43) },
          { uo: 0, vo: 64, dU: 64, dV: 64, scale: fr(0.9), alpha: fr(0.76) },
          { uo: 64, vo: 64, dU: 64, dV: 64, scale: fr(0.6), alpha: fr(0.9) },
          { uo: 128, vo: 0, dU: 64, dV: 64, scale: fr(1.3), alpha: fr(1.75) },
          { uo: 64, vo: 0, dU: 64, dV: 64, scale: fr(2.1), alpha: fr(2.2) },
      ]; void flareSpritesInline;
      // W->Lights is a prepend chain: world.nodes order (newest-first)
      // filtered to lights reproduces the C's flare iteration order.
      const lights = world.nodes.filter((n) => n.type === 'light').map((n) => n.data);
      for (const [id, scale] of [[5, 370], [2, 450]]) {
        const n = world.nodes.find((nd) => nd.id === id && nd.type === 'light');
        if (n) { n.data.flare = flare; n.data.flareScale = scale; }
      }
      // Pre_Cmp_6 Search_Node_From_Name handles (Loop_6_05 toggles these
      // with OBJ_ON/OBJ_OFF between its two render passes).
      const byName = (n) => world.nodes.find((nd) => nd.name === n);
      st.t6 = {
        world, vista, camNode, lights,
        obj1: byName('mur01.2'), obj2: byName('Torus2'), obj3: byName('tun01'),
        pil: ['p01', 'p02', 'p03', 'p04'].map(byName),
        fc: ['f01', 'f02', 'f03'].map(byName),
        zbBase: new Int16Array(W * H),    // ZB_Base, allocated once in Init_6
        renderer: newRenderer(world, camNode),
      };
    }
    // lines4.h -> unpacked/lines.json (see work/unpack.mjs)
    {
      const raw = assets.lines ?? [];
      st.lines = [];
      for (let i = 0; i < 130; i++) {
        st.lines.push({
          xo: fr(raw[i * 4]), yo: fr(raw[i * 4 + 1]),
          x: fr(raw[i * 4 + 2]), y: fr(raw[i * 4 + 3]),
          scale: 0, rot: 0,
        });
      }
    }
    // Pre_Cmp_Destroy_Screen (janim.c): the Or ramp + hash tables. The 1000
    // random() draws Init_Hash1 makes are the ones reset() already had to
    // consume for stream position; now they build real tables.
    {
      const orTab = new Uint16Array(128);
      const cm = { u16: orTab, u32: new Uint32Array(orTab.buffer) };
      st.ramp(cm, 0, 64, 0, 0, 0, 20, 200, 100, 0x2565);
      for (let i = 64; i < 128; ++i) orTab[i] = orTab[63];
      st.orTab = orTab;
    }
    // Pre_Cmp_Blb (dem/src/blb.c): 5 metaballs on a 32^3 grid, env-mapped
    // through the single 256x256 cache-II map.
    {
      const blob = createBlob();
      for (let i = 0; i < 5; i++) mcBlobify2(blob, fr(0.8), fr(1.0));
      blob.K = fr(0.35);
      setMCDimensions(blob, 0, 0, 0, 3, 3, 3);
      const mesh = mcFinishBlob(blob, 4000, 5000);
      mesh.shader = 'env';
      const cache = loadTxtCacheIIbis(assets.dat.get('maps1.lzw'), jpg('maps1.jpg', 0x2565));

      // The blob node's transform: Id_Transform leaves TRANSF_QUATERNION
      // with QRot = (0,0,0,-1), and Set_Node_Rot_II then writes EULER angles
      // into Rot.Rot[0..2] — the same union bytes as QRot[0..2]. The type is
      // never changed, so Transform_To_Matrix reads them as a quaternion
      // (w,x,y) with z left at -1. Constant every frame; reproduced exactly.
      const qrot = new Float32Array([
        fr(40.0 * M_PI_180), fr(134.0 * M_PI_180), fr(30.0 * M_PI_180), -1,
      ]);
      const node = {
        type: 'mesh', name: 'blob', id: 1, parent: null, children: [],
        flags: { notActive: false, dontRender: false, doAnim: false },
        transform: null, mo: idMatrix2(), m: idMatrix2(), curTime: -Infinity,
        data: mesh,
        box: { mins: [-3, -3, -3], maxs: [3, 3, 3], radius: fr(Math.sqrt(3 * 36)) },
      };
      quatToMatrix(qrot, node.mo);
      const camNode = {
        type: 'camera', name: 'blobcam', id: 2, parent: null, children: [],
        flags: { notActive: false, dontRender: false, doAnim: false },
        mo: idMatrix2(), m: idMatrix2(), curTime: -Infinity,
        data: { pos: new Float32Array(3), target: new Float32Array(3), bank: 0, lens: fr(1.6), d: 0, splines: {} },
      };
      const world = { nodes: [node, camNode], framesStart: 0, framesEnd: 1 };
      const renderer = newRenderer(world, camNode);
      renderer.setEnvCache(cache, cache.maps[0]);
      st.blob = blob;
      st.tBlob = { world, node, camNode, renderer, cache };
    }
    // Pre_Cmp_World4 (dem/src/world4.c): two small worlds, cache III, no
    // vista — so they render through the default scene-graph path. Flares
    // hang off two lights per world (scale 0.37*550 / 0.45*550).
    {
      const mk = (wname, cname, objName, lens, lid1, lid2) => {
        const world = parse3ds(assets.dat.get(wname));
        finishMeshes(world, { weld: weldMeshes });
        finishWorld(world);
        loadTxtCacheIII(assets.dat.get(cname), world, 9);
        // Search_Node_From_Name(..., "Camera01") — shad5.3ds carries TWO
        // cameras and the demo names the one it wants, so picking the first
        // camera in the chain renders greets 1 and 3 from the wrong view.
        const camNode = world.nodes.find((n) => n.type === 'camera' && n.name === 'Camera01');
        camNode.data.lens = fr(lens);
        const byName = (n) => world.nodes.find((nd) => nd.name === n);
        const lights = world.nodes.filter((n) => n.type === 'light').map((n) => n.data);
        for (const [id, sc] of [[lid1, fr(0.37 * 550)], [lid2, fr(0.45 * 550)]]) {
          const n = world.nodes.find((nd) => nd.id === id && nd.type === 'light');
          if (n) { n.data.flare = flare; n.data.flareScale = sc; }
        }
        return {
          world, camNode, lights,
          obj1: byName(objName), obj2: byName('Object02'),
          renderer: newRenderer(world, camNode),
        };
      };
      st.greets = {
        w6: mk('shad5.3ds', 'shad5.lzw', 'Object01', 1.0, 6, 7),
        w7: mk('box4.3ds', 'box4.lzw', 'Object03', 2.1, 4, 3),
        zbBase: new Int16Array(W * H),
        cur: null,
      };
      st.greets.cur = st.greets.w6;
    }
    // Pre_Cmp_World2 (dem/src/world2.c): the city, on the vista path.
    {
      const world = parse3ds(assets.dat.get('villef3.3ds'));
      const vista = parseVista(assets.dat.get('villef3.vst'));
      finishMeshes(world, { weld: weldMeshes });
      finishWorld(world);
      loadTxtCacheIV(assets.dat.get('v42.lzw'), jpg('v42.jpg', 0x2565), world, 9);
      const camNode = world.nodes.find((n) => n.type === 'camera');
      camNode.data.lens = fr(1.9);
      st.t4 = { world, vista, camNode, renderer: newRenderer(world, camNode) };
    }
    // Pre_Cmp_World3 (dem/src/world3.c): st.3ds through the same
    // finish/cache pipeline, but rendered by the default scene-graph path
    // (no vista is ever loaded for this world).
    {
      const world = parse3ds(assets.dat.get('st.3ds'));
      finishMeshes(world, { weld: weldMeshes });
      finishWorld(world);
      loadTxtCacheIV(assets.dat.get('st.lzw'), jpg('st.jpg', 0x2565), world, 9);
      const camNode = world.nodes.find((n) => n.type === 'camera');
      camNode.data.lens = fr(1.3);
      st.t5 = { world, camNode, renderer: newRenderer(world, camNode) };
    }
    startDemo();
  }

  // ---- mixers (dem/fx1/mix4.c; UNIX C variants) --------------------------

  const B = () => st.buffers;

  // outOff/srcOff/rows let Paste_DMask_Move_II shift the layers against
  // each other (VBUFFER Bits/H fiddling in the C); defaults = full screen.
  function mix8_8BlendMaskII(out, mask, anim, outOff = 0, srcOff = 0, rows = H) {
    const dst = out.bits, m = mask.bits, a = anim.bits;
    const cm = mask.cmap.u32, ca = anim.cmap.u16;
    const n = rows * W;
    for (let i = 0; i < n; i++) {
      const s = cm[m[srcOff + i]];
      const b = s & 0xff;
      const c = ca[a[srcOff + i]];
      const s777 = (s >>> 8) + T.lo[b][c & 0xff] + T.hi[b][c >> 8];
      dst[outOff + i] = to16(s777);
    }
  }

  // dem/fx1/mix4.c Mix_16_8_Blend_Mask_II: mask palette entry carries a
  // premultiplied 777 color + 0-15 fade level for the 565 source under it.
  function mix16_8BlendMaskII(out, mask, in16) {
    cap.composite('mix16_8BlendMaskII', { mask: mask._layer, in16: in16._layer }, {}, out);
    const dst = out.bits, m = mask.bits, s = in16.bits;
    const cm = mask.cmap.u32;
    for (let i = 0; i < W * H; i++) {
      const e = cm[m[i]];
      const b = e & 0xff;
      const c = s[i];
      dst[i] = to16((e >>> 8) + T.lo[b][c & 0xff] + T.hi[b][c >> 8]);
    }
    cap.out(out);
  }

  function mixSat16_8(out, in16, in8) {
    cap.composite('mixSat16_8', { in16: in16._layer, in8: in8._layer },
      { off8: in8.off || 0 }, out);
    const dst = out.bits, s1 = in16.bits, s2 = in8.bits;
    const cm = in8.cmap.u32;
    const off = in8.off || 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const p = s1[i];
        const s777 = T.lo[0][p & 0xff] + T.hi[0][p >> 8] + cm[s2[off + i]];
        dst[i] = to16(s777);
      }
    }
    cap.out(out);
  }

  function mix16_16Mix(out, in1, in2, l1, h1, l2, h2, copyBack) {
    cap.composite('mix16_16Mix', { in1: in1._layer, in2: in2._layer },
      { lvl1: l1._level ?? 0, lvl2: l2._level ?? 0, copyBack: copyBack ? 1 : 0 }, out);
    const dst = out.bits, s1 = in1.bits, s2 = in2.bits;
    for (let i = 0; i < W * H; i++) {
      const a = s1[i], b = s2[i];
      const s777 = l1[a & 0xff] + h1[a >> 8] + l2[b & 0xff] + h2[b >> 8];
      const v = to16(s777);
      dst[i] = v;
      if (copyBack) s1[i] = v;
    }
    cap.out(out);
  }

  // dem/fx1/mixdsp.c Mix_Dsp_Off_Safe_X/Y (separable displacement).
  function dspSafeX(out, inp, dspMap, offTab) {
    // offTab is rebuilt in place per call, so the recording needs its own
    // copy -- taken only when something is recording, since the argument
    // would otherwise be built and thrown away on every CPU frame.
    if (cap.active) cap.composite('dspX', { src: inp._layer, dspMap: '#dspX' }, { off: offTab.slice() }, out);
    const dst = out.bits, src = inp.bits, sat = st.dspSatX;
    for (let y = 0; y < H; y++) {
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const k = offTab[dspMap[row + x]] + x;
        dst[row + x] = src[row + sat[256 + k]];
      }
    }
    cap.out(out);
  }
  function dspSafeY(out, inp, dspMap, offTab) {
    if (cap.active) cap.composite('dspY', { src: inp._layer, dspMap: '#dspY' }, { off: offTab.slice(), rowStep: W }, out);
    const dst = out.bits, src = inp.bits, sat = st.dspSatY;
    for (let y = 0; y < H; y++) {
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const k = offTab[dspMap[row + x]];
        dst[row + x] = src[x + sat[256 + y + k]];
      }
    }
    cap.out(out);
  }

  // ---- effect helpers ----------------------------------------------------

  function getBeat() {
    if (st.timer > st.nextBeat) { st.timer0 = st.nextBeat; st.nextBeat += 1924; }
    const x = (st.timer - st.timer0) / 1924;
    st.animCnt = Math.trunc(x * st.animMaxCnt);
    if (st.animCnt >= st.animMaxCnt) st.animCnt = st.animMaxCnt - 1;
    else if (st.animCnt < 0) st.animCnt = 0;
    return x;
  }

  // janim.c Get_Beat_3 — four sub-pulses per beat, sharp 1->0 decay.
  function getBeat3() {
    if (st.timer > st.nextBeat) { st.timer0 = st.nextBeat; st.nextBeat += 1924; }
    let x = fr((st.timer - st.timer0) / 1924);
    x = fr((4 * x) % 1);                        // (FLT)fmod(4.0*x, 1.0)
    if (x > 0 && x < 0.3) return fr(1 - fr(x / fr(0.3)));
    return 0;
  }

  // janim.c Get_Beat_2 — twin pulses per beat (1->0 attack, 0->1 swell).
  // Shares Timer0/Next_Beat with Get_Beat; the 0.6/0.8 window bounds are
  // double literals in the C, 0.2f is a float — mirrored exactly.
  function getBeat2() {
    if (st.timer > st.nextBeat) { st.timer0 = st.nextBeat; st.nextBeat += 1924; }
    let x = fr((st.timer - st.timer0) / 962);   // 1.0f*(INT)/(1924/2)
    x = fr(x % 1);                              // (FLT)fmod(x, 1.0)
    if (x > 0 && x < fr(0.2)) x = fr(fr(fr(0.2) - x) / fr(0.2));
    else if (x > 0.6 && x < 0.8) x = fr(fr(x - fr(0.6)) / fr(0.2));
    else x = 0;
    return x;
  }

  // janim.c JPaste_I/II/III/IV — four interleaved 2x upscale variants,
  // cycled per frame; JANIM is a persistent feedback buffer.
  function jpaste(variant, bits, srcOff) {
    const d = B().janim.bits;
    const w = st.curAnimW, h = st.curAnimH; // 160x100 source
    if (variant === 0) {                    // JPaste_I: even rows/cols
      for (let y = 0; y < h; y++) {
        const dr = y * 2 * W, sr = srcOff + y * w;
        for (let x = 0; x < w; x++) d[dr + 2 * x] = bits[sr + x];
        // the doubling writes even columns up to 2(w-1); fill the last odd
        // one from its neighbour. SOURCE-derived, like the row copy below —
        // it bounds the doubled strip, not the screen.
        d[dr + 2 * w - 1] = d[dr + 2 * w - 2];
      }
      // last-row copy (janim.c tail memcpy: row 199 <- row 198)
      d.copyWithin((2 * h - 1) * W, (2 * h - 2) * W, (2 * h - 1) * W);
    } else if (variant === 1) {             // JPaste_II: odd cols = avg
      for (let y = 0; y < h; y++) {
        const dr = y * 2 * W, sr = srcOff + y * w;
        for (let x = 0; x < w; x++) {
          d[dr + 2 * x + 1] = (d[dr + 2 * x] + d[dr + 2 * x + 2]) >> 1;
          d[dr + 2 * x] = bits[sr + x];
        }
      }
    } else if (variant === 3) {             // JPaste_III (janim.c:62-80)
      // C keeps a pointer P at row-even start + 318 and walks i=-159..-1:
      // odd cols of the NEXT row become horizontal averages (of that row's
      // old even cols), even cols 0..316 take source bytes sequentially,
      // then the tail writes col 318 from the row's last source byte.
      let s = srcOff;
      for (let y = 0; y < h; y++) {
        // P starts at the END of the doubled row: 2*(w-1) is SOURCE-derived
        // (159*2 = 318 for the 160-wide anim strips), while the +W/+W+1/+W+2
        // offsets are the DESTINATION row stride. The two are independent and
        // only coincide here because 2*160 happens to equal W.
        const P = y * 2 * W + 2 * (w - 1);
        for (let i = -(w - 1); i < 0; i++) {
          d[P + 2 * i + W + 1] = (d[P + 2 * i + W] + d[P + 2 * i + W + 2]) >> 1;
          d[P + 2 * i] = bits[s++];
        }
        d[P + W + 1] = (d[P + W] + d[P + W + 1]) >> 1;
        d[P] = bits[s++];
      }
    } else {                                // JPaste_IV: odd rows = vert avg
      for (let y = 0; y < h - 1; y++) {
        const dr = y * 2 * W, sr = srcOff + y * w;
        for (let x = 0; x < w; x++) {
          d[dr + 2 * x + W] = (d[dr + 2 * x + 2 * W] + d[dr + 2 * x]) >> 1;
          d[dr + 2 * x] = bits[sr + x];
        }
      }
    }
  }

  // A_Paste[] = { JPaste_I, JPaste_II, JPaste_IV, JPaste_III } (janim.c:112)
  // jpaste() dispatch: 0->I, 1->II, 3->III, anything else ->IV.
  const A_PASTE_VARIANT = [0, 1, 2 /* ->IV */, 3];
  function pasteJpegAnim(move) {
    if (!st.curAnim) return;
    let srcOff = st.curAnimSize * st.animCnt;
    if (move && st.jaParity) {
      if (st.animCnt !== 0) srcOff -= st.curAnimW * (st.trandom.at(0, st.timer) & 0x03);
      st.jaParity--;
    }
    jpaste(A_PASTE_VARIANT[st.aCnt], st.curAnim.bytes, srcOff);
    if (st.isSub25) st.aCnt = (st.aCnt + 1) & 0x03;
  }

  // scratch.c
  const CYCLE = [0, 4, 7, 8, 10, 11, 12, 15, 14, 13, 12, 10, 7, 5, 3, 2];
  function scratchNextRamp(off) {
    const j = CYCLE[st.cOff];
    st.ramp(B().scratch.cmap, 0, 8, off, off, off,
      Math.trunc((off + 55) * j / 16), Math.trunc((off + 45) * j / 16),
      Math.trunc((off + 40) * j / 16), 0x777);
    const cAdv = st.trandom.at(1, st.timer) & 0x01;
    if (st.isSub25) st.cOff += cAdv;
    if (st.cOff >= 16) st.cOff = 0;
  }
  function scratchAndFlash(x) {
    st.cOff = Math.trunc(x * 4) & 0x0f;
    if (x > 0.8) scratchNextRamp(Math.trunc(240 * (x - 0.8) / 0.2));
    else scratchNextRamp(0);
  }
  function getNextScratch(adv) {
    const aAdv = adv + (st.trandom.at(2, st.timer) & 0x03);
    if (st.isSub25) st.adv0 = (st.adv0 + aAdv) & 0xff;
    if (st.adv0 >= 0xc0) st.adv0 = 0;
    B().scratch.bits = scratchGifs[st.adv0 >> 6].bits;
    B().scratch.off = st.scrOff;
    const jump = (st.trandom.at(0, st.timer) & 0x07) === 0;
    const jumpBy = (st.trandom.at(1, st.timer) & 0x7f) - 64;
    const drift = (st.trandom.at(2, st.timer) & 0x07) - 3;
    if (jump && st.isSub25) {
      st.scrOff += jumpBy;
      if (st.scrOff >= W) st.scrOff -= W;
      else if (st.scrOff < 0) st.scrOff += W;
    }
    if (st.isSub25) st.scrOff += drift;
    if (st.scrOff >= W) st.scrOff = 0;
    else if (st.scrOff < 0) st.scrOff = W - 1;
    // NOTE: the window pointer above uses the PRE-mutation Off — scratch.c
    // computes Bits+Off first and advances Off for the NEXT frame.
  }

  // dsp.c
  function initDeformBlur() {
    satTable5(st.dspSatX, 0, W - 1, 1, 256, W - 1, 257);
    satTable5(st.dspSatY, 0, H - 1, W, 256, H - 1, 257);
  }
  const offX = new Int32Array(256);
  function initDspXY2(amp) { satTable4(offX, -amp, amp, 1, 256); }

  function deformBlur(amp, i0, j0) {        // dsp.c Deform_Blur
    const [i, j] = decayLevels(i0, j0);
    initDspXY2(Math.trunc(amp));
    dspSafeX(B().vblurI, B().vblurII, st.dspRadial, offX);
    dspSafeY(B().vblurII, B().vblurI, st.dspRadial.subarray(W * H), offX);
    mix16_16Mix(B().screen, B().vblurII, B().screen, T.lo[i], T.hi[i], T.lo[j], T.hi[j], false);
  }

  // ---- appear / disappear wipes (dem/src/appear.c) -----------------------
  //
  // A luminance-keyed wipe: a greyscale JPEG ("appear2/3.jpg") is remapped
  // through a sliding 256-entry window of a 768-entry ramp, giving each
  // pixel a 0..31 blend level that crossfades two 565 buffers through
  // dedicated fade-to-white / fade-to-black tables.

  // appear.c Pre_Cmp_Appear
  function buildAppear() {
    const map = new Uint8Array(256 + 2 * 256);
    let j = 0;
    for (let i = 0; i < 256; i++, j++) map[j] = 0;
    for (let i = 0; i < 256; i++, j++) {
      let x = fr(fr(1 / 256) * i);
      x = fr(Math.sin(x * Math.PI / 2));
      x = fr(x * x);
      if (x < 0) map[j] = 0;
      else if (x >= 1) map[j] = 31;
      else map[j] = Math.floor(31.0 * x) & 0xff;
    }
    for (let i = 0; i < 256; i++, j++) map[j] = 31;

    // 32x256 blend tables: rows 0-15 fade to white, rows 16-31 to black
    const lo = new Uint32Array(32 * 256), hi = new Uint32Array(32 * 256);
    for (let l = 0; l < 16; ++l) {
      const x = fr(1.0 * (16 - l) / 16);
      const Bw = Math.trunc(0x7f * l / 16), Gw = Math.trunc(0x7f * l / 16), Rw = Math.trunc(0x7f * l / 16);
      for (let i = 0; i < 256; ++i) {
        const Bo = 4 * (i & 0x1f);
        const G1o = 2 * (i >> 5), G2o = (i & 0x07) << 4;
        const Ro = 4 * (i >> 3);
        // fade to black
        const B0 = Math.trunc(fr(Bw * x)), G0 = Math.trunc(fr(Gw * x)), R0 = Math.trunc(fr(Rw * x));
        lo[(l + 16) * 256 + i] = B0 >>> 0;
        hi[(l + 16) * 256 + i] = ((G0 << 8) | (R0 << 16)) >>> 0;
        // fade to white
        let B = Bo + Bw; if (B > 0x7f) B = 0x7f;
        let G1 = G1o + Gw; if (G1 > 0x7f) G1 = 0x7f;
        let G2 = G2o + Gw; if (G2 > 0x7f) G2 = 0x7f;
        let R = Ro + Rw; if (R > 0x7f) R = 0x7f;
        G2 &= 0x78; G1 &= 0x07;
        lo[l * 256 + i] = (B | (G1 << 8)) >>> 0;
        hi[l * 256 + i] = (((G2 << 8) | (R << 16)) >>> 0);
      }
    }
    return { map, lo, hi };
  }
  const appearT = buildAppear();
  const appear = jpg('appear2.jpg', 0x0000);
  const appear3 = jpg('appear3.jpg', 0x0000);
  const blur012 = jpg('blur012.jpg', 0x0000);
  const cnt4 = jpg('cnt4.jpg', 0x0000);
  const cnt5 = jpg('cnt5.jpg', 0x0000);
  const cnt1 = jpg('count1.jpg', 0x0000);
  const cnt2 = jpg('count2.jpg', 0x0000);
  const cnt3 = jpg('count3.jpg', 0x0000);
  const restb1 = jpg('restb1.jpg', 0x0000);
  const endPic = jpg('endb01.jpg', 0x2565);
  const restb2 = jpg('restb2.jpg', 0x0000);
  // part2.c Pre_Cmp_Ptcl: sp.jpg is decoded as 888 then converted to 777 with
  // the alpha byte masked off; the four "sprites" are 16-row bands of it.
  const sprsGlxy = (() => {
    const src = jpg('sp.jpg', 0x3888);
    const px = new Uint32Array(src.w * src.h);
    for (let i = 0; i < src.w * src.h; i++) {
      const B0 = src.bytes[3 * i], G0 = src.bytes[3 * i + 1], R0 = src.bytes[3 * i + 2];
      px[i] = (((R0 & 0xfe) << 15) | ((G0 & 0xfe) << 7) | ((B0 & 0xfe) >> 1)) >>> 0;
    }
    return [0, 1, 2, 3].map((k) => ({ bits: px, w: src.w, dU: src.w, dV: 16, uo: 0, vo: k * 16.0 }));
  })();
  const ufos = ['mac6.gif', 'coca3.gif', 'dollar2.gif', 'yes.gif'].map((n) => {
    const g = parseGif(assets.dat.get(n));
    const cmap = new Uint32Array(256);
    for (let i = 0; i < g.nbCol; i++) cmap[i] = rgbTo565(g.pal[3*i], g.pal[3*i+1], g.pal[3*i+2]);
    return { bits: g.bits, w: g.w, h: g.h, cmap, x: 0, y: 0 };
  });

  // demo_src Remap_Byte through a sliding window of Appear_Map
  function remapAppear(dst, src, inv) {
    const gx = st.globalX;
    const x = gx < 0 ? 0 : gx >= 1 ? 512 : Math.floor(gx * 512);
    const off = inv ? 512 - x : x;
    const map = appearT.map;
    for (let i = 0; i < W * H; i++) dst[i] = map[off + src[i]];
  }

  // mix4.c Mix_16_16_Blend — per-pixel blend level picks a table row; the
  // second source uses the complement (b ^ 31).
  function mix16_16Blend(out, in1, in2, blend) {
    cap.composite('mix16_16Blend', { in1: in1._layer, in2: in2._layer, blend: blend._layer }, {}, out);
    const dst = out.bits, s1 = in1.bits, s2 = in2.bits, bl = blend.bits;
    const lo = appearT.lo, hi = appearT.hi;
    for (let i = 0; i < W * H; i++) {
      let b = bl[i];
      const a = s1[i];
      let s777 = lo[(b << 8) | (a & 0xff)] + hi[(b << 8) | (a >> 8)];
      b ^= 31;
      const c = s2[i];
      s777 += lo[(b << 8) | (c & 0xff)] + hi[(b << 8) | (c >> 8)];
      dst[i] = to16(s777);
    }
    cap.out(out);
  }

  // appear.c Init_Appear. The clear in case 2 is
  //   memset( VB(VBLUR_I).Bits, 0, VB(1).Size )
  // — it wipes VBLUR_I for **VB(1)'s** size, not its own. VB(1) is retyped by
  // whichever part ran last: SELECT_CMAP(1) gives a real 320x200 buffer
  // (Size 64000 bytes = HALF of the 565 VBLUR_I), while SELECT_CMAP_V(1) —
  // used by Init_Blur_II for the sponsor part — makes it VIRTUAL, and
  // Select_VBuffer sets Size = 0 for virtual buffers. So coming out of the
  // city (which follows the sponsor) this clears NOTHING and the wipe
  // dissolves from the live city image. Getting this wrong renders the
  // city-out transition black.
  // Select_VBuffer keeps Size as a HIGH-WATER MARK: it only reallocates when
  // the new Total exceeds the current Size, so a later, smaller select leaves
  // Size at the older, bigger value. Selecting VIRTUAL resets it to 0.
  function selectVB1(totalBytes, virtual) {
    if (virtual) { st.vb1Size = 0; return; }
    if (totalBytes > st.vb1Size) st.vb1Size = totalBytes;
  }

  function initAppear(param) {
    if (param === 1) {
      selectVB1(W * H, false);              // SELECT_CMAP(1)
    } else if (param === 2) {
      B().vblurI.bits.fill(0, 0, st.vb1Size >> 1);   // bytes -> u16 entries
    }
  }

  function loopAppearII() {                 // appear.c Loop_Appear_II
    remapAppear(B().vb1b.bits, appear.bytes, false);
    loop14AnimMaskRaw();
    mix16_16Blend(B().screen, B().vblurII, B().screen, B().vb1b);
  }

  function loopDisappear() {                // appear.c Loop_Disappear
    remapAppear(B().vb1b.bits, appear3.bytes, true);
    mix16_16Blend(B().screen, B().vblurII, B().vblurI, B().vb1b);
  }

  // part1.c Scroll_3 — up to four parallax marquee bands over VSCREEN
  const BASE_Y = 195;
  function scroll3(n) {
    if (!st.jaParity && (st.trandom.at(2, st.timer) & 0x07) === 0x03) {
      st.jaParity = 3 + (st.trandom.at(1, st.timer) & 0x07);
    }
    let src;
    if (st.jaParity) { src = cnt5; st.jaParity--; } else src = cnt4;
    const cmap = new Uint32Array(256);
    st.ramp({ u16: new Uint16Array(cmap.buffer), u32: cmap }, 0, 256, 0, 0, 0, 200, 220, 255, 0x777);
    if (n > 0) paste256((BASE_Y - 20) * W, W, 20, src.bytes, 212 * src.w, st.uScroll, cmap);
    if (n > 1) paste256((BASE_Y - 35) * W, W, 30, src.bytes, 182 * src.w, (st.uScroll * 2) & 0xff, cmap);
    if (n > 2) paste256((BASE_Y - 50) * W, W, 40, src.bytes, 138 * src.w, (st.uScroll * 3) & 0xff, cmap);
    if (n > 3) paste256((BASE_Y - 85) * W, W, 68, src.bytes, 68 * src.w, (st.uScroll * 4) & 0xff, cmap);
  }

  function loop14AnimMaskSat() {             // part14.c Loop_14_Anim_Mask_Sat
    scratchAndFlash(getBeat());
    let x = st.globalX;
    if (x < fr(0.1)) x = fr(x * fr(0.4));
    else if (x > fr(0.9)) x = fr(fr(1.0 - x) * fr(0.4));
    else x = 1.0;
    saturateDMaskPal(fr(x * 2.0), st.curMask, 0xe0c0b2);
    pasteJpegAnim(false);
    mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  function loop14Ready() {                   // part14.c Loop_14_Ready
    loop14AnimMaskMoveII();
    if (!st.readyCnt) {
      // Rolled on the 25 Hz sub-clock so the authored 1-in-64 chance is
      // tested 25x/second at any rate; the burst length still scales.
      if (st.isSub25 && (st.random() & 0x3f) === 0x01) {
        st.readyCnt = durFrames(st.random() & 0x1f);
      }
    }
    if (st.readyCnt) {
      destroyScreen(getBeat());
      st.readyCnt--;
    }
  }

  function loop14Anim() {                    // part14.c Loop_14_Anim
    scratchAndFlashII(getBeat());
    pasteJpegAnim(false);
    getNextScratch(0x08);
    mixSat8_8(B().screen, B().janim, B().scratch);
  }

  function loop14NoWay() {                   // part14.c Loop_14_No_Way
    loop14AnimMaskMoveII();
    scroll3(4);
    st.uScroll = Math.trunc(fr(st.globalX * 2.0) * 256.0) & 0xff;
  }

  function loop14NoWayII() {                 // part14.c Loop_14_No_Way_II
    loop14Anim();
    scroll3(Math.trunc(fr(fr(1.0 - st.tickX) * 4)));
    st.uScroll = Math.trunc(fr(st.globalX * 2.0) * 256.0) & 0xff;
  }

  function loopDeformBlurIV() {              // dsp.c Loop_Deform_Blur_IV
    st.dspRadial = dsp.r2;
    let x = fr(Math.cos(st.tickX * Math.PI));
    x = fr(1 - fr(x * x));
    const amp = fr(-fr(x * 7.0));
    x = fr(1 - st.tickX); x = fr(1 - fr(x * x));
    // Deform_Blur_II
    initDspXY2(Math.trunc(amp));
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    dspSafeY(B().screen, B().vblurI, st.dspRadial.subarray(W * H), offX);
    const i = Math.trunc(fr(6 + fr(4 * x))), j = Math.trunc(fr(6 + fr(5 * x)));
    mix16_16Mix(B().screen, B().vblurII, B().screen, T.lo[i], T.hi[i], T.lo[j], T.hi[j], true);
  }

  function init03(param) {                   // parts.c Init_03
    if (param === 0) { initD2(0); initScroll(0); }
    else if (param === 1) {
      st.curMask = masks.logo; setupMaskII(st.curMask);
      setupAnim(anims.anim93, 160, 100);
      initD2(1);
      setAnimCMap(190, 220, 170);
    } else if (param === 2) {
      st.curMask = masks.noway; setupMaskII(st.curMask);
      setupAnim(anims.anim72, 160, 100);
      setAnimCMap(190, 220, 170);
      initScroll(2);
    } else if (param === 3) {
      st.curMask = null;
      setAnimCMap(180, 220, 150, 0x4777);
    } else if (param === 11) {                // "ready?"
      st.curMask = masks.rest; setupMaskII(st.curMask);
      setupAnim(anims.anim93, 160, 100);
      setAnimCMap(180, 220, 150);
      initCaustics(0);                        // sets up VB(1)
      st.ramp(B().vb2.cmap, 0, 256, 0, 0, 0, 190, 255, 230, 0x4777);
      st.animCnt = 0;
    } else if (param === 13) {
      st.curMask = masks.end; setupMaskII(st.curMask);
      setupAnim(anims.anim82, 160, 100);
      setAnimCMap(180, 220, 150);
    } else st.unported = true;
  }

  // ---- credits / fade / end (parts.c Init_05, dsp.c, fx1/fade.c) ---------

  function init05(param) {                   // parts.c Init_05
    if (param === 0) {
      initDeformBlur();
      st.dspRadial = dsp.r2;
      // VB(1) becomes a 565 copy of the end picture, with the counter strip
      // pasted into a virtual window at x=210 before the part starts.
      const src = new Uint16Array(endPic.bytes.buffer, endPic.bytes.byteOffset, endPic.bytes.byteLength >> 1);
      B().vb1.bits.set(src.subarray(0, W * H));
      const cmap = new Uint32Array(256);
      st.ramp({ u16: new Uint16Array(cmap.buffer), u32: cmap }, 0, 256, 0, 0, 0, 200, 220, 255, 0x777);
      paste256YInto(B().vb1.bits, 0 * W + 210, 68, 200, cnt1.bytes, 0x00, cmap, cnt1.w);
    } else if (param === 1) {
      st.curMask = masks.resist;
      setupMaskII(st.curMask);
      const src = new Uint16Array(endPic.bytes.buffer, endPic.bytes.byteOffset, endPic.bytes.byteLength >> 1);
      B().vb1.bits.set(src.subarray(0, W * H));
      mix16_8BlendMaskII(B().screen, B().vmask, B().vb1);
      st.curMask = null;                     // Clear_DMask
    } else st.unported = true;
  }

  // Paste_256_Y against an explicit destination (Init_05 targets VB(1))
  function paste256YInto(dst, dstOff, w, h, src, v1, cmap, srcW) {
    let V = v1 & 0xff;
    let s = (v1 & 0xff) * srcW;
    let d = dstOff;
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const p = dst[d + i];
        dst[d + i] = to16(T.lo[0][p & 0xff] + T.hi[0][p >> 8] + cmap[src[s + i]]);
      }
      V++;
      if (V === 256) { s = 0; V = 0; } else s += srcW;
      d += W;
    }
  }

  function loopDeformBlurIII() {             // dsp.c Loop_Deform_Blur_III
    const x = fr(1 - st.tickX);
    const amp = fr(fr(7.0) + fr(fr(x * 25.0) + fr(fr(50.0 * x) * fr(Math.sin(x * Math.PI * 6.0)))));
    mix16_16Mix(B().vblurII, B().screen, B().vb1, T.lo[9], T.hi[9], T.lo[8], T.hi[8], false);
    // Deform_Blur_III: radial #2, VBLUR_II -> VBLUR_I -> VBLUR_II -> screen
    const i0 = Math.trunc(fr(5.5 + fr(5.5 * x)));
    const [i, j] = decayLevels(i0, i0);
    initDspXY2(Math.trunc(amp));
    dspSafeX(B().vblurI, B().vblurII, st.dspRadial, offX);
    dspSafeY(B().vblurII, B().vblurI, st.dspRadial.subarray(W * H), offX);
    mix16_16Mix(B().screen, B().vblurII, B().screen, T.lo[i], T.hi[i], T.lo[j], T.hi[j], false);
    scratchAndFlash(getBeat());
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  // fade.c Init_Fade_Tab_16b + Mix_To_Fade_Feedback: the screen fades into
  // itself, so each frame dims what the previous one left behind.
  const fadeLo = new Uint16Array(256), fadeHi = new Uint16Array(256);
  function initFadeTab16b(x) {
    for (let i = 0; i < 256; ++i) {
      const R = Math.trunc((i & 0xf8) * x) >>> 0;
      const G2 = Math.trunc(((i & 0x07) << 8) * x) >>> 0;
      const G1 = Math.trunc((i & 0xe0) * x) >>> 0;
      const Bv = Math.trunc((i & 0x1f) * x) >>> 0;
      fadeHi[i] = (((R & 0xf8) << 8) | (G2 & 0x07e0)) & 0xffff;
      fadeLo[i] = ((G1 & 0xe0) | (Bv & 0x1f)) & 0xffff;
    }
  }
  function loopFade() {                      // fx1/fade.c Loop_Fade
    initFadeTab16b(fr(decayConst(0.98)));
    const d = B().screen.bits;
    for (let i = 0; i < W * H; i++) {
      const v = d[i];
      d[i] = (fadeLo[v & 0xff] + fadeHi[v >> 8]) & 0xffff;
    }
  }

  function loop14End() {                     // part14.c Loop_14_End
    loop14AnimMask();
  }

  // ---- world4: the greets (dem/src/world4.c) -----------------------------
  //
  // Structurally Loop_6_05 again: two passes over one world time with
  // OBJ_ON/OFF between them, the z-buffer shader on, mask blend after the
  // first pass and flares on the second. But this world has NO vista, so it
  // goes through the default scene-graph path, and its textures come from
  // cache III (every mip level LZW'd in-file, no atlas).

  function initGreets(param) {
    const g = st.greets;
    if (param === 0) {
      st.timeScale = 1.0; st.timeOff = 0;
      st.curMask = masks.gr1;
      g.cur = g.w6;
    } else if (param === 1) {
      st.curMask = masks.gr2;
      g.cur = g.w7;
    } else if (param === 2) {
      st.timeScale = fr(0.98);
      st.curMask = masks.gr3;
      g.cur = g.w6;
    } else { st.unported = true; return; }
    setupMaskII(st.curMask);
    g.cur.renderer.setupCamScreen(0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H);
  }

  function loopW402() {                      // world4.c Loop_W4_02
    const g = st.greets, c = g.cur;
    const OFF = (n) => { if (n) n.flags.dontRender = true; };
    const ON = (n) => { if (n) n.flags.dontRender = false; };
    c.renderer.setZBuffer(g.zbBase);

    const upd = () => {
      const { time, lx } = updateWorld(c.world, c.camNode, st.globalX, st.timeScale, st.timeOff, W);
      if (lx !== null) c.renderer.rc.lx = lx;
      return time;
    };
    c.renderer.setupClipping(6.0, W - 6.0, 10.0, H - 10.0, 1.0, 2000.0);
    ON(c.obj2); OFF(c.obj1);
    upd();
    c.renderer.renderScene(B().screen.bits);

    saturateDMaskPal(getBeat(), st.curMask, 0xd8d8e0);
    mix16_8BlendMaskII(B().screen, B().vmask, B().screen);

    c.renderer.setupClipping(0, W, 0, H, 1.0, 2000.0);
    OFF(c.obj2); ON(c.obj1);
    upd();
    c.renderer.renderScene(B().screen.bits);
    c.renderer.renderLightFlare(c.lights, B().screen.bits, FLARE_TABLES);
    ON(c.obj2);

    c.renderer.setZBuffer(null);

    // --- native layer -----------------------------------------------------
    // The same two passes again at 1600x1200. Geometry is resolution
    // independent, so this re-projects and re-rasterises rather than scaling
    // anything: renderScene memsets Vertex_State per mesh, so a second call
    // with a different camera transforms cleanly. The 320x200 pass above is
    // left intact because downstream parts read the screen buffer and the
    // gates verify it. The dmask between the passes is 320x200 and gets
    // block-replicated -- see work/re/RESOLUTION.md.
    if (native3d) {
      const R = c.renderer;
      nativePass(R, (clip) => {
        // As above: world4.c clears neither the screen nor ZB_Base.
        R.setZBuffer(nativeZBuf);
        clip(6.0, W - 6.0, 10.0, H - 10.0, 1.0, 2000.0);
        ON(c.obj2); OFF(c.obj1);
        R.renderScene(nativeScreen);
        blendMaskNative(nativeScreen, B().vmask);
        clip(0, W, 0, H, 1.0, 2000.0);
        OFF(c.obj2); ON(c.obj1);
        R.renderScene(nativeScreen);
        R.renderLightFlare(c.lights, nativeScreen, FLARE_TABLES);
        ON(c.obj2);
      });
      st.nativeValid = true;
    }
  }

  // ---- caustics + the D2 logo (caustics.c, part1.c) ----------------------

  const CLAMP_OFF = 320;

  function initCaustics(param) {
    if (!st.caustics) {
      st.caustics = {
        base: buildBasePlasma(),
        tabI: new Uint16Array(2 * CLAMP_OFF + 256),
        tabII: new Uint16Array(2 * CLAMP_OFF + 256),
        plasma: new Uint8Array(256 * 256),
        bil1: new Uint8Array(257 * 257),
      };
    }
    const c = st.caustics;
    if (param === 0) {
      B().vb1b.bits = st.vb1Own;             // SELECT_CMAP(1)
      selectVB1(W * H, false);
      st.ramp(B().vb1b.cmap, 0, 256, 0, 0, 0, 150, 180, 255, 0x4777);
      satTable3(c.tabI, 0, 40 * 2, 1.0, 1, 256);
      satTable3(c.tabII, 0, 25 * 2, 1.0, 40 * 2 + 1, 256);
    } else if (param === 1) {
      satTable3(c.tabI, 0, 64, 1.0, 1, 256);
      satTable3(c.tabII, 0, 64, 1.0, 64 + 1, 256);
    }
  }

  function prepareCaustics(t) {
    const c = st.caustics;
    // The C's angle constants (2.12, 1.244, ...) are DOUBLE literals, so
    // each of these expressions evaluates in double all the way to the
    // (BYTE) cast — only `x` itself is float.
    let x = fr(t * fr(1.0));
    const U1 = Math.trunc(60.0 + 40.0 * Math.cos(2.12 * x * M_PI_180)) & 0xff;
    const V1 = Math.trunc(120.0 + 70.0 * Math.sin(1.244 * x * M_PI_180)) & 0xff;
    x = fr(t * fr(1.5));
    const U2 = Math.trunc(110.0 - 90.0 * Math.cos(2.87 * x * M_PI_180)) & 0xff;
    const V2 = Math.trunc(90.0 + 120.0 * Math.sin(0.985 * x * M_PI_180)) & 0xff;
    doPlasma256(c.plasma, c.base, U1, V1, U2, V2);
  }

  // caustics.c Set_Tunnel_Position
  function setTunnelPosition(x, m) {
    const po = new Float32Array(3), dir = new Float32Array(3);
    po[2] = fr(st.timeScale * x);
    dir[2] = fr(fr(70.0 * x) - 35.0);
    const s = fr(Math.sin(8.0 * x * Math.PI));
    po[0] = fr(fr(3.8) * fr(Math.sin(2.0 * Math.PI * s)));
    po[1] = fr(fr(3.8) * fr(Math.cos(2.0 * Math.PI * s)));
    dir[0] = 5.0; dir[1] = 0.0;
    cameraMatrix(po, dir, 0.0, m);
  }

  function loopCausticsII() {
    const c = st.caustics;
    prepareCaustics(fr(st.globalX * 500.0));
    doDaFunkyCaustics(c.plasma, 64, 64, c.bil1, c.tabI, c.tabII);
    doBilinear4(c.plasma, c.bil1, 64, 64, 256);
    setTunnelPosition(st.globalX, st.causticsM);
    // The ray tunnel's UV map is the last thing in the show whose cost scales
    // with output resolution: everything upstream of it (the plasma, the
    // photon scatter, the 4x expansion) produces a FIXED 256x256 texture and
    // is correctly left on the CPU. The mapping itself becomes one pass.
    const uvCap = capturing ? (c.cap ??= { acc: new Uint16Array(82 * 26) }) : null;
    doRayTunnel(B().vb3.bits, st.causticsM, c.plasma, uvCap);
    if (uvCap) {
      cap.composite('bilinUVMap', { map: '#plasma', grid: '#uvgrid', acc: '#uvacc' },
        {}, B().vb3, { uv: { map: c.plasma, grid: uvCap.grid, acc: uvCap.acc, n: uvCap.n } });
      cap.out(B().vb3);
    }
  }

  // part1.c Do_Blur_Y — the logo plate blurred/offset into VB(1)
  const blurSat1 = new Uint16Array(350 + 2 * CLAMP_OFF);
  const blurInt1 = new Int32Array(256);
  function doBlurY(out) {
    let off;
    const sCnt = Math.trunc(fr(st.globalX * 256));
    if (st.globalX < 0.3) {
      off = Math.trunc(fr(fr(256.0 / 0.3) * st.globalX));
      if (off > 255) off = 255;
    } else off = 255;
    satTable3(blurSat1, 0, off, 1.0, 1, 350);

    let W2 = W, H2 = H, offX, src, y;
    if (sCnt & 0x08) {
      src = restb1.bytes;
      off = 0;
      offX = (st.trandom.at(1, st.timer) & 0x0f) - 7;
      if (!(st.trandom.at(0, st.timer) & 0x1f)) offX += (st.trandom.at(2, st.timer) & 0x1f) - 16;
      y = fr(st.globalX * 255.0);
    } else {
      src = restb2.bytes;
      const x = fr(fr((st.trandom.at(0, st.timer) & 0xff) / 128.0) - 1.0);
      off = Math.trunc(fr(x * 24.0));
      offX = (st.trandom.at(1, st.timer) & 0x07) - 3;
      // C: 256.0f*(FLT)exp(-x*x*3.5f) — the argument is a float chain
      y = fr(fr(256.0) * fr(Math.exp(fr(fr(fr(-x) * x) * fr(3.5)))));
    }
    satTable4(blurInt1, 0, Math.trunc(y), 1, 256);

    let dOff = 0, sOff = 0;
    if (off > 0) { H2 -= off; dOff += off * W; }
    else if (off < 0) { H2 += off; sOff -= off * W; }
    if (offX > 0) { W2 -= offX; dOff += offX; }
    else if (offX < 0) { W2 += offX; sOff += offX; }

    const dst = out.bits;
    for (let j = 0; j < H2; j++) {
      const dr = dOff + j * W + W2, sr = sOff + j * W;
      for (let i = -W2; i < 0; ++i) {
        dst[dr + i] = blurSat1[CLAMP_OFF + dst[dr + i] + blurInt1[src[sr + W2 + i]]] & 0xff;
      }
    }
  }

  // janim.c Destroy_Screen — horizontal squash, mirrored per row pair
  function destroyScreen(x) {
    const dst = B().screen.bits;
    const off = Math.trunc(x * 0x100);
    // Pure gather despite writing in place -- see EFFECTS.destroyScreenIII.
    cap.composite('destroyScreenIII', { screen: 'screen' }, { off, w1: W - 1 }, B().screen);
    let d = 0;
    for (let j = H / 2; j > 0; j--) {
      let U = (W - 1) * off;
      for (let i = W - 1; i >= 0; --i) { dst[d + i] = dst[d + (U >> 8)]; U -= off; }
      d += W;
      U = (W - 1) * 0x100 - (W - 1) * off;
      for (let i = 0; i < W; ++i) { dst[d + i] = dst[d + (U >> 8)]; U += off; }
      d += W;
    }
    cap.out(B().screen);
  }

  function initD2(param) {                   // part1.c Init_D2
    if (param === 0) {
      // SELECT_CMAP(1): VB(1) becomes a REAL 8-bit buffer again. The
      // sponsor part had pointed it straight at the blur012 JPEG
      // (SELECT_CMAP_V + VB(1).Bits = Blur012->Bits), and Do_Blur_Y both
      // reads and writes its destination — so without this it accumulates
      // into the JPEG instead of a fresh plane.
      B().vb1b.bits = st.vb1Own;
      st.vb1Own.fill(0);
      selectVB1(W * H, false);
      initCaustics(1);
      st.timeScale = fr(2600.0);
    }
    st.ramp(B().vb1b.cmap, 0, 256, 0, 0, 0, 180, 180, 180, 0x777);
  }

  function loopD2() {                        // part1.c Loop_D2
    let x = getBeat();
    scratchAndFlash(x);
    loopCausticsII();
    doBlurY(B().vb1b);
    x = fr(fr(2.0 * x) % 1);
    // NOTE: the C writes `150.0 + 65.0+sin(..)` — the '+' where a '*' was
    // surely meant, so the sine barely moves the colour. Preserved.
    const R = Math.trunc(150.0 + 65.0 + Math.sin(x * Math.PI));
    const G = Math.trunc(250.0 - 85.0 + Math.sin(x * Math.PI));
    const Bc = Math.trunc(180.0 - 65.0 + Math.sin(x * Math.PI));
    st.ramp(B().vb3.cmap, 0, 256, 0, 0, 0, R, G, Bc, 0x777);
    mixSat8_8(B().screen, B().vb1b, B().vb3);
    scroll1();
    st.uScroll = Math.trunc(fr(25.0 * st.globalX) * 256.0) & 0xff;
  }

  function loopD2Destroy() {                 // part1.c Loop_D2_Destroy
    loopD2();
    destroyScreen(st.tickX);
  }

  // ---- particles: galaxy + UFOs (dem/src/part2.c) ------------------------
  //
  // 900 sprites orbiting in a disc, accumulated additively into a 777
  // (32-bit) buffer over the bilinear sky, then packed down to 565. The
  // saturation is a per-channel remap table: values >= 128 clamp to 255.

  const [B_REMAP, G_REMAP, R_REMAP] = T.sat;
  const sat777 = (v) => (R_REMAP[(v >>> 16) & 0xff] | G_REMAP[(v >>> 8) & 0xff] | B_REMAP[v & 0xff]) >>> 0;

  // part2.c Set_Ptcl_I — every draw here is the raw LCG, so the particle
  // setup advances the shared seed ~5x900 times.
  const BETA = fr(5.2), R0 = fr(1.0), Z0 = fr(0.26);
  const K_12_256 = fr(fr(1.2) / fr(256.0));    // 1.2f/256.0f, folded in float
  // -ffast-math does not only contract, it REASSOCIATES. Set_Ptcl_I as
  // compiled (clang -O2 -ffast-math, arm64) differs structurally from the C:
  //
  //   * `r*r/(0.3f*0.3f)` becomes `Rraw * -(1/0.09f/65536) * r`, with r's own
  //     /65536 scaling folded into the constant  (0xB931C71C)
  //   * `Ptr->z = Z0 * (Rz/32768 - 1) * factor` is distributed to
  //     `fmaf(Rz, Z0/32768, -Z0) * factor` -- two roundings, not four
  //   * both branches converge with only the FACTOR live; the C's `z` local
  //     is never materialised
  //
  // Porting the C literally leaves ~40% of the 900 particles one ulp off,
  // which then integrates over the part's 370 frames into visible drift.
  const K_RR = fr(fr(-frcp(fr(fr(0.3) * fr(0.3)))) / 65536.0);  // 0xB931C71C
  const K_Z0_32768 = fr(Z0 / 32768.0);                          // 0x37051EB8
  function setPtclI(p) {
    const rnd = st.trandom.raw;
    const rRaw = rnd() & 0xffff, tRaw = rnd() & 0xffff, zRaw = rnd() & 0xffff;
    let r = fr(rRaw / 65536.0);            // ucvtf s, w, #16
    let t = fr(tRaw / 65536.0);
    let factor;
    if (r < 0.3) {                         // .3 is a double literal
      factor = fmaf(fr(rRaw * K_RR), r, 1.0);
      t = fr(t * (2.0 * Math.PI));         // 2.0f*M_PI promotes to double
      r = fr(Math.pow(r, fr(0.9)));        // 0.9f widened for pow()
    } else {
      r = fr(Math.pow(r, 1.4));            // 1.4 is a double literal
      factor = fr(Math.sqrt(fr(fr(1.0 - r) / fmaf(r, BETA, 1.0))));
      t = fr(fr(fr(t * t) * t) * Math.PI);
      if (rnd() & 0x01) t = fr(t + Math.PI);
      const ct = fr(Math.cos(t));          // clang narrows cos() to cosf()
      r = fmaf(fr(r * fr(0.5)), fmaf(ct, ct, 1.0), fr(0.1));
    }
    // dt associates as (-2r) * (M_PI * term), both multiplies in double
    const term = fmaf((rnd() & 0xffff), fr(0.01 / 65536.0), 1.0);
    p.t = t;
    p.dt = fr(fr(r * fr(-2.0)) * (Math.PI * term));
    p.z = fr(fmaf(zRaw, K_Z0_32768, fr(-Z0)) * factor);
    p.r = fr(R0 * r);
    p.dr = fmaf((rnd() & 0xff), K_12_256, fr(0.5));
    p.sp = (rnd() & 0x10) ? 0 : 1 + (rnd() % 3);
  }

  function setupAllPtcls() {
    st.ptcl = Array.from({ length: 900 }, () => ({ z: 0, t: 0, dt: 0, r: 0, dr: 0, x: 0, y: 0, s: 0, sp: 0 }));
    for (const p of st.ptcl) setPtclI(p);
    st.ptclTrf = {
      rot: new Float32Array([fr(Math.PI / 2 - 0.42), fr(0.25), 0]),
      pos: new Float32Array([0, 0, fr(1.0)]),
    };
  }

  // transf.c Transform_To_Matrix, TRANSF_RZYX branch (doubles)
  function transformRZYX(rot, pos, m) {
    const Cx = Math.cos(rot[0]), Sx = Math.sin(rot[0]);
    const Cy = Math.cos(rot[1]), Sy = Math.sin(rot[1]);
    const Cz = Math.cos(rot[2]), Sz = Math.sin(rot[2]);
    const A = Cz * Sy, Bv = Sz * Sy;
    m[0] = m[9] = fr(Cy * Cz);
    m[1] = m[12] = fr(-Cy * Sz);
    m[2] = m[15] = fr(Sy);
    m[3] = m[10] = fr(Sz * Cx + A * Sx);
    m[4] = m[13] = fr(Cz * Cx - Bv * Sx);
    m[5] = m[16] = fr(-Cy * Sx);
    m[6] = m[11] = fr(Sz * Sx - A * Cx);
    m[7] = m[14] = fr(Cz * Sx + Bv * Cx);
    m[8] = m[17] = fr(Cy * Cx);
    m[18] = pos[0]; m[19] = pos[1]; m[20] = pos[2];
    m[21] = fr(-fr(fr(fr(m[9] * m[18]) + fr(m[10] * m[19])) + fr(m[11] * m[20])));
    m[22] = fr(-fr(fr(fr(m[12] * m[18]) + fr(m[13] * m[19])) + fr(m[14] * m[20])));
    m[23] = fr(-fr(fr(fr(m[15] * m[18]) + fr(m[16] * m[19])) + fr(m[17] * m[20])));
  }

  function setPtclMatrix(phi, phi2, theta) {
    const t = st.ptclTrf;
    t.rot[0] = fr(fr(fr(Math.PI / 2 - fr(0.32))) + fr(fr(0.10) * fr(Math.sin(theta * 8.0))));
    t.rot[1] = fr(fr(0.15) + fr(fr(0.10) * fr(Math.sin(theta * 5.0))));
    t.rot[0] = fr(t.rot[0] * phi2);
    t.rot[1] = fr(t.rot[1] * phi2);
    t.rot[2] = fr(theta * 4.0);
    t.pos[2] = fr(1.0 - fr(fr(0.3) * phi));
    transformRZYX(t.rot, t.pos, st.ptclM);
  }

  // sprite.c Setup_Paste_Sprite + Paste_Sprite_Asm_Sat_32 (additive, 777)
  const spOff = new Int32Array(1024), spRow = new Int32Array(1024);
  function pasteSpriteSat32(dst, xo, yo, R, sp) {
    const clips = st.ptclClips;
    let tmp = fr(fr(0.5 * R) * sp.dU);
    const xi = fr(xo - tmp);
    let x = xi, xf = fr(xo + tmp);
    tmp = fr(fr(0.5 * R) * sp.dV);
    const yi = fr(yo - tmp);
    let y = yi, yf = fr(yo + tmp);
    // Capture the UNCLIPPED rect: the sprite maps its source linearly across
    // it, so pairing a clipped rect with unclipped UVs would stretch the
    // sprite. GL clips the quad itself.
    if (capturing) {
      cap.sprite(xi, yi, fr(xo + fr(fr(0.5 * R) * sp.dU)), yf,
        sp.uo, sp.vo, sp.uo + sp.dU, sp.vo + sp.dV, sp, 'add777');
    }
    if (xf <= clips[0]) return;
    if (xf > clips[1]) xf = clips[1];
    if (yf <= clips[2]) return;
    if (yf > clips[3]) yf = clips[3];
    if (x > clips[1]) return;
    if (x < clips[0]) x = clips[0];
    if (y > clips[3]) return;
    if (y < clips[2]) y = clips[2];
    const Rinv = fr(1 / R);

    const X = Math.ceil(x);
    let U = fr(sp.uo + fr(fr(X - xi) * Rinv));
    let dF = Math.trunc(Rinv * 65536.0);
    const dErr = dF & 0xffff;
    dF >>= 16;
    let F = Math.trunc(U * 65536.0);
    let err = F & 0xffff;
    F >>= 16;
    const stepsX = Math.ceil(xf) - X;
    if (stepsX <= 0) return;
    for (let i = 0; i < stepsX; ++i) {
      spOff[i] = F; F += dF; err += dErr;
      if (err & 0x10000) { err &= 0xffff; F++; }
    }

    const Y = Math.ceil(y);
    let V = fr(sp.vo + fr(fr(Y - yi) * Rinv));
    F = Math.trunc(V * 65536.0);
    err = F & 0xffff;
    F >>= 16;
    const bps = sp.w;                        // BpS in UINTs
    const dFy = dF * bps;
    let what = F * bps;
    const stepsY = Math.ceil(yf) - Y;
    if (stepsY <= 0) return;
    for (let i = 0; i < stepsY; ++i) {
      spRow[i] = what; what += dFy; err += dErr;
      if (err & 0x10000) { err &= 0xffff; what += bps; }
    }

    const dW0 = stepsX - 1, dH0 = stepsY - 1;
    const bits = sp.bits;
    let d = Y * W + X + dW0;
    for (let dH = -dH0; dH <= 0; ++dH) {
      const row = spRow[dH + dH0];
      for (let dW = -dW0; dW <= 0; ++dW) {
        const s = (bits[row + spOff[dW + dW0]] + dst[d + dW]) >>> 0;
        dst[d + dW] = sat777(s);
      }
      d += W;
    }
  }

  function doPtclAnim(phi, phi2, theta) {
    // Compiled -ffast-math (see fp.js), Do_Ptcl_Anim differs from its C in
    // four places: cos/sin collapse to one __sincosf_stret (FLOAT precision,
    // not double-then-narrow); the t integration contracts to one fmadd;
    // `(2.0f-V2)/2.0f` reassociates to fmaf(V2,-0.5,1.0); and the screen
    // mapping hoists V2*95.0f so x/y become a single fmadd/fmsub each.
    const M = st.ptclM;
    const v = new Float32Array(3);
    const theta4 = fr(theta * fr(4.0));    // loop-invariant, hoisted by clang
    for (const p of st.ptcl) {
      const r = fr(p.r * phi);
      const t = fr(p.t + theta4);
      v[0] = fr(fr(Math.cos(t)) * r);
      v[1] = fr(fr(Math.sin(t)) * r);
      v[2] = fr(p.z * phi);
      p.t = fmaf(p.dt, st.globalDx, p.t);
      aV(v, M, v);
      if (v[2] <= 0 || v[2] > 2.0) { p.s = 0; continue; }
      p.s = fmaf(v[2], fr(-0.5), 1.0);
      const iz = fr(fr(1.6) / fr(1.0 + v[2]));
      const iz95 = fr(iz * fr(95.0));
      p.x = fmaf(iz95, v[0], fr(100.0));
      if (p.x < -30.0 || p.x > 350.0) { p.s = 0; continue; }
      p.y = fmaf(fr(-iz95), v[1], fr(100.0));   // fmsub
      if (p.y < -30.0 || p.y > 230.0) { p.s = 0; continue; }
      // 2.0f*0.5f folds to 1.0f and vanishes: r is just s*dr
      pasteSpriteSat32(st.vb1_777, p.x, p.y, fr(p.s * p.dr), sprsGlxy[p.sp]);
    }
  }

  function loopPtclI() {                     // part2.c Loop_Ptcl_I
    st.ptclClips = [1.0, 319.0, 1.0, 199.0];
    const gx = st.globalX;
    let phi;
    if (gx < fr(0.4)) phi = fr(gx / fr(0.4));
    else if (gx < fr(0.8)) phi = 1.0;
    else if (gx < fr(0.85)) phi = fr(1.0 - fr(fr(gx - fr(0.80)) / fr(0.05)));
    else phi = fr(fr(10.0 * fr(gx - fr(0.85))) / fr(0.15));
    let phi2;
    if (gx < fr(0.3)) phi2 = 0;
    else if (gx < fr(0.5)) phi2 = fr(fr(gx - fr(0.3)) / fr(0.2));
    else phi2 = 1.0;
    const theta = fr(gx * fr(2.0 * Math.PI));
    setPtclMatrix(phi, phi2, theta);
    setBck(B().vb3.bits, sky.bits, st.ptclM, W);
    let amp = 1.0;
    if (gx >= fr(0.35)) amp = fr(1.0 - fr(fr(gx - fr(0.35)) / fr(0.65)));
    B().vb3.cmap.u32.set(cmapTo777Fx(sky.pal, sky.nbCol, 0x000000, Math.trunc(fr(amp * 255.0)) & 0xff));
    // Map_32_Bits: sky through its 777 palette into the 32-bit accumulator
    { const d = st.vb1_777, s = B().vb3.bits, cm = B().vb3.cmap.u32;
      for (let i = 0; i < W * H; i++) d[i] = cm[s[i]]; }
    const sprFrom = cap.spriteCount;
    doPtclAnim(phi, phi2, theta);
    // Mix_777_To_16
    { const d = B().screen.bits, s = st.vb1_777;
      for (let i = 0; i < W * H; i++) d[i] = to16(s[i]); }
    // The whole tail -- sky through its 777 palette, 900 saturating sprite
    // adds, pack to 565 -- is one GPU pass over an additively blended
    // accumulation buffer. Recorded AFTER the CPU has done it, so the mirror
    // matches and the destroyXOr that follows chains instead of syncing.
    if (cap.sprites('spriteOver', { base: 'vb3' }, B().screen, sprFrom)) cap.out(B().screen);
    destroyScreenXOr(getBeat());
  }

  // janim.c Destroy_Screen_XOr
  const XOR_TAB = [
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x8410, 0x8410,
    0xf79e, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xf79e, 0xe71c, 0x8410,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  ];
  function destroyScreenXOr(x) {
    const or = XOR_TAB[Math.floor(fr(x * fr(31.9)))] ?? 0;
    if (or === 0x0000) return;
    cap.composite('destroyXOr', { screen: 'screen' }, { orMask: or }, B().screen);
    const d = B().screen.bits;
    for (let i = 0; i < W * H; i++) d[i] ^= or;
    cap.out(B().screen);
  }

  function loopPtclII() {                    // part2.c Loop_Ptcl_II
    loopPtclI();
    const u = st.ufo;
    if (u) {
      // VBuf_Map_8_Bits_Transp into the virtual VB(2) window. This is a 1:1
      // blit that skips index 0 and maps through a 565 palette -- the same
      // semantics as Print_Letter's map16 mode, so it captures as a glyph
      // quad and needs no separate GPU path.
      const d = B().screen.bits, s = u.bits, cm = u.cmap;
      if (capturing) {
        cap.glyph(u.x, u.y, u.x + u.w, u.y + u.h, 0, 0, u.w, u.h,
          { bits: u.bits, w: u.w, convert: cm }, 'map16', 'screen');
      }
      for (let j = 0; j < u.h; j++) {
        const dr = (u.y + j) * W + u.x, sr = j * u.w;
        for (let i = 0; i < u.w; i++) if (s[sr + i]) d[dr + i] = cm[s[sr + i]] & 0xffff;
      }
    }
    const p = st.ptcl[st.ptclZero];
    if (p && p.s > 0) {
      const d = B().screen.bits;
      const ux = st.ufoX, uy = st.ufoY;
      let base = uy * W + ux;
      // Each run below is captured as one solid quad; the CPU still writes
      // the pixels, because frame565 is the verified artifact.
      const RED = 0xf800;
      const solid = { solid: RED };
      const runQ = (x0, y0, x1, y1) => {
        if (capturing) cap.glyph(x0, y0, x1, y1, 0, 0, 1, 1, solid, 'solid', 'screen');
      };
      runQ(ux, uy - 1, ux + 4, uy + 3);
      for (const off of [-W, 0, W, 2 * W]) for (let i = 0; i < 4; i++) d[base + off + i] = 0xf800;
      let xo = Math.trunc(p.x);
      if (xo < 5) xo = 5; else if (xo >= ux) xo = ux;
      p.x = fr(1.0 * xo);
      let a = xo;
      base = uy * W + a;
      a = ux - a;
      runQ(xo, uy, ux, uy + 2);
      while (a--) { d[base + a] = 0xf800; d[base + a + W] = 0xf800; }
      let yo = Math.trunc(p.y);
      if (yo < 5) yo = 5; else if (yo > H - 5) yo = H - 5;
      p.y = fr(1.0 * yo);
      let a2;
      if (uy < yo) a2 = uy; else { a2 = yo; yo = uy; }
      let dp = a2 * W + Math.trunc(p.x);
      a2 = yo - a2;
      runQ(Math.trunc(p.x), a2 === 0 ? yo : yo - a2, Math.trunc(p.x) + 2, yo);
      while (a2--) { d[dp] = 0xf800; d[dp + 1] = 0xf800; dp += W; }
      dp = Math.trunc(p.y) * W + Math.trunc(p.x);
      runQ(Math.trunc(p.x) - 1, Math.trunc(p.y) - 1, Math.trunc(p.x) + 3, Math.trunc(p.y) + 2);
      for (const off of [-W, 0, W]) for (let i = -1; i < 3; i++) d[dp + off + i] = 0xf800;
    }
  }

  function initPtcl(param) {                 // part2.c Init_Ptcl
    if (param === 0) {
      st.vb1_777 = new Uint32Array(W * H);
      setupAllPtcls();
    } else if (param === 1) { st.ufo = ufos[0]; st.ufo.x = 165; st.ufo.y = 5; st.ptclZero = 1; st.ufoX = 165 + 20; st.ufoY = 5 + 23; }
    else if (param === 3) { st.ufo = ufos[1]; st.ufo.x = 200; st.ufo.y = 0; st.ptclZero = 4; st.ufoX = 200 + 6; st.ufoY = 5 + 15; }
    else if (param === 5) { st.ufo = ufos[2]; st.ufo.x = 240; st.ufo.y = 5; st.ptclZero = 7; st.ufoX = 240 - 4; st.ufoY = 5 + 10; }
    else if (param === 7) { st.ufo = ufos[3]; st.ufo.x = 145; st.ufo.y = 10; st.ptclZero = 15; st.ufoX = 145 - 4; st.ufoY = 10 + 10; }
    else st.ufo = null;                      // even params just drop the UFO
  }

  // ---- the lines part (dem/src/part10.c + draw16/line16.c) ---------------
  //
  // 130 vector segments tumbling under a per-line random scale/rotation,
  // drawn with an anti-aliased Bresenham that writes through a palette
  // (Line_16_CMap) rather than blending — so the AA is a colour LUT lookup.

  const aaTable = new Uint8Array(256);
  function initAntiAliasTable(c0, c1, gamma) {
    aaTable[0] = c0;
    for (let i = 1; i < 256; ++i) {
      let x = fr(fr(1.0 * i) / fr(256.0));
      if (gamma !== 1.0) x = fr(Math.pow(x, gamma));
      aaTable[i] = Math.floor(c0 + fr(x * fr(c1 - c0))) & 0xff;
    }
  }

  // line16.c _Draw_ALine_16. Pointer arithmetic is in BYTES over a 565
  // buffer, so ±2 bytes is ±1 pixel and ±Pix_BpS is ±one row.
  function drawALine16(dst, cmap, clips, xo, yo, x1, y1, stride = W, wid = 1) {
    let tmp;
    if (y1 < yo) { tmp = yo; yo = y1; y1 = tmp; tmp = xo; xo = x1; x1 = tmp; }
    if (yo >= clips[3] || y1 < clips[2]) return;
    let dy = fr(y1 - yo), dx = fr(x1 - xo);
    const MIX = (p, c) => { dst[p] = cmap[c]; };
    // Pen width, in pixels across the line. The core is `wid` pixels thick
    // along the perpendicular axis with the two anti-aliasing taps just
    // outside it. wid = 1 collapses to the original three-tap pattern
    // exactly, so the 320x200 path runs the same arithmetic it always did.
    //
    // This exists because a vector line is NOT automatically
    // resolution-independent: drawn one pixel wide at 1600x1200 it is 5-6x
    // thinner in relative terms than the same line at 320x200, so the
    // artwork gets sparser and dimmer instead of sharper.
    const CORE = (p, c, step) => {
      for (let n = 0; n < wid; n++) dst[p + n * step] = cmap[c];
    };

    if (dy < 0.001) {                        // horizontal
      if (dx < 0) { tmp = xo; xo = x1; x1 = tmp; }
      if (xo >= clips[1] || x1 < clips[0]) return;
      if (xo < clips[0]) xo = clips[0];
      if (x1 > clips[1]) x1 = clips[1];
      const y = Math.ceil(yo);
      const err = Math.floor(255.0 * (y - yo)) & 0xff;
      const C1 = aaTable[err], C2 = aaTable[0xff], C3 = aaTable[0xff ^ err];
      let x = Math.ceil(xo);
      const xf = Math.ceil(x1);
      let p = (y + 1) * stride - x;
      for (; x < xf; ++x) {
        MIX(p - stride, C1); CORE(p, C2, stride); MIX(p + wid * stride, C3);
        p -= 1;
      }
      return;
    }

    let S = fr(Math.abs(dx));
    if (S < fr(0.001)) {                     // vertical
      if (xo >= clips[1] || xo < clips[0] + 1.0) return;
      const x = Math.ceil(xo);
      const err = Math.floor(255.0 * (x - xo)) & 0xff;
      const C1 = aaTable[0xff ^ err], C2 = aaTable[0xff], C3 = aaTable[err];
      if (y1 >= clips[3]) y1 = clips[3];
      if (yo < clips[2]) yo = clips[2];
      let y = Math.ceil(yo);
      const yf = Math.ceil(y1);
      let p = (y + 1) * stride - x;
      for (; y < yf; ++y) {
        MIX(p - 1, C1); CORE(p, C2, 1); MIX(p + wid, C3);
        p += stride;
      }
      return;
    }

    if (yo < clips[2]) { xo = fr(xo + fr(fr(clips[2] - yo) * dx / dy)); yo = clips[2]; }
    if (y1 > clips[3]) { x1 = fr(x1 + fr(fr(clips[3] - y1) * dx / dy)); y1 = clips[3]; }

    let T;
    if (S < dy) {                            // steeper than 45 degrees
      S = fr(dx / dy);
      T = fr(65536.0 / Math.sqrt(1.0 + S * S));
      if (S < 0) {
        if (xo < clips[0] || x1 >= clips[1]) return;
        if (xo > clips[1]) { yo = fr(yo + fr(fr(clips[1] - xo) / S)); xo = clips[1]; }
        if (x1 < clips[0]) { y1 = fr(y1 + fr(fr(clips[0] - x1) / S)); x1 = clips[0]; }
      } else {
        if (xo >= clips[1] || x1 < clips[0]) return;
        if (xo < clips[0]) { yo = fr(yo + fr(fr(clips[0] - xo) / S)); xo = clips[0]; }
        if (x1 > clips[1]) { y1 = fr(y1 + fr(fr(clips[1] - x1) / S)); x1 = clips[1]; }
      }
      let yi = Math.ceil(yo);
      xo = fr(xo + fr(S * fr(yi - yo)));
      const xi = Math.ceil(xo);
      let p = (yi + 1) * stride - xi;
      let err = Math.floor(fr(T * fr(xi - xo))) >>> 0;
      let yf = Math.ceil(y1);
      if (yi >= clips[3] || yf <= clips[2]) return;
      const dErr = Math.floor(Math.abs(fr(T * S))) >>> 0;
      const tErr = Math.floor(T) >>> 0;
      const C0 = aaTable[0xff];
      yf -= yi;
      if (dx > 0) {
        err = (tErr - err) >>> 0;
        while (yf--) {
          MIX(p - 1, aaTable[(err >>> 8) & 0xff]);
          MIX(p + wid, aaTable[0xff ^ ((err >>> 8) & 0xff)]);
          CORE(p, C0, 1);
          p += stride; err = (err + dErr) >>> 0;
          if (err >= tErr) { err = (err - tErr) >>> 0; p -= 1; }
        }
      } else {
        while (yf--) {
          MIX(p + wid, aaTable[(err >>> 8) & 0xff]);
          MIX(p - 1, aaTable[0xff ^ ((err >>> 8) & 0xff)]);
          CORE(p, C0, 1);
          p += stride; err = (err + dErr) >>> 0;
          if (err >= tErr) { err = (err - tErr) >>> 0; p += 1; }
        }
      }
    } else {                                 // shallower than 45 degrees
      S = fr(dy / dx);
      T = fr(65536.0 / Math.sqrt(1.0 + S * S));
      if (S < 0) {
        if (xo < clips[0] || x1 >= clips[1]) return;
        if (xo > clips[1]) { yo = fr(yo + fr(fr(clips[1] - xo) * S)); xo = clips[1]; }
        if (x1 < clips[0]) { y1 = fr(y1 + fr(fr(clips[0] - x1) * S)); x1 = clips[0]; }
      } else {
        if (xo >= clips[1] || x1 < clips[0]) return;
        if (xo < clips[0]) { yo = fr(yo + fr(fr(clips[0] - xo) * S)); xo = clips[0]; }
        if (x1 > clips[1]) { y1 = fr(y1 + fr(fr(clips[1] - x1) * S)); x1 = clips[1]; }
      }
      let xi = Math.ceil(xo);
      yo = fr(yo + fr(fr(xi - xo) * S));
      const yi = Math.ceil(yo);
      let p = (yi + 1) * stride - xi;
      let err = Math.floor(fr(T * fr(yi - yo))) >>> 0;
      let xf = Math.ceil(x1);
      const dErr = Math.floor(Math.abs(fr(T * S))) >>> 0;
      const tErr = Math.floor(T) >>> 0;
      err = (tErr - err) >>> 0;
      const C0 = aaTable[0xff];
      if (dx > 0) {
        p -= 1;
        xf -= xi;
        while (xf--) {
          MIX(p + wid * stride, aaTable[(err >>> 8) & 0xff]);
          MIX(p - stride, aaTable[0xff ^ ((err >>> 8) & 0xff)]);
          CORE(p, C0, stride);
          p -= 1; err = (err + dErr) >>> 0;
          if (err >= tErr) { err = (err - tErr) >>> 0; p += stride; }
        }
      } else {
        xi -= xf;
        while (xi--) {
          MIX(p + wid * stride, aaTable[(err >>> 8) & 0xff]);
          MIX(p - stride, aaTable[0xff ^ ((err >>> 8) & 0xff)]);
          CORE(p, C0, stride);
          p += 1; err = (err + dErr) >>> 0;
          if (err >= tErr) { err = (err - tErr) >>> 0; p += stride; }
        }
      }
    }
  }

  function randomizeLines() {                // part10.c Randomize_Lines
    for (let i = 0; i < 130; i++) {
      st.lines[i].scale = fr(1.0 * (st.trandom.raw() & 0xff) / 256.0);
      st.lines[i].rot = fr(0.4 * Math.PI * (st.trandom.raw() & 0xff) / 256.0);
    }
  }

  // `tw`/`th` are the target's pixel dimensions and `stride` its row pitch.
  // The line endpoints arrive in NORMALISED 0..1 space (lines4.h stores them
  // that way), so the only thing binding this to 320x200 is the final scale
  // and the 2-pixel clip margin -- which is why this part can render at any
  // resolution while most of the show cannot.
  function doLines(eps, zoom, dst, cmap, tw, th, stride, wid = 1) {
    const mx = fr(2.0 * tw / W), my = fr(2.0 * th / H);   // the 2px margin, scaled
    const clips = [mx, tw - mx, my, th - my];
    for (let i = 0; i < 130; i++) {
      const L = st.lines[i];
      const scale = fr(1.0 + fr(L.scale * eps));
      const rot = fr(L.rot * eps);
      const C = fr(Math.cos(rot)), S = fr(Math.sin(rot));
      let xi = fr(fr(L.xo - fr(0.5)) * zoom + fr(0.5));
      let yi = fr(fr(L.yo - fr(0.5)) * zoom + fr(0.5));
      let xf = fr(fr(L.x - fr(0.5)) * zoom + fr(0.5));
      let yf = fr(fr(L.y - fr(0.5)) * zoom + fr(0.5));
      const Cx = fr(fr(xi + xf) / 2.0), Cy = fr(fr(yi + yf) / 2.0);
      xi = fr(fr(xi - Cx) * scale); yi = fr(fr(yi - Cy) * scale);
      xf = fr(fr(xf - Cx) * scale); yf = fr(fr(yf - Cy) * scale);
      const xxi = fr(fr(fr(xi * C) + fr(yi * S)) + Cx);
      const yyi = fr(fr(fr(-xi * S) + fr(yi * C)) + Cy);
      const xxf = fr(fr(fr(xf * C) + fr(yf * S)) + Cx);
      const yyf = fr(fr(fr(-xf * S) + fr(yf * C)) + Cy);
      drawALine16(dst, cmap, clips,
        fr(xxi * tw), fr(yyi * th), fr(xxf * tw), fr(yyf * th), stride, wid);
    }
  }

  function init10(param) {                   // part10.c Init_10
    if (param === 0) randomizeLines();
    B().vb1.bits.set(B().screen.bits);       // Copy_Buffer_Short(VB(1), VSCREEN)
    initAntiAliasTable(0, 255, fr(1.7));
  }

  function loop10() {                        // part10.c Loop_10
    B().vb2b.bits.fill(0);                   // VBuffer_Clear(VB(2))
    let x = fr(Math.sin(st.tickX * Math.PI));
    const zoom = fr(fr(0.9) * fr(1.0 - fr(fr(1.0 - x) * fr(1.0 - x))));
    const rot = fr(x * x);
    st.ramp(B().vb2b.cmap, 0, 256, 0, 0, 0, 50, 240, 200, 0x2565);
    const eps = fr(3.0 * rot), zm = fr(1 + zoom);
    doLines(eps, zm, B().vb2b.bits, B().vb2b.cmap.u16, W, H, W);
    const i = Math.trunc(fr(4.0 - fr(rot * 3.0)));
    const j = Math.trunc(fr(rot * 14.0));
    mix16_16Mix(B().screen, B().vb1, B().vb2b, T.lo[j], T.hi[j], T.lo[i], T.hi[i], false);

    // --- native layer -----------------------------------------------------
    // The lines are redrawn at 1600x1200 from the same normalised geometry
    // and the same per-frame eps/zoom, so this is a higher-resolution
    // rendering of the same image, not a filtered copy of the 320x200 one.
    // VB(1) -- a frozen 320x200 screen grab taken at Init_10, which here is
    // the last blob frame -- has no more detail to give, so it is block
    // replicated 5x6 during the mix. See work/re/RESOLUTION.md.
    if (native) {
      nativeLines.fill(0);
      // A uniform pen: one source pixel covers 5 native pixels across and 6
      // down, so its apparent width depends on line orientation purely as a
      // pixel-aspect artifact. Native pixels are square, so the corrected
      // choice is one width for every orientation -- consistent with the
      // decision recorded in work/re/RESOLUTION.md 6.
      doLines(eps, zm, nativeLines, B().vb2b.cmap.u16, NW, NH, NW, NATIVE_PEN);
      mixNativeOverBlocks(nativeScreen, B().vb1.bits, nativeLines,
        T.lo[j], T.hi[j], T.lo[i], T.hi[i]);
      st.nativeValid = true;
    }
  }

  // mix16_16Mix, but `in1` is a 320x200 buffer sampled with 5x6 block
  // replication while `in2` is already native. One pass, no intermediate
  // upscaled copy.
  function mixNativeOverBlocks(dst, in1, in2, l1, h1, l2, h2) {
    const sx = NW / W, sy = NH / H;
    for (let y = 0; y < NH; y++) {
      const srcRow = ((y / sy) | 0) * W;
      const dstRow = y * NW;
      for (let x = 0; x < NW; x++) {
        const a = in1[srcRow + ((x / sx) | 0)], b = in2[dstRow + x];
        dst[dstRow + x] = to16(l1[a & 0xff] + h1[a >> 8] + l2[b & 0xff] + h2[b >> 8]);
      }
    }
  }

  // ---- the blob (dem/src/blb.c + marching cubes) -------------------------


  // janim.c Pre_Cmp_Destroy_Screen tables. Init_Hash1(10,126) is what
  // consumes the 1000 BSD random() draws at load; until this part they were
  // only consumed for stream position, now the tables are actually read.
  const HTAB_I = new Uint8Array(256);
  [3, 2, 3, 3, 3, 4, 4, 3, 3, 4, 29, 32, 16, 15, 14, 11, 8, 5, 3, 1, 0, 0]
    .forEach((v, i) => { HTAB_I[i] = v; });

  function initHash1(amp, seed, rnd) {
    const hash1 = new Uint8Array(256), randTab = new Uint8Array(256);
    for (let i = 0; i < 256; i++) { hash1[i] = i; randTab[i] = i % amp; }
    for (let i = 0; i < 500; i++) {
      const j = (rnd() >> 9) & 0xff;
      const k = (rnd() >> 5) & 0xff;
      if (k === j) continue;
      const sw = hash1[k]; hash1[k] = hash1[j]; hash1[j] = sw;
    }
    return { hash1, randTab, hashSeed: seed & 0xff };
  }

  // janim.c Destroy_Screen_III — hashed green-ramp corruption of the screen
  function destroyScreenIII() {
    const dst = B().screen.bits;
    const phase0 = st.dsPhase0;
    let phase = (phase0 + (st.trandom.at(2, st.timer) & 0x03)) & 0xff;
    st.dsPhase0 = (phase0 + 3) & 0xff;
    for (let y = 0; y < H; y++) {
      const ko = HTAB_I[phase] + 8 + (st.trandom.at(1, st.timer) & 0x03);
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const k = (ko + st.hash.randTab[st.hashSeed]) & 0xff;
        dst[row + x] = st.orTab[((dst[row + x] >> 5) & 0x003f) + k];
        st.hashSeed = st.hash.hash1[st.hashSeed];
      }
      phase = (phase + 1) & 0xff;
    }
  }

  // part1.c Paste_256_Y — like Paste_256 but scrolling vertically
  function paste256Y(dstOff, w, h, src, v1, cmap, srcW) {
    const dst = B().screen.bits;
    let V = v1 & 0xff;
    let s = (v1 & 0xff) * srcW;
    let d = dstOff;
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const p = dst[d + i];
        dst[d + i] = to16(T.lo[0][p & 0xff] + T.hi[0][p >> 8] + cmap[src[s + i]]);
      }
      V++;
      if (V === 256) { s = 0; V = 0; } else s += srcW;
      d += W;
    }
  }

  function scroll2() {                      // part1.c Scroll_2
    if (!st.jaParity && (st.trandom.at(2, st.timer) & 0x07) === 0x03) {
      st.jaParity = 3 + (st.trandom.at(0, st.timer) & 0x07);
      st.scrollSrc = (st.trandom.at(1, st.timer) & 0x08) ? cnt2 : cnt3;
    }
    const sc = st.scroll;
    if (st.jaParity) {
      paste256Y(sc.off, sc.w, sc.h, st.scrollSrc.bytes, st.uScroll, B().vb4cmap.u32, st.scrollSrc.w);
      st.jaParity--;
    } else {
      paste256Y(sc.off, sc.w, sc.h, cnt1.bytes, st.uScroll, B().vb4cmap.u32, cnt1.w);
    }
  }

  // blb.c Set_Blob_Position — the five metaball centres on their own orbits
  const M_PI_180 = Math.PI / 180.0;
  function setBlobPosition(bRot, R) {
    const G = st.blob.G;
    G[0][0] = fr(fr(fr(R * fr(0.66)) * Math.sin(fr(0.243) * bRot * M_PI_180)));
    G[0][1] = fr(fr(fr(R * fr(0.6)) * Math.cos(fr(1.9012) * bRot * M_PI_180)));
    G[0][2] = fr(fr(fr(R * fr(0.7)) * Math.sin(fr(0.543) * bRot * M_PI_180 + fr(0.2142))));
    G[1][0] = fr(fr(fr(R * fr(0.9212)) * Math.sin(fr(0.7613) * bRot * M_PI_180)));
    G[2][2] = fr(fr(fr(R * fr(0.883)) * Math.sin(fr(1.443) * bRot * M_PI_180)));
    G[3][2] = fr(fr(fr(R * fr(0.9524)) * Math.sin(fr(1.64) * bRot * M_PI_180)));
    G[3][1] = fr(fr(fr(R * fr(0.92151)) * Math.cos(fr(1.64) * bRot * M_PI_180)));
    G[4][0] = fr(fr(fr(R * 2 * fr(0.4)) * Math.sin(fr(1.312) * bRot * M_PI_180)));
    G[4][2] = fr(fr(fr(R * 2 * fr(0.3151)) * Math.cos(fr(0.735) * bRot * M_PI_180 + fr(0.124))));
  }

  function setupBlb(flag) {                 // blb.c Setup_Blb
    const t = st.tBlob;
    t.renderer.setupClipping(0, W, 0, H, fr(0.2), 0);
    if (flag) {
      const cy = fr(0.5 + fr(fr(fr(0.08) * (st.trandom.at(1, st.timer) & 0x07)) / 16));
      t.renderer.setupCamScreen(0.3 * W, 0.5 * H, 0.3 * W, fr(cy * H));
    } else t.renderer.setupCamScreen(0.3 * W, 0.5 * H, 0.3 * W, 0.5 * H);
    let x = getBeat();
    x = fr(Math.cos(x * 2.0 * Math.PI));
    x = fr(x * x); x = fr(x * x);
    const y = fr(st.globalX * 1500);
    setBlobPosition(y, fr(0.5 + fr(fr(0.69) * x)));
    doMCube3(st.blob);
  }

  function loopAnimBlbI() {                 // blb.c Loop_Anim_Blb_I
    loop14AnimMaskRaw();
    if (st.blbCount) {
      setupBlb(true);
      st.blbCount--;
    } else {
      setupBlb(false);
      if (st.isSub25 && st.tickX < 0.90
          && (st.trandom.at(1, st.timer) & 0x3f) === 0x00) {
        st.blbCount = durFrames(10 + (st.trandom.at(0, st.timer) & 0x0f));
      }
    }
    // Render_Obj: the node is normally flagged DONT_RENDER; it is cleared
    // just for this call (and W->Time bumped so everything re-evaluates).
    // Render_World's camera setup, which this part gets through the default
    // path rather than updateWorld: build Cam->Mo, then M = Cam.Mo x Node.Mo.
    const t = st.tBlob;
    cameraMatrix(t.camNode.data.pos, t.camNode.data.target, t.camNode.data.bank, t.camNode.mo);
    multMatrixABC(t.camNode.mo, t.node.mo, t.node.m);
    t.renderer.renderScene(B().screen.bits);
    if (st.blbCount) destroyScreenIII();
    scroll2();
    st.uScroll = Math.trunc(fr(25.0 * st.globalX) * 256.0) & 0xff;
  }

  function initBlb() {                       // blb.c Init_Blb
    const t = st.tBlob;
    t.renderer.setupCamScreen(0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H);
    t.camNode.data.pos[0] = 0; t.camNode.data.pos[1] = 0; t.camNode.data.pos[2] = -2;
    t.camNode.data.target[0] = 0; t.camNode.data.target[1] = 0; t.camNode.data.target[2] = 0;
  }

  function init02(param) {
    if (param !== 0) { st.unported = true; return; }
    st.curMask = masks.blob;
    setupMaskII(st.curMask);
    setupAnim(anims.anim32, 160, 100);
    setAnimCMap(200, 250, 170);
    initBlb();
    initScroll(1);
  }

  // ---- world2: the city (dem/src/world2.c) -------------------------------
  //
  // Vista path (villef3.vst), so this reuses the tunnel's renderer wholesale.
  // Note Init_04(0) points Setup_Rendering at VBLUR_II — the 3D lands in the
  // blur buffer and Deform_Blur composites it onto VSCREEN; Init_04(1)
  // switches the target back to VSCREEN for the plain pass.

  function init04(param) {
    const t4 = st.t4;
    if (param === 0) {
      initDeformBlur();
      st.dspRadial = dsp.r1;
      t4.vista.curTime = Infinity;          // Select_Vista_World rewind
      t4.renderer.setupClipping(0, W, 0, H, 1, 2000);
      t4.renderer.setupCamScreen(0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H);
      st.timeScale = 1; st.timeOff = 0;
      st.w2Target = 'vblurII';
    } else if (param === 1) {
      st.w2Target = 'screen';
    } else st.unported = true;
  }

  function renderCity() {
    const t4 = st.t4;
    t4.renderer.setupClipping(0, W, 0, H, 1, 3000);
    const { time, lx } = updateWorld(t4.world, t4.camNode, st.globalX, st.timeScale, st.timeOff, W);
    if (lx !== null) t4.renderer.rc.lx = lx;
    t4.renderer.render(t4.vista, time, B()[st.w2Target].bits);
    return time;
  }

  // The vista path at native resolution. Same PVS walk and the same world
  // time -- only the projection and the rasteriser target change.
  function renderCityNative(time) {
    const t4 = st.t4;
    nativePass(t4.renderer, (clip) => {
      nativeScreen.fill(0);
      clip(0, W, 0, H, 1, 3000);
      t4.renderer.render(t4.vista, time, nativeScreen);
    });
  }

  function loopW201() {                     // world2.c Loop_W2_01
    renderCity();
    let x = fr(1 - st.tickX);
    const amp = fr(fr(128 * x) * x);
    x = st.tickX; x = fr(x * x);
    deformBlur(amp, Math.trunc(fr(2 + fr(6 * x))), Math.trunc(fr(2 + fr(4 * x))));
  }

  function loopW203() {                     // world2.c Loop_W2_03
    const time = renderCity();
    scratchAndFlash(getBeat());
    getNextScratch(0x0e);
    mixSat16_8(B().screen, B().screen, B().scratch);
    if (native3d) { renderCityNative(time); mixSatNative(nativeScreen, B().scratch); }
  }

  // ---- "official sponsor" part (dsp.c Blur_II + part1.c scroller) --------
  //
  // NOTE ON DUPLICATE SYMBOLS: Extract_DMask_Raw_Col and Saturate_DMask_Pal
  // are each defined twice — in dem/fx1/dmask.c and dem/fx1/mmask.c. Only
  // mmask.c is in the build (Makefile FX1 list); dem/fx1/dmask.c is not
  // compiled at all. The mmask versions are byte-based and quirk-free; the
  // dmask.c ones read the 8-bit mask through a USHORT* and look like UB.
  function extractDMaskRawCol(mask, dst) {
    const m = (1 << (8 - mask.mBits)) - 1;
    for (let i = 0; i < mask.w * mask.h; i++) dst[i] = (mask.bits[i] & m) ? 0xff : 0x00;
  }

  function initBlurII() {                   // dsp.c Init_Blur_II
    selectVB1(W * H, true);                 // SELECT_CMAP_V(1): VIRTUAL
    B().vb1b.bits = blur012.bytes;          // VB(1) points AT the JPEG
    extractDMaskRawCol(st.curMask, B().vb3.bits);
  }

  // dsp.c Loop_Blur_II — three-stage crossfade from the blur plate, through
  // a mixed plate, to the masked anim.
  const T1 = fr(0.45), T2 = fr(0.60);
  function loopBlurII(x) {
    if (x < T1) {
      x = fr(x / T1);
      const A = Math.trunc(fr(x * 255));
      st.ramp(B().vb1b.cmap, 0, A, 0, 0, 0, 255, 255, 250, 0x777);
      st.ramp(B().vb1b.cmap, A, 256, 255, 255, 250, 255, 255, 250, 0x777);
      // writes VMASK's cmap as a plain 777 palette; nothing in this branch
      // reads it (the third branch rewrites it via DMask_Setup_Pal)
      B().vmask.cmap.u32.set(cmapTo777Fx(st.curMask.pal, st.curMask.nbCol, 0x000000, A ^ 0xff));
      setAnimCMap(230, 220, 180, 0x777);
      mixSat8_8(B().screen, B().vb1b, B().janim);
    } else if (x < T2) {
      x = fr(fr(x - T1) / fr(T2 - T1));
      const A = Math.trunc(fr(x * 255));
      st.ramp(B().vb2.cmap, 0, A, 0, 0, 0, 0, 0, 0, 0x777);
      st.ramp(B().vb2.cmap, A, 240, 0, 0, 0, 155, 155, 150, 0x777);
      st.ramp(B().vb2.cmap, 240, 256, 155, 155, 150, 255, 255, 250, 0x777);
      const Bv = A ^ 0xff;
      const offX = new Uint8Array(256), offY = new Uint8Array(256);
      for (let i = 0; i < 256; ++i) {
        offX[i] = (A * i) >> 8;
        offY[i] = (Bv * i) >> 8;
      }
      const src = B().vb3.bits, src2 = B().vb1b.bits, dst = B().vb2.bits;
      for (let i = 0; i < W * H; i++) dst[i] = (offY[src2[i]] + offX[src[i]]) & 0xff;
      setAnimCMap(230, 220, 180, 0x777);
      mixSat8_8(B().screen, B().vb2, B().janim);
    } else {
      x = fr(fr(x - T2) / fr(1 - T2));
      x = fr(1 - x);
      x = fr(1 - fr(x * x));
      const A = Math.trunc(fr(x * 255));
      setAnimCMap(230, 220, 180, 0x2565);
      dmaskSetupPal(B().vmask.cmap.u32, st.curMask,
        cmapTo777Fx(st.curMask.pal, st.curMask.nbCol, 0xdfdfdf, A));
      mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
    }
  }

  // scratch.c Scratch_And_Flash_II — two flash windows over a doubled x
  function scratchAndFlashII(x) {
    st.cOff = Math.trunc(fr(x * 4)) & 0x0f;
    x = x * 2.0;                            // FLT *= double
    if (x < 1.0 && x > 0.8) scratchNextRamp(Math.trunc(140.0 * (x - 0.8) / 0.2));
    else if (x > 1.8) scratchNextRamp(Math.trunc(240.0 * (x - 1.8) / 0.2));
    else scratchNextRamp(0);
  }

  // part1.c Paste_256 — additive 256-wide overlay, U wraps as a BYTE
  function paste256(dstOff, w, h, src, srcOff, u1, cmap) {
    const dst = B().screen.bits;
    // Paste_256's arithmetic IS Mix_Sat16_8 -- to16(lo0[p&255] + hi0[p>>8] +
    // cmap[s]) -- so the GPU reuses that effect and differs only in how the
    // 8-bit source is addressed: a 256-wide window wrapping horizontally.
    // The band is a few rows of the screen, so the pass is scissored.
    if (capturing) {
      cap.composite('mixSat16_8', { in16: 'screen', in8: '#scroll' }, {}, B().screen, {
        scroll: {
          bits: src, w: 256, cmap: Uint32Array.from(cmap),
          y0: Math.floor(dstOff / W), h,
          srcRow: Math.floor(srcOff / 256), u1: u1 & 0xff,
        },
      });
    }
    let s = srcOff;
    let d = dstOff;
    for (let j = 0; j < h; j++) {
      let U = u1 & 0xff;
      for (let i = 0; i < w; i++) {
        const p = dst[d + i];
        const s777 = T.lo[0][p & 0xff] + T.hi[0][p >> 8] + cmap[src[s + U]];
        U = (U + 1) & 0xff;
        dst[d + i] = to16(s777);
      }
      s += 256;
      d += W;
    }
    // Keep the mirror level with what the CPU just produced, exactly as the
    // mixers do -- otherwise the next composite sees a changed screen and
    // ships all 64000 pixels over.
    if (capturing) cap.out(B().screen);
  }

  // part1.c Init_Scroll / Scroll_1 — VB(4) is a VIRTUAL window into VSCREEN
  // (rows 5..53 for param 3), so the paste composites straight onto screen.
  // part1.c Init_Scroll — VB(4) is a VIRTUAL window into VSCREEN, so the
  // marquee composites straight onto the screen with no extra buffer.
  function initScroll(param) {
    st.jaParity = 0;
    // Param 2 ("blobz") is an EMPTY branch in the C — Init_Scroll does the
    // JA_Parity reset and the VB(4) select, then nothing. Treating it as a
    // missing case made the engine report `unported` from 122.8 s to the end
    // of the show (2411 of 5481 frames) even though the gate scores that
    // window bit-exact, so the browser status line and the ?debug readout
    // both claimed the last 45 % of a fully verified port was unimplemented.
    const win = param === 0 ? [0, 145, W, 49]
      : param === 1 ? [0, 0, 68, 200]
        : param === 3 ? [0, 5, W, 49] : null;
    if (param === 2) return;
    if (!win) { st.unported = true; return; }
    st.scroll = { off: win[1] * W + win[0], w: win[2], h: win[3] };
    st.ramp(B().vb4cmap, 0, 256, 0, 0, 0, 200, 220, 255, 0x777);
  }
  function scroll1() {
    if ((st.trandom.at(3, st.timer) & 0x07) === 0x03) {
      st.jaParity = 3 + (st.trandom.at(1, st.timer) & 0x03);
    }
    const sc = st.scroll;
    if (st.jaParity) {
      paste256(sc.off, sc.w, sc.h, cnt5.bytes, st.jaParity * 256, st.uScroll, B().vb4cmap.u32);
      scratchAndFlashII(getBeat());
      getNextScratch(0x08);
      mixSat16_8(B().screen, B().screen, B().scratch);
      st.jaParity--;
    } else {
      paste256(sc.off, sc.w, sc.h, cnt4.bytes, 7 * 256, st.uScroll, B().vb4cmap.u32);
    }
  }

  function loop14BlurI() {                  // part14.c Loop_14_Blur_I
    getBeat();
    if ((st.random() & 0x0f) === 0x05) st.jaParity = 5;
    pasteJpegAnim(true);
    if (st.tickX > fr(0.3)) loopBlurII(fr(fr(1 - st.tickX) / fr(0.7)));
    else loopBlurII(fr(st.tickX / fr(0.3)));
  }

  function loop14Powered() {                // part14.c Loop_14_Powered
    loop14BlurI();
    scroll1();
    st.uScroll = Math.trunc(st.globalX * 7.0 * 256.0) & 0xff;
  }

  // dsp.c Loop_Dsp_I — radial beat displacement. Init_Dsp_XY also builds a
  // Dsp_Off_Y table (row-stride steps) but the Y mixer is handed Dsp_Off_X;
  // the Y table is dead — quirk preserved by simply not building it.
  function loopDspI() {
    const x = getBeat2();
    if (x === 0) return;
    initDspXY2(Math.trunc(16 * x));         // (INT)(16.0*x), double product
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    dspSafeY(B().screen, B().vblurI, st.dspRadial.subarray(W * H), offX);
  }

  // dsp.c Loop_Dsp_III — beat displacement with a random radial choice per
  // pulse train; Dsp_Radial nulls between pulses and re-picks on re-entry.
  function loopDspIII() {
    const x = getBeat3();
    if (x === 0) { st.dspRadial = null; return; }
    if (st.dspRadial === null)
      st.dspRadial = (st.trandom.at(2, st.timer) & 0x08) ? dsp.r1 : dsp.r2;
    initDspXY2(Math.trunc(15 * x));         // (INT)(15.0*x)
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    dspSafeY(B().screen, B().vblurI, st.dspRadial.subarray(W * H), offX);
  }

  // dsp.c Init_Blur_I / Loop_Deform_Blur_I (the "disappear" wipe).
  function initBlurI(param) {
    initDeformBlur();
    B().vblurII.bits.set(B().screen.bits);  // Copy_Buffer_Short(VBLUR_II, VSCREEN)
    if (param === 0) st.dspRadial = dsp.r1;
    else if (param === 1) st.dspRadial = dsp.r2;
  }
  function loopDeformBlurI() {
    let x = fr(Math.cos(st.tickX * Math.PI));
    x = fr(1 - fr(x * x));
    const amp = fr(x * 28);
    x = fr(1 - st.tickX);
    x = fr(1 - fr(x * x));
    const i = Math.trunc(fr(6 + fr(9 * x)));
    const j = Math.trunc(fr(5 + fr(6 * x)));
    // Deform_Blur_II: displace VSCREEN, then Mix_16_16_Mix_Copy blends the
    // frozen VBLUR_II snapshot in and refreshes it with the result.
    initDspXY2(Math.trunc(amp));
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    dspSafeY(B().screen, B().vblurI, st.dspRadial.subarray(W * H), offX);
    mix16_16Mix(B().screen, B().vblurII, B().screen, T.lo[i], T.hi[i], T.lo[j], T.hi[j], true);
  }

  function setupMaskII(mask) {
    uncompressDMask(mask);
    B().vmask.bits = mask.bits;
    dmaskSetupPal(B().vmask.cmap.u32, mask, cmapTo777(mask.pal, mask.nbCol));
  }
  function saturateDMaskPal(x, mask, color) {
    let pal777;
    if (x >= 0.4) pal777 = cmapTo777(mask.pal, mask.nbCol);
    else if (x < 0.1) pal777 = cmapTo777Fx(mask.pal, mask.nbCol, 0x000000, Math.trunc(255 * x / 0.1));
    else pal777 = cmapTo777Fx(mask.pal, mask.nbCol, color, Math.trunc(255 * (0.4 - x) / 0.3));
    dmaskSetupPal(B().vmask.cmap.u32, mask, pal777);
  }
  function setAnimCMap(r, g, b, format = 0x2565) {
    st.ramp(B().janim.cmap, 0, 256, 0, 0, 0, r, g, b, format);
  }
  function setupAnim(anim, w, h) {
    st.curAnim = anim; st.curAnimW = w; st.curAnimH = h;
    st.animCnt = 0;
    st.curAnimSize = w * h;
    st.animMaxCnt = anim ? Math.trunc((anim.w * anim.h) / st.curAnimSize) : 0;
  }

  // ---- lyric text system (dem/src/fonts.c, fonts2.c, demo_src/sprite.c) --

  const frv = Math.fround;

  // Setup_Paste_Sprite + Paste_Sprite_Asm_8* (sprite.c): scaled blit via a
  // 16.16 DDA, float setup with ceil edges, clipped to st.clips.
  // mode: 'set8' (byte dst), 'map16' (565 via cmap), 'or16' (565 OR).

  function pasteSprite(dst, dstIsU16, mode, cmapU16, xo, yo, R, sp) {
    const [c0, c1, c2, c3] = st.clips;
    let tmp = frv(frv(0.5 * R) * sp.dU);
    const xi = frv(xo - tmp);
    let x = xi, xf = frv(xo + tmp);
    tmp = frv(frv(0.5 * R) * sp.dV);
    const yi = frv(yo - tmp);
    let y = yi, yf = frv(yo + tmp);
    // Capture the UNCLIPPED rect with its source window, as the particle path
    // does: the sprite maps its source linearly across the rect, so pairing a
    // clipped rect with unclipped UVs would stretch the glyph. GL clips it.
    //
    // Unlike the particles this is NOT an additive blend -- `set8` and
    // `map16` overwrite and `or16` ORs -- so these cannot accumulate into one
    // buffer. They are recorded so the mode and the overlap can be measured
    // before a drawing path is designed for them.
    if (capturing) {
      // A glyph addresses its sheet through bitsOff, a LINEAR offset, while
      // Uo/Vo are always 0 -- Print_Letter packs the whole font into one GIF
      // and points at each glyph by byte position. Convert to the 2-D window
      // the GPU needs, or every glyph samples from the sheet's origin and
      // the typography renders as the same letter repeated.
      const u0 = sp.bitsOff % sp.BpS, v0 = Math.floor(sp.bitsOff / sp.BpS);
      cap.glyph(xi, yi, frv(xo + frv(frv(0.5 * R) * sp.dU)), yf,
        u0, v0, u0 + sp.dU, v0 + sp.dV,
        { bits: sp.bits, w: sp.BpS, convert: cmapU16 }, mode, dst);
    }
    if (xf <= c0 || x > c1 || yf <= c2 || y > c3) return;
    if (xf > c1) xf = c1;
    if (yf > c3) yf = c3;
    if (x < c0) x = c0;
    if (y < c2) y = c2;
    R = frv(1 / R);

    const X = Math.ceil(x);
    let U = frv(sp.Uo + frv(frv(X - xi) * R));
    let dF = Math.trunc(R * 65536.0);
    const dErr = dF & 0xffff;
    dF >>= 16;
    let F = Math.trunc(U * 65536.0);
    let Err = F & 0xffff;
    F >>= 16;
    const stepsX = Math.ceil(xf) - X;
    if (stepsX <= 0) return;
    const offsets = new Int32Array(stepsX);
    for (let i = 0; i < stepsX; i++) {
      offsets[i] = F;
      F += dF; Err += dErr;
      if (Err & 0x10000) { Err &= 0xffff; F++; }
    }

    const Y = Math.ceil(y);
    let V = frv(sp.Vo + frv(frv(Y - yi) * R));
    F = Math.trunc(V * 65536.0);
    Err = F & 0xffff;
    F >>= 16;
    const stepsY = Math.ceil(yf) - Y;
    if (stepsY <= 0) return;
    let srcRow = sp.bitsOff + F * sp.BpS;
    const rows = new Int32Array(stepsY);
    const dFv = dF * sp.BpS;
    for (let i = 0; i < stepsY; i++) {
      rows[i] = srcRow;
      srcRow += dFv;
      Err += dErr;
      if (Err & 0x10000) { Err &= 0xffff; srcRow += sp.BpS; }
    }

    const src = sp.bits;
    for (let dh = 0; dh < stepsY; dh++) {
      const row = rows[dh];
      const dRow = (Y + dh) * W + X;
      for (let dw = 0; dw < stepsX; dw++) {
        const c = src[row + offsets[dw]];
        if (!c) continue;
        if (mode === 'set8') dst[dRow + dw] = c;
        else if (mode === 'map16') dst[dRow + dw] = cmapU16[c];
        else dst[dRow + dw] |= cmapU16[c];
      }
    }
  }

  // fonts.c Print_Letter (8-bit), _II (Map16_Or), _III (Map16)
  function printLetter(variant, dstBits, i, X, Y, R, F) {
    if (i === 255) return frv(R * F.avrg);
    const sp = {
      Uo: 0, Vo: 0, dU: F.fWidth[i], dV: F.gif.h,
      BpS: F.gif.w, bits: F.gif.bits, bitsOff: F.fPos[i],
    };
    const xc = frv(X + frv(sp.dU / 2));
    const yc = frv(Y + frv(sp.dV / 2));
    if (variant === 1) pasteSprite(dstBits, false, 'set8', null, xc, yc, R, sp);
    else if (variant === 2) pasteSprite(dstBits, true, 'or16', B().vletters.cmap.u16, xc, yc, R, sp);
    else pasteSprite(dstBits, true, 'map16', B().vletters.cmap.u16, xc, yc, R, sp);
    return frv(R * F.fWidth[i]);
  }

  // truecol.c CMap_To_16bits, 0x2565 target (u16 entries) — used by the
  // letter cmaps.
  function cmapTo565(cmap, pal, nbCol) {
    const n = Math.min(nbCol, 256);
    for (let i = 0; i < n; i++) cmap.u16[i] = rgbTo565(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]);
  }

  // fonts2.c Split_Text / Store_Words_Time / Advance_Word
  function splitText(text) {
    st.words = []; st.wordsL = [];
    const sLen = text.length + 1;      // C scans through the NUL
    const at = (i) => (i < text.length ? text[i] : '\0');
    let i = 0;
    while (i < sLen) {
      while (at(i) === ' ') { if (++i === sLen) break; }
      if (i >= sLen) break;
      const start = i;
      let j = 0;
      while (at(i) !== ' ') { if (++i === sLen) break; j++; }
      st.words.push(text.substr(start, j));
      st.wordsL.push(j);
      if (st.words.length === 16) break;
    }
  }
  function storeWordsTime() {
    st.wordsS = [];
    const n = st.words.length;
    for (let i = 0; i <= n; i++) st.wordsS.push(frv(i / n));
  }
  function advanceWord(tickX) {
    while (tickX < st.wordsS[st.curWord] || tickX >= st.wordsS[st.curWord + 1]) {
      if (st.curWord < st.words.length) st.curWord++;
      else break;
    }
  }

  // fonts2.c Print_Word_3 (Map16_Or into a 565 buffer)
  function printWord3(dstBits, tickX, F, randAmp) {
    let R = frv(st.curEv.scale);
    R = frv(R * frv(1 + frv(frv(frv(frv(randAmp * (st.trandom.at(0, st.timer) & 7)) * frv(1 - tickX))) / 16)));
    let X, Y;
    if (st.curEv.x !== 0) { X = st.curEv.x; Y = st.curEv.y; }
    else { X = 0.5 * W; Y = frv(0.5 * H - frv(R * 10)); }
    Y = frv(Y - (st.trandom.at(1, st.timer) & 3));
    X = frv(X - frv(frv(R * F.avrg) * st.wordsL[st.curWord]));
    st.clips = [0, W, 0, H];
    cmapTo565(B().vletters.cmap, F.gif.pal, F.gif.nbCol);
    const word = st.words[st.curWord];
    for (let i = 0; i < st.wordsL[st.curWord]; i++) {
      const C = F.charMap[word.charCodeAt(i) & 0x7f];
      X = frv(X + printLetter(2, dstBits, C, X, Y, R, F));
    }
  }

  // Font_FX[3] = { Init_Fx3(Store_Words_Time), NULL, Post_Fx3 }
  function postFx3(tickX) {
    if (st.curWord === st.words.length) return 0;
    if ((st.trandom.at(2, st.timer) & 0x03) === 0)
      printWord3(B().screen.bits, tickX, fonts[2], 1.0);
    else printWord3(B().screen.bits, tickX, fonts[3], 1.0);
    advanceWord(tickX);
    return 1;
  }

  // fonts2.c Print_Word_2 — 8-bit letters into VLETTERS (cleared), cmap in
  // 0x777 for the Mix_Sat8_8 blend with the scratch layer.
  function printWord2(tickX, F, randAmp) {
    let R = frv(st.curEv.scale);
    R = frv(R * frv(1 + frv(frv(frv(frv(randAmp * (st.trandom.at(0, st.timer) & 7)) * frv(1 - tickX))) / 16)));
    let X, Y;
    if (st.curEv.x !== 0) { X = st.curEv.x; Y = st.curEv.y; }
    else { X = frv(0.5 * W); Y = frv(0.5 * H - frv(R * 10)); }
    Y = frv(Y - (st.trandom.at(1, st.timer) & 7));
    X = frv(X - frv(frv(R * F.avrg) * st.wordsL[st.curWord]));
    // VBuffer_Clear(VLETTERS). Captured as a full-layer solid quad: without
    // it the clear's pixels change OUTSIDE every glyph rect, the coverage
    // test refuses the range, and the whole Loop_14_* family falls back to
    // shipping the CPU's 320x200 letters -- which is what it did.
    if (capturing) {
      cap.glyph(0, 0, W, H, 0, 0, 1, 1, { solid: 0 }, 'solid',
        B().vletters.bits);
    }
    B().vletters.bits.fill(0);
    st.clips = [0, W, 0, H];
    B().vletters.cmap.u32.set(cmapTo777(F.gif.pal, F.gif.nbCol));
    const word = st.words[st.curWord];
    for (let i = 0; i < st.wordsL[st.curWord]; i++) {
      const C = F.charMap[word.charCodeAt(i) & 0x7f];
      X = frv(X + printLetter(1, B().vletters.bits, C, X, Y, R, F));
    }
  }

  // fonts2.c Letter_Scratch — gray ramp into the scratch cmap + advance.
  function letterScratch() {
    const c1 = (st.trandom.at(0, st.timer) & 0x17) + 70;
    const c2 = st.trandom.at(1, st.timer) & 0x07;
    st.ramp(B().scratch.cmap, 0, 8, c1, c1, c1, c2, c2, c2, 0x777);
    getNextScratch(0x17);
  }

  // mix4.c Mix_Sat8_8 — two 8-bit sources through their 777 cmaps, packed.
  function mixSat8_8(out, in1, in2) {
    cap.composite('mixSat8_8', { in1: in1._layer, in2: in2._layer },
      { off2: in2.off || 0 }, out);
    const dst = out.bits, s1 = in1.bits, s2 = in2.bits;
    const cm1 = in1.cmap.u32, cm2 = in2.cmap.u32;
    const off2 = in2.off || 0;
    for (let i = 0; i < W * H; i++) dst[i] = to16(cm1[s1[i]] + cm2[s2[off2 + i]]);
    cap.out(out);
  }

  // Font_FX[1] = { Init_Fx2(Store_Words_Time), Do_Fx2, NULL }
  function doFx2(tickX) {
    const F = fonts[st.trandom.at(2, st.timer) & 0x01];  // arg eval first
    if (st.curWord === st.words.length) return 0;
    printWord2(tickX, F, 1.0);
    advanceWord(tickX);
    letterScratch();
    mixSat8_8(B().screen, B().vletters, B().scratch);
    return 1;
  }

  // Font_FX[8] = { Init_Fx7, NULL, Post_Fx7 } — the marquee scroller.
  function initFx7() {
    st.sLen = (st.curEv.text.length + 1) * fonts[5].avrg + W;   // INT math
  }
  function printWord4(dstBits, F, X) {      // fonts2.c Print_Word_4
    const R = frv(st.curEv.scale);
    let Y = st.curEv.y !== 0 ? st.curEv.y : 5;
    Y = frv(Y - frv(frv(F.gif.h * R) / 2));
    const text = st.curEv.text;
    for (let i = 0; i < text.length; i++) {
      const C = text[i] === '_' ? 255 : F.charMap[text.charCodeAt(i) & 0x7f];
      X = frv(X + printLetter(3, dstBits, C, X, Y, R, F));
    }
  }
  function postFx7(tickX) {
    st.clips = [0, W, 0, H];
    cmapTo565(B().vletters.cmap, fonts[5].gif.pal, fonts[5].gif.nbCol);
    printWord4(B().screen.bits, fonts[5], frv(W - frv(frv(frv(1.3) * tickX) * st.sLen)));
    return 1;
  }

  // truecol.c CMap_To_16bits, 0x4565 target (565 values in u32 entries) —
  // what Print_Word_1 builds for VBuf_Map_8_Bits to read.
  function cmapTo565U32(cmap, pal, nbCol) {
    const n = Math.min(nbCol, 256);
    for (let i = 0; i < n; i++) cmap.u32[i] = rgbTo565(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]);
  }

  // vbuf.c VBuf_Map_8_Bits — 8-bit source through a u32 cmap into 565.
  function vbufMap8Bits(out, in8) {
    const dst = out.bits, src = in8.bits, cm = in8.cmap.u32;
    for (let i = 0; i < W * H; i++) dst[i] = cm[src[i]] & 0xffff;
  }

  // fonts2.c Print_Word_1 — one word, 8-bit, into a cleared VLETTERS.
  function printWord1(wordIdx, F) {
    let X, Y;
    if (st.curEv.x !== 0) { X = st.curEv.x; Y = st.curEv.y; }
    else { X = 0.5 * W; Y = 0.5 * H - 10; }
    const R = frv(st.curEv.scale);
    X = frv(X - frv(frv(R * F.avrg) * st.wordsL[wordIdx]));
    B().vletters.bits.fill(0);
    st.clips = [0, W, 0, H];
    cmapTo565U32(B().vletters.cmap, F.gif.pal, F.gif.nbCol);
    const word = st.words[wordIdx];
    for (let i = 0; i < st.wordsL[wordIdx]; i++) {
      const C = F.charMap[word.charCodeAt(i) & 0x7f];
      X = frv(X + printLetter(1, B().vletters.bits, C, X, Y, R, F));
    }
  }

  // Font_FX[0] = { Init_Fx1(Store_Words_Time), Do_Fx1, NULL }
  function doFx1(tickX) {
    if (st.curWord >= st.words.length) st.curWord = st.words.length - 1;
    const F = (st.trandom.at(2, st.timer) & 0x03) === 0 ? fonts[2] : fonts[3];
    printWord1(st.curWord, F);
    advanceWord(tickX);
    vbufMap8Bits(B().screen, B().vletters);
    return 1;
  }

  // Font_FX[2] = Do_Fx22 — same body as fx1's Do_Fx2 but the other font pair
  function doFx22(tickX) {
    const F = fonts[2 + (st.trandom.at(2, st.timer) & 0x01)];
    if (st.curWord === st.words.length) return 0;
    printWord2(tickX, F, 1.0);
    advanceWord(tickX);
    letterScratch();
    mixSat8_8(B().screen, B().vletters, B().scratch);
    return 1;
  }

  // Font_FX[7] = Do_Fx23 — Print_Word_3 straight into a cleared VSCREEN
  function doFx23(tickX) {
    if (st.curWord === st.words.length) return 0;
    B().screen.bits.fill(0);
    printWord3(B().screen.bits, tickX, fonts[st.trandom.at(1, st.timer) & 0x01], frv(0.1));
    advanceWord(tickX);
    return 1;
  }

  // fonts2.c Print_Words_Base (fx4: 8-bit into VLETTERS, ':' ends a block)
  function printWordsBase(F, Xo, Y, Ro, spc) {
    cmapTo777(F.gif.pal, F.gif.nbCol).forEach((v, i) => { B().vletters.cmap.u32[i] = v; });
    for (; st.curWord < st.words.length; ++st.curWord) {
      let X = Xo;
      if (st.words[st.curWord][0] === ':') { st.curWord++; break; }
      const word = st.words[st.curWord];
      for (let i = 0; i < st.wordsL[st.curWord]; i++) {
        const ch = word.charCodeAt(i) & 0x7f;
        const C = word[i] === '_' ? 255 : F.charMap[ch];
        if (C === 255) { X = frv(X + fonts[0].avrg); continue; }
        printLetter(1, B().vletters.bits, C, X, Y, Ro, F);
        X = frv(X + spc[C]);
      }
      Y = frv(Y + 32);
    }
    return Y;
  }

  // Font_FX[4] = { Init_Fx4, NULL, Post_Fx4 }. Init snapshots VSCREEN into
  // VLETTERS_II exactly ONCE for the whole run (a C function-static).
  function initFx4() {
    if (!st.fx4Snapshot) {
      B().vlettersII.bits.set(B().screen.bits);
      st.fx4Snapshot = true;
    }
    B().vletters.bits.fill(0);
  }
  function postFx4(tickX) {
    st.clips = [0, W, 0, H];
    letterScratch();
    { // Dst = (Dst>>1) + 3 - (Src>>1), 8-bit wrap
      const d = B().vletters.bits, s = B().scratch.bits, off = B().scratch.off || 0;
      for (let i = 0; i < W * H; i++) d[i] = ((d[i] >> 1) + 3 - (s[off + i] >> 1)) & 0xff;
    }
    let R = frv(st.curEv.scale);
    let Xo, Y;
    if (st.curEv.x !== 0) { Xo = st.curEv.x; Y = st.curEv.y; }
    else { Xo = frv(0.5 * W); Y = frv(0.5 * H - frv(R * 10)); }
    let F;
    if ((st.trandom.at(2, st.timer) & 0x03) === 0) {
      F = fonts[2];
      R = frv(R * frv(1 + frv(frv(frv(frv(0.08) * (st.trandom.at(0, st.timer) & 7)) * frv(1 - tickX)) / 16)));
    } else F = fonts[3];
    Y = frv(Y - frv(frv(F.gif.h * R) * frv(0.5)));
    Y = frv(Y - frv(frv(0.1) * (st.trandom.at(1, st.timer) & 0x07)));
    st.curWord = 0;
    printWordsBase(F, Xo, Y, R, fonts[2].fWidth);
    mixSat16_8(B().screen, B().vlettersII, B().vletters);
    return 1;
  }

  // fonts2.c Print_Words_Base_II + Print_Words_II (fx5)
  function printWordsBaseII(F, Xo, Y, Ro, spc) {
    for (; st.curWord < st.words.length; ++st.curWord) {
      let X = Xo;
      if (st.words[st.curWord][0] === ':') { st.curWord++; break; }
      const word = st.words[st.curWord];
      for (let i = 0; i < st.wordsL[st.curWord]; i++) {
        const C = word[i] === '_' ? 255 : F.charMap[word.charCodeAt(i) & 0x7f];
        if (C === 255) { X = frv(X + fonts[0].avrg); continue; }
        printLetter(2, B().screen.bits, C, X, Y, Ro, F);
        X = frv(X + spc[C]);
      }
      Y = frv(Y + 32);
    }
    return Y;
  }
  function printWordsII() {
    const Ro = frv(st.curEv.scale);
    let Xo, Yo;
    if (st.curEv.x !== 0) { Xo = st.curEv.x; Yo = st.curEv.y; }
    else { Xo = frv(0.5 * W); Yo = frv(0.5 * H - frv(Ro * 10)); }
    let F, dY;
    if ((st.trandom.at(2, st.timer) & 0x07) === 0) { F = fonts[1]; dY = st.trandom.at(1, st.timer) & 0x03; }
    else { F = fonts[0]; dY = 0; }
    Yo = frv(Yo + dY);
    cmapTo565(B().vletters.cmap, F.gif.pal, F.gif.nbCol);
    st.curWord = 0;
    Yo = printWordsBaseII(F, Xo, Yo, Ro, fonts[0].fWidth);
    Yo = frv(Yo + frv(32 - dY));
    F = (st.trandom.at(1, st.timer) & 0x03) === 0 ? fonts[2] : fonts[3];
    Yo = frv(Yo - frv(frv(F.gif.h * Ro) * frv(0.5)));
    cmapTo565(B().vletters.cmap, F.gif.pal, F.gif.nbCol);
    for (; st.curWord < st.words.length; ++st.curWord) {
      let X = Xo;
      const word = st.words[st.curWord];
      for (let i = 0; i < st.wordsL[st.curWord]; i++) {
        const C = word[i] === '_' ? 255 : F.charMap[word.charCodeAt(i) & 0x7f];
        if (C === 255) { X = frv(X + fonts[3].avrg); continue; }
        printLetter(2, B().screen.bits, C, X, Yo, Ro, F);
        X = frv(X + (fonts[3].fWidth[C] - 3));
      }
      Yo = frv(Yo + 32);
    }
    letterScratch();
    mixSat16_8(B().screen, B().screen, B().scratch);
  }
  function initFx5() { st.clips = [0, W, 0, H]; }
  function doFx5() {
    B().screen.bits.fill(0);
    printWordsII();
    return 1;
  }

  // Font_FX[6] = { Init_Fx6, NULL, Post_Fx6 } — per-character reveal with a
  // brightness ramp on the newest glyph; ':' starts a new line.
  function initFx6() {
    st.clips = [0, W, 0, H];
    st.sLen = st.curEv.text.length + 1;
  }
  function postFx6(tickX) {
    const F = fonts[4];
    let xo = frv(tickX * st.sLen);
    const E = Math.floor(xo);
    let ro = frv(xo - frv(1.0 * E));
    const Co = ((Math.trunc(126 * ro) ^ 0x7f) + 0x50);
    cmapTo565(B().vletters.cmap, F.gif.pal, F.gif.nbCol);
    ro = frv(st.curEv.scale);
    let Xo, Yo;
    if (st.curEv.x !== 0) { Xo = st.curEv.x; Yo = st.curEv.y; }
    else { Xo = frv(0.08 * W); Yo = frv(0.5 * H - frv(ro * 10)); }
    let X = Xo;
    const text = st.curEv.text;
    for (let i = 0; i < E; ++i) {
      const ch = text.charCodeAt(i) & 0x7f;
      let C;
      if (text[i] === '_') C = 255;
      else if (text[i] === ':') { Yo = frv(Yo + 30); X = Xo; continue; }
      else C = F.charMap[ch];
      if (C === 255) { X = frv(X + F.avrg); continue; }
      if (i === E - 1) st.ramp(B().vletters.cmap, 0, F.gif.nbCol, Co, Co, Co, 255, 255, 255, 0x2565);
      printLetter(2, B().screen.bits, C, X, Yo, ro, F);
      X = frv(X + F.fWidth[C]);
    }
    return 1;
  }

  // fonts2.c FX dispatch table (Font_FX[] at fonts2.c:638)
  const FONT_FX = {
    0: { init: storeWordsTime, do: doFx1, post: null },
    1: { init: storeWordsTime, do: doFx2, post: null },
    2: { init: storeWordsTime, do: doFx22, post: null },
    3: { init: storeWordsTime, do: null, post: postFx3 },
    4: { init: initFx4, do: null, post: postFx4 },
    5: { init: initFx5, do: doFx5, post: null },
    6: { init: initFx6, do: null, post: postFx6 },
    7: { init: storeWordsTime, do: doFx23, post: null },
    8: { init: initFx7, do: null, post: postFx7 },
  };

  // fonts2.c Bypass_I / Bypass_II / Post_FX
  function bypassI(timer) {
    if (st.curText === -1 || st.curText >= ffxEvents.length) return 0;
    const ev = () => ffxEvents[st.curText];
    while (timer > ev().start + ev().len) {
      st.curText++;
      if (st.curText >= ffxEvents.length) { st.curText = -1; return 0; }
    }
    if (timer < ev().start) return 0;
    st.curEv = ev();
    splitText(st.curEv.text);
    st.curWord = 0;
    st.fxNb = st.curEv.fx;
    const fx = FONT_FX[st.fxNb];
    if (!fx) { st.unported = true; st.curText++; return 0; }
    if (fx.init) fx.init();
    st.postLoop = null;
    st.bypassLoop = null;
    if (fx.post) {
      st.postLoop = 'Post_FX';
      return 0;
    }
    st.bypassLoop = 'Bypass_II';
    return bypassII(st.curEv.start);
  }
  function bypassII(timer) {
    let tickX = frv((timer - st.curEv.start) / st.curEv.len);
    if (timer > st.curEv.start + st.curEv.len) {
      st.bypassLoop = 'Bypass_I';
      tickX = 1.0;
    }
    return FONT_FX[st.fxNb].do(tickX);
  }
  function postFX(timer) {
    let tickX = frv((timer - st.curEv.start) / st.curEv.len);
    if (timer > st.curEv.start + st.curEv.len) {
      st.bypassLoop = 'Bypass_I';
      st.postLoop = null;
      tickX = 1.0;
    }
    FONT_FX[st.fxNb].post(tickX);
  }
  const BYPASS = { Bypass_I: bypassI, Bypass_II: bypassII };

  // ---- part loops (dem/src/part14.c, dsp.c) ------------------------------

  function loop14AnimMask() {
    pasteJpegAnim(false);
    scratchAndFlash(getBeat());
    mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  function loop14AnimMaskRaw() {           // part14.c Loop_14_Anim_Mask_Raw
    scratchAndFlash(getBeat());
    pasteJpegAnim(false);
    mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
  }

  // janim.c Paste_JPEG_Anim_Sprites — 4x4 source blocks averaged to a
  // luminance index, each drawn as an 8x8 tile from the baked sprite table
  // (dem/fx1/sprs2.h, extracted to unpacked/sprites.json).
  function pasteJpegAnimSprites() {
    if (!st.curAnim) return;
    const bits = st.curAnim.bytes;
    const w = st.curAnimW;                  // 160
    let src = st.curAnimSize * st.animCnt + w;
    const d = B().janim.bits;
    let dst1 = 2 * w;
    for (let H = st.curAnimH; H > 0; H -= 4) {
      for (let i = -w; i < 0; i += 4) {
        let C = bits[src + i] + bits[src + i + 1] + bits[src + i + 2] + bits[src + i + 3];
        C += bits[src + w + i] + bits[src + w + i + 1] + bits[src + w + i + 2] + bits[src + w + i + 3];
        C += bits[src + 2 * w + i] + bits[src + 2 * w + i + 1] + bits[src + 2 * w + i + 2] + bits[src + 2 * w + i + 3];
        C += bits[src + 3 * w + i] + bits[src + 3 * w + i + 1] + bits[src + 3 * w + i + 2] + bits[src + 3 * w + i + 3];
        C = Math.floor(C / (16 * 16));
        let sp = C * 64;
        let dst = dst1;
        for (let row = 0; row < 8; row++) {
          for (let k = 0; k < 8; k++) d[dst + 2 * i + k] = sprites[sp++];
          dst += W;
        }
      }
      dst1 += 8 * W;
      src += 4 * w;
    }
  }

  // mmask.c Paste_DMask_Move_II — the LIVE definition. An earlier `#if 0`
  // block in the same file holds a trivial version that just forwards to
  // Mix_8_8_Blend_Mask; that one is dead code and reading it instead costs a
  // debugging leg (the non-II mixer reads an 8-bit mask as USHORT*, which
  // renders the mask doubled). The live one jitters mask+anim against the
  // screen by a random -2..1 scanlines and runs the ordinary
  // Mix_8_8_Blend_Mask_II over the overlapping band. Quirks preserved:
  // Off_X is computed and never used, and the TRandom slots are drawn
  // 1-then-0 — order matters, they share one LCG.
  function pasteDMaskMoveII(out, mask, anim) {
    st.trandom.at(1, st.timer);             // Off_X: drawn, then discarded
    const offY = (st.trandom.at(0, st.timer) & 0x03) - 2;
    let outOff = 0, srcOff = 0, h;
    if (offY > 0) { outOff = offY * W; h = H - offY; }
    else { srcOff = -offY * W; h = H + offY; }
    mix8_8BlendMaskII(out, mask, anim, outOff, srcOff, h);
  }

  function loop14AnimMaskMoveII() {        // part14.c
    const x = getBeat();
    scratchAndFlash(x);
    if ((st.random() & 0x1f) === 0x0a) st.jaParity = 16;
    pasteJpegAnim(true);
    if (st.jaParity > 5) {
      const t = fr((st.jaParity - 6) / fr(15 - 6));
      let col = Math.trunc(0x80 * (1 - t) + 0xe0 * t) << 16;
      col |= Math.trunc(0xa0 * (1 - t) + 0xe0 * t) << 8;
      col |= Math.trunc(0xc2 * (1 - t) + 0xf0 * t);
      saturateDMaskPal(fr(0.3), st.curMask, col >>> 0);
      pasteDMaskMoveII(B().screen, B().vmask, B().janim);
    } else {
      dmaskSetupPal(B().vmask.cmap.u32, st.curMask, cmapTo777(st.curMask.pal, st.curMask.nbCol));
      mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
    }
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  function loop14AnimMaskMoveSprites() {   // part14.c
    const x = getBeat();
    scratchAndFlash(x);
    pasteJpegAnimSprites();
    saturateDMaskPal(x, st.curMask, 0xc8c8c8);
    mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
    loopDspI();
  }

  function loop14AnimMaskMove() {
    const x = getBeat();
    scratchAndFlash(x);
    if ((st.random() & 0x1f) === 0x05) st.jaParity = 5;
    pasteJpegAnim(true);
    saturateDMaskPal(x, st.curMask, 0xc8c8c8);
    mix8_8BlendMaskII(B().screen, B().vmask, B().janim);
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  function loopDeformBlurII() {             // dsp.c:414 — Parental advisory
    const fr = Math.fround;
    const x = fr(1 - st.tickX);
    B().vblurII.bits.set(B().vb1.bits);     // Copy_Buffer_Short(VBLUR_II, VB1)
    deformBlur(fr(x * 90), Math.trunc(fr(4 + fr(8 * x))), Math.trunc(fr(4 + fr(9 * x))));
    if (st.tracePremix) st.tracePremix = B().screen.bits.slice();
    scratchAndFlash(getBeat());
    getNextScratch(0x08);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  function doDissolve() {                   // dsp.c:280
    // (INT)(9.0f + Tick_x*3.0): the 3.0 literal is double, so this pair is
    // double math over the float Tick_x; Amp = Tick_x*50.0f is float.
    const i = Math.trunc(9 + st.tickX * 3);
    const j = Math.trunc(7 + st.tickX * 3);
    initDspXY2(Math.trunc(Math.fround(st.tickX * 50)));
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    mix16_16Mix(B().screen, B().vblurI, B().screen, T.lo[i], T.hi[i], T.lo[j], T.hi[j], false);
  }

  // ---- phase inits (dem/src/parts.c Init_00, dsp.c Init_Dissolve) --------

  function init00(param) {
    if (param === 0) {                      // Parental I
      st.timer0 = st.timer;
      st.nextBeat = st.timer0 + 1924;
      st.curMask = masks.advisory;
      setupMaskII(st.curMask);
      setupAnim(anims.anim32, 160, 100);
      setAnimCMap(180, 240, 160);
      initDeformBlur();
      st.dspRadial = dsp.r1;
      selectVB1(2 * W * H, false);          // SELECT_565(1): sets the mark
      loop14AnimMask();                     // one priming frame at init
      B().vb1.bits.set(B().screen.bits);    // Copy_Buffer_Short(VB1, VSCREEN)
      B().screen.bits.fill(0);              // VBuffer_Clear(VSCREEN)
    } else if (param === 1) {               // work-in
      st.curMask = masks.work;
      setupMaskII(st.curMask);
      setupAnim(anims.anim32, 160, 100);
      setAnimCMap(180, 220, 160);
      initAppear(1);
      B().vblurII.bits.set(B().screen.bits);
      initDeformBlur();                     // Init_Dsp(0), "for safety"
    } else if (param === 2) {               // work-2
      st.curMask = masks.work;
      setupMaskII(st.curMask);
      setupAnim(anims.anim32, 160, 100);
      setAnimCMap(180, 220, 160);
    } else if (param === 3) {               // buy-in
      st.curMask = masks.buy;
      setupMaskII(st.curMask);
      setupAnim(anims.anim12, 160, 100);
      setAnimCMap(220, 210, 170);
      initAppear(1);
      B().vblurII.bits.set(B().screen.bits);
    } else if (param === 7) {               // official sponsor
      st.curMask = masks.sponsor;
      setupMaskII(st.curMask);
      setupAnim(anims.anim32, 160, 100);    // Set_Anim_CMap is commented out
      initBlurII();
      initScroll(3);
    } else if (param === 8) {               // Welcome(c)...
      st.curMask = masks.welcome;
      setupMaskII(st.curMask);
      setupAnim(anims.anim12, 160, 100);
      setAnimCMap(220, 220, 220);
    } else if (param === 9) {               // disappear
      initAppear(2);
      B().vblurII.bits.set(B().screen.bits);
    } else if (param === 10) {              // consume-in
      st.curMask = masks.consume;
      setupMaskII(st.curMask);
      setupAnim(anims.anim32, 160, 100);
      setAnimCMap(180, 220, 160);
      st.jaParity = 0;
    } else if (param === 11) {              // die-in
      st.curMask = masks.die;
      setupMaskII(st.curMask);
      setupAnim(anims.anim12, 160, 100);
      setAnimCMap(240, 220, 150);
      st.jaParity = 0;
    } else st.unported = true;
  }

  function initDissolve(param) {
    st.dspRadial = dsp.r1;
    satTable5(st.dspSatX, 0, W - 1, 1, 256, W - 1, 257);
    if (param === 1) st.dspParity = 0;
    else if (param === 2) st.dspParity = 1;
  }

  // ---- tunnel (dem/src/part6.c) ------------------------------------------

  function init01(param) {
    const t6 = st.t6;
    if (param === 1) {                    // "a world of technology"
      t6.vista.curTime = Infinity;        // Init_6/Select_Vista_World rewind
      t6.renderer.setupClipping(0, W, 0, H, fr(0.2), 0);
      t6.renderer.setupCamScreen(0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H);
      st.curMask = masks.techno;
      setupMaskII(st.curMask);
      st.timeScale = fr(0.85); st.timeOff = 0;
    } else if (param === 2) {             // tunnel + beat
      st.curMask = null;
      initDeformBlur();                   // Init_Dsp(0): same two tables
    } else if (param === 3) { st.curMask = masks.join; setupMaskII(st.curMask); }
    else if (param === 4) { st.curMask = masks.modern; setupMaskII(st.curMask); }
    else if (param === 6) { st.curMask = masks.improve; setupMaskII(st.curMask); }
    else if (param === 7) { st.curMask = masks.kiss; setupMaskII(st.curMask); }
    else if (param === 8) {               // "hot stuff inside"
      st.timer0 = st.timer;               // beat anchor reset
      st.nextBeat = st.timer0 + 1924;
      st.timeScale = fr(0.32); st.timeOff = fr(0.2);
      st.curMask = masks.upgrade;
      setupMaskII(st.curMask);
      initDeformBlur();                   // Init_Dsp(0)
    } else st.unported = true;
  }

  function loop602() {                    // part6.c Loop_6_02
    const t6 = st.t6;
    t6.renderer.setupClipping(25, 300, 20, H - 20, fr(0.2), 0);
    const { time, lx } = updateWorld(t6.world, t6.camNode, st.globalX, st.timeScale, st.timeOff, W);
    if (lx !== null) t6.renderer.rc.lx = lx;
    t6.renderer.render(t6.vista, time, B().screen.bits);
    if (st.trace62) st.trace62 = { screen: B().screen.bits.slice(), time, lx, nbPoly: t6.renderer.stats.selected };
    saturateDMaskPal(getBeat(), st.curMask, 0xc4c0c2);
    mix16_8BlendMaskII(B().screen, B().vmask, B().screen);
    if (native3d) {
      nativePass(t6.renderer, (clip) => {
        nativeScreen.fill(0);
        clip(25, 300, 20, H - 20, fr(0.2), 0);
        t6.renderer.render(t6.vista, time, nativeScreen);
      });
      blendMaskNative(nativeScreen, B().vmask);
    }
  }

  // Tables the flare paste blends through (flare16.c RGB_16_TO_777 /
  // RGB_777_TO_16 use the level-0 fade tables + the saturating pack).
  const FLARE_TABLES = { to16, lo777: T.lo[0], hi777: T.hi[0] };

  // Shared body of Loop_6_06 / Loop_6_08: zbuf shader on, full-screen clip,
  // render with the z second pass, then the lens flares (Render_World runs
  // Render_Light_Flare after Renderer.Render).
  function renderTunnelZbufFlares() {
    const t6 = st.t6;
    t6.renderer.setZBuffer(t6.zbBase);
    t6.renderer.setupClipping(0, W, 0, H, fr(0.2), 0);
    const { time, lx } = updateWorld(t6.world, t6.camNode, st.globalX, st.timeScale, st.timeOff, W);
    if (lx !== null) t6.renderer.rc.lx = lx;
    t6.renderer.render(t6.vista, time, B().screen.bits);
    t6.renderer.renderLightFlare(t6.lights, B().screen.bits, FLARE_TABLES);
    if (st.trace62) st.trace62 = { screen: B().screen.bits.slice(), time, lx, nbPoly: t6.renderer.stats.selected };
    return t6;
  }

  function loop606() {                    // part6.c Loop_6_06 (tunnel + bump)
    const t6 = renderTunnelZbufFlares();
    t6.renderer.setZBuffer(null);
    scratchAndFlash(getBeat());
    getNextScratch(0x0e);
    mixSat16_8(B().screen, B().screen, B().scratch);
    loopDspI();
  }

  // ---- world3: the statue (dem/src/world3.c) -----------------------------
  //
  // Uses the engine's DEFAULT (non-vista) render path: unlike part6 this
  // part never assigns Renderer.Render = Render_Vista, so the scene-graph
  // walk in raster3d.renderScene applies.

  function initWorld3(param) {
    const t5 = st.t5;
    t5.renderer.setupClipping(0, W, 0, H, 10, 0);
    t5.renderer.setupCamScreen(0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H);
    st.timeScale = fr(1.08); st.timeOff = 0;
    st.curMask = masks.tomorrow;            // Head_For
    setupMaskII(st.curMask);
    if (param === 1) {
      cmapTo565Fx(B().vb3.cmap.u16, sky.pal, sky.nbCol, 0x000000, 100);
    }
  }

  function loopW303() {                     // world3.c Loop_W3_03
    const t5 = st.t5;
    // WORLD_SET_FRAME + camera anim/matrix, then the camera matrix is kept
    // as Ptcl_M for the sky projection (the particle parts reuse it).
    const { time, lx } = updateWorld(t5.world, t5.camNode, st.globalX, st.timeScale, st.timeOff, W);
    if (lx !== null) t5.renderer.rc.lx = lx;
    st.ptclM = t5.camNode.mo;
    setBck(B().vb3.bits, sky.bits, st.ptclM, W);

    mix8_8BlendMaskII(B().screen, B().vmask, B().vb3);

    t5.renderer.setupClipping(0, W, 0, H, 20, 0);
    t5.renderer.renderScene(B().screen.bits);

    scratchAndFlash(getBeat());
    getNextScratch(0x0e);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  // part6.c Loop_6_05 — two render passes over the same world time: the
  // architecture through the inset clip window (masked, no flares), then
  // the torus alone full-screen with flares on. OBJ_ON/OBJ_OFF is the
  // OBJ_DONT_RENDER flag Vista_Select_Poly honours.
  function loop605() {
    const t6 = st.t6;
    const OFF = (n) => { if (n) n.flags.dontRender = true; };
    const ON = (n) => { if (n) n.flags.dontRender = false; };

    t6.renderer.setZBuffer(t6.zbBase);
    // WORLD_SET_FRAME runs once, before pass 1; pass 2 re-renders the same
    // world time (Render_World re-animates to the identical value).
    const upd = () => {
      const { time, lx } = updateWorld(t6.world, t6.camNode, st.globalX, st.timeScale, st.timeOff, W);
      if (lx !== null) t6.renderer.rc.lx = lx;
      return time;
    };
    OFF(t6.obj2);
    t6.fc.forEach(ON); t6.pil.forEach(ON); ON(t6.obj1); ON(t6.obj3);
    t6.renderer.setupClipping(25, 300, 20, H - 20, fr(0.2), 0);
    const time = upd();
    t6.renderer.render(t6.vista, time, B().screen.bits);   // no flares yet

    saturateDMaskPal(getBeat(), st.curMask, 0xc4c0c2);
    mix16_8BlendMaskII(B().screen, B().vmask, B().screen);

    ON(t6.obj2);
    t6.fc.forEach(OFF); t6.pil.forEach(OFF); OFF(t6.obj1); OFF(t6.obj3);
    t6.renderer.setupClipping(0, W, 0, H, fr(0.2), 0);
    t6.renderer.render(t6.vista, upd(), B().screen.bits);
    t6.renderer.renderLightFlare(t6.lights, B().screen.bits, FLARE_TABLES);
    OFF(t6.obj2);

    t6.renderer.setZBuffer(null);
    loopDspIII();
  }

  function loop608() {                    // part6.c Loop_6_08 (tunnel + flashes)
    const t6 = renderTunnelZbufFlares();
    saturateDMaskPal(getBeat(), st.curMask, 0xc4c0c2);
    mix16_8BlendMaskII(B().screen, B().vmask, B().screen);
    t6.renderer.setZBuffer(null);
    scratchAndFlash(getBeat());
    getNextScratch(0x0e);
    mixSat16_8(B().screen, B().screen, B().scratch);
  }

  // NO NATIVE PATH for Loop_6_06, Loop_6_05 or Loop_6_08, for two different
  // reasons -- both cases where presenting the block-replicated 320x200 frame
  // is the better answer:
  //
  //   6_06 and 6_05 end in screen-space displacement (Loop_Dsp_I, the
  //   beat-driven smear). Reproducing it natively needs a Dsp_Radial table
  //   built for 1.92M pixels and costs the full 30x; rendering the tunnel
  //   natively and silently dropping the displacement would lose a visible
  //   effect on every beat.
  //
  //   6_08 measured 47 ms/frame natively -- the tunnel with a z-buffer pass,
  //   flares, a mask blend and a scratch composite, all at 1600x1200. That
  //   misses not just the 16.67 ms of 60 Hz but the 40 ms of 25 Hz, so it
  //   would drop frames at any rate. 6_02 renders the same tunnel natively
  //   and fits comfortably at 14.98 ms -- the difference is entirely the
  //   z-buffer pass, the flares and the two extra full-screen composites.

  // dsp.c Do_Dissolve_X / _X2 / _Y — screen smear dissolves with TRandom
  // amplitude jitter; X uses the -2A..0 table (Init_Dsp_X_3), X2/Y flip
  // sign via Dsp_Parity. The Y variant fills the same X offset table.
  function doDissolveX() {
    const i = Math.trunc(6 + st.tickX * 4.0);
    const j = Math.trunc(7 + st.tickX * 4.0);
    let amp = fr(fr(fr(st.tickX * fr(1 - st.tickX)) * 4) * 8);
    amp = fr(amp + fr(fr((st.trandom.at(2, st.timer) & 0xff) * fr(5 / 256)) - fr(2.5)));
    const a = Math.trunc(amp);
    satTable4(offX, -2 * a, 0, 1, 256);
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    mix16_16Mix(B().screen, B().vblurI, B().screen, T.lo[i], T.hi[i], T.lo[j], T.hi[j], false);
  }
  function doDissolveX2() {
    let amp = fr(fr(fr(st.tickX * fr(1 - st.tickX)) * 4) * 10);
    amp = fr(amp + fr(fr((st.trandom.at(2, st.timer) & 0xff) * fr(4 / 256)) - 2));
    if (st.dspParity) amp = fr(-amp);
    st.dspParity ^= 1;
    initDspXY2(Math.trunc(amp));
    dspSafeX(B().vblurI, B().screen, st.dspRadial, offX);
    mix16_16Mix(B().screen, B().vblurI, B().screen, T.lo[7], T.hi[7], T.lo[9], T.hi[9], false);
  }
  function doDissolveY() {
    let amp = fr(fr(fr(st.tickX * fr(1 - st.tickX)) * 4) * 15);
    amp = fr(amp + fr(fr((st.trandom.at(1, st.timer) & 0xff) * fr(4 / 256)) - 2));
    if (st.dspParity) amp = fr(-amp);
    st.dspParity ^= 1;
    const a = Math.trunc(amp);
    satTable4(offX, -2 * a, 0, 1, 256);   // Init_Dsp_Y_3 == X_3 (macro quirk)
    dspSafeY(B().vblurI, B().screen, st.dspRadial.subarray(W * H), offX);
    mix16_16Mix(B().screen, B().vblurI, B().screen, T.lo[7], T.hi[7], T.lo[8], T.hi[8], false);
  }

  // ---- phase driver ------------------------------------------------------

  const INITS = {
    Init_00: init00, Init_Dissolve: initDissolve, Init_01: init01,
    Init_Blur_I: initBlurI, Init_World3: initWorld3, Init_04: init04,
    Init_02: init02, Init_10: init10, Init_Blb: initBlb, Init_Ptcl: initPtcl,
    Init_03: init03, Init_Greets: initGreets, Init_05: init05,
  };
  const LOOPS = {
    Loop_Deform_Blur_I: loopDeformBlurI,
    Loop_Deform_Blur_II: loopDeformBlurII,
    Loop_14_Anim_Mask: loop14AnimMask,
    Do_Dissolve: doDissolve,
    Loop_14_Anim_Mask_Move: loop14AnimMaskMove,
    Loop_6_02: loop602,
    Loop_W3_03: loopW303,
    Loop_Appear_II: loopAppearII,
    Loop_Disappear: loopDisappear,
    Loop_14_Anim_Mask_Move_II: loop14AnimMaskMoveII,
    Loop_14_Anim_Mask_Move_Sprites: loop14AnimMaskMoveSprites,
    Loop_14_Powered: loop14Powered,
    Loop_Anim_Blb_I: loopAnimBlbI,
    Loop_10: loop10,
    Loop_14_No_Way: loop14NoWay,
    Loop_14_No_Way_II: loop14NoWayII,
    Loop_Deform_Blur_IV: loopDeformBlurIV,
    Loop_W4_02: loopW402,
    Loop_Deform_Blur_III: loopDeformBlurIII,
    Loop_Fade: loopFade,
    Loop_14_End: loop14End,
    Loop_14_Ready: loop14Ready,
    Loop_14_Anim_Mask_Sat: loop14AnimMaskSat,
    Loop_D2: loopD2,
    Loop_D2_Destroy: loopD2Destroy,
    Loop_Ptcl_I: loopPtclI,
    Loop_Ptcl_II: loopPtclII,
    Loop_W2_01: loopW201,
    Loop_W2_03: loopW203,
    Loop_6_05: loop605,
    Loop_6_06: loop606,
    Loop_6_08: loop608,
    Do_Dissolve_X: doDissolveX,
    Do_Dissolve_X2: doDissolveX2,
    Do_Dissolve_Y: doDissolveY,
  };
  const CLOSES = {
    // part6.c Close_6: Destroy_World + free ZB_Base + Clear_DMask. Nothing
    // later in the show reads the tunnel context, so dropping it is enough.
    Close_6: () => { st.curMask = null; st.t6 = null; },
    // world3.c Close_World3: Destroy_World + Clear_DMask
    Close_World3: () => { st.curMask = null; st.t5 = null; },
    Clear_DMask: () => { st.curMask = null; },
    // appear.c: DISPOSE(1) only frees the blend buffer
    Close_Appear_II: () => {},
    Close_Disappear: () => {},
    // dsp.c Close_Blur_II: frees the blur plate + Clear_DMask (NOT the anim)
    Close_Blur_II: () => { st.curMask = null; },
    Close_World2: () => { st.t4 = null; },
    // appear.c Destroy_Appear: DISPOSE(1) + free the two wipe plates
    Destroy_Appear: () => {},
    Close_Blb: () => {},
    Close_Ptcl: () => { st.ptcl = null; },
    Close_D2: () => {},
    Close_Caustics: () => { st.caustics = null; },
    Close_World4: () => { st.curMask = null; st.greets = null; },
    Clear_DMask_And_Anim: () => { st.curMask = null; st.curAnim = null; },
  };

  let PH = null; // phase table, injected via setSchedule
  function setSchedule(phases) { PH = phases; }

  function runInit(ph) {
    const fn = INITS[ph.init];
    if (ph.init) { if (fn) fn(ph.param); else st.unported = true; }
  }

  function startDemo() {                    // demo_src/demo_drv.c Start_Demo(0)
    st.phase = 0;
    runInit(PH[0]);
    st.timer = st.tickIn = PH[0].t;
    st.tickOut = PH[1].t;
    const len = PH[0].len;
    const finalTick = len > 0 ? PH[len].t : st.tickOut;
    st.startingTick = st.tickIn;
    const nb = st.tickOut - st.tickIn;
    st.tdx = nb ? Math.fround(1 / nb) : 0;
    st.tickX = 0; st.globalX = 0;
    const nb2 = finalTick - st.tickIn;
    st.gdx = nb2 ? Math.fround(1 / nb2) : 0;
  }

  function advanceFrame() {                 // demo_drv.c Advance_Timer
    const prevTimer = st.timer;
    // synthetic clock (oracle timers.c My_getitimer). Integer division is a
    // floor here because frame numbers are non-negative, so the ms grid is
    // monotone and lands exactly on phase boundaries that are multiples of
    // the step.
    st.timer = Math.floor((st.frame * rate.num) / rate.den);
    // Global_dx = _G_dx_*(Timer - previous Timer): the per-frame advance of
    // Global_x, which the particle part integrates its orbits with.
    st.globalDx = fr(st.gdx * (st.timer - prevTimer));
    if (st.timer >= st.tickOut) {
      const closer = CLOSES[PH[st.phase].close];
      if (PH[st.phase].close) { if (closer) closer(); else st.unported = true; }
      st.phase++;
      // demo_drv.c:268 — on transition Timer is pinned to the new phase's
      // Timing for this frame (Tick_x = 0, beat/TRandom see the pinned
      // value); the real clock resumes next frame. The oracle goldens name
      // transition frames by this pinned value.
      st.timer = st.tickIn = PH[st.phase].t;
      if (st.tickIn === -1) return -1;
      st.tickOut = PH[st.phase + 1].t;
      const nb = st.tickOut - st.tickIn;
      st.tdx = nb ? Math.fround(1 / nb) : 0;
      st.tickX = 0;
      const len = PH[st.phase].len;
      if (len > 0) {
        const finalTick = PH[st.phase + len].t;
        st.startingTick = st.tickIn;
        st.globalX = 0;
        // NOT Global_dx: the C zeroes it only in the one-time Init_Timing
        // path (demo_drv.c:178), never here. On a transition frame it keeps
        // the value computed at the top of Advance_Timer -- with the OUTGOING
        // group's _G_dx_, since that assignment happens before this block.
        // The particle part integrates its orbits with Global_dx, so zeroing
        // it here freezes every particle for one frame and puts the whole
        // 15-second galaxy a step behind the oracle for good.
        const nb2 = finalTick - st.startingTick;
        st.gdx = nb2 ? Math.fround(1 / nb2) : 0;
      }
      runInit(PH[st.phase]);
    }
    // FLT semantics: Tick_x/Global_x are float products (demo_drv.c:302-306);
    // downstream (INT) casts sit on quantization boundaries, so the float
    // rounding is load-bearing (Amp flipped 71/72 on frame 29 without it).
    st.tickX = Math.min(1, Math.fround(st.tdx * (st.timer - st.tickIn)));
    st.globalX = Math.min(1, Math.fround(st.gdx * (st.timer - st.startingTick)));
    // 25 Hz sub-clock. Free-running counters and the marching-cube growth
    // advance only on frames that would have existed at 25 Hz, which keeps
    // the film grain and the blob at their authored speed instead of
    // running them 2.4x fast. Always true at 25 Hz and with remaster off.
    const sub = Math.floor(st.timer / 40);
    st.isSub25 = !retimed || sub !== st.sub25;
    st.sub25 = sub;
    cap.begin();
    // Cleared every frame: a part opts IN by rendering its native layer and
    // setting this. Anything without a native path leaves it false and the
    // caller presents the 320x200 frame, block-replicated by the presenter
    // exactly as before.
    st.nativeValid = false;
    // demo_drv.c:312-318: Bypass_Loop may suppress the phase render
    // entirely; Post_Loop overlays after it.
    let skip = false;
    if (st.bypassLoop) skip = !!BYPASS[st.bypassLoop](st.timer);
    if (!skip) {
      const loop = LOOPS[PH[st.phase].loop];
      if (loop) loop(); else if (PH[st.phase].loop !== 'Loop_0') st.unported = true;
    }
    if (st.postLoop) postFX(st.timer);
    cap.endFrame();
    st.frame++;
    return st.phase;
  }

  // NOTE on the transition clock: the original sets Timer to the phase's
  // Timing on transition, then Tick_x uses that same Timer for the frame.
  // With the oracle's synthetic clock, Timer == frame*STEP both before and
  // after (My_getitimer is pure), so re-reading is equivalent — this
  // mirrors demo_drv.c:227 exactly under SOM_ORACLE semantics.

  return {
    reset,
    setSchedule,
    // Render frame N (timer = N*STEP_MS). Seeking backwards replays from
    // zero — determinism by construction, matching the repo's pure-render
    // adapter contract.
    //
    // st.frame is the NEXT frame to render, so the last rendered frame is
    // st.frame-1. Re-requesting that same frame must be a no-op: a 60 Hz
    // rAF loop driving this 25 Hz show asks for the same frame ~2.4x in a
    // row, and comparing against st.frame instead of st.frame-1 made every
    // one of those a full replay from zero — O(show length) per tick,
    // degrading as the show ran (58 ms/tick by 8 s in).
    advanceTo(frameNo) {
      if (!st || frameNo < st.frame - 1) reset();
      while (st.frame <= frameNo) {
        if (advanceFrame() === -1) break;
      }
    },
    // ---- seeking ---------------------------------------------------------
    // A snapshot of everything advanceFrame() reads or writes, so a caller can
    // jump BACKWARDS without the replay-from-zero above. ~1.9 MB each,
    // measured. `frame` is the next frame to render, which is what advanceTo
    // wants back.
    //
    // The correctness of this is not obvious and is not assumed:
    // work/verify/snapshot-replay.mjs requires a restored frame to be
    // BIT-IDENTICAL to a replayed one, because a snapshot that missed a field
    // would render something plausible rather than fail.
    //
    // AND IT DOES NOT HOLD EVERYWHERE, WHICH IS WHY THE GATE NAMES BOTH SETS.
    // Measured: every 2D part round-trips exactly; every part with a live 3D
    // context (tunnel, city, statue, blob) does not. The cause is not a missed
    // field. A per-part context holds `renderer: newRenderer(world, camNode)`
    // and the meshes it draws are registered in texture caches that live in
    // this closure, OUTSIDE `st` -- so copying `st` yields a world the caches
    // no longer point at. Rebuilding the renderer on restore was tried and
    // fails harder (a null texture slot in the blob part): the entanglement is
    // architectural, not a bug. Seeking therefore replays for those parts;
    // making them snapshotable means giving the caches an identity that
    // survives a copy, which is a larger change than it looks.
    snapshot() {
      if (!st) reset();
      return { frame: st.frame, state: captureState(st) };
    },
    restore(snap) {
      if (!st) reset();
      applyState(st, snap.state);
    },
    get frame565() { return st.buffers.screen.bits; },
    // The native-resolution frame, valid only when nativeValid is true.
    get frameNative() { return nativeScreen; },
    get nativeValid() { return !!(st && st.nativeValid); },
    nativeSize: { w: NW, h: NH },
    _debug() { return st; },   // bring-up introspection; not part of the API
    // --- the capture boundary --------------------------------------------
    // The injected capture, so a caller that supplied one can read back what
    // was recorded. Null-object by default, so this is always safe to touch.
    get capture() { return cap; },
    // Two things the capture cannot compute for itself because they are the
    // engine's own tables, handed over rather than copied: the whole point is
    // that the shaders index the same data the CPU mixers do.
    //
    // The displacement map in play (dspRadial is two W*H halves in one
    // array: X first, Y second) plus the saturation tables. Which of r1/r2
    // is current changes per part, so this is read per frame rather than
    // uploaded once.
    get dspMaps() {
      return st && st.dspRadial
        ? { x: st.dspRadial.subarray(0, W * H), y: st.dspRadial.subarray(W * H, 2 * W * H),
            satX: st.dspSatX, satY: st.dspSatY }
        : null;
    },
    get tables() {
      return {
        lo: T.lo, hi: T.hi, rgb71: T.rgb71, sat: T.sat,
        appearLo: appearT?.lo ?? null, appearHi: appearT?.hi ?? null,
        fadeLo, fadeHi,
      };
    },
    get state() {
      return st && {
        frame: st.frame - 1, timer: st.timer, phase: st.phase,
        tickX: st.tickX, globalX: st.globalX, unported: st.unported,
      };
    },
  };
}
