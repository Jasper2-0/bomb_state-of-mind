// state-of-mind — caustics + ray tunnel (dem/src/caustics.c).
//
// A 256x256 plasma is sampled twice at drifting offsets, the sum is treated
// as a height field whose gradient scatters "photons" into a 65x65
// accumulator (Do_Da_Funky_Caustics — this is the caustic), that is blown up
// bilinearly back to 256x256, and finally used as the texture for an
// analytically ray-traced cylinder (Do_Ray_Tunnel) whose UVs are again
// bilinearly interpolated over a 41x26 grid.
//
// Used by the logo part (Loop_D2) and the "ready" part.

import { doBilinearUVMap8 } from './bilin.js';
import { W } from './screen.js';

const fr = Math.fround;

const BILX = 64, BILY = 64, BILX2 = 40, BILY2 = 25;
const CLAMP_OFF = 320;
const TRAD = fr(10.0);
// BILX2/2 and BILY2/2 are INTEGER divisions in C: 40/2 = 20 (exact either
// way) but 25/2 = 12, not 12.5. The two forms happen to agree at j = 0 —
// (0.4/a)*0.625*(0-a) = -0.25 for any a — so getting this wrong leaves the
// grid's first row correct and every later row subtly off.
const HALF_X = Math.trunc(BILX2 / 2), HALF_Y = Math.trunc(BILY2 / 2);
const NX = fr(0.4 / HALF_X);
const NY = fr(fr(0.4 / HALF_Y) * fr(0.625));

// caustics.c Pre_Cmp_Caustics — the base plasma (doubles; the C sums six
// transcendentals per texel and truncates to a byte)
export function buildBasePlasma() {
  const out = new Uint8Array(256 * 256);
  let p = 0;
  for (let j = 0; j < 256; ++j) {
    for (let i = 0; i < 256; ++i) {
      const norm = 64.0 + 64.0 * (
        Math.sin(i / 21.0) + Math.cos(j / 11.0) + Math.cos(i / 20.0) + Math.sin(j / 19.0)
        + Math.sin((i + j) / 25.0)
        + Math.cos(Math.sqrt((128.0 - i) * (128.0 - i) + (128.0 - j) * (128.0 - j)) / 10.0)
      );
      // (PIXEL)Norm on the oracle's arm64: fcvtzu saturates a NEGATIVE
      // double to 0 (it does not wrap), then the low byte is stored. Norm
      // spans 64 +/- 384 here, so the negative case is common and wrapping
      // it instead of clamping corrupts the whole plasma.
      out[p++] = norm < 0 ? 0 : Math.trunc(norm) & 0xff;
    }
  }
  return out;
}

// clamp.c Init_Saturation_Table3
export function satTable3(tab, c0, c1, gamma, w, amp) {
  let j = 0;
  for (let i = 0; i <= CLAMP_OFF; ++i, ++j) tab[j] = (w * c0) & 0xffff;
  for (let i = 1; i < amp; ++i, ++j) {
    let x = fr(fr(1.0 * i) / fr(amp));
    if (gamma !== 1.0) x = fr(Math.pow(x, gamma));
    tab[j] = (w * (Math.floor(c0 + (x * (c1 - c0) + 0.5)) & 0xffff)) & 0xffff;
  }
  for (let i = 0; i < CLAMP_OFF; ++i, ++j) tab[j] = (w * c1) & 0xffff;
}

// plasma.c Do_Plasma_256 — two wrapping 256x256 taps summed as SIGNED chars
export function doPlasma256(dst, base, u1, v1, u2, v2) {
  let d = 0;
  let V1 = v1 & 0xff, V2 = v2 & 0xff;
  for (let j = 256; j > 0; --j) {
    d += 256;
    const s1 = V1 << 8, s2 = V2 << 8;
    V1 = (V1 + 1) & 0xff; V2 = (V2 + 1) & 0xff;
    let U1 = u1 & 0xff, U2 = u2 & 0xff;
    for (let i = -256; i < 0; ++i) {
      // `char` is signed here, so the taps add as signed bytes
      const a = (base[s1 + U1] << 24) >> 24;
      const b = (base[s2 + U2] << 24) >> 24;
      U1 = (U1 + 1) & 0xff; U2 = (U2 + 1) & 0xff;
      dst[d + i] = (a + b) & 0xff;
    }
  }
}

// caustics.c Do_Da_Funky_Caustics — gradient-scattered photon accumulation
const causTmp = new Uint16Array(65 * 65);
export function doDaFunkyCaustics(caustics, w, h, base, tabI, tabII) {
  causTmp.fill(0, 0, (w + 1) * (h + 1));
  let p = 0;
  for (let j = 0; j < 255; ++j) {
    for (let i = 0; i < 255; ++i) {
      let dx = caustics[p + i + 1] - caustics[p + i] + i;
      dx = tabI[CLAMP_OFF + dx];
      let dy = caustics[p + i + 256] - caustics[p + i] + j;
      dy = dx + tabII[CLAMP_OFF + dy];
      causTmp[dy] = (causTmp[dy] + 3) & 0xffff;
    }
    p += 256;
  }
  for (let i = 0; i < (w + 1) * (h + 1); ++i) {
    base[i] = causTmp[i] >= 256 ? 255 : causTmp[i] & 0xff;
  }
}

// bilin2.c Do_Bilinear_4 — 4x expansion, 8.8 interpolation, USHORT wrap
const b4V = new Uint16Array(512 + 4), b4dV = new Uint16Array(512 + 4);
export function doBilinear4(out, inp, w, h, dstBps) {
  const n = w + 1;
  for (let i = 0; i <= w; ++i) b4V[i] = (inp[i] << 8) & 0xffff;
  let inOff = 0, o = 0;
  for (let j = h; j > 0; --j) {
    inOff += n;
    for (let i = 0; i <= w; ++i) {
      b4dV[i] = (Math.trunc(((inp[inOff + i] << 8) - b4V[i]) / 4)) & 0xffff;
    }
    for (let k = 0; k < 4; k++) { bilinLine4(out, o, w); o += dstBps; }
  }
}
function bilinLine4(out, outOff, w) {
  let o = outOff, c = 0;
  let V = b4V[c]; b4V[c] = (V + b4dV[c]) & 0xffff; c++;
  for (let W = w; W > 0; --W) {
    const Vo0 = V;
    V = b4V[c];
    const dV = (((V - Vo0) << 16 >> 16) >> 2) & 0xffff;
    b4V[c] = (b4V[c] + b4dV[c]) & 0xffff; c++;
    let Vo = Vo0;
    for (let k = 0; k < 4; k++) { out[o++] = (Vo >> 8) & 0xff; Vo = (Vo + dV) & 0xffff; }
  }
}

// caustics.c Do_Ray_Tunnel — analytic cylinder intersection per grid node
const rtUV = new Int32Array(51 * 81 * 2);
// `capture`, when given, is filled with the accumulator state the GPU needs
// and handed back with the UV grid; see EFFECTS.bilinUVMap.
export function doRayTunnel(out, M, map, capture) {
  let d = 0;
  for (let j = 0; j <= BILY2; ++j) {
    const ny = fr(NY * (j - HALF_Y));
    for (let i = 0; i <= BILX2; ++i) {
      let p0 = fr(NX * (i - HALF_X)), p1 = ny, p2 = 1.0;
      // nA_V_Eq( P, M ) is nA_V, which reads the matrix's INVERSE rows
      // ([9..17]), not the transpose of the direct part ([0..8]).
      const q0 = fr(fr(M[9] * p0) + fr(M[12] * p1) + fr(M[15] * p2));
      const q1 = fr(fr(M[10] * p0) + fr(M[13] * p1) + fr(M[16] * p2));
      const q2 = fr(fr(M[11] * p0) + fr(M[14] * p1) + fr(M[17] * p2));
      p0 = fr(Math.abs(q0)); p1 = fr(Math.abs(q1)); p2 = fr(Math.abs(q2));
      if (p1 > p0) { const t = p1; p1 = p0; p0 = t; }
      const b = fr(fr(M[18] * p0) + fr(M[19] * p1));
      const a = fr(fr(p0 * p0) + fr(p1 * p1));
      const c = fr(fr(fr(M[18] * M[18]) + fr(M[19] * M[19])) - fr(TRAD * TRAD));
      let t = fr(fr(4.0) * fr(fr(b * b) - fr(a * c)));
      let U = 0, V = 0;
      if (a > 0 && t >= 0) {
        t = fr(fr(-b + Math.sqrt(t)) / a);
        const x = fr(fr(t * p0) + M[18]);
        const y = fr(fr(t * p1) + M[19]);
        const z = fr(fr(t * p2) + M[20]);
        U = Math.trunc(fr(z * fr(340.0)));
        // (65536.0f/2.0f/M_PI) is a DOUBLE constant (M_PI is double), so the
        // product stays double until it lands in the FLT `t`.
        t = fr((65536.0 / 2.0 / Math.PI) * fr(Math.PI - Math.atan2(y, x)));
        V = Math.trunc(t);
      }
      rtUV[d++] = U; rtUV[d++] = V;
    }
  }
  doBilinearUVMap8(out, rtUV, BILX2, BILY2, W, map, capture?.acc);
  if (capture) {
    const n = 2 * BILX2 + 2;
    capture.grid = Uint16Array.from(rtUV.subarray(0, n * (BILY2 + 1)), (v) => v & 0xffff);
    capture.n = n; capture.rows = BILY2 + 1;
  }
}

export { BILX, BILY, BILX2, BILY2, CLAMP_OFF };
