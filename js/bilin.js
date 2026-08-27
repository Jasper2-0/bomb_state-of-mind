// state-of-mind — bilinear background mapper (dem/fx1/bilin2.c + part2.c).
//
// The sky/environment background is drawn by projecting a coarse grid of
// UV samples (41x26 cells of 8x8 pixels) through the camera rotation, then
// bilinearly interpolating the UVs across each cell and point-sampling a
// 256x256 map with an 8.8 fixed-point index. It is used by world3 (statue),
// the particle parts (galaxy/UFO) and caustics.
//
// FIDELITY NOTE: this is the v1.3 source's C path, which is what the oracle
// builds and therefore what the golden frames encode. It is known to differ
// visibly from the shipped 1998 binary in the galaxy sky (~110 s) — see
// work/re/NOTES.md. Per the campaign decision we match the oracle here and
// log the deviation rather than reconstructing the binary's behaviour.
//
// All interpolation is 16-bit wrapping arithmetic: the UV accumulators are
// USHORT and the per-step deltas are SHORT (arithmetic >>3). The wrap is
// load-bearing — it is what makes the sky tile.

import { W, H } from './screen.js';

const fr = Math.fround;

export const BILX2 = 40, BILY2 = 25;
const DX = fr(2 * 8 / W);                 // NB: the C steps *both* axes by DX
const MAX_W = 512 + 4;

// sign-extend the low 16 bits, then arithmetic-shift right 3
const d3 = (x) => ((x << 16) >> 16) >> 3;

// bilin2.c Do_Bilin_Line_UV_Map_8 — one 8-pixel-wide scanline pass. Mutates
// uv in place: successive calls walk the cell downwards by duv.
function bilinLine(out, outOff, uv, duv, w, map) {
  let i = 0;
  let U = uv[0]; uv[0] = (U + duv[0]) & 0xffff;
  let V = uv[1]; uv[1] = (V + duv[1]) & 0xffff;
  i += 2;
  let o = outOff;
  for (; w > 0; --w) {
    const Uo0 = U, Vo0 = V;
    U = uv[i]; uv[i] = (uv[i] + duv[i]) & 0xffff;
    const dU = d3(U - Uo0) & 0xffff;
    V = uv[i + 1]; uv[i + 1] = (uv[i + 1] + duv[i + 1]) & 0xffff;
    const dV = d3(V - Vo0) & 0xffff;
    i += 2;
    let Uo = Uo0, Vo = Vo0;
    for (let k = 0; k < 8; k++) {
      out[o + k] = map[(Vo & 0xff00) | (Uo >> 8)];
      Uo = (Uo + dU) & 0xffff;
      Vo = (Vo + dV) & 0xffff;
    }
    o += 8;
  }
}

const lineUV = new Uint16Array(MAX_W * 2);
const lineDUV = new Uint16Array(MAX_W * 2);

// bilin2.c Do_Bilinear_UV_Map_8 — 8 rows per cell, deltas recomputed per
// cell row from the next grid row.
// `capture`, when given, receives the accumulator's state at the START of
// each grid row. That state is the ONE thing a shader cannot cheaply derive:
// lineUV is a running total updated eight times per grid row by a TRUNCATING
// divide (d3), so after a row it does NOT equal the next grid row's value --
// it drifts, and the drift is part of the image. Re-deriving it per pixel
// would mean replaying every earlier grid row inside the fragment shader;
// handing over 26 rows of it costs a few kilobytes and makes the shader a
// closed form. See EFFECTS.bilinUVMap.
export function doBilinearUVMap8(out, grid, w, h, dstBps, map, capture) {
  const n = 2 * w + 2;                    // the C loops i<=2*W+1 inclusive
  for (let i = 0; i < n; i++) lineUV[i] = grid[i] & 0xffff;
  let inOff = 0, o = 0;
  for (let j = h; j > 0; --j) {
    if (capture) capture.set(lineUV.subarray(0, n), (h - j) * n);
    inOff += n;
    for (let i = 0; i < n; i++) lineDUV[i] = d3(grid[inOff + i] - lineUV[i]) & 0xffff;
    for (let k = 0; k < 8; k++) {
      bilinLine(out, o, lineUV, lineDUV, w, map);
      o += dstBps;
    }
  }
}


// part2.c PROJ — spherical projection of a view-space ray through the
// particle/camera matrix. Doubles where the C uses double (atan2/sin), with
// the float narrowing at the same places.
function proj(uv, x, y, z, m) {
  // nA_Inv_V_Eq( V, Ptcl_M ): V = transpose(A).V
  const v0 = fr(fr(fr(m[0] * x) + fr(m[3] * y)) + fr(m[6] * z));
  const v1 = fr(fr(fr(m[1] * x) + fr(m[4] * y)) + fr(m[7] * z));
  const v2 = fr(fr(fr(m[2] * x) + fr(m[5] * y)) + fr(m[8] * z));
  void v2;
  // V[0]*M[8] is a float*float product; the second argument mixes in the
  // double literal 1.0, so it evaluates in double over float sub-products.
  let yy = fr(Math.atan2(fr(v0 * m[8]), 1.0 - fr(v0 * m[6]) - fr(v1 * m[7])));
  if (yy < 0) yy = fr(Math.PI - yy);
  const xx = fr(Math.atan2(v1 * Math.sin(yy), v0));
  uv[0] = fr(0.5 + (1.0 / Math.PI) * xx);   // UAMP/M_PI is a double constant
  uv[1] = fr((1.0 / Math.PI) * yy);
}

const gridUV = new Int32Array((BILX2 + 1) * (BILY2 + 1) * 2);
const projUV = new Float32Array(2);

// part2.c Set_Bck — build the coarse UV grid, then bilinearly expand it.
// Quirks preserved: the vertical step uses DX (not the defined DY), and the
// horizontal origin is -100/160 rather than -1, which shifts the sky.
export function setBck(dst, map, m, w) {
  let p = 0;
  let y = fr((H / 2) / (W / 2));
  for (let j = 0; j <= BILY2; ++j, y = fr(y - DX)) {
    let x = fr(-(H / 2) / (W / 2));
    for (let i = 0; i <= BILX2; ++i, x = fr(x + DX)) {
      proj(projUV, x, y, 1.0, m);
      // (SHORT) cast: 16-bit truncation, sign-extended into the INT array
      gridUV[p++] = (Math.trunc(fr(fr(projUV[0] * 65536) * 2)) << 16) >> 16;
      gridUV[p++] = (Math.trunc(fr(fr(projUV[1] * 65536) * 2)) << 16) >> 16;
    }
  }
  doBilinearUVMap8(dst, gridUV, BILX2, BILY2, w, map);
}
