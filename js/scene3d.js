// state-of-mind — scene animation & transform layer.
//
// Reimplements the runtime half of the 3D scene system for loaded worlds:
// TCB spline evaluation (3d_src/anim.c), node animation (camera.c, mesh.c,
// light.c Anim), Transform_To_Matrix (transf.c), the camera look-at matrix
// (camera.c Camera_Matrix), and the per-frame world update that
// Render_World performs before rasterization (render.c:258-324).
//
// Numeric contract: the C computes tangents/Hermite in double and narrows
// results to FLT — mirrored here with Math.fround at the same boundaries.

import { idMatrix2, quatToMatrix, aV, nAV } from './world3ds.js';

const fr = Math.fround;

// ---- spline evaluation (3d_src/anim.c) -----------------------------------

function doEase(val, ein, eout) {          // doubles throughout
  let s = ein + eout;
  if (s === 0) return val;
  if (s > 1) { const is = 1 / s; ein *= is; eout *= is; s = 1; }
  const k = 1 / (2 - s);
  if (val < ein) return (k / ein) * val * val;
  if (val < 1 - eout) return k * (2 * val - ein);
  val = 1 - val;
  return 1 - (k / eout) * val * val;
}

function fetchSplineKey(sp, t) {           // returns [n, localT]
  const keys = sp.keys, last = keys.length - 1;
  if (sp.flags === 'loop') t = fr(t % keys[last].frame);
  if (t >= keys[last].frame) { sp.cache = last; return [-1, keys[last].frame]; }
  if (t < keys[0].frame) {
    sp.cache = 0;
    t = keys[0].frame;
    const frames = fr(keys[1].frame - keys[0].frame);
    const lt = frames > 0 ? fr(fr(t - keys[0].frame) / frames) : 0;  // FLT division
    return [0, fr(doEase(lt, keys[0].easeOut, keys[1].easeIn))];
  }
  let n = sp.cache;
  if (keys[n].frame > t) n = 0;
  while (keys[n + 1].frame < t) n++;
  if (n > keys.length - 2) { n = keys.length - 2; t = keys[n + 1].frame; }
  sp.cache = n;
  const frames = fr(keys[n + 1].frame - keys[n].frame);
  const lt = frames > 0 ? fr(fr(t - keys[n].frame) / frames) : 0;    // FLT division
  return [n, fr(doEase(lt, keys[n].easeOut, keys[n + 1].easeIn))];
}

function tangent(sp, out, cur, side) {     // side: 'left' | 'right'
  const keys = sp.keys, ds = sp.dataSize;
  const k = keys[cur];
  if (cur === 0) {
    const next = keys[1];
    if (keys.length === 2) {
      for (let i = 0; i < ds; i++) out[i] = fr(fr(next.values[i] - k.values[i]) * fr(1 - k.tens));
      return;
    }
    if (sp.flags !== 'loop') {
      const tmp = new Float32Array(4);
      tangent(sp, tmp, 1, 'left');
      for (let i = 0; i < ds; i++) {
        out[i] = fr(fr(fr(fr(next.values[i] - k.values[i]) * 1.5) - fr(tmp[i] * 0.5)) * fr(1 - k.tens));
      }
      return;
    }
    // loop: prev = keys[len-2], fall through to the TCB case
    return tcb(sp, out, cur, keys[keys.length - 2], next, side);
  }
  if (cur === keys.length - 1) {
    const prev = keys[cur - 1];
    if (keys.length === 2) {
      for (let i = 0; i < ds; i++) out[i] = fr(fr(k.values[i] - prev.values[i]) * fr(1 - k.tens));
      return;
    }
    if (sp.flags !== 'loop') {
      const tmp = new Float32Array(4);
      tangent(sp, tmp, cur - 1, 'right');
      for (let i = 0; i < ds; i++) {
        out[i] = fr(fr(fr(fr(k.values[i] - prev.values[i]) * 1.5) - fr(tmp[i] * 0.5)) * fr(1 - k.tens));
      }
      return;
    }
    return tcb(sp, out, cur, prev, keys[1], side);
  }
  return tcb(sp, out, cur, keys[cur - 1], keys[cur + 1], side);
}

function tcb(sp, out, cur, prev, next, side) {
  const k = sp.keys[cur], ds = sp.dataSize;
  const cont = side === 'right' ? 0.5 * (1 - k.cont) : 0.5 * (1 + k.cont);
  for (let i = 0; i < ds; i++) {
    // C: (float - float) is a single-precision subtract BEFORE the double
    // promotion for the (1.0 +/- Bias) product — narrow the differences.
    const g1 = fr(k.values[i] - prev.values[i]) * (1 + k.bias);
    const g2 = fr(next.values[i] - k.values[i]) * (1 - k.bias);
    out[i] = fr((g1 + (g2 - g1) * cont) * (1 - k.tens));
  }
}

export function setupSplineDerivates(sp) {
  if (!sp) return;
  sp.cache = 0;
  if (sp.keys.length < 2) return;
  for (let i = 0; i < sp.keys.length; i++) {
    tangent(sp, sp.keys[i].left, i, 'left');
    tangent(sp, sp.keys[i].right, i, 'right');
  }
}

export function interpolateSpline(t, sp, out) {
  if (!sp) return;
  const keys = sp.keys;
  if (keys.length <= 1) {                  // Void_Spline
    for (let i = 0; i < sp.dataSize; i++) out[i] = keys.length ? keys[0].values[i] : 0;
    return;
  }
  const [n, lt] = fetchSplineKey(sp, t);
  if (n === -1) {
    for (let i = 0; i < sp.dataSize; i++) out[i] = keys[keys.length - 1].values[i];
    return;
  }
  const t2 = lt * lt, s = 1 - lt, s2 = s * s;
  const h1 = t2 * (3 - 2 * lt);
  const h0 = 1 - h1;
  const h2 = lt * s2;
  const h3 = -t2 * s;
  for (let i = 0; i < sp.dataSize; i++) {
    out[i] = fr(h0 * keys[n].values[i] + h1 * keys[n + 1].values[i]
      + h2 * keys[n].right[i] + h3 * keys[n + 1].left[i]);
  }
}

// ---- rotation splines (anim.c) -------------------------------------------
//
// Rotation tracks do NOT use the TCB Hermite path above. They get their own
// tangents built from quaternion interpolation (Get_Rot_Spline_Tangent) and
// are evaluated with cubic BEZIER coefficients, not Hermite. Quaternion
// layout is w,x,y,z (quat.c defines _W_ as 0).

const Q_EPSILON = fr(1.0e-5);

// quat.c Quaternion_Linear_Interp — a slerp. QUIRK: it computes `Flip` from
// the sign of the dot product but never applies it (both `Dot = -Dot` and
// `Scale_Q2 *= Flip` are commented out in the source), so antipodal pairs
// take the long way round. Preserved.
function quaternionLinearInterp(out, q1, q2, t) {
  let dot = fr(q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]);
  dot = fr(dot + fr(q1[0] * q2[0]));
  let s1, s2;
  if (fr(1 + dot) > Q_EPSILON) {
    if (fr(1 - dot) > Q_EPSILON) {
      const omega = fr(Math.acos(dot));
      const sin = fr(1 / Math.sin(omega));
      s1 = fr(fr(Math.sin(fr(1 - t) * omega)) * sin);
      s2 = fr(fr(Math.sin(t * omega)) * sin);
    } else { s1 = fr(1 - t); s2 = t; }
    for (let i = 0; i < 4; i++) out[i] = fr(fr(s1 * q1[i]) + fr(s2 * q2[i]));
    return;
  }
  // antipodal fallback: build an orthogonal quaternion and rotate by t*PI
  const tmp = new Float32Array(4);
  const omega = fr(Math.acos(q1[0]));
  const sin = fr(Math.sin(omega));
  if (sin < Q_EPSILON) { tmp[0] = 0; tmp[1] = 1; tmp[2] = 0; tmp[3] = 0; }
  else {
    const th = fr(1 / Math.tan(omega));
    tmp[0] = fr(-sin);
    tmp[1] = fr(th * q1[1]); tmp[2] = fr(th * q1[2]); tmp[3] = fr(th * q1[3]);
  }
  s1 = fr(Math.cos(t * Math.PI)); s2 = fr(Math.sin(t * Math.PI));
  for (let i = 0; i < 4; i++) out[i] = fr(fr(s1 * q1[i]) + fr(s2 * tmp[i]));
}

const g1q = new Float32Array(4), g2q = new Float32Array(4), g3q = new Float32Array(4);

// anim.c Get_Rot_Spline_Tangent
function rotTangent(sp, out, cur, side) {
  const keys = sp.keys, k = keys[cur];
  if (cur === 0) {
    const next = keys[1];
    if (sp.flags !== 'loop' || keys.length <= 2) {
      quaternionLinearInterp(out, k.values, next.values,
        fr(fr(fr(1 - k.tens) * fr(1 - fr(k.cont * k.bias))) / 3));
      return;
    }
  } else if (cur === keys.length - 1) {
    const prev = keys[cur - 1];
    if (sp.flags !== 'loop' || keys.length <= 2) {
      quaternionLinearInterp(out, k.values, prev.values,
        fr(fr(fr(1 - k.tens) * fr(1 - fr(k.cont * k.bias))) / 3));
      return;
    }
  }
  const prev = cur === 0 ? keys[keys.length - 2] : keys[cur - 1];
  const next = cur === keys.length - 1 ? keys[1] : keys[cur + 1];
  quaternionLinearInterp(g1q, k.values, prev.values, fr(-fr(1 + k.bias) / 3));
  quaternionLinearInterp(g2q, k.values, next.values, fr(fr(1 - k.bias) / 3));
  let cont, tens;
  if (side === 'right') { cont = fr(0.5 - fr(0.5 * k.cont)); tens = fr(1 - k.tens); }
  else { cont = fr(0.5 + fr(0.5 * k.cont)); tens = fr(k.tens - 1); }
  quaternionLinearInterp(g3q, g1q, g2q, cont);
  quaternionLinearInterp(out, k.values, g3q, tens);
}

export function setupRotSplineDerivates(sp) {
  if (!sp) return;
  sp.cache = 0;
  if (sp.keys.length < 2) return;
  for (let i = 0; i < sp.keys.length; i++) {
    rotTangent(sp, sp.keys[i].left, i, 'left');
    rotTangent(sp, sp.keys[i].right, i, 'right');
  }
}

// anim.c Interpolate_Rot_Spline_3ds_Values (Bezier_Coeffs, doubles)
export function interpolateRotSpline(t, sp, out) {
  if (!sp) return;
  const keys = sp.keys;
  if (keys.length === 0) { out[0] = 1; out[1] = out[2] = out[3] = 0; return; }
  if (keys.length === 1) {
    for (let i = 0; i < sp.dataSize; i++) out[i] = keys[0].values[i];
    return;
  }
  const [n, lt] = fetchSplineKey(sp, t);
  if (n === -1) {
    for (let i = 0; i < sp.dataSize; i++) out[i] = keys[keys.length - 1].values[i];
    return;
  }
  const S = 1.0 - lt, S2 = S * S, T2 = lt * lt;
  const h0 = S2 * S, h2 = 3.0 * S2 * lt, h3 = 3.0 * S * T2, h1 = T2 * lt;
  for (let i = 0; i < sp.dataSize; i++) {
    out[i] = fr(h0 * keys[n].values[i] + h1 * keys[n + 1].values[i]
      + h2 * keys[n].right[i] + h3 * keys[n + 1].left[i]);
  }
}

export function interpolateHidden(t, sp) {
  if (!sp) return 0;
  if (sp.keys.length <= 1) return Math.trunc(sp.keys[0].values[0]);
  const [n] = fetchSplineKey(sp, t);
  if (n === -1) return Math.trunc(sp.keys[sp.keys.length - 1].values[0]);
  return Math.trunc(sp.keys[n].values[0]);
}

// ---- Transform_To_Matrix (transf.c, quaternion case) ---------------------

export function transformToMatrix(t, m) {
  quatToMatrix(t.qrot, m);
  const sx = t.scale[0], sy = t.scale[1], sz = t.scale[2];
  m[0] = fr(m[0] * sx); m[3] = fr(m[3] * sx); m[6] = fr(m[6] * sx);
  m[1] = fr(m[1] * sy); m[4] = fr(m[4] * sy); m[7] = fr(m[7] * sy);
  m[2] = fr(m[2] * sz); m[5] = fr(m[5] * sz); m[8] = fr(m[8] * sz);
  const ix = 1 / sx, iy = 1 / sy, iz = 1 / sz;
  m[9] = fr(m[9] * ix); m[10] = fr(m[10] * ix); m[11] = fr(m[11] * ix);
  m[12] = fr(m[12] * iy); m[13] = fr(m[13] * iy); m[14] = fr(m[14] * iy);
  m[15] = fr(m[15] * iz); m[16] = fr(m[16] * iz); m[17] = fr(m[17] * iz);
  m[18] = fr(t.pos[0] + t.pivot[0]);
  m[19] = fr(t.pos[1] + t.pivot[1]);
  m[20] = fr(t.pos[2] + t.pivot[2]);
  m[21] = fr(-fr(fr(fr(m[9] * m[18]) + fr(m[10] * m[19])) + fr(m[11] * m[20])));
  m[22] = fr(-fr(fr(fr(m[12] * m[18]) + fr(m[13] * m[19])) + fr(m[14] * m[20])));
  m[23] = fr(-fr(fr(fr(m[15] * m[18]) + fr(m[16] * m[19])) + fr(m[17] * m[20])));
}

// ---- camera matrix (camera.c Camera_Matrix, double math) -----------------

export function cameraMatrix(pos, tgt, omega, m) {
  // D is a VECTOR (FLT[3]) in the C — differences narrow to float
  const d0 = fr(tgt[0] - pos[0]), d1 = fr(tgt[1] - pos[1]), d2 = fr(tgt[2] - pos[2]);
  const a0 = d0 * d0 + d1 * d1;
  if (a0 < 0.00001) {
    m.fill(0);
    m[0] = m[4] = m[8] = m[9] = m[13] = m[17] = 1;
    m[18] = -pos[0]; m[21] = pos[0];
    m[19] = -pos[1]; m[22] = pos[1];
    m[20] = -pos[2]; m[23] = pos[2];
    return;
  }
  const theta = fr(Math.atan2(Math.sqrt(a0), d2));
  const phi = fr(Math.atan2(d1, d0));
  const cx = Math.cos(phi), sx = Math.sin(phi);
  const cy = Math.cos(theta), sy = Math.sin(theta);
  const cz = Math.cos(omega), sz = Math.sin(omega);
  const a = cx * cy, b = sx * cy;
  m[9] = m[0] = fr(-a * sz - sx * cz);
  m[12] = m[1] = fr(-b * sz + cx * cz);
  m[15] = m[2] = fr(sy * sz);
  m[10] = m[3] = fr(-a * cz + sx * sz);
  m[13] = m[4] = fr(-b * cz - cx * sz);
  m[16] = m[5] = fr(sy * cz);
  m[11] = m[6] = fr(cx * sy);
  m[14] = m[7] = fr(sx * sy);
  m[17] = m[8] = fr(cy);
  // Camera_Matrix_Set_Position: invT = Pos; T = R.(-Pos)
  m[21] = pos[0]; m[22] = pos[1]; m[23] = pos[2];
  const v = [fr(-pos[0]), fr(-pos[1]), fr(-pos[2])];
  const o = [0, 0, 0];
  nAV(o, m, v);
  m[18] = o[0]; m[19] = o[1]; m[20] = o[2];
}

// ---- world setup + per-frame animation -----------------------------------

// Finish/Select_World-time work relevant to animation: spline derivative
// setup (camera.c Finish, mesh Finish equivalents).
export function finishWorld(world) {
  for (const node of world.nodes) {
    const sp = node.data?.splines;
    if (!sp) continue;
    for (const key of Object.keys(sp)) {
      // rotation tracks get quaternion tangents, not TCB Hermite ones
      if (key === 'rot') setupRotSplineDerivates(sp[key]);
      else setupSplineDerivates(sp[key]);
    }
  }
}

// Camera Anim (camera.c): pos/target/roll/fov; D = Lens*tan(Fov/2).
// Returns projection half-width factor (engine stores it in _RCst_._Lx_).
export function animCamera(camNode, frame, pixWidth) {
  const cam = camNode.data;
  if (camNode.curTime === frame) return null;
  interpolateSpline(frame, cam.splines.pos, cam.pos);
  if (cam.dummy) interpolateSpline(frame, cam.dummy.data.splines.pos, cam.target);
  else interpolateSpline(frame, cam.splines.target, cam.target);
  if (cam.splines.roll) {
    const out = new Float32Array(1);
    interpolateSpline(frame, cam.splines.roll, out);
    cam.bank = out[0];
  }
  let lx = null;
  if (cam.splines.fov) {
    const out = new Float32Array(1);
    interpolateSpline(frame, cam.splines.fov, out);
    cam.d = fr(cam.lens * Math.tan(out[0] / 2));
    lx = fr(0.5 * pixWidth * cam.d);
  }
  return lx;
}

// Mesh/light Anim (mesh.c/light.c): hide flag + transform splines.
export function animMeshNode(node, frame) {
  const mesh = node.data, sp = mesh.splines;
  if (sp.hide) {
    let hide;
    if (frame < sp.hide.keys[0].frame) hide = Math.trunc(sp.hide.keys[0].values[0]) ^ 1;
    else hide = interpolateHidden(fr(frame + 0.01), sp.hide);
    node.flags.notActive = !hide;          // Hide==1 -> visible (mesh.c Anim)
  }
  if (node.curTime === frame) return;
  if (sp.rot) interpolateRotSpline(frame, sp.rot, node.transform.qrot);
  if (sp.pos) interpolateSpline(frame, sp.pos, node.transform.pos);
  if (sp.scale) interpolateSpline(frame, sp.scale, node.transform.scale);
  if (mesh.morphNodes) throw new Error(`morph not ported yet (${node.name})`);
}

// The pre-render update Render_World performs (render.c): animate + build
// the camera matrix, then per-node camera-space matrices Mo/M.
// Returns { time } for instrumentation.
export function updateWorld(world, camNode, globalX, timeScale, timeOff, pixWidth) {
  // Time_Scale / Time_Off are FLT globals (dem/src/panim.c) — 0.85f is NOT
  // 0.85; narrow the constants before use or the show time drifts ~3e-5.
  timeScale = fr(timeScale);
  timeOff = fr(timeOff);
  const x = fr(fr(globalX * timeScale) + timeOff);
  const time = fr(world.framesStart + fr(x * fr(world.framesEnd - world.framesStart)));

  // camera (render.c: NODE_ANIM if changed, then NODE_TRANSFORM).
  // Node->Cur_Time is initialized to -HUGE and never assigned anywhere in
  // the engine, so the "if changed" guards never fire — anim runs every
  // frame. Mirrored: we never set curTime.
  let lx = null;
  if (camNode.flags.doAnim) lx = animCamera(camNode, time, pixWidth);
  cameraMatrix(camNode.data.pos, camNode.data.target, camNode.data.bank, camNode.mo);

  // mesh nodes: Anim + Transform_To_Matrix (STORE_TRANSFORM_MATRIX) +
  // camera-space M = Cam.Mo x Node.Mo (STORE_NODE_MATRIX / Mult_Matrix_ABC)
  for (const node of world.nodes) {
    if (node.type !== 'mesh') continue;
    if (node.flags.doAnim) animMeshNode(node, time);
    transformToMatrix(node.transform, node.mo);
    multMatrixABC(camNode.mo, node.mo, node.m);
  }

  // lights: Setup_Light_II (render.c) — Check_Node_Transform + Compose_
  // Light_And_Camera. Light->Pos = translation of the node's world matrix;
  // Light->P = A_V(Cam.Mo, Pos), the camera-space position the flare pass
  // projects (raster3d.selectLightVisibility).
  for (const node of world.nodes) {
    if (node.type !== 'light') continue;
    if (node.flags.doAnim) animMeshNode(node, time);   // light.c Anim: same shape
    transformToMatrix(node.transform, node.mo);
    multMatrixABC(camNode.mo, node.mo, node.m);        // STORE_NODE_MATRIX
    const light = node.data;
    if (!light.p) light.p = new Float32Array(3);
    if (!light.posW) light.posW = new Float32Array(3);
    light.posW[0] = node.mo[18]; light.posW[1] = node.mo[19]; light.posW[2] = node.mo[20];
    aV(light.p, camNode.mo, light.posW);
  }
  return { time, lx };
}

// Mult_Matrix_ABC (transf.c): C = A.B for MATRIX2.
export function multMatrixABC(a, b, c) {
  // direct 3x3: C = A.B
  for (let r = 0; r < 3; r++) {
    for (let col = 0; col < 3; col++) {
      c[r * 3 + col] = fr(fr(fr(a[r * 3] * b[col]) + fr(a[r * 3 + 1] * b[3 + col])) + fr(a[r * 3 + 2] * b[6 + col]));
    }
  }
  // inverse 3x3: C^-1 = B^-1.A^-1
  for (let r = 0; r < 3; r++) {
    for (let col = 0; col < 3; col++) {
      c[9 + r * 3 + col] = fr(fr(fr(b[9 + r * 3] * a[9 + col]) + fr(b[9 + r * 3 + 1] * a[9 + 3 + col])) + fr(b[9 + r * 3 + 2] * a[9 + 6 + col]));
    }
  }
  // translations: Tc = A.Tb + Ta ; invTc = B^-1.invTa + invTb
  const tb = [b[18], b[19], b[20]], out = [0, 0, 0];
  aV(out, a, tb);
  c[18] = out[0]; c[19] = out[1]; c[20] = out[2];
  const ita = [a[21], a[22], a[23]];
  // A_Inv_V(out, B, invTa): rows [9..17] of B + invT of B
  c[21] = fr(fr(fr(fr(b[9] * ita[0]) + fr(b[10] * ita[1])) + fr(b[11] * ita[2])) + b[21]);
  c[22] = fr(fr(fr(fr(b[12] * ita[0]) + fr(b[13] * ita[1])) + fr(b[14] * ita[2])) + b[22]);
  c[23] = fr(fr(fr(fr(b[15] * ita[0]) + fr(b[16] * ita[1])) + fr(b[17] * ita[2])) + b[23]);
}
