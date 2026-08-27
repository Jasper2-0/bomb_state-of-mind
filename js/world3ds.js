// state-of-mind — 3DS world loader.
//
// Reimplements the demo's own .3ds reader (3d_src/3d_io.c) — the chunks it
// consumes, in its consumption order, with its quirks. This is NOT a
// general 3DS loader: fields the engine ignores are not part of its format
// (METHOD.md), and deviations from the documented 3DS spec here mirror the
// engine (negated rotation angle, cumulative rot keys, pivot baking into
// vertices, V flip on UVs, forced full edge flags...).
//
// All floats are f32 (DataView getFloat32) matching the engine's FLT reads;
// derived math (normals, quaternions) follows the C float/double mix and is
// fround-disciplined where the C narrows to FLT.

import { fmaf } from './fp.js';

const fr = Math.fround;

// ---- MATRIX2 conventions (include/transf.h, 3d_src/transf.c) -------------
// 24 floats: [0..8] direct 3x3 row-major (row.vector), [9..17] inverse,
// [18..20] forward translation, [21..23] inverse translation.

export function idMatrix2() {
  const m = new Float32Array(24);
  m[0] = m[4] = m[8] = 1;
  m[9] = m[13] = m[17] = 1;
  return m;
}

// The oracle is built -ffast-math, so these two are NOT the sum-then-add the C
// writes. clang contracts each term into an fmadd AND reassociates, seeding
// the accumulator with the translation:
//
//   C:      ( A0*x + A1*y + A2*z ) + A18        -- 5 roundings, T added last
//   binary: fmadd(x,A0,A18); fmadd(y,A1,.); fmadd(z,A2,.)  -- 3, T added FIRST
//
// A_V is the engine's universal vector transform, so this rounding order
// propagates into every projected vertex, every normal and every camera-space
// position in the show.
export function aV(out, a, v) {           // Out = A.V + T
  const x = v[0], y = v[1], z = v[2];
  out[0] = fmaf(z, a[2], fmaf(y, a[1], fmaf(x, a[0], a[18])));
  out[1] = fmaf(z, a[5], fmaf(y, a[4], fmaf(x, a[3], a[19])));
  out[2] = fmaf(z, a[8], fmaf(y, a[7], fmaf(x, a[6], a[20])));
}

export function nAV(out, a, v) {          // Out = t(A^-1).V (normals/rotation)
  const x = v[0], y = v[1], z = v[2];
  out[0] = fmaf(z, a[15], fmaf(y, a[12], fr(a[9] * x)));
  out[1] = fmaf(z, a[16], fmaf(y, a[13], fr(a[10] * x)));
  out[2] = fmaf(z, a[17], fmaf(y, a[14], fr(a[11] * x)));
}

// ---- quaternions (3d_src/quat.c); QUATERNION layout [w,x,y,z] ------------

export function quatToMatrix(q, m) {      // fills [0..8] + transpose [9..17]
  const xx0 = fr(q[1] * q[1]), yy0 = fr(q[2] * q[2]), zz0 = fr(q[3] * q[3]);
  const s = fr(2 / fr(fr(xx0 + yy0) + fr(zz0 + fr(q[0] * q[0]))));
  const xx = fr(xx0 * s), yy = fr(yy0 * s), zz = fr(zz0 * s);
  const xs = fr(q[1] * s), wx = fr(q[0] * xs);
  const ys = fr(q[2] * s), wy = fr(q[0] * ys), xy = fr(q[1] * ys);
  const zs = fr(q[3] * s), wz = fr(q[0] * zs), xz = fr(q[1] * zs), yz = fr(q[2] * zs);
  m[9] = m[0] = fr(1 - fr(yy + zz));
  m[10] = m[3] = fr(xy + wz);
  m[11] = m[6] = fr(xz - wy);
  m[12] = m[1] = fr(xy - wz);
  m[13] = m[4] = fr(1 - fr(xx + zz));
  m[14] = m[7] = fr(yz + wx);
  m[15] = m[2] = fr(xz + wy);
  m[16] = m[5] = fr(yz - wx);
  m[17] = m[8] = fr(1 - fr(xx + yy));
}

export function matrixToQuat(m, q) {
  const trace = fr(fr(1 + m[0]) + fr(m[4] + m[8]));
  if (trace > 0.00001) {
    let s = fr(Math.sqrt(trace));
    q[0] = fr(0.5 * s);
    s = fr(0.5 / s);
    q[1] = fr(fr(m[7] - m[5]) * s);
    q[2] = fr(fr(m[2] - m[6]) * s);
    q[3] = fr(fr(m[3] - m[1]) * s);
  } else {
    const diag = [0, 0, 4, 8], next = [0, 2, 3, 1];
    let i = 1;
    if (m[4] > m[0]) i = 2;
    if (m[8] > m[diag[i]]) i = 3;
    const s = Math.sqrt(2 * (1 + m[diag[i]]));
    q[i] = fr(0.5 * s);
    const is = 1 / s;
    if (i === 1) { q[2] = fr(m[1] * is); q[3] = fr(m[2] * is); }
    else if (i === 2) { q[3] = fr(m[5] * is); q[1] = fr(m[3] * is); }
    else { q[1] = fr(m[6] * is); q[2] = fr(m[7] * is); }
    q[0] = 0;
  }
  let n = fr(fr(fr(q[0] * q[0]) + fr(q[1] * q[1])) + fr(fr(q[2] * q[2]) + fr(q[3] * q[3])));
  if (n < 0.0001) { q[0] = 0; q[1] = 1; q[2] = 0; q[3] = 0; }
  else {
    n = fr(1 / Math.sqrt(n));
    q[0] = fr(q[0] * n); q[1] = fr(q[1] * n); q[2] = fr(q[2] * n); q[3] = fr(q[3] * n);
  }
}

function thetaVToQuat(out, rot) {         // rot = [angle, x, y, z] -> returns |axis|^2
  out[0] = fr(Math.cos(rot[0] / 2));
  let tmp = Math.sin(rot[0] / 2);
  const n = rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3];
  if (n === 0) { out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0; }
  else {
    tmp /= Math.sqrt(n);
    out[1] = fr(tmp * rot[1]);
    out[2] = fr(tmp * rot[2]);
    out[3] = fr(tmp * rot[3]);
  }
  return n;
}

export function multQuatEq(a, b) {        // A = A.B
  const t0 = fr(fr(fr(a[0] * b[0]) - fr(a[1] * b[1])) - fr(fr(a[2] * b[2]) + fr(a[3] * b[3])));
  const t1 = fr(fr(fr(a[0] * b[1]) + fr(a[1] * b[0])) + fr(fr(a[2] * b[3]) - fr(a[3] * b[2])));
  const t2 = fr(fr(fr(a[0] * b[2]) + fr(a[2] * b[0])) + fr(fr(a[3] * b[1]) - fr(a[1] * b[3])));
  const t3 = fr(fr(fr(a[0] * b[3]) + fr(a[3] * b[0])) + fr(fr(a[1] * b[2]) - fr(a[2] * b[1])));
  a[0] = t0; a[1] = t1; a[2] = t2; a[3] = t3;
}

// ---- normals (3d_src/normal.c) -------------------------------------------

function storeNormals2(mesh) {
  for (const p of mesh.polys.slice(0, mesh.nbPolys ?? mesh.polys.length)) {
    const v0 = p.pt[0] * 3, v1 = p.pt[1] * 3, v2 = p.pt[2] * 3;
    const V = mesh.vertex;
    const dx1 = fr(V[v1] - V[v0]), dx2 = fr(V[v2] - V[v0]);
    const dy1 = fr(V[v1 + 1] - V[v0 + 1]), dy2 = fr(V[v2 + 1] - V[v0 + 1]);
    const dz1 = fr(V[v1 + 2] - V[v0 + 2]), dz2 = fr(V[v2 + 2] - V[v0 + 2]);
    p.no[0] = fr(fr(dy1 * dz2) - fr(dy2 * dz1));
    p.no[1] = fr(fr(dx2 * dz1) - fr(dx1 * dz2));
    p.no[2] = fr(fr(dx1 * dy2) - fr(dx2 * dy1));
    const n2 = fr(fr(fr(p.no[0] * p.no[0]) + fr(p.no[1] * p.no[1])) + fr(p.no[2] * p.no[2]));
    if (n2 > 0) {
      const n = fr(1 / Math.sqrt(n2));
      p.no[0] = fr(p.no[0] * n); p.no[1] = fr(p.no[1] * n); p.no[2] = fr(p.no[2] * n);
    }
  }
}

function averageNormals(mesh) {
  const ns = mesh.normals;
  ns.fill(0);
  for (const p of mesh.polys.slice(0, mesh.nbPolys ?? mesh.polys.length)) {
    for (let k = 0; k < 3; k++) {
      const v = p.pt[k] * 3;
      ns[v] = fr(ns[v] + p.no[0]);
      ns[v + 1] = fr(ns[v + 1] + p.no[1]);
      ns[v + 2] = fr(ns[v + 2] + p.no[2]);
    }
  }
  for (let i = 0; i < mesh.nbVertex; i++) {
    const v = i * 3;
    let n = fr(fr(fr(ns[v] * ns[v]) + fr(ns[v + 1] * ns[v + 1])) + fr(ns[v + 2] * ns[v + 2]));
    if (n > 0) {
      n = fr(1 / Math.sqrt(n));
      ns[v] = fr(ns[v] * n); ns[v + 1] = fr(ns[v + 1] * n); ns[v + 2] = fr(ns[v + 2] * n);
    }
  }
}

// ---- world / node model --------------------------------------------------

const POLY_U_WRAP = 0x0400, POLY_V_WRAP = 0x0800; // placeholders; real bit
// values come from include/poly.h flags — only wrap flags matter downstream
// and they are consumed by our own code, so symbolic is fine.

function newSpline(dataSize, n) {
  const keys = [];
  for (let i = 0; i < n; i++) {
    keys.push({ frame: 0, tens: 0, cont: 0, bias: 0, easeIn: 0, easeOut: 0,
      values: new Float32Array(4), left: new Float32Array(4), right: new Float32Array(4) });
  }
  return { flags: 0, dataSize, keys, cache: 0 };
}

function newTransform() {
  return {
    type: 'quaternion',
    pivot: new Float32Array(3),
    pos: new Float32Array(3),
    scale: new Float32Array([1, 1, 1]),
    qrot: new Float32Array([1, 0, 0, 0]),
  };
}

function newNode(world, type, name) {
  const node = {
    type, name, id: world._nodeId--,
    parent: null, children: [],
    flags: { notActive: false, dontRender: false, doAnim: false, hidden3ds: false },
    transform: newTransform(),
    mo: idMatrix2(),
    m: idMatrix2(),
    curTime: -Infinity,
    data: null,
  };
  world.nodes.push(node);
  return node;
}

// ---- the parser ----------------------------------------------------------

export function parse3ds(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u16 = (p) => dv.getUint16(p, true);
  const i16 = (p) => dv.getInt16(p, true);
  const i32 = (p) => dv.getInt32(p, true);
  const f32 = (p) => dv.getFloat32(p, true);

  const world = {
    framesStart: 0, framesEnd: 0,
    nodes: [], materials: [],
    _nodeId: 0xfffe,
  };

  let curNode = null, curObj = null, curName = null, curDummy = null;
  let currentData = 0, currentId = 0xffff;

  const findByName = (name) => world.nodes.find((n) => n.name === name) ?? null;
  const findById = (id) => world.nodes.find((n) => n.id === id) ?? null;

  const readString = (p) => {
    let s = '';
    while (p.o < bytes.length) {
      const c = bytes[p.o++];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  };

  // -- individual readers; each gets {o} cursor and chunk end ------------

  function meshBlock(p, end) {             // CHK_TRIMESH
    curNode = newNode(world, 'mesh', curName);
    curNode.data = {
      vertex: null, normals: null, uv: null, polys: [], nbVertex: 0,
      splines: {}, morphNodes: null,
    };
    curObj = curNode.data;
    curName = null;
    walk(p, end);
  }

  function vertList(p) {                   // CHK_VERTLIST
    const n = i16(p.o); p.o += 2;
    const m = curObj;
    m.nbVertex = n;
    m.vertex = new Float32Array(n * 3);
    m.normals = new Float32Array(n * 3);
    if (!m.uv) m.uv = new Float32Array(n * 2);
    for (let i = 0; i < n * 3; i++) { m.vertex[i] = f32(p.o); p.o += 4; }
    for (let i = 0; i < n; i++) m.normals[i * 3 + 2] = 1;
  }

  function mapList(p) {                    // CHK_MAPLIST — V flipped
    const n = i16(p.o); p.o += 2;
    const m = curObj;
    if (!m.uv || m.uv.length < n * 2) m.uv = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      m.uv[i * 2] = f32(p.o); p.o += 4;
      m.uv[i * 2 + 1] = fr(1 - f32(p.o)); p.o += 4;
    }
    m.hasUV = true;
  }

  function faceList(p, end) {              // CHK_FACELIST + nested chunks
    const n = i16(p.o); p.o += 2;
    const m = curObj;
    m.polys = [];
    for (let i = 0; i < n; i++) {
      const a = u16(p.o), b = u16(p.o + 2), c = u16(p.o + 4), fl = u16(p.o + 6);
      p.o += 8;
      m.polys.push({
        id: 0xffff, pt: [a, b, c], nbPts: 3, colors: 0,
        flags: { uWrap: !!(fl & 0x08), vWrap: !!(fl & 0x10) },
        no: new Float32Array(3), material: null, uvI: null,
      });
    }
    storeNormals2(m);
    averageNormals(m);
    walk(p, end);                          // FACEMAT / SMOOLIST live inside
  }

  function faceMat(p) {                    // CHK_FACEMAT
    const name = readString(p);
    const n = i16(p.o); p.o += 2;
    for (let i = 0; i < n; i++) {
      const idx = u16(p.o); p.o += 2;
      curObj.polys[idx].material = name;
    }
  }

  function trMatrix(p) {                   // CHK_TRMATRIX — bake into verts
    const m9 = new Float32Array(9);
    for (let i = 0; i < 9; i++) { m9[i] = f32(p.o); p.o += 4; }
    const pivot = new Float32Array(3);
    for (let i = 0; i < 3; i++) { pivot[i] = f32(p.o); p.o += 4; }
    const q = new Float32Array(4);
    matrixToQuat(m9, q);
    quatToMatrix(q, curNode.mo);
    const mesh = curNode.data;
    // Sub_Vertex_Eq then Matrix_Vertex_Eq (nA_V, transpose-of-inverse rows)
    const v = new Float32Array(3), o = new Float32Array(3);
    for (let i = 0; i < mesh.nbVertex; i++) {
      v[0] = fr(mesh.vertex[i * 3] - pivot[0]);
      v[1] = fr(mesh.vertex[i * 3 + 1] - pivot[1]);
      v[2] = fr(mesh.vertex[i * 3 + 2] - pivot[2]);
      nAV(o, curNode.mo, v);
      mesh.vertex[i * 3] = o[0];
      mesh.vertex[i * 3 + 1] = o[1];
      mesh.vertex[i * 3 + 2] = o[2];
    }
    curNode.transform.qrot.set(q);
    curNode.transform.type = 'quaternion';
    curNode.transform.pivot.set(pivot);
    curNode.transform.pos.fill(0);
    curNode.transform.scale.set([1, 1, 1]);
  }

  function cameraRd(p) {                   // CHK_CAMERA
    curNode = newNode(world, 'camera', curName);
    curName = null;
    const b = new Float32Array(8);
    for (let i = 0; i < 8; i++) { b[i] = f32(p.o); p.o += 4; }
    curNode.data = {
      pos: b.slice(0, 3), target: b.slice(3, 6),
      bank: fr(b[6] * Math.PI / 180),
      lens: 1.0, lx: 1.0, ly: 1.0,
      d: fr(1.0 * Math.tan(fr(b[7] * Math.PI / 180) / 2)),
      splines: {}, dummy: null, vista: null,
    };
    curObj = curNode.data;
  }

  function lightRd(p, end) {               // CHK_LIGHT (+nested)
    curNode = newNode(world, 'light', curName);
    curName = null;
    const b = new Float32Array(3);
    for (let i = 0; i < 3; i++) { b[i] = f32(p.o); p.o += 4; }
    curNode.data = {
      pos: b, dir: new Float32Array([0, 0, 1]), lightType: 'point',
      zMin: 0, zMax: 0, mult: 1, off: false, splines: {}, dummy: null,
      flare: null,
    };
    curObj = curNode.data;
    walk(p, end);
  }

  // -- keyframer ---------------------------------------------------------

  function splineFlagsRead(p, flags, key) {
    key.tens = key.cont = key.bias = key.easeIn = key.easeOut = 0;
    for (let i = 0; i < 16; i++) {
      if (!(flags & (1 << i))) continue;
      const d = f32(p.o); p.o += 4;
      if (i === 0) key.tens = d;
      else if (i === 1) key.cont = d;
      else if (i === 2) key.bias = d;
      else if (i === 3) key.easeIn = d;
      else if (i === 4) key.easeOut = d;
    }
  }

  function trackHead(p) {                  // 5 unknown shorts + n (int32)
    const unknown0 = u16(p.o);
    p.o += 10;
    const n = i32(p.o); p.o += 4;
    let flags = 0;
    if ((unknown0 & 3) === 2) flags = 'repeat';
    else if ((unknown0 & 3) === 3) flags = 'loop';
    return { n, flags };
  }

  function readTrackVector(p, sp) {
    for (const key of sp.keys) {
      key.frame = i32(p.o); p.o += 4;
      const flags = u16(p.o); p.o += 2;
      splineFlagsRead(p, flags, key);
      for (let j = 0; j < sp.dataSize; j++) { key.values[j] = f32(p.o); p.o += 4; }
    }
    curNode.flags.doAnim = true;
  }

  function readTrackQuat(p, sp) {
    const rot = new Float32Array(4);
    for (let i = 0; i < sp.keys.length; i++) {
      const key = sp.keys[i];
      key.frame = i32(p.o); p.o += 4;
      const flags = u16(p.o); p.o += 2;
      splineFlagsRead(p, flags, key);
      rot[0] = fr(-f32(p.o)); p.o += 4;    // 3ds angle sign quirk
      rot[1] = f32(p.o); p.o += 4;
      rot[2] = f32(p.o); p.o += 4;
      rot[3] = f32(p.o); p.o += 4;
      if (thetaVToQuat(key.values, rot) !== 0) {
        if (i > 0) multQuatEq(key.values, sp.keys[i - 1].values);
      } else if (i > 0) key.values.set(sp.keys[i - 1].values);
    }
    curNode.flags.doAnim = true;
  }

  function trackTargetSplineSlot() {
    // which spline slot does a POS track fill, per Current_Data
    if (currentData === 0xb002) return curNode.type === 'dummy' ? 'pos' : 'pos';
    if (currentData === 0xb003) { curObj.dummy = curDummy; return 'pos'; }
    if (currentData === 0xb004) return curNode.type === 'dummy' ? 'pos' : 'target';
    if (currentData === 0xb005) { if (curNode.type === 'dummy') return null; curObj.dummy = curDummy; return 'pos'; }
    if (currentData === 0xb006) return curNode.type === 'dummy' ? 'pos' : 'target';
    return null;
  }

  function trackPos(p) {
    const { n, flags } = trackHead(p);
    const slot = trackTargetSplineSlot();
    if (slot === null) return;             // light-track dummy: skipped
    const sp = newSpline(3, n);
    sp.flags = flags;
    readTrackVector(p, sp);
    for (const key of sp.keys) {           // sub-pivot offset
      key.values[0] = fr(key.values[0] - curNode.transform.pivot[0]);
      key.values[1] = fr(key.values[1] - curNode.transform.pivot[1]);
      key.values[2] = fr(key.values[2] - curNode.transform.pivot[2]);
    }
    curObj.splines[slot] = sp;
  }

  function trackRot(p) {
    const { n, flags } = trackHead(p);
    const sp = newSpline(4, n);
    sp.flags = flags;
    readTrackQuat(p, sp);
    curObj.splines.rot = sp;
    curNode.transform.type = 'quaternion';
  }

  function trackScale(p) {
    const { n, flags } = trackHead(p);
    const sp = newSpline(3, n);
    sp.flags = flags;
    readTrackVector(p, sp);
    curObj.splines.scale = sp;
  }

  function degreeTrack(p, slot) {
    const { n, flags } = trackHead(p);
    const sp = newSpline(1, n);
    sp.flags = flags;
    readTrackVector(p, sp);
    for (const key of sp.keys) key.values[0] = fr(key.values[0] * fr(Math.PI / 180));
    curObj.splines[slot] = sp;
  }

  function trackHide(p) {
    const { n, flags } = trackHead(p);
    const sp = newSpline(1, n);
    sp.flags = flags;
    let hide = curNode.flags.notActive ? 1 : 0;
    for (const key of sp.keys) {
      key.frame = i32(p.o); p.o += 4;
      const fl = u16(p.o); p.o += 2;
      splineFlagsRead(p, fl, key);
      key.values[0] = hide;
      hide ^= 1;
    }
    curObj.splines.hide = sp;
    curNode.flags.doAnim = true;
  }

  function trackMorph(p) {
    const { n, flags } = trackHead(p);
    const sp = newSpline(1, n);
    sp.flags = flags;
    const mesh = curObj;
    mesh.morphNodes = [];
    for (let i = 0; i < n; i++) {
      const key = sp.keys[i];
      key.frame = i32(p.o); p.o += 4;
      const fl = u16(p.o); p.o += 2;
      splineFlagsRead(p, fl, key);
      key.values[0] = i;
      const name = readString(p);
      const node = findByName(name);
      mesh.morphNodes.push(node);
      if (node && node !== curNode) node.flags.dontRender = true;
    }
    mesh.splines.morph = sp;
    if (mesh.morphNodes.includes(curNode)) {
      mesh.vertexCopy = mesh.vertex.slice();
      mesh.normalsCopy = mesh.normals.slice();
    }
    curNode.flags.doAnim = true;
  }

  function trackObjName(p) {               // CHK_NODE_HDR
    const name = readString(p);
    const w0 = u16(p.o); p.o += 4;
    const parent = u16(p.o); p.o += 2;
    curObj = null;
    if (currentData === 0xb002 || currentData === 0xb004 || currentData === 0xb006) {
      curNode = findByName(name);
      if (!curNode) {                      // dummy object
        curNode = newNode(world, 'dummy', name);
        curNode.data = { splines: {} };
        curDummy = curNode;
        curNode.flags.notActive = true;
      } else curDummy = null;
    } else if (currentData === 0xb003 || currentData === 0xb005) {
      curNode = findByName(name);
      if (!curNode) throw new Error(`keyframer: unknown ${name}`);
      // dummy promotion (camera/light tracks arriving before the object):
      // nsal.3ds does not exercise this path; guard and fail loudly.
      if ((currentData === 0xb003 && curNode.type !== 'camera') ||
          (currentData === 0xb005 && curNode.type !== 'light')) {
        throw new Error(`keyframer: dummy promotion needed for ${name} — not implemented`);
      }
      curNode.data.dummy = curDummy;
      curDummy = null;
    } else { curNode = null; return; }
    curNode.transform.type = 'quaternion';
    curNode.id = currentId;
    curObj = curNode.data;
    if (w0 & 0x0800) curNode.flags.notActive = true;
    if (parent !== 0xffff) {
      const upper = findById(parent);
      if (upper && upper !== curNode.parent) {
        if (curNode.parent) {
          const i = curNode.parent.children.indexOf(curNode);
          if (i >= 0) curNode.parent.children.splice(i, 1);
        }
        curNode.parent = upper;
        upper.children.push(curNode);
        upper.flags.dontRender = false;
      }
    }
  }

  function pivotRd(p) {                    // CHK_PIVOT
    const pv = new Float32Array(3);
    for (let i = 0; i < 3; i++) { pv[i] = f32(p.o); p.o += 4; }
    if (curNode) {
      const t = curNode.transform;
      t.pivot[0] = fr(t.pivot[0] + pv[0]);
      t.pivot[1] = fr(t.pivot[1] + pv[1]);
      t.pivot[2] = fr(t.pivot[2] + pv[2]);
      if (curNode.type === 'mesh') {
        const m = curNode.data;
        for (let i = 0; i < m.nbVertex; i++) {
          m.vertex[i * 3] = fr(m.vertex[i * 3] - pv[0]);
          m.vertex[i * 3 + 1] = fr(m.vertex[i * 3 + 1] - pv[1]);
          m.vertex[i * 3 + 2] = fr(m.vertex[i * 3 + 2] - pv[2]);
        }
      }
      if (curNode.parent) {
        const pp = curNode.parent.transform.pivot;
        t.pivot[0] = fr(t.pivot[0] - pp[0]);
        t.pivot[1] = fr(t.pivot[1] - pp[1]);
        t.pivot[2] = fr(t.pivot[2] - pp[2]);
      }
    }
  }

  function instanceName(p) {               // CHK_INSTANCE_NAME
    const name = readString(p);
    if (curNode) curNode.name = name;
  }

  // -- chunk dispatch (3d_io.c Chunk_Names[]) ----------------------------

  const READERS = {
    0x4d4d: null, 0x3d3d: null, 0xb000: null,     // containers
    0x4000: (p, end) => { curObj = null; curName = readString(p); walk(p, end); },
    0x4100: meshBlock,
    0x4110: vertList,
    0x4120: faceList,
    0x4130: faceMat,
    0x4140: mapList,
    0x4150: () => {},                              // smoothing: ignored
    0x4160: trMatrix,
    0x4700: cameraRd,
    0x4600: lightRd,
    0x4659: (p) => { curObj.zMin = f32(p.o); p.o += 4; },
    0x465a: (p) => { curObj.zMax = f32(p.o); p.o += 4; },
    0x465b: (p) => { curObj.mult = f32(p.o); p.o += 4; },
    0x4620: () => { curObj.off = true; },
    0xafff: null,                                  // material container
    0xa000: (p, end) => { world.materials.push(readString(p)); walk(p, end); },
    0xa300: (p) => { readString(p); },
    0xb008: (p) => {
      world.framesStart = i32(p.o);
      world.framesEnd = i32(p.o + 4);
      p.o += 8;
      curDummy = null;
    },
    0xb002: (p, end) => { currentData = 0xb002; walk(p, end); },
    0xb003: (p, end) => { currentData = 0xb003; walk(p, end); },
    0xb004: (p, end) => { currentData = 0xb004; walk(p, end); },
    0xb005: (p, end) => { currentData = 0xb005; walk(p, end); },
    0xb006: (p, end) => { currentData = 0xb006; walk(p, end); },
    0xb007: (p, end) => { currentData = 0xb005; walk(p, end); },
    0xb010: trackObjName,
    0xb011: instanceName,
    0xb013: pivotRd,
    0xb030: (p) => { currentId = i16(p.o) & 0xffff; p.o += 2; },
    0xb020: trackPos,
    0xb021: trackRot,
    0xb022: trackScale,
    0xb023: (p) => degreeTrack(p, 'fov'),
    0xb024: (p) => degreeTrack(p, 'roll'),
    0xb026: trackMorph,
    0xb029: trackHide,
  };

  function walk(p, finalPos) {             // Chunk_Read
    while (p.o < finalPos) {
      const id = u16(p.o);
      const len = i32(p.o + 2);
      const chunkStart = p.o;
      p.o += 6;
      if (!(id in READERS)) {
        if (len < 2) return;               // engine gives up here
        p.o = chunkStart + len;
        continue;
      }
      const end = chunkStart + len;
      const reader = READERS[id];
      if (reader) reader(p, end);
      else walk(p, end);
      p.o = end;
    }
  }

  walk({ o: 0 }, bytes.length);

  // world root: the engine hangs parentless nodes off Root; our nodes array
  // + parent links carry the same information.
  return world;
}

// ---- Finish-time mesh processing (mesh.c Finish, called via Set_Up_World)

// mesh.c Mesh_Store_Normals: computes face normals AND DELETES degenerate
// polys (duplicate points or zero-area) by compacting the array — poly
// indices that vista/cache files reference are POST-deletion indices.
function storeNormalsTrash(mesh) {
  const polys = mesh.polys, V = mesh.vertex;
  if (mesh.nbPolys === undefined) mesh.nbPolys = polys.length;
  for (let i = 0; i < mesh.nbPolys;) {
    const p = polys[i];
    let bad = p.pt[0] === p.pt[1] || p.pt[1] === p.pt[2] || p.pt[2] === p.pt[0];
    if (!bad) {
      const v0 = p.pt[0] * 3, v1 = p.pt[1] * 3, v2 = p.pt[2] * 3;
      const dx1 = fr(V[v1] - V[v0]), dx2 = fr(V[v2] - V[v0]);
      const dy1 = fr(V[v1 + 1] - V[v0 + 1]), dy2 = fr(V[v2 + 1] - V[v0 + 1]);
      const dz1 = fr(V[v1 + 2] - V[v0 + 2]), dz2 = fr(V[v2 + 2] - V[v0 + 2]);
      p.no[0] = fr(fr(dy1 * dz2) - fr(dy2 * dz1));
      p.no[1] = fr(fr(dx2 * dz1) - fr(dx1 * dz2));
      p.no[2] = fr(fr(dx1 * dy2) - fr(dx2 * dy1));
      let n = fr(fr(fr(p.no[0] * p.no[0]) + fr(p.no[1] * p.no[1])) + fr(p.no[2] * p.no[2]));
      if (n <= 0) bad = true;
      else {
        n = fr(1 / Math.sqrt(n));
        p.no[0] = fr(p.no[0] * n); p.no[1] = fr(p.no[1] * n); p.no[2] = fr(p.no[2] * n);
        i++;
      }
    }
    if (bad) {
      // C: Nb_Polys--; for(I=i;I<Nb_Polys;++I) Polys[I]=Polys[I+1];
      // The ARRAY IS NOT SHRUNK — the slots past Nb_Polys keep the stale
      // structs, and the vista still indexes them for meshes whose polys
      // were all trashed (see the tail snapshot below).
      mesh.nbPolys--;
      for (let I = i; I < mesh.nbPolys; ++I) polys[I] = polys[I + 1];
    }
  }
  // C copies POLY structs by value, so the stale tail is independent of the
  // live polys. Our shift moved references, so snapshot the tail to break
  // the aliasing — otherwise later per-poly writes (UVs, ids) would leak
  // into slots the C left untouched.
  for (let k = mesh.nbPolys; k < polys.length; k++) {
    const q = polys[k];
    polys[k] = { ...q, pt: q.pt.slice(), no: q.no.slice(), uvI: q.uvI ? q.uvI.slice() : q.uvI };
  }
}

function averageNormalsPub(mesh) { averageNormals(mesh); }

// box.c Set_Up_Box + Set_BBox_From_Min_Max (FLT chains). Mins/Maxs live on
// the NODE's box in the C (Node->Box) and are object-space: the default
// (non-vista) render path projects the 8 corners per frame to cull and to
// key the object sort (box.c Project_Box).
function boxRadius(mesh, node) {
  const mins = [Infinity, Infinity, Infinity], maxs = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.nbVertex; i++) {
    for (let a = 0; a < 3; a++) {
      const v = mesh.vertex[i * 3 + a];
      if (v < mins[a]) mins[a] = v;
      if (v > maxs[a]) maxs[a] = v;
    }
  }
  let r = 0;
  for (const a of [1, 2, 0]) {            // C's accumulation order y,z,x
    const t = fr(maxs[a] - mins[a]);
    r = fr(r + (t < 0.01 ? fr(0.01) : fr(t * t)));
  }
  const radius = fr(Math.sqrt(r));
  if (node) node.box = { mins, maxs, radius };
  return radius;
}

// optimesh.c Optimize_Mesh: O(n^2) weld. Quirks preserved: threshold is
// LINEAR (0.05*radius) compared against SQUARED distance, and the midpoint
// nudge writes vertex j (the scan index), not the weld target k.
function optimizeMesh(mesh, thresh) {
  if (!mesh || mesh.nbVertex === 0 || mesh.polys.length === 0 || thresh === 0) return;
  const V = mesh.vertex, n0 = mesh.nbVertex;
  const vtxIndex = new Uint16Array(n0);
  for (let i = 0; i < n0; i++) {
    vtxIndex[i] = i;
    for (let j = 0; j < i; j++) {
      const k = vtxIndex[j];
      const d0 = fr(V[i * 3] - V[k * 3]);
      let r = fr(d0 * d0);
      const d1 = fr(V[i * 3 + 1] - V[k * 3 + 1]);
      r = fr(r + fr(d1 * d1));
      const d2 = fr(V[i * 3 + 2] - V[k * 3 + 2]);
      if (fr(r + fr(d2 * d2)) <= thresh) {
        vtxIndex[i] = k;
        V[j * 3] = fr(V[j * 3] + fr(d0 / 2));
        V[j * 3 + 1] = fr(V[j * 3 + 1] + fr(d1 / 2));
        V[j * 3 + 2] = fr(V[j * 3 + 2] + fr(d2 / 2));
      }
    }
  }
  let n = 0;
  for (let i = 0; i < n0; i++) if (vtxIndex[i] === i) n++;
  if (n === n0) return;
  const neu = new Float32Array(n * 3);
  n = 0;
  for (let i = 0; i < n0; i++) {
    const j = vtxIndex[i];
    if (j === i) {
      vtxIndex[i] = n;
      neu[n * 3] = V[i * 3]; neu[n * 3 + 1] = V[i * 3 + 1]; neu[n * 3 + 2] = V[i * 3 + 2];
      n++;
    } else vtxIndex[i] = vtxIndex[j];
  }
  mesh.nbVertex = n;
  mesh.vertex = neu;
  mesh.normals = new Float32Array(n * 3);
  for (let i = 0; i < mesh.nbPolys; i++) {          // C: i < Nb_Polys
    const p = mesh.polys[i];
    for (let k = 0; k < p.nbPts; k++) p.pt[k] = vtxIndex[p.pt[k]];
  }
  storeNormalsTrash(mesh);                // post-weld degenerates deleted
  averageNormals(mesh);
}

// mesh.c Finish (the parts the vista render path depends on)
/**
 * `opts.weld` mirrors the original's own `_SKIP_OPTIM_MESH_` compile switch
 * (include/main3d.h:26, consumed at 3d_src/optimesh.c:163 and mesh.c:467).
 * It defaults to ON because no makefile in the release defines that symbol, so
 * the weld ran in the shipped binary and the restored show must run it too.
 *
 * It exists as a parameter only so the weld can be MEASURED -- work/tools/
 * weld-census.mjs loads each mesh both ways and reports what the weld costs.
 * Nothing in the show passes it.
 */
export function finishMeshes(world, { weld = true } = {}) {
  for (const node of world.nodes) {
    if (node.type !== 'mesh') continue;
    const mesh = node.data;
    storeNormalsTrash(mesh);
    averageNormalsPub(mesh);
    const radius = boxRadius(mesh, node);   // Set_Up_Box, pre-weld (mesh.c:434)
    if (!mesh.hasUV) throw new Error(`spherical mapping not ported (${node.name})`);
    // Mesh_Store_UV_Mapping: per-poly USHORT UVs from vertex UVs; the
    // cache overwrites these for its polys — kept for the uncovered rest.
    for (const p of mesh.polys.slice(0, mesh.nbPolys)) {
      p.uvI = [];
      for (let k = 0; k < p.nbPts; k++) {
        let mu = mesh.uv[p.pt[k] * 2] % 1;
        p.uvI.push((mu * 65535) & 0xffff);
        mu = mesh.uv[p.pt[k] * 2 + 1] % 1;
        p.uvI.push((mu * 65535) & 0xffff);
      }
      const P0 = p.pt[0], P1 = p.pt[1], P2 = p.pt[2];
      const dU1 = fr(mesh.uv[P1 * 2] - mesh.uv[P0 * 2]);
      const dU2 = fr(mesh.uv[P2 * 2] - mesh.uv[P1 * 2]);
      const dV1 = fr(mesh.uv[P1 * 2 + 1] - mesh.uv[P0 * 2 + 1]);
      const dV2 = fr(mesh.uv[P2 * 2 + 1] - mesh.uv[P1 * 2 + 1]);
      p.u1v2u2v1 = fr(fr(dU1 * dV2) - fr(dU2 * dV1));
    }
    if (weld) optimizeMesh(mesh, fr(fr(0.05) * radius));
    for (let i = 0; i < mesh.nbPolys; i++) mesh.polys[i].id = i;
  }
}
