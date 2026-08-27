// state-of-mind — marching cubes (3d_src/mcube/mcube3.c).
//
// The blob part builds an implicit surface from 5 metaball centres on a
// 32x32x32 grid every frame. This is "Marching cubes IV" in Skal's numbering:
// an incremental variant that carries a per-column bitmask of which cells
// were active last frame (Flags1/Flags2), grows it by one cell in every
// direction (MC_Grow_Neighbours), and only evaluates the field there. The
// growth step is what makes it cheap; it also means the surface can only
// move one cell per frame, and the source itself notes it "WILL bug on
// boundary" — both preserved.
//
// The field is sampled through a 256-entry table rather than evaluated
// analytically, so the isosurface has the table's quantization baked in.

const fr = Math.fround;

export const MCX = 32, MCY = 32, MCZ = 32;
const BLOB_TABLE_SIZE = 256;
const MC_PTS = 32;
export const MAX_CENTERS = 8;

// ---- static tables -------------------------------------------------------

const polyOrder = Array.from({ length: 256 }, () => new Uint8Array(MC_PTS).fill(255));
const edgeOrder = Array.from({ length: 256 }, () => new Uint8Array(16).fill(255));
const blobTable = new Float32Array(BLOB_TABLE_SIZE);
const blobTable2 = new Float32Array(BLOB_TABLE_SIZE);
const nghTab = new Uint32Array(256);

const MC_MASKS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
const MC_ORDER = [11, 1, 2, 0, 4, 6, 3, 7, 5, 8, 9, 10];
const R1_TAB = [4, 5, 1, 0, 7, 6, 2, 3];
const R2_TAB = [1, 5, 6, 2, 0, 4, 7, 3];
const R3_TAB = [1, 2, 3, 0, 5, 6, 7, 4];
const R1_E = [4, 9, 0, 8, 6, 10, 2, 11, 7, 5, 1, 3];
const R2_E = [9, 5, 10, 1, 8, 7, 11, 3, 0, 4, 6, 2];
const R3_E = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8];

// The 27 base configurations, rotated into all 256 by Build_MCube_Tab.
// Transcribed verbatim, including the entries the source marks /* XXX */
// (0xde and 0x5e contain what look like typos — 0x5e lists "4,5,5" — but
// they are what the demo renders).
const MC_15_POLYS = [
  [0x01, 0, 3, 8],
  [0x03, 1, 3, 8, 1, 8, 9],
  [0x83, 1, 3, 8, 1, 8, 9, 6, 7, 11],
  [0x07, 8, 9, 10, 8, 10, 3, 2, 3, 10],
  [0x87, 8, 9, 10, 8, 10, 3, 2, 3, 10, 6, 7, 11],
  [0x0f, 8, 9, 10, 8, 10, 11],
  [0x21, 0, 3, 8, 4, 5, 9],
  [0xa1, 0, 3, 8, 4, 5, 9, 6, 7, 11],
  [0x41, 0, 3, 8, 5, 6, 10],
  [0x1b, 4, 9, 1, 4, 1, 2, 2, 7, 4, 2, 11, 7],
  [0x1d, 0, 1, 4, 1, 11, 4, 1, 10, 11, 11, 7, 4],
  [0x17, 2, 7, 9, 2, 3, 7, 9, 7, 4, 2, 9, 10],
  [0x55, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6],
  [0xa5, 0, 3, 8, 4, 5, 9, 6, 7, 11, 2, 1, 10],
  [0xFE, 0, 8, 3],
  [0xFC, 1, 8, 3, 1, 9, 8],
  [0x7C, 1, 8, 3, 1, 9, 8, 6, 11, 7, 3, 8, 7, 3, 7, 11],
  [0xf8, 8, 10, 9, 8, 3, 10, 2, 10, 3],
  [0x78, 8, 10, 9, 8, 3, 10, 2, 10, 3, 6, 11, 7],
  [0xf0, 8, 10, 9, 8, 11, 10],
  [0xde, 0, 8, 3, 4, 9, 5, 4, 0, 9, 4, 8, 0],
  [0x5e, 0, 8, 3, 4, 5, 5, 6, 11, 7],
  [0xbe, 0, 8, 3, 5, 10, 6],
  [0xe2, 0, 4, 1, 1, 4, 11, 1, 11, 10, 11, 4, 7],
  [0xe8, 2, 9, 7, 2, 7, 3, 9, 4, 7, 2, 10, 9],
  [0xaa, 0, 5, 1, 0, 4, 5, 2, 7, 3, 2, 6, 7],
  [0x5a, 0, 8, 3, 4, 9, 5, 6, 11, 7, 2, 10, 1],
];

let tablesBuilt = false;
function buildTables() {
  if (tablesBuilt) return;
  tablesBuilt = true;

  for (const cfg of MC_15_POLYS) {
    const I = cfg[0];
    if (I === 0x00 || I === 0xff) continue;
    for (let r1 = 0; r1 < 4; r1++) {
      for (let r2 = 0; r2 < 4; r2++) {
        for (let r3 = 0; r3 < 4; r3++) {
          let newCfg = 0;
          for (let j = 7; j >= 0; --j) {
            if (!(I & MC_MASKS[j])) continue;
            let k = j;
            for (let r = 0; r < r1; ++r) k = R1_TAB[k];
            for (let r = 0; r < r2; ++r) k = R2_TAB[k];
            for (let r = 0; r < r3; ++r) k = R3_TAB[k];
            newCfg |= MC_MASKS[k];
          }
          if (polyOrder[newCfg][0] !== 255) continue;
          for (let j = 0; j < 16; ++j) {
            let k = cfg[j + 1];
            if (k === undefined) break;         // C reads a 255 terminator
            for (let r = 0; r < r1; ++r) k = R1_E[k];
            for (let r = 0; r < r2; ++r) k = R2_E[k];
            for (let r = 0; r < r3; ++r) k = R3_E[k];
            polyOrder[newCfg][j] = k;
          }
        }
      }
    }
  }

  // Compute_Edge_Order: the edges each config touches, in MC_Order sequence
  for (let i = 0; i < 256; ++i) {
    let total = 0;
    for (let k = 0; k < 12; ++k) {
      for (let n = 0; n < 12; ++n) {
        if (polyOrder[i][n] === MC_ORDER[k]) { edgeOrder[i][total++] = MC_ORDER[k]; break; }
      }
    }
    while (total !== 16) edgeOrder[i][total++] = 0xff;
  }

  // Build_Default_Blob_Table (doubles, like the C: cos/sin are double here)
  for (let i = BLOB_TABLE_SIZE - 1; i >= 0; --i) {
    let x = i / BLOB_TABLE_SIZE;
    x = fr(Math.sqrt(x));
    blobTable[i] = fr(Math.cos(x * Math.PI) / 2.0 + 0.5);
    if (i !== 0) blobTable2[i] = fr(Math.sin(x * Math.PI) / x);
  }

  // Build_Ngh_Table: 3 neighbours along z per set bit
  for (let i = 255; i >= 0; --i) {
    let v = 0;
    for (let j = 7; j >= 0; --j) {
      let bit = 1 << (j + 8);
      bit = (bit | (bit << 1) | (bit >>> 1)) >>> 0;
      if (i & (1 << j)) v = (v | bit) >>> 0;
    }
    nghTab[i] = v;
  }
}

// ---- blob instance -------------------------------------------------------

export function createBlob() {
  buildTables();
  const b = {
    nbCenters: 0,
    radius: new Float32Array(MAX_CENTERS),
    radius2: new Float32Array(MAX_CENTERS),
    invRadius2: new Float32Array(MAX_CENTERS),
    strength: new Float32Array(MAX_CENTERS),
    G: Array.from({ length: MAX_CENTERS }, () => new Float32Array(3)),
    K: 0,
    // grid state, persistent across frames (this is the whole point of the
    // incremental variant)
    lastFlags: new Uint32Array(2 * MCX * MCY),
    rawFlags: new Uint8Array(2 * MCX * MCY),
    rawSlice: new Float32Array(2 * MCX * MCY),
    xEdge: new Uint16Array(MCX * MCY),
    yEdge: new Uint16Array(MCX * MCY),
    zEdge: new Uint16Array(MCY),
    topYEdge: new Uint16Array(MCY),
    f1: 0, f2: MCX * MCY,                     // offsets into lastFlags
    dims: null,
    mesh: null,
  };
  return b;
}

export function mcBlobify2(blob, radius, strength) {
  if (blob.nbCenters === MAX_CENTERS) return;
  const i = blob.nbCenters++;
  blob.radius[i] = radius;
  blob.radius2[i] = fr(radius * radius);
  blob.invRadius2[i] = fr(fr(1.0 * BLOB_TABLE_SIZE) / fr(radius * radius));
  blob.strength[i] = strength;
  blob.G[i][0] = blob.G[i][1] = blob.G[i][2] = 0;
}

export function setMCDimensions(blob, xo, yo, zo, dx, dy, dz) {
  const d = {
    offX: fr(xo - dx), offY: fr(yo - dy), offZ: fr(zo - dz),
    scaleX: fr(2.0 * dx / MCX), scaleY: fr(2.0 * dy / MCY), scaleZ: fr(2.0 * dz / MCZ),
  };
  d.offEndX = fr(d.offX + fr((MCX - 1) * d.scaleX));
  d.offEndY = fr(d.offY + fr((MCY - 1) * d.scaleY));
  d.offEndZ = fr(d.offZ + fr((MCZ - 1) * d.scaleZ));
  blob.dims = d;
}

// MC_Finish_Blob2: allocate the mesh and prime the flags so the first
// Do_MCube evaluates the whole interior region.
export function mcFinishBlob(blob, maxVtx, maxPolys) {
  const polys = [];
  for (let i = 0; i < maxPolys; i++) {
    polys.push({ pt: new Uint16Array(3), no: new Float32Array(3), nbPts: 3, uvI: null, mips: null, clipped: 0, id: i });
  }
  blob.mesh = {
    vertex: new Float32Array(maxVtx * 3),
    normals: new Float32Array(maxVtx * 3),
    uv: null, hasUV: false,
    nbVertex: 0, polys, nbPolys: 0,
  };
  // Build_Ngh_Table's priming half: all interior columns marked live
  const F = blob.lastFlags;
  F.fill(0);
  for (let i = 2; i < MCX - 2; ++i) {
    for (let j = 2; j < MCY - 2; ++j) F[i * MCY + j] = 0x3FFFFFFC;
  }
  blob.f1 = 0; blob.f2 = MCX * MCY;
  blob.rawFlags.fill(0);
  blob.rawSlice.fill(0);
  // the C seeds vertices 0/1 with the grid corners so the bbox comes out right
  const d = blob.dims;
  blob.mesh.vertex[0] = d.offX; blob.mesh.vertex[1] = d.offY; blob.mesh.vertex[2] = d.offZ;
  blob.mesh.vertex[3] = fr(d.offX + fr(d.scaleX * MCX));
  blob.mesh.vertex[4] = fr(d.offY + fr(d.scaleY * MCY));
  blob.mesh.vertex[5] = fr(d.offZ + fr(d.scaleZ * MCZ));
  return blob.mesh;
}

// ---- the per-frame surface build ----------------------------------------

const slice = new Float32Array(3);

// MC_Grow_Neighbours must run EVERY frame, at any frame rate. It is not an
// animation whose speed can be re-timed: it is a spatial tracker paired with
// the prune below, where the emit pass clears the bit for any cell whose
// field turned out to be outside (`F[f1+off] ^= mcMask`). Dilate and prune
// are a matched pair. Skipping the dilate on some frames while the prune
// keeps running erodes the one-cell boundary ring that detects cells the
// surface has just moved into, and the blob visibly breaks up.
//
// The blob's size and position are already time-driven — the five centres are
// pure functions of Global_x (blb.c Set_Blob_Position) — so nothing here
// encodes speed. At a higher rate the surface moves LESS per frame, so
// growing every frame tracks strictly better; the pair is self-limiting
// (after prune the set is just the inside cells again), so it cannot run
// away. work/re/FRAMERATE.md 3.4 originally claimed otherwise and was wrong.
export function doMCube3(blob) {
  const mesh = blob.mesh;
  mesh.nbVertex = 0;
  mesh.nbPolys = 0;

  let flagsA = 0, flagsB = MCX * MCY;          // offsets into rawFlags
  let sliceA = 0, sliceB = MCX * MCY;          // offsets into rawSlice
  const RF = blob.rawFlags, RS = blob.rawSlice;
  const F = blob.lastFlags;
  const d = blob.dims;

  // --- MC_Grow_Neighbours: grow f1 into f2, then swap so f1 is current
  {
    const base = MCY + 1;
    let f1 = blob.f1, f2 = blob.f2;
    // C flips the pointers BEFORE growing, so the write target is the old f1
    const t = f1; f1 = f2; f2 = t;
    blob.f1 = f1; blob.f2 = f2;
    // now F[f2 + ...] is the "previous" table and F[f1 + ...] the target
    for (let k = MCX * MCY - MCY - 2 - MCY - 1; k >= 0; --k) {
      let prev = F[f2 + base + k];
      if (prev === 0) continue;
      let column = nghTab[prev & 0xff] >>> 8;
      prev >>>= 8;
      column |= nghTab[prev & 0xff];
      prev >>>= 8;
      column = (column | (nghTab[prev & 0xff] << 8)) >>> 0;
      prev >>>= 8;
      column = (column | (nghTab[prev & 0xff] << 16)) >>> 0;
      const o = f1 + base + k;
      F[o - 1 - MCY] |= column; F[o - MCY] |= column; F[o + 1 - MCY] |= column;
      F[o - 1] |= column; F[o] |= column; F[o + 1] |= column;
      F[o - 1 + MCY] |= column; F[o + MCY] |= column; F[o + 1 + MCY] |= column;
    }
  }

  // --- Emit_MC_Polys2
  let mcMask = 0x00000001;
  slice[2] = d.offZ;
  const f1 = blob.f1;

  for (let k = MCZ - 1; k > 0; --k) {
    { const t = sliceA; sliceA = sliceB; sliceB = t; }
    { const t = flagsA; flagsA = flagsB; flagsB = t; }
    mcMask = (mcMask << 1) >>> 0;
    slice[2] = fr(slice[2] + d.scaleZ);
    slice[0] = d.offX;
    let off = MCY + 1;

    for (let i = MCX - 2; i > 0; --i) {
      slice[0] = fr(slice[0] + d.scaleX);
      slice[1] = d.offY;
      for (let j = MCY - 2; j > 0; --j) {
        slice[1] = fr(slice[1] + d.scaleY);
        if (F[f1 + off] & mcMask) {
          if (!doV(blob, RS, sliceB + off)) {   // field <= 0 -> inside
            RF[flagsB + off + 1 + MCY] |= 0x01;
            RF[flagsB + off + MCY] |= 0x08;
            RF[flagsB + off + 1] |= 0x02;
            RF[flagsB + off] |= 0x04;
            RF[flagsA + off + 1 + MCY] |= 0x10;
            RF[flagsA + off + MCY] |= 0x80;
            RF[flagsA + off + 1] |= 0x20;
            RF[flagsA + off] |= 0x40;
          } else F[f1 + off] ^= mcMask;
        }
        const fa = RF[flagsA + off];
        if (fa !== 0x00) {
          if (fa !== 0xff) emitPolys3(blob, RS, sliceA, sliceB, off, fa, j);
          RF[flagsA + off] = 0x00;
        }
        off++;
      }
      off += 2;
    }
  }

  storeNormals2(mesh);
}

// Do_V: sum the metaball field at slice[] into Slice_B[Off]
function doV(blob, RS, dst) {
  let V = fr(-blob.K);
  for (let k = blob.nbCenters - 1; k >= 0; --k) {
    let tmp = fr(slice[0] - blob.G[k][0]);
    let x = fr(tmp * tmp);
    if (x >= blob.radius2[k]) continue;
    tmp = fr(slice[1] - blob.G[k][1]);
    x = fr(x + fr(tmp * tmp));
    if (x >= blob.radius2[k]) continue;
    tmp = fr(slice[2] - blob.G[k][2]);
    x = fr(x + fr(tmp * tmp));
    if (x >= blob.radius2[k]) continue;
    x = fr(x * blob.invRadius2[k]);
    V = fr(V + blobTable[Math.floor(x)]);
  }
  RS[dst] = V;
  return !(V > 0);
}

function computeNormal(blob, vtx) {
  const mesh = blob.mesh, N = mesh.normals, Vx = mesh.vertex;
  const o = vtx * 3;
  let n0 = 0, n1 = 0, n2 = 0;
  const x = Vx[o], y = Vx[o + 1], z = Vx[o + 2];
  for (let i = blob.nbCenters - 1; i >= 0; --i) {
    const X = fr(x - blob.G[i][0]), Y = fr(y - blob.G[i][1]), Z = fr(z - blob.G[i][2]);
    const tmp = fr(fr(fr(X * X) + fr(Y * Y)) + fr(Z * Z));
    const t2 = Math.floor(fr(tmp * blob.invRadius2[i]));
    if (t2 < BLOB_TABLE_SIZE) {
      const b = blobTable2[t2];
      n0 = fr(n0 + fr(X * b)); n1 = fr(n1 + fr(Y * b)); n2 = fr(n2 + fr(Z * b));
    }
  }
  let s = fr(fr(fr(n0 * n0) + fr(n1 * n1)) + fr(n2 * n2));
  if (s > 0) {
    s = fr(1 / Math.sqrt(s));
    n0 = fr(n0 * s); n1 = fr(n1 * s); n2 = fr(n2 * s);
  }
  N[o] = n0; N[o + 1] = n1; N[o + 2] = n2;
}

const edgeTable = new Uint16Array(12);

function emitPolys3(blob, RS, sliceA, sliceB, off, index, jo) {
  const mesh = blob.mesh, d = blob.dims;
  const eo = edgeOrder[index];
  let i = 0, vtx = 0;
  for (;;) {
    const edge = eo[i];
    if (edge === 0xff) break;
    switch (edge) {
      case 0: vtx = blob.xEdge[off]; break;
      case 1: vtx = blob.yEdge[off + MCY]; break;
      case 2: vtx = blob.xEdge[off + 1]; break;
      case 3: vtx = blob.yEdge[off]; break;
      case 4: vtx = blob.xEdge[off] = blob.vtx6; break;
      case 5: {
        vtx = mesh.nbVertex++;
        blob.topYEdge[jo] = vtx;
        mesh.vertex[vtx * 3] = slice[0];
        mesh.vertex[vtx * 3 + 2] = slice[2];
        const y = fr(RS[sliceB + off] / fr(RS[sliceB + off - 1] - RS[sliceB + off]));
        mesh.vertex[vtx * 3 + 1] = fr(slice[1] + fr(y * d.scaleY));
        computeNormal(blob, vtx);
        break;
      }
      case 6: {
        vtx = blob.vtx6 = mesh.nbVertex++;
        mesh.vertex[vtx * 3 + 1] = slice[1];
        mesh.vertex[vtx * 3 + 2] = slice[2];
        const x = fr(RS[sliceB + off] / fr(RS[sliceB + off - MCY] - RS[sliceB + off]));
        mesh.vertex[vtx * 3] = fr(slice[0] + fr(x * d.scaleX));
        computeNormal(blob, vtx);
        break;
      }
      case 7: vtx = blob.yEdge[off] = blob.topYEdge[jo]; break;
      case 8: vtx = blob.zEdge[jo]; break;
      case 9: vtx = blob.zEdge[jo] = blob.vtx10; break;
      case 10: {
        vtx = blob.vtx10 = mesh.nbVertex++;
        mesh.vertex[vtx * 3] = slice[0];
        mesh.vertex[vtx * 3 + 1] = slice[1];
        const z = fr(RS[sliceB + off] / fr(RS[sliceA + off] - RS[sliceB + off]));
        mesh.vertex[vtx * 3 + 2] = fr(slice[2] + fr(z * d.scaleZ));
        computeNormal(blob, vtx);
        break;
      }
      case 11: vtx = blob.zEdge[jo - 1]; break;
      default: break;
    }
    edgeTable[edge] = vtx;
    i++;
  }

  const po = polyOrder[index];
  i = 0;
  while (po[i] !== 0xff) {
    const p = mesh.polys[mesh.nbPolys++];
    p.pt[0] = edgeTable[po[i++]];
    p.pt[1] = edgeTable[po[i++]];
    p.pt[2] = edgeTable[po[i++]];
  }
}

// normal.c Mesh_Store_Normals2 — face normals, no degenerate deletion
function storeNormals2(mesh) {
  const V = mesh.vertex;
  for (let i = 0; i < mesh.nbPolys; i++) {
    const p = mesh.polys[i];
    const v0 = p.pt[0] * 3, v1 = p.pt[1] * 3, v2 = p.pt[2] * 3;
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
