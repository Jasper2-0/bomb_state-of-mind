// state-of-mind — vista renderer (the tunnel/city draw path).
//
// Reimplements the precomputed-visibility render pipeline: the .vst chunk
// walk (3d_src/vista.c), per-poly camera transform + projection
// (3d_src/vmesh.c Vista_Select_Poly), the counting sort (sort.c), the
// X/Z Sutherland-Hodgman clip with 1/z near splits (clip.c), the
// fixed-point edge rasterizer (shaders/raster.c), the perspective-correct
// UV gradient setup (sh_uvc.c + cache3.c) and the span loops
// (draw16/loops216). FLT-typed math is fround-disciplined; doubles stay
// doubles. Deliberate engine quirks preserved: near-clip splits lerp the
// pre-projection xp/yp; RENDER_UVC_2 dispatches to the 4-pixel span
// function (Primitives_16 table).

import { aV, nAV } from './world3ds.js';
import { MIP_WIDTH, MIP_MASKS } from './cache3d.js';
import { retreiveTextureII } from './cache2.js';

const fr = Math.fround;

import { W, H, NH } from './screen.js';   // was a second, independent declaration

// Per-scanline scratch. Sized for the NATIVE height, not the 320x200 one:
// these are indexed by screen y, and a typed array silently DROPS
// out-of-bounds writes, so at 1200 rows an undersized array does not throw --
// it just stops rasterising everything below row 255. That failure is
// invisible except as missing geometry.
const SCAN_ROWS = NH + 8;
const VTX_XO = 0x04, VTX_X1 = 0x08, VTX_YO = 0x10, VTX_Y1 = 0x20, VTX_ZO = 0x40, VTX_Z1 = 0x80;
const VTX_TESTED = 0x02, VTX_VISIBLE = 0x03;
const CLIP_USEFUL = VTX_XO | VTX_X1 | VTX_ZO | VTX_Z1;
const B_CULL_EPSILON = fr(0.75), Z_EPSILON = fr(0.00001);
const AREA_EPSILON2 = fr(0.25), AREA_EPSILON3 = fr(0.2);
const UVC_LIN = fr(0.07), UVC_32 = fr(0.20), UVC_16 = fr(0.30), UVC_8 = fr(0.44), UVC_4 = fr(0.55);
const EPS_MIP = [1 / 256, 4 / 256, 8 / 256, 16 / 256, 32 / 256, 64 / 256, 128 / 256].map(fr);

// ---- .vst parsing (vista.c Load_Vista) -----------------------------------

import { compressRead } from './cache3d.js';

function loadLzwU16(bytes, posRef) {
  // Load_LZW_Block_II framing: u32 size + Compress_Read stream; the
  // block-buffered decoder returns the exact stream position, matching
  // where the engine's FILE* lands for the next record.
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const total = dv.getUint32(posRef.o, true); posRef.o += 4;
  const out = new Uint8Array(total);
  posRef.o = compressRead(bytes, posRef.o, out, total);
  return new Uint16Array(out.buffer, 0, total >> 1);
}

export function parseVista(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const p = { o: 0 };
  const maxPolys0 = dv.getInt32(0, true);
  const cameraId = dv.getUint16(4, true);
  const lens = dv.getFloat32(8, true);
  const clip = dv.getFloat32(12, true);
  const nbChunks = dv.getInt32(16, true);
  p.o = 20;
  const chunks = [];
  for (let i = 0; i < nbChunks; i++) {
    const time = dv.getFloat32(p.o, true);
    const type = dv.getInt32(p.o + 4, true);
    const zMin = dv.getFloat32(p.o + 8, true);
    const zMax = dv.getFloat32(p.o + 12, true);
    const nbPolys = dv.getInt32(p.o + 16, true);
    p.o += 20;
    let objId = null, polyId = null;
    if (nbPolys) {
      objId = loadLzwU16(bytes, p);
      polyId = loadLzwU16(bytes, p);
    }
    chunks.push({ time, type, zMin, zMax, nbPolys, objId, polyId });
  }
  return {
    maxPolys: maxPolys0 * 2 + 20, cameraId, lens, clip, chunks,
    curTime: Infinity, curChunkIdx: -1, curNb: 0,
    objId: new Uint16Array(maxPolys0 * 2 + 20),
    polyId: new Uint16Array(maxPolys0 * 2 + 20),
  };
}

// ---- counting sort (sort.c Counting_Sort, UNIX C version) ----------------

export function countingSort(a, nb) {
  if (!(nb - 1)) return;                  // C: if (!--Nb) return — 1 poly stays
  const n0 = nb - 1;
  const c = new Uint16Array(512);
  const b = new Uint32Array(nb);
  for (let n = n0; n >= 0; --n) c[a[n] & 0xff]++;
  for (let n = 1; n < 256; ++n) c[n] += c[n - 1];
  for (let n = n0; n >= 0; --n) {
    b[--c[a[n] & 0xff]] = a[n];
    c[256 + ((a[n] & 0xff00) >> 8)]++;
  }
  for (let n = 257; n < 512; ++n) c[n] += c[n - 1];
  for (let n = n0; n >= 0; --n) {
    a[--c[256 + ((b[n] & 0xff00) >> 8)]] = b[n];
  }
}

// ---- the renderer --------------------------------------------------------

export function createVistaRenderer(world, camNode) {
  const byId = new Map(world.nodes.map((n) => [n.id, n]));

  // render state (_RCst_ subset)
  const rc = {
    clips: new Float32Array(8),           // x0 x1 y0 y1 zmin 1/zmin zmax 1/zmax
    outClips: new Float32Array(6),
    lx: W / 2, ly: H / 2, cx: W / 2, cy: H / 2,   // defaults; setupCamScreen overrides
    // Destination geometry. The renderer takes its target buffer per call, so
    // these are all that bind it to a resolution: bps is the row pitch and
    // rows the height used for the flare visibility bounds. setupTarget()
    // switches them for a native-resolution pass.
    bps: W, rows: H,
    // Screen-space sprite scale. Lens flares are fixed-size bitmaps pasted in
    // screen space, so at 1600x1200 they would keep their pixel size and
    // shrink to dots relative to the frame -- the same trap as drawing vector
    // line art one pixel wide. setupTarget scales this with the target.
    spriteScale: 1,
    zMin: 0, zMax: 0, zScale: 0, zVistaScale: 0, zOff: 0,
    // INV_Z_SCALE (raster.h): 65536*32768*64 = 2^37, exact in f32.
    globalZScale: fr(65536 * 32768 * 64),
  };

  function setupClipping(x0, x1, y0, y1, zmin, zmax) {
    rc.clips[0] = x0; rc.clips[1] = x1; rc.clips[2] = y0; rc.clips[3] = y1;
    rc.clips[4] = zmin; rc.clips[5] = fr(1 / zmin);
    if (zmax < zmin) zmax = 1e30;
    rc.clips[6] = zmax; rc.clips[7] = fr(1 / zmax);
  }
  function setupTarget(w, h) {
    rc.bps = w; rc.rows = h;
    // Uniform: native pixels are square, so a flare should be round rather
    // than carrying the source's 5:6 pixel aspect (work/re/RESOLUTION.md 6).
    rc.spriteScale = fr((w / W + h / H) / 2);
  }
  function setupCamScreen(lx, ly, cx, cy) {
    rc.lx = lx; rc.ly = ly; rc.cx = cx; rc.cy = cy;
    rc.outClips[0] = fr(-4 * lx); rc.outClips[1] = fr(4 * lx);
    rc.outClips[2] = fr(-4 * ly); rc.outClips[3] = fr(4 * ly);
    rc.outClips[4] = fr(-10 * lx); rc.outClips[5] = 1e30;
  }

  // vertex pools
  const MAXV = 8192;
  const pV = Array.from({ length: MAXV }, () => ({ n0: 0, n1: 0, n2: 0, invZ: 0, xp: 0, yp: 0, u: 0, v: 0 }));
  const vtxState = new Uint8Array(MAXV);
  const polyKeys = new Uint32Array(4096);
  const sorted = new Uint32Array(4096);
  let nbPolySorted = 0, nbSorted = 0;

  // scanline state
  const scanPt1 = new Int32Array(SCAN_ROWS), scanPt2 = new Int32Array(SCAN_ROWS);
  const scanUiZ = new Float32Array(SCAN_ROWS), scanViZ = new Float32Array(SCAN_ROWS), scanIZ = new Float32Array(SCAN_ROWS);
  let scanStart = 0, scanH = 0;

  // per-poly emission state
  let outVtx = [], vtxFlags = [], nbOutEdges = 0;
  let area = 0, dx0 = 0, dy0 = 0, dx1 = 0, dy1 = 0, aZ1 = 0, aZ2 = 0;
  let uvSrc = null, uvOff = 0, mipMask = 0, mipScale = 0, uoMip = 0, voMip = 0, curMip = 0;
  let flatColor = 0;
  let dU1 = 0, dU2 = 0, dV1 = 0, dV2 = 0;
  let DxU = 0, DyU = 0, DxV = 0, DyV = 0, DxiZ = 0, DyiZ = 0;
  let curPoly = null, curNode = null, curMesh = null;
  let curShader = 'uvc';                  // 'uvc' (default) | 'env' (blob)
  let pEdges = [null, null, null];    // P_Edges: pre-clip vertex trio (Setup_Poly_Vtx)
  let screen = null;

  // z-buffer pass state (UVc_zBuf_Shader / shaders/sh_z.c). ZBuffer is a
  // pointer the part installs per frame (part6.c: ZBuffer = ZB_Base before
  // Render_World, NULL after) — never cleared: the painter's sort already
  // orders surfaces; the z values exist only for the flare occlusion probe.
  let zBuffer = null;

  // ---- geometry capture for the GPU path ---------------------------------
  //
  // The CPU keeps doing the vertex work -- transform, clip, sort, mip select
  // -- because none of it scales with resolution and it is where the show's
  // exactness lives. What is captured here is the FILL: the clipped triangle
  // fan in screen space with its texture page, which is the part that costs
  // 30x at 1600x1200.
  //
  // Measured across the show: at most 14541 triangles and 82 distinct pages
  // per frame, so bucketing by page and issuing one draw call each is well
  // within budget.
  let geoSink = null;
  function setGeoSink(s) { geoSink = s; }

  function captureGeo() {
    const env = curShader === 'env';
    const src = env ? envSt.uvSrc : uvSrc;
    const v0 = outVtx[0];
    // Fan from vertex 0, matching the order the span rasteriser walks.
    for (let j = 1; j < nbOutEdges - 1; j++) {
      geoSink.tri(v0, outVtx[j], outVtx[j + 1], src,
        env ? envSt.uvOff ?? 0 : uvOff,
        env ? envSt.mipMask ?? 0xffff : mipMask,
        // uoMip is the half-texel bias the CPU adds before sampling; it is
        // 1.5 rather than 0.5 for the block's own "down" polygon, so it has
        // to be captured rather than derived from the mip level.
        env ? 0.5 : uoMip / 65536,
        src ? 0 : flatColor);
    }
  }
  const scanS = new Int32Array(SCAN_ROWS);
  let DxS = 0, DyS = 0, dySInt = 0;

  // edge arrays (Emit_Poly)
  const E = 16;
  const eX = new Float32Array(E), eY = new Float32Array(E), eZ = new Float32Array(E);
  const eDx = new Float32Array(E), eDy = new Float32Array(E), eDz = new Float32Array(E);
  const eEy = new Float32Array(E), eYo = new Int32Array(E), eYf = new Int32Array(E);
  const eXe = new Int32Array(E), eDXe = new Int32Array(E);
  const rightEdges = new Int32Array(E), leftEdges = new Int32Array(E);
  let nbRight = 0, nbLeft = 0;
  const statsRef = { culledBack: 0, culledDot: 0, emitted: 0 };
  const extra = Array.from({ length: 64 }, () => ({ n0: 0, n1: 0, n2: 0, invZ: 0, xp: 0, yp: 0, u: 0, v: 0 }));

  // ---- Sort_Vista_Polys (vista.c) ----------------------------------------

  function sortVistaPolys(vista, time) {
    let idx = vista.curChunkIdx;
    let needApply = false;
    if (vista.curTime > time) { idx = 0; vista.curTime = -Infinity; needApply = true; }
    else {
      if (idx < 0) return;
      const next = vista.chunks[idx + 1];
      if (!next) {
        if (vista.curTime !== vista.chunks[idx].time) needApply = true;
      } else if (next.time <= time) {
        // advance: apply every chunk we pass over
        while (vista.chunks[idx + 1] && vista.chunks[idx + 1].time <= time) {
          idx++;
          applyChunk(vista, vista.chunks[idx]);
        }
        finishChunk(vista, idx);
        selectVisible(vista, time);
        return;
      }
    }
    if (needApply) {
      // walk forward from idx applying until next chunk exceeds time
      applyChunk(vista, vista.chunks[idx]);
      while (vista.chunks[idx + 1] && vista.chunks[idx + 1].time <= time) {
        idx++;
        applyChunk(vista, vista.chunks[idx]);
      }
      finishChunk(vista, idx);
    }
    selectVisible(vista, time);
  }

  function applyChunk(vista, chunk) {
    if (chunk.type === 1) {               // VISTA_INIT
      vista.objId.set(chunk.objId.subarray(0, chunk.nbPolys));
      vista.polyId.set(chunk.polyId.subarray(0, chunk.nbPolys));
      vista.curNb = chunk.nbPolys;
    } else {                              // VISTA_DELTA — ordered insert/remove
      let min = 0;
      for (let j = 0; j < chunk.nbPolys; j++) {
        const srcId = ((chunk.objId[j] << 16) | chunk.polyId[j]) >>> 0;
        for (let i = min; i <= vista.curNb; i++) {
          if (i === vista.curNb) { insertAt(vista, i, srcId); min = i + 1; break; }
          const dstId = ((vista.objId[i] << 16) | vista.polyId[i]) >>> 0;
          if (dstId === srcId) {          // remove
            vista.curNb--;
            vista.objId.copyWithin(i, i + 1, vista.curNb + 1);
            vista.polyId.copyWithin(i, i + 1, vista.curNb + 1);
            min = i;
            break;
          }
          if (dstId > srcId) { insertAt(vista, i, srcId); min = i + 1; break; }
        }
      }
    }
  }
  function insertAt(vista, i, srcId) {
    vista.objId.copyWithin(i + 1, i, vista.curNb);
    vista.polyId.copyWithin(i + 1, i, vista.curNb);
    vista.objId[i] = srcId >>> 16;
    vista.polyId[i] = srcId & 0xffff;
    vista.curNb++;
  }
  function finishChunk(vista, idx) {
    const chunk = vista.chunks[idx];
    vista.curTime = chunk.time;
    vista.curChunkIdx = idx;
    rc.zMin = chunk.zMin;
    rc.zMax = chunk.zMax;
    rc.zScale = fr(rc.globalZScale * rc.zMin);
    rc.zVistaScale = fr(65536 / fr(fr(rc.zMax - rc.zMin) + fr(0.0001)));
    rc.zOff = fr(-rc.zScale / rc.zMax);
  }

  function selectVisible(vista) {
    nbPolySorted = 0;
    nbSorted = 0;
    for (let i = 0; i < vista.curNb; i++) {
      polyKeys[i] = (nbPolySorted << 16) >>> 0;
      const node = byId.get(vista.objId[i]);
      if (!node || node.flags.dontRender || node.type === 'camera') continue;
      if (node.flags.notActive) continue;
      if (node.type === 'mesh') {
        curNode = node;
        curMesh = node.data;
        vistaSelectPoly(i, vista, node, node.data.polys[vista.polyId[i]]);
      }
      // lights: flare bookkeeping — next leg
    }
    if (nbPolySorted) countingSort(polyKeys, nbPolySorted);
  }

  // ---- Vista_Select_Poly (vmesh.c) ---------------------------------------

  const tmpN = new Float32Array(3), tmpV = new Float32Array(3), tmpV0 = new Float32Array(3), tmpIn = new Float32Array(3);

  function vistaSelectPoly(order, vista, node, poly) {
    poly.clipped = 0;
    nAV(tmpN, node.m, poly.no);
    if (tmpN[2] > B_CULL_EPSILON) { statsRef.culledBack++; return; }
    const nbSortedSave = nbSorted;

    tmpIn[0] = curMesh.vertex[poly.pt[0] * 3];
    tmpIn[1] = curMesh.vertex[poly.pt[0] * 3 + 1];
    tmpIn[2] = curMesh.vertex[poly.pt[0] * 3 + 2];
    aV(tmpV0, node.m, tmpIn);             // vertex 0's camera-space position
    const dot = fr(fr(fr(tmpV0[0] * tmpN[0]) + fr(tmpV0[1] * tmpN[1])) + fr(tmpV0[2] * tmpN[2]));
    if (dot > 0) { statsRef.culledDot++; return; }

    let zo = poly.nbPts, z1 = poly.nbPts, xo = poly.nbPts, x1 = poly.nbPts, yo = poly.nbPts, y1 = poly.nbPts;
    let key = 0;
    let clippedAccum = 0;

    for (let j = poly.nbPts - 1; j >= 0; --j) {
      const pvIdx = nbSortedSave + j;
      const pv = pV[pvIdx];
      let src = tmpV0;                   // j==0 reuses the dot-test vector
      if (j !== 0) {
        tmpIn[0] = curMesh.vertex[poly.pt[j] * 3];
        tmpIn[1] = curMesh.vertex[poly.pt[j] * 3 + 1];
        tmpIn[2] = curMesh.vertex[poly.pt[j] * 3 + 2];
        aV(tmpV, node.m, tmpIn);
        src = tmpV;
      }
      pv.n0 = src[0]; pv.n1 = src[1]; pv.n2 = src[2];

      if (pv.n2 < rc.outClips[4] || pv.n2 > rc.outClips[5]) { nbSorted = nbSortedSave; return; }
      if (Math.abs(pv.n2) < Z_EPSILON) pv.n2 = Z_EPSILON;
      pv.invZ = fr(1 / pv.n2);
      let state = VTX_TESTED;
      const iZL = fr(pv.invZ * rc.lx);
      pv.xp = fr(fr(pv.n0 * iZL) + rc.cx);
      pv.yp = fr(fr(-pv.n1 * iZL) + rc.cy);

      if (pv.n2 < rc.clips[4]) {
        state |= VTX_VISIBLE | VTX_ZO;
        zo--;
        if (rc.clips[4] > key) key = rc.clips[4];
      } else {
        if (pv.n2 > rc.clips[6]) {
          state |= VTX_VISIBLE | VTX_Z1;
          z1--;
          if (rc.clips[6] > key) key = rc.clips[6];
        } else if (pv.n2 > key) key = pv.n2;

        if (pv.xp < rc.clips[0]) { state |= VTX_VISIBLE | VTX_XO; xo--; }
        else if (pv.xp >= rc.clips[1]) { state |= VTX_VISIBLE | VTX_X1; x1--; }
        if (pv.yp < rc.clips[2]) { state |= VTX_VISIBLE | VTX_YO; yo--; }
        if (pv.yp >= rc.clips[3]) { state |= VTX_VISIBLE | VTX_Y1; y1--; }
      }
      vtxState[pvIdx] = state;
      clippedAccum |= state;
      nbSorted++;
    }
    if (!xo || !x1 || !yo || !y1 || !zo || !z1) { poly.clipped = 0; nbSorted = nbSortedSave; return; }

    poly.clipped = (clippedAccum & ~(VTX_VISIBLE | VTX_YO | VTX_Y1)) | 0x02;
    let qKey = fr(fr(rc.zMax - key) * rc.zVistaScale);
    if (qKey < 0) qKey = 0; else if (qKey > 65535) qKey = 65535;
    polyKeys[nbPolySorted] = (((nbPolySorted << 16) | Math.trunc(qKey)) >>> 0);
    sorted[nbPolySorted] = ((nbSortedSave | (order << 16)) >>> 0);
    nbPolySorted++;
  }

  // ---- Render_Vista main walk --------------------------------------------

  function renderVista(vista, time) {
    sortVistaPolys(vista, time);
    if (!nbPolySorted) return;
    for (let i = 0; i < nbPolySorted; i++) {
      const keyIdx = polyKeys[i] >>> 16;
      const id = sorted[keyIdx] & 0xffff;
      const order = sorted[keyIdx] >>> 16;
      const node = byId.get(vista.objId[order]);
      if (node.type !== 'mesh') continue;
      curNode = node;
      curMesh = node.data;
      curPoly = curMesh.polys[vista.polyId[order]];
      // Init_From_Poly_III
      uvSrc = null;
      // Vista_Setup_Poly_Vtx
      nbOutEdges = curPoly.nbPts;
      outVtx = [];
      vtxFlags = [];
      for (let j = 0; j <= nbOutEdges; j++) {
        const k = j === nbOutEdges ? 0 : j;
        outVtx[j] = pV[id + k];
        vtxFlags[j] = vtxState[id + k] & ~(0x01 | VTX_YO | VTX_Y1);
        // FLAG_FOR_CLIPPING_II mask: visible-bit 0x01 + Y flags cleared
      }
      pEdges = [outVtx[0], outVtx[1], outVtx[2]];
      dy1 = fr(outVtx[2].yp - outVtx[1].yp);
      dx0 = fr(outVtx[1].xp - outVtx[0].xp);
      dy0 = fr(outVtx[1].yp - outVtx[0].yp);
      dx1 = fr(outVtx[2].xp - outVtx[1].xp);
      area = fr(fr(dx0 * dy1) - fr(dx1 * dy0));
      if (globalThis.SOM_DBG && i < 3) console.log('poly', i, 'area', area,
        outVtx.slice(0, 3).map((v) => `(${v.xp.toFixed(1)},${v.yp.toFixed(1)},z${v.n2.toFixed(1)})`).join(' '));
      if (Math.abs(area) <= AREA_EPSILON3) continue;
      statsRef.emitted++;
      if (curPoly.uvI) {                  // POLY_MAP_UV (UVc shader marks all)
        for (let j = curPoly.nbPts - 1; j >= 0; --j) {
          outVtx[j].u = fr((1 / 65535) * curPoly.uvI[j * 2]);
          outVtx[j].v = fr((1 / 65535) * curPoly.uvI[j * 2 + 1]);
        }
      }
      emitPoly();
    }
  }

  // ---- default (non-vista) scene-graph path -------------------------------
  //
  // The engine has ONE shader/raster core and two front-ends: the vista PVS
  // walk above (part6 tunnel, world2 city) and this scene-graph walk
  // (world3 statue, world4 greets), which parts select by leaving
  // Renderer.Render at its default instead of Render_Vista.
  //
  // Both feed the same emitPoly/clip/rasterizer below. They differ in three
  // ways that matter bit-for-bit:
  //   1. Area formula. vista.c:414 uses dx0*dy1 - dx1*dy0; shader.c:127 uses
  //      P_Edges[0].yp*dx1 + P_Edges[2].yp*dx0 + P_Edges[1].yp*(x0-x2).
  //      Algebraically equal, NOT equal in f32. Each file has the other
  //      commented out.
  //   2. Vertices are transformed once per MESH and cached in vertexState,
  //      shared across that mesh's polys (mesh.c Transform_Vertices_I).
  //   3. Sorting is two-level: polys within a mesh against that mesh's own
  //      z range, then meshes against each other by projected box.

  const objSorted = new Uint32Array(4096);
  const objToSort = [];
  let nbObjSorted = 0;
  const meshPolyKeys = new Uint32Array(8192);
  const polyToSort = [];                  // {poly, key}
  let nbMeshPolys = 0;
  const boxPt = Array.from({ length: 8 }, () => new Float32Array(3));
  const boxClip = new Float32Array(6);

  // box.c BBox_Planes_To_Points + Project_Box. Returns the object sort key
  // (nearest/farthest 1/z), and fills boxClip with the screen extent.
  function projectBox(box, m) {
    const { mins, maxs } = box;
    for (let i = 0; i < 8; i++) {
      boxPt[i][0] = (i & 4) ? mins[0] : maxs[0];
      boxPt[i][1] = (i & 2) ? mins[1] : maxs[1];
      boxPt[i][2] = (i & 1) ? mins[2] : maxs[2];
    }
    boxClip[0] = rc.clips[1]; boxClip[1] = rc.clips[0];
    boxClip[2] = rc.clips[3]; boxClip[3] = rc.clips[2];
    boxClip[4] = 1e10; boxClip[5] = 0;
    let min = 1e10, max = -1e10;
    for (let i = 7; i >= 0; --i) {
      aV(boxPt[i], m, boxPt[i]);          // A_V_Eq: in-place (aV caches xyz)
      let invZ = fr(1 / boxPt[i][2]);
      if (invZ < 0 || invZ > rc.clips[5]) continue;
      if (invZ < min) min = invZ;
      if (invZ > max) max = invZ;
      if (invZ < boxClip[4]) boxClip[4] = invZ;
      if (invZ > boxClip[5]) boxClip[5] = invZ;
      invZ = fr(invZ * rc.lx);
      const xp = fr(rc.cx + fr(boxPt[i][0] * invZ));
      if (xp < boxClip[0]) boxClip[0] = xp;
      if (xp > boxClip[1]) boxClip[1] = xp;
      const yp = fr(rc.cy - fr(boxPt[i][1] * invZ));
      if (yp < boxClip[2]) boxClip[2] = yp;
      if (yp > boxClip[3]) boxClip[3] = yp;
    }
    if (boxClip[0] < rc.clips[0]) boxClip[0] = rc.clips[0];
    if (boxClip[1] > rc.clips[1]) boxClip[1] = rc.clips[1];
    if (boxClip[2] < rc.clips[2]) boxClip[2] = rc.clips[2];
    if (boxClip[3] > rc.clips[3]) boxClip[3] = rc.clips[3];
    if (min >= 0) return min;
    if (max <= 0) return 0;
    boxClip[0] = rc.clips[0]; boxClip[1] = rc.clips[1];
    boxClip[2] = rc.clips[2]; boxClip[3] = rc.clips[3];
    return max;
  }

  // camera.c Sort_Objects — node walk with box cull and object keying.
  function sortObjects(nodes) {
    for (const node of nodes) {
      if (node.flags.dontRender) continue;
      if (node.type === 'camera' || node.type === 'material') continue;
      // Check_Node_Transform + STORE_NODE_MATRIX already done by updateWorld
      if (!node.box) continue;              // lights/dummies carry no mesh box
      let key = projectBox(node.box, node.m);
      if (key === 0) continue;
      if (key < fr(1 / rc.outClips[4])) continue;
      if (boxClip[1] < rc.clips[0]) continue;
      if (boxClip[0] > rc.clips[1]) continue;
      if (boxClip[1] - boxClip[0] < 1) continue;
      if (boxClip[3] < rc.clips[2]) continue;
      if (boxClip[2] > rc.clips[3]) continue;
      if (boxClip[3] - boxClip[2] < 1) continue;
      if (node.flags.notActive) continue;
      if (node.type !== 'mesh') continue;   // Methods->Render == NULL otherwise
      key = fr(key * 65536);
      if (key > 65535) key = 65535; else if (key < 0) key = 0;
      objSorted[nbObjSorted++] = ((nbObjSorted - 1) << 16 | (key >>> 0) & 0xffff) >>> 0;
      objToSort[nbObjSorted - 1] = node;
    }
  }

  // mesh.c Transform_Vertices_I — per-mesh vertex transform + poly select,
  // then a per-mesh poly sort against that mesh's own z range.
  const vtxScratch = new Float32Array(3);
  function transformVerticesI(node, m) {
    const mesh = node.data;
    curShader = mesh.shader === 'env' ? 'env' : 'uvc';
    nbMeshPolys = 0;
    let meshMin = 1e10, meshMax = -1e10;
    const nbV = mesh.nbVertex;
    for (let i = 0; i < nbV; i++) vtxState[i] = 0;   // memset(Vertex_State,0,..)

    for (let pi = 0; pi < mesh.nbPolys; pi++) {
      const poly = mesh.polys[pi];
      poly.clipped = 0;                              // FLAG_POLY_NOT_VISIBLE
      nAV(tmpN, m, poly.no);
      if (tmpN[2] > B_CULL_EPSILON) continue;
      tmpIn[0] = mesh.vertex[poly.pt[0] * 3];
      tmpIn[1] = mesh.vertex[poly.pt[0] * 3 + 1];
      tmpIn[2] = mesh.vertex[poly.pt[0] * 3 + 2];
      aV(tmpV0, m, tmpIn);
      if (fr(fr(fr(tmpV0[0] * tmpN[0]) + fr(tmpV0[1] * tmpN[1])) + fr(tmpV0[2] * tmpN[2])) > 0) continue;

      let zo = poly.nbPts, z1 = poly.nbPts, xo = poly.nbPts;
      let x1 = poly.nbPts, yo = poly.nbPts, y1 = poly.nbPts;
      let key = 0, rejected = false;
      for (let j = poly.nbPts - 1; j >= 0; --j) {
        const K = poly.pt[j];
        const st = vtxState[K];
        if ((st & VTX_VISIBLE) === 0x01) { rejected = true; break; }   // REJECTED
        if ((st & VTX_VISIBLE) === 0x00) {                             // NOT_TESTED
          const pv = pV[K];
          let src;
          if (j !== 0) {
            tmpIn[0] = mesh.vertex[K * 3];
            tmpIn[1] = mesh.vertex[K * 3 + 1];
            tmpIn[2] = mesh.vertex[K * 3 + 2];
            aV(vtxScratch, m, tmpIn);
            src = vtxScratch;
          } else src = tmpV0;              // vertex 0 reuses the dot-test vector
          pv.n0 = src[0]; pv.n1 = src[1]; pv.n2 = src[2];

          if (pv.n2 < rc.outClips[4] || pv.n2 > rc.outClips[5]) {
            vtxState[K] = 0x01; rejected = true; break;
          }
          if (Math.abs(pv.n2) < Z_EPSILON) pv.n2 = Z_EPSILON;
          pv.invZ = fr(1 / pv.n2);
          let state = VTX_TESTED;
          const iZL = fr(pv.invZ * rc.lx);
          pv.xp = fr(fr(pv.n0 * iZL) + rc.cx);
          pv.yp = fr(fr(-pv.n1 * iZL) + rc.cy);

          if (pv.n2 <= rc.clips[4]) {
            state |= VTX_VISIBLE | VTX_ZO; zo--;
            if (rc.clips[4] > key) key = rc.clips[4];
            // C goto Skip: xp/yp are wrong, so no x/y flags for this vertex
          } else {
            if (pv.n2 > rc.clips[6]) {
              state |= VTX_VISIBLE | VTX_Z1; z1--;
              if (rc.clips[6] > key) key = rc.clips[6];
            } else if (pv.n2 > key) key = pv.n2;
            if (pv.xp < rc.clips[0]) { state |= VTX_VISIBLE | VTX_XO; xo--; }
            else if (pv.xp >= rc.clips[1]) { state |= VTX_VISIBLE | VTX_X1; x1--; }
            if (pv.yp < rc.clips[2]) { state |= VTX_VISIBLE | VTX_YO; yo--; }
            if (pv.yp >= rc.clips[3]) { state |= VTX_VISIBLE | VTX_Y1; y1--; }
          }
          if (curShader === 'env') shadeEnvVertex(pv, mesh, K, m);
          vtxState[K] = state;
          poly.clipped |= state;
        } else {                            // already transformed this mesh
          if (st & VTX_ZO) zo--;
          if (st & VTX_Z1) z1--;
          if (st & VTX_XO) xo--;
          if (st & VTX_X1) x1--;
          if (st & VTX_YO) yo--;
          if (st & VTX_Y1) y1--;
          if (pV[K].n2 > key) key = pV[K].n2;
          poly.clipped |= st;
        }
      }
      if (rejected) { poly.clipped = 0; continue; }
      if (!xo || !x1 || !yo || !y1 || !zo || !z1) { poly.clipped = 0; continue; }

      poly.clipped = (poly.clipped & ~(VTX_VISIBLE | VTX_YO | VTX_Y1)) | 0x02;
      polyToSort[nbMeshPolys] = poly;
      polyToSort[nbMeshPolys].sortKey = key;
      nbMeshPolys++;
      if (key > meshMax) meshMax = key;
      if (key < meshMin) meshMin = key;
    }
    if (!nbMeshPolys) return;

    // per-mesh z range gives the sort maximum precision (mesh.c:215)
    const zScale = fr(65536 / fr(fr(meshMax - meshMin) + fr(0.0001)));
    for (let i = 0; i < nbMeshPolys; i++) {
      const k = Math.floor(fr(fr(meshMax - polyToSort[i].sortKey) * zScale));
      meshPolyKeys[i] = ((i << 16) | (k & 0xffff)) >>> 0;
    }
    countingSort(meshPolyKeys, nbMeshPolys);
  }

  // shader.c Setup_Poly_Vtx — note the Area formula differs from the vista
  // path's (see the block comment above); this one is load-bearing here.
  function setupPolyVtxMesh(poly) {
    nbOutEdges = poly.nbPts;
    outVtx = []; vtxFlags = [];
    outVtx[nbOutEdges] = pV[poly.pt[0]];
    vtxFlags[nbOutEdges] = vtxState[poly.pt[0]] & ~(0x01 | VTX_YO | VTX_Y1);
    for (let j = nbOutEdges - 1; j >= 0; --j) {
      outVtx[j] = pV[poly.pt[j]];
      vtxFlags[j] = vtxState[poly.pt[j]] & ~(0x01 | VTX_YO | VTX_Y1);
    }
    pEdges = [outVtx[0], outVtx[1], outVtx[2]];
    dy1 = fr(pEdges[2].yp - pEdges[1].yp);
    dx0 = fr(pEdges[1].xp - pEdges[0].xp);
    dy0 = fr(pEdges[1].yp - pEdges[0].yp);
    dx1 = fr(pEdges[2].xp - pEdges[1].xp);
    area = fr(fr(pEdges[0].yp * dx1) + fr(pEdges[2].yp * dx0));
    area = fr(area + fr(pEdges[1].yp * fr(pEdges[0].xp - pEdges[2].xp)));
  }

  function renderMesh(node) {              // mesh.c Render_Mesh
    curNode = node;
    curMesh = node.data;
    curShader = curMesh.shader === 'env' ? 'env' : 'uvc';
    if (!curMesh.polys || !curMesh.nbPolys) return;
    transformVerticesI(node, node.m);
    for (let i = 0; i < nbMeshPolys; i++) {
      const poly = polyToSort[meshPolyKeys[i] >>> 16];
      curPoly = poly;
      uvSrc = null;                        // Init_From_Poly
      envSt.uvSrc = null;
      setupPolyVtxMesh(poly);
      if (Math.abs(area) <= AREA_EPSILON3) continue;
      statsRef.emitted++;
      if (poly.uvI) {
        for (let j = poly.nbPts - 1; j >= 0; --j) {
          outVtx[j].u = fr((1 / 65535) * poly.uvI[j * 2]);
          outVtx[j].v = fr((1 / 65535) * poly.uvI[j * 2 + 1]);
        }
      }
      emitPoly();
    }
  }

  // camera.c Render_Camera
  function renderCamera(nodes) {
    nbObjSorted = 0;
    sortObjects(nodes);
    if (!nbObjSorted) return;
    countingSort(objSorted, nbObjSorted);
    for (let i = 0; i < nbObjSorted; i++) renderMesh(objToSort[objSorted[i] >>> 16]);
  }

  // render.c Init_Render, the non-vista branch: z range comes from the clip
  // planes once per frame instead of per vista chunk.
  function initRenderNoVista() {
    rc.zMin = rc.clips[4];
    rc.zMax = rc.clips[6];
    rc.zScale = fr(rc.globalZScale * rc.zMin);
    rc.zVistaScale = fr(65536 / fr(fr(rc.zMax - rc.zMin) + fr(0.0001)));
    rc.zOff = fr(-rc.zScale / rc.zMax);
  }

  // ---- env-mapped path (shaders/sh_env.c + cache1.c) ---------------------
  //
  // The blob renders with Env_Shader: UVs come from the transformed vertex
  // NORMAL rather than from the mesh, the clip splitter is the affine
  // Split_UV (not the 1/z-correct Split_UVc), and the span loop is the
  // affine _Draw_UV_16b — which deliberately applies NO mip mask ("*NO*
  // Mip_Mask HERE!!" in the source), so U/V wrap naturally at 8.8.

  const scanU = new Int32Array(SCAN_ROWS), scanV = new Int32Array(SCAN_ROWS);
  let DxU2 = 0, DyU2 = 0, DxV2 = 0, DyV2 = 0, dyUi = 0, dyVi = 0;
  let envCache = null, envTexture = null;
  const envSt = {
    curMip: 0, mipMask: 0, mipScale: 0, uoMip: 0, voMip: 0,
    flatColor: 0, uvSrc: null, uvOff: 0,
  };

  function setEnvCache(cache, texture) { envCache = cache; envTexture = texture; }

  // sh_env.c Shader_Env_Mapper — per-vertex, during Transform_Vertices_I
  const envN = new Float32Array(3);
  function shadeEnvVertex(pv, mesh, k, m) {
    envN[0] = mesh.normals[k * 3]; envN[1] = mesh.normals[k * 3 + 1]; envN[2] = mesh.normals[k * 3 + 2];
    nAV(envN, m, envN);
    let n = fr(fr(fr(envN[0] * envN[0]) + fr(envN[1] * envN[1])) + fr(envN[2] * envN[2]));
    if (n > 0) {
      n = fr(1 / Math.sqrt(n));
      envN[0] = fr(envN[0] * n); envN[1] = fr(envN[1] * n);
    }
    pv.u = fr(fr(0.5) * fr(1 + envN[0]));
    pv.v = fr(fr(0.5) * fr(1 + envN[1]));
  }

  // cache1.c Compute_MipMap_Env — uses the live UV cross product, since the
  // poly's stored U1V2_U2V1 is meaningless for env mapping.
  function computeMipMapEnv() {
    dU1 = fr(pEdges[1].u - pEdges[0].u);
    dU2 = fr(pEdges[2].u - pEdges[1].u);
    dV1 = fr(pEdges[1].v - pEdges[0].v);
    dV2 = fr(pEdges[2].v - pEdges[1].v);
    const R = fr(Math.abs(fr(area * fr(fr(dU1 * dV2) - fr(dU2 * dV1)))));
    return retreiveTextureII(envSt, envCache, envTexture, R) < 3;
  }

  // sh_env.c Shader_Deal_UV_Env
  function dealUVEnv() {
    if (Math.abs(area) > AREA_EPSILON2) { flatColor = envSt.flatColor; return drawFlatI; }
    if (envSt.uvSrc === null && computeMipMapEnv()) { flatColor = envSt.flatColor; return drawFlatI; }
    const area2 = fr(area * envSt.mipScale);
    DyU2 = fr(fr(fr(dU1 * dy1) - fr(dU2 * dy0)) * area2); dyUi = cInt(DyU2);
    DyV2 = fr(fr(fr(dV1 * dy1) - fr(dV2 * dy0)) * area2); dyVi = cInt(DyV2);
    DxU2 = fr(fr(fr(dU2 * dx0) - fr(dU1 * dx1)) * area2);
    DxV2 = fr(fr(fr(dV2 * dx0) - fr(dV1 * dx1)) * area2);
    rasterizeUV();
    return drawUV16;
  }

  // raster.c Rasterize_UV (right edges only, like the UVc variant)
  function rasterizeUV() {
    for (let i = nbRight - 1; i >= 0; --i) {
      const j = rightEdges[i];
      let y = eYo[j];
      const yf = eYf[j];
      const dy = eEy[j];
      const dx = fr(scanPt2[y] - eX[j]);
      const dXe0 = eDXe[j];
      const slope = dXe0 >= 0 ? fr(dXe0 >> 16) : fr(-((-dXe0) >> 16));
      let t = fr(DxU2 + fr(slope * DyU2));
      const dU = cInt(t);
      t = fr(outVtx[j].u * envSt.mipScale);
      t = fr(t + fr(fr(dy * DxU2) + fr(dx * DyU2)));
      let U = cInt(fr(t + envSt.uoMip));
      t = fr(DxV2 + fr(slope * DyV2));
      const dV = cInt(t);
      t = fr(outVtx[j].v * envSt.mipScale);
      t = fr(t + fr(fr(dy * DxV2) + fr(dx * DyV2)));
      let V = cInt(fr(t + envSt.voMip));
      let xe = eXe[j], dxe = dXe0, incU, incV;
      if (dxe < 0) { dxe = -dxe; xe = -xe; incU = -dyUi; incV = -dyVi; }
      else { incU = dyUi; incV = dyVi; }
      dxe &= 0xffff; xe &= 0xffff;
      while (y < yf) {
        scanU[y] = U; U = (U + dU) | 0;
        scanV[y] = V; V = (V + dV) | 0;
        xe += dxe;
        if (xe & 0x10000) { xe &= 0xffff; U = (U + incU) | 0; V = (V + incV) | 0; }
        y++;
      }
    }
  }

  function drawUV16() {                    // draw16/loops216.c _Draw_UV_16b
    const src = envSt.uvSrc, base = envSt.uvOff;
    let y = scanStart, dy = scanH;
    while (dy > 0) {
      const p1 = scanPt1[y];
      let len = p1 - scanPt2[y];
      const ptr = (y + 1) * rc.bps - p1;
      let U = scanU[y], V = scanV[y];
      while (len < 0) {
        const off = ((U >> 16) & 0x00ff) | ((V >> 8) & 0xff00);
        screen[ptr + len] = src[base + off];   // NO mip mask, by design
        len++;
        U = (U - dyUi) | 0; V = (V - dyVi) | 0;
      }
      y++; dy--;
    }
  }

  function drawFlatI() {                   // RENDER_FLAT_I
    const color = flatColor & 0xffff;
    let y = scanStart, dy = scanH;
    while (dy > 0) {
      const ptr = (y + 1) * rc.bps - scanPt1[y];
      let len = scanPt1[y] - scanPt2[y];
      while (len < 0) { screen[ptr + len] = color; len++; }
      y++; dy--;
    }
  }

  // clip.c Split_UV — affine, no 1/z correction on the UVs
  function splitUV(x, from, to, res) {
    res.invZ = fr(from.invZ + fr(x * fr(to.invZ - from.invZ)));
    res.n2 = fr(1 / res.invZ);
    res.u = fr(from.u + fr(x * fr(to.u - from.u)));
    res.v = fr(from.v + fr(x * fr(to.v - from.v)));
    res.xp = fr(from.xp + fr(x * fr(to.xp - from.xp)));
    res.yp = fr(from.yp + fr(x * fr(to.yp - from.yp)));
  }

  // ---- clip (clip.c SH_Clip: Z1, Zo, Xo, X1 passes) ----------------------

  let nbExtra = 0;

  function splitUVc(x, from, to, res) {   // clip.c Split_UVc
    res.invZ = fr(from.invZ + fr(x * fr(to.invZ - from.invZ)));
    const tmp = fr(1 / res.invZ);
    res.n2 = tmp;
    let x1 = fr(from.u * from.invZ), x2 = fr(to.u * to.invZ);
    res.u = fr(fr(x1 + fr(x * fr(x2 - x1))) * tmp);
    x1 = fr(from.v * from.invZ); x2 = fr(to.v * to.invZ);
    res.v = fr(fr(x1 + fr(x * fr(x2 - x1))) * tmp);
    res.xp = fr(from.xp + fr(x * fr(to.xp - from.xp)));
    res.yp = fr(from.yp + fr(x * fr(to.yp - from.yp)));
  }

  function getVertexFlags(p) {
    let s = 0;
    if (p.xp < rc.clips[0]) s |= VTX_XO;
    if (p.xp >= rc.clips[1]) s |= VTX_X1;
    if (p.yp < rc.clips[2]) s |= VTX_YO;
    if (p.yp >= rc.clips[3]) s |= VTX_Y1;
    return s;
  }

  // One Sutherland-Hodgman pass. plane: {bit, valueOf(p), clipVal, fix(p)}
  function clipPass(bit, coordOf, clipVal, fixup) {
    const inVtx = outVtx.slice(0, nbOutEdges + 1);
    const inFlags = vtxFlags.slice(0, nbOutEdges + 1);
    let cur = 0, allFlags = 0;
    for (let e = 0; e < nbOutEdges; e++) {
      const ps = inVtx[e], pf = inVtx[e + 1];
      const sOut = inFlags[e] & bit, fOut = inFlags[e + 1] & bit;
      if (sOut && fOut) continue;
      if (!sOut && fOut) {                // in -> out: emit split
        const iv = extra[nbExtra++];
        const x = fr(fr(clipVal - coordOf(ps)) / fr(coordOf(pf) - coordOf(ps)));
        (curShader === 'env' ? splitUV : splitUVc)(x, ps, pf, iv);
        if (fixup) fixup(iv);
        outVtx[cur] = iv;
        const nf = getVertexFlags(iv);
        vtxFlags[cur] = nf; allFlags |= nf;
        cur++;
      } else if (sOut && !fOut) {         // out -> in: split + endpoint
        const iv = extra[nbExtra++];
        const x = fr(fr(clipVal - coordOf(ps)) / fr(coordOf(pf) - coordOf(ps)));
        (curShader === 'env' ? splitUV : splitUVc)(x, ps, pf, iv);
        if (fixup) fixup(iv);
        const nf = getVertexFlags(iv);
        outVtx[cur] = iv; vtxFlags[cur] = nf; cur++; allFlags |= nf;
        outVtx[cur] = pf; vtxFlags[cur] = inFlags[e + 1]; cur++;
        allFlags |= inFlags[e + 1];
      } else {                            // in -> in: endpoint
        outVtx[cur] = pf; vtxFlags[cur] = inFlags[e + 1]; cur++;
        allFlags |= inFlags[e + 1];
      }
    }
    nbOutEdges = cur;
    if (cur) { outVtx[cur] = outVtx[0]; vtxFlags[cur] = vtxFlags[0]; }
    return allFlags;
  }

  function shClip(allFlags) {
    nbExtra = 0;
    if (allFlags & VTX_Z1) allFlags = clipPass(VTX_Z1, (p) => p.invZ, rc.clips[7], null);
    if (allFlags & VTX_ZO) allFlags = clipPass(VTX_ZO, (p) => p.invZ, rc.clips[5], null);
    if (allFlags & VTX_XO) allFlags = clipPass(VTX_XO, (p) => p.xp, rc.clips[0], (p) => { p.xp = rc.clips[0]; });
    if (allFlags & VTX_X1) allFlags = clipPass(VTX_X1, (p) => p.xp, rc.clips[1], (p) => { p.xp = rc.clips[1]; });
  }

  // ---- Emit_Poly (shader.c) + Rasterize_Edges (raster.c) -----------------

  function emitPoly() {
    const useful = curPoly.clipped & CLIP_USEFUL;
    if (useful) {
      shClip(useful);
      if (!nbOutEdges) return;
    }
    aZ1 = 0; aZ2 = 0;
    for (let j = nbOutEdges - 1; j >= 0; --j) {
      eY[j] = outVtx[j].yp;
      eDy[j] = fr(outVtx[j + 1].yp - eY[j]);
      eX[j] = outVtx[j].xp;
      eDx[j] = fr(outVtx[j + 1].xp - eX[j]);
      eZ[j] = outVtx[j].invZ;
      aZ2 = fr(aZ2 + eZ[j]);
      eDz[j] = fr(outVtx[j + 1].invZ - eZ[j]);
      aZ1 = fr(aZ1 + fr(Math.abs(eDz[j])));
    }
    eX[nbOutEdges] = outVtx[nbOutEdges].xp;
    eY[nbOutEdges] = outVtx[nbOutEdges].yp;
    eZ[nbOutEdges] = outVtx[nbOutEdges].invZ;
    aZ2 = fr(1 / aZ2);
    aZ1 = fr(aZ1 * aZ2);
    aZ2 = fr(aZ2 * nbOutEdges);
    area = fr(1 / area);

    rasterizeEdges();
    if (scanH <= 0) return;

    const draw = curShader === 'env' ? dealUVEnv() : dealUVc();
    if (geoSink) captureGeo();
    if (draw) draw();
    if (zBuffer) {                        // Deal_With_Poly2 = Shader_Deal_zBuf
      const draw2 = dealZBuf();
      draw2();
    }
  }

  // ---- z-buffer pass (shaders/sh_z.c + raster.c Rasterize_iZ) ------------

  // C int-cast semantics of the oracle build (arm64 fcvtzs): out-of-range
  // saturates, NaN converts to 0. The x87/i386 release would give
  // 0x80000000 instead — the oracle is the golden reference here.
  function cInt(x) {
    if (Number.isNaN(x)) return 0;
    if (x >= 2147483648) return 2147483647;
    if (x < -2147483648) return -2147483648;
    return Math.trunc(x);
  }

  function dealZBuf() {
    // sh_z.c quirk: SIGNED Area test (Deal_UVc uses fabs, Deal_zBuf doesn't)
    if (area > AREA_EPSILON2) {
      const s = fr(fr(pEdges[0].invZ * rc.zScale) + rc.zOff);
      flatColor = (cInt(s) >> 16) & 0xffff;
      return drawFlatZBuf;
    }
    const dS1 = fr(fr(pEdges[1].invZ - pEdges[0].invZ) * rc.zScale);
    const dS2 = fr(fr(pEdges[2].invZ - pEdges[1].invZ) * rc.zScale);
    DyS = fr(fr(fr(dS1 * dy1) - fr(dS2 * dy0)) * area);
    DxS = fr(fr(fr(dS2 * dx0) - fr(dS1 * dx1)) * area);
    rasterizeIZ();
    return drawZBuf;
  }

  function rasterizeIZ() {                // raster.c Rasterize_iZ (right edges)
    for (let i = nbRight - 1; i >= 0; --i) {
      const j = rightEdges[i];
      let y = eYo[j];
      const yf = eYf[j];
      const dy = eEy[j];
      const dx = fr(scanPt2[y] - eX[j]);
      const dXe0 = eDXe[j];
      const slope = dXe0 >= 0 ? fr(dXe0 >> 16) : fr(-((-dXe0) >> 16));
      let t = fr(DxS + fr(slope * DyS));
      const dS = cInt(t);
      t = fr(outVtx[j].invZ * rc.zScale);
      t = fr(t + rc.zOff);
      t = fr(t + fr(fr(dy * DxS) + fr(dx * DyS)));   // sub-pixel
      let S = cInt(t);
      dySInt = cInt(DyS);
      let xe = eXe[j], dxe = dXe0, incS;
      if (dxe < 0) { dxe = -dxe; xe = -xe; incS = -dySInt; }
      else incS = dySInt;
      dxe &= 0xffff; xe &= 0xffff;
      while (y < yf) {
        scanS[y] = S; S = (S + dS) | 0;
        xe += dxe;
        if (xe & 0x10000) { xe &= 0xffff; S = (S + incS) | 0; }
        y++;
      }
    }
  }

  function drawZBuf() {                   // shaders/loops2.c _Draw_zBuf
    let y = scanStart, dy = scanH;
    while (dy > 0) {
      const p1 = scanPt1[y];
      let len = p1 - scanPt2[y];
      const ptr = (y + 1) * rc.bps - p1;
      let S = scanS[y] >>> 0;             // UINT accumulator
      while (len < 0) {
        zBuffer[ptr + len] = S >>> 16;    // (SHORT)(S>>16), no z-test
        len++;
        S = (S - dySInt) >>> 0;
      }
      y++; dy--;
    }
  }

  function drawFlatZBuf() {               // shaders/loops2.c _Draw_Flat_zBuf
    const color = flatColor;
    let y = scanStart, dy = scanH;
    while (dy > 0) {
      const p1 = scanPt1[y];
      let len = p1 - scanPt2[y];
      const ptr = (y + 1) * rc.bps - p1;
      while (len < 0) { zBuffer[ptr + len] = color; len++; }
      y++; dy--;
    }
  }

  // ---- lens flares (3d_src/flare.c + flare16.c) --------------------------

  const FLARE_FACTOR = [0.1, 0.15, 0.3, 0.45, 0.6, 0.85, 0.65, 1.0].map(fr);
  const spOffsets = new Int32Array(512);
  const spWhats = new Int32Array(512);
  let flareTo16 = null, flareLo777 = null, flareHi777 = null;

  // (USHORT) cast of an out-of-range float: arm64 fcvtzu saturates the
  // 32-bit register, then the low 16 bits are kept.
  function cUShort(x) {
    if (Number.isNaN(x)) return 0;
    if (x < 0) return 0;
    if (x >= 4294967296) return 0xffff;
    return Math.trunc(x) & 0xffff;
  }

  // light.c Select_Light_Visibility — project the camera-space position.
  function selectLightVisibility(light) {
    light.where = -1;
    if (light.p[2] <= rc.clips[4]) return;
    light.invZ = fr(1 / light.p[2]);
    const iZ = fr(light.invZ * rc.lx);
    light.xp = Math.trunc(fr(rc.cx - fr(light.p[0] * iZ)));
    if (light.xp >= rc.clips[1] || light.xp <= rc.clips[0]) return;
    light.yp = Math.trunc(fr(rc.cy - fr(light.p[1] * iZ)));
    // C tests Xp against Clips[2] (the y-min) — typo preserved.
    if (light.yp >= rc.clips[3] || light.xp <= rc.clips[2]) return;
    light.where = light.yp * rc.bps + light.xp;
  }

  // flare.c Compute_Flare_Visibility — probe the z-buffer at the light's
  // pixel and its four neighbours; scale by how many saw the light.
  function flareVisibility(light) {
    const t = fr(fr(fr(light.invZ * rc.zScale) + rc.zOff) / 65536);
    const zLight = cUShort(t);
    const X = light.xp, Y = light.yp, w = light.where;
    let pts = zBuffer[w] < zLight ? 4 : 0;
    if (X > 1 && zBuffer[w - 1] < zLight) pts += 1;
    if (X < rc.bps - 1 && zBuffer[w + 1] < zLight) pts += 1;
    if (Y > 1 && zBuffer[w - rc.bps] < zLight) pts += 1;
    if (Y < rc.rows - 1 && zBuffer[w + rc.bps] < zLight) pts += 1;
    if (pts === 0) return 0;
    return fr(fr(FLARE_FACTOR[pts - 1] * light.invZ) * rc.clips[4]);
  }

  // flare.c Setup_Paste_Flare + flare16.c Paste_Flare_Asm_16_Sat: scaled
  // additive blit through the 0x777 convert palette, DDA offset tables.
  function pasteFlareSprite(xo, yo, R, flare, spr) {
    let tmp = fr(fr(0.5 * R) * spr.dU);
    const xi = fr(xo - tmp);
    let x = xi, xf = fr(xo + tmp);
    tmp = fr(fr(0.5 * R) * spr.dV);
    const yi = fr(yo - tmp);
    let y = yi, yf = fr(yo + tmp);
    // The sprite maps its source rect linearly across the destination rect,
    // so the GPU needs only the two rects rather than the DDA the CPU walks.
    if (geoSink) {
      geoSink.sprite(xi, yi, xf, yf,
        spr.uo, spr.vo, spr.uo + spr.dU, spr.vo + spr.dV,
        flare, 'add777');
    }
    const c = rc.clips;
    if (xf <= c[0]) return;
    if (xf > c[1]) xf = c[1];
    if (yf <= c[2]) return;
    if (yf > c[3]) yf = c[3];
    if (x > c[1]) return;
    if (x < c[0]) x = c[0];
    if (y > c[3]) return;
    if (y < c[2]) y = c[2];
    const Rinv = fr(1 / R);

    const X = Math.ceil(x);
    let U = fr(spr.uo + fr(fr(X - xi) * Rinv));
    let dF = Math.trunc(Rinv * 65536);    // (INT)(R*65536.0) — double product
    const dErr = dF & 0xffff;
    dF >>= 16;
    let F = Math.trunc(U * 65536);
    let err = F & 0xffff;
    F >>= 16;
    const dWo = Math.ceil(xf) - X;
    for (let i = 0; i < dWo; i++) {
      spOffsets[i] = F; F += dF; err += dErr;
      if (err & 0x10000) { err &= 0xffff; F++; }
    }

    const Y = Math.ceil(y);
    let V = fr(spr.vo + fr(fr(Y - yi) * Rinv));
    F = Math.trunc(V * 65536);
    err = F & 0xffff;
    F >>= 16;
    const bps = flare.bps;
    dF *= bps;
    let what = F * bps;
    const dHo = Math.ceil(yf) - Y;
    for (let i = 0; i < dHo; i++) {
      spWhats[i] = what; what += dF; err += dErr;
      if (err & 0x10000) { err &= 0xffff; what += bps; }
    }

    // paste loop — same pointer walk as the C (post-decrement bounds)
    const dW0 = dWo - 1, dH0 = dHo - 1;
    const bits = flare.bits, cmap = flare.convert;
    let where = Y * rc.bps + X + dW0;
    for (let dH = -dH0; dH <= 0; ++dH) {
      const row = spWhats[dH + dH0];
      for (let dW = -dW0; dW <= 0; ++dW) {
        const s = bits[row + spOffsets[dW + dW0]];
        if (s) {
          const s777 = cmap[s] + flareLo777[screen[where + dW] & 0xff]
            + flareHi777[screen[where + dW] >> 8];
          screen[where + dW] = flareTo16(s777);
        }
      }
      where += rc.bps;                     // Sp_BpS = 640 bytes = one row
    }
  }

  function doLightFlare(light) {          // flare.c Do_Light_Flare
    if (light.invZ >= rc.clips[5]) return;
    const base = fr(fr(light.flareScale * flareVisibility(light)) * rc.spriteScale);
    if (!(base > 0)) return;
    for (const spr of light.flare.sprites) {  // Draw_Light_Flares
      const A = spr.alpha;
      const fx = fr(fr(A * rc.cx) + fr(fr(1 - A) * light.xp));
      const fy = fr(fr(A * rc.cy) + fr(fr(1 - A) * light.yp));
      pasteFlareSprite(fx, fy, fr(base * spr.scale), light.flare, spr);
    }
  }

  // render.c/flare.c Render_Light_Flare over the W->Lights chain. The
  // projection (Setup_Light_II) is computed here lazily — rc.lx/clips do
  // not change between the C's pre-render setup and the post-render flare
  // pass, so the results are identical.
  function renderLightFlare(lights, screenU16, tables) {
    screen = screenU16;
    flareTo16 = tables.to16; flareLo777 = tables.lo777; flareHi777 = tables.hi777;
    for (const light of lights) {
      // light.p (camera-space position) is set by updateWorld's light pass
      // (Compose_Light_And_Camera equivalent).
      selectLightVisibility(light);
      if (!light.flare) continue;
      if (light.where < 0) continue;
      doLightFlare(light);
    }
  }

  const SLOPE_EPSILON = fr(0.001);

  function rasterizeEdges() {
    scanStart = Math.ceil(rc.clips[3]);
    scanH = 0;
    nbRight = nbLeft = 0;
    for (let s = nbOutEdges - 1; s >= 0; --s) {
      if (eDy[s] > SLOPE_EPSILON) {       // right edge
        if (eY[s] >= rc.clips[3] || eY[s + 1] <= rc.clips[2]) continue;
        rightEdges[nbRight++] = s;
        const slope = fr(eDx[s] * fr(1 / eDy[s]));
        let y;
        const yo = eY[s];
        y = Math.ceil(rc.clips[2] > yo ? rc.clips[2] : yo);
        const adjY = fr(y - yo);
        eEy[s] = adjY;
        let xo = fr(eX[s] + fr(adjY * slope));
        if (y < scanStart) scanStart = y;
        eYo[s] = y;
        let yfF = eY[s + 1];
        if (yfF > rc.clips[3]) yfF = rc.clips[3];
        const yf = Math.ceil(yfF);
        let xe = Math.ceil(xo * 65536);
        const dxe = Math.ceil(slope * 65536);
        eXe[s] = xe; eDXe[s] = dxe;
        scanH += yf - y;
        while (y < yf) { scanPt2[y] = xe >> 16; xe += dxe; y++; }
        eYf[s] = yf;
      } else if (eDy[s] < -SLOPE_EPSILON) { // left edge
        if (eY[s] <= rc.clips[2] || eY[s + 1] >= rc.clips[3]) continue;
        leftEdges[nbLeft++] = s;
        const slope = fr(eDx[s] * fr(1 / eDy[s]));
        let y;
        const yo = eY[s + 1];
        y = Math.ceil(rc.clips[2] > yo ? rc.clips[2] : yo);
        eEy[s] = fr(y - yo);
        let xo = fr(eX[s + 1] + fr(eEy[s] * slope));
        eYo[s] = y;
        let yfF = eY[s];
        if (yfF > rc.clips[3]) yfF = rc.clips[3];
        const yf = Math.ceil(yfF);
        let xe = Math.ceil(xo * 65536);
        const dxe = Math.ceil(slope * 65536);
        eXe[s] = xe; eDXe[s] = dxe;
        while (y < yf) { scanPt1[y] = xe >> 16; xe += dxe; y++; }
        eYf[s] = yf;
      }
    }
  }

  // ---- cache III texture fetch (cache3.c Retreive_Texture) ---------------

  function polyFlatColor() {
    const m = curPoly.mips;
    if (!m) return 0;
    let blk = null;
    for (let i = 0; i < m.maxMip; i++) { if (m.levels[i]) { blk = m.levels[i]; break; } }
    if (!blk) return 0;
    uvSrc = blk.bits; uvOff = 0;
    return blk.bits[blk.uv + 257];
  }

  function retreiveTexture(r) {
    const m = curPoly.mips;
    if (!m) { curMip = 2; flatColor = polyFlatColor(); mipMask = MIP_MASKS[curMip]; return curMip; }
    if (r < EPS_MIP[0] * EPS_MIP[0]) curMip = 8;
    else if (r < EPS_MIP[1] * EPS_MIP[1]) curMip = 8;
    else if (r < EPS_MIP[2] * EPS_MIP[2]) curMip = 7;
    else if (r < EPS_MIP[3] * EPS_MIP[3]) curMip = 6;
    else if (r < EPS_MIP[4] * EPS_MIP[4]) curMip = 5;
    else if (r < EPS_MIP[5] * EPS_MIP[5]) curMip = 4;
    else if (r < EPS_MIP[6] * EPS_MIP[6]) curMip = 3;
    else curMip = 2;
    if (curMip > m.maxMip) curMip = m.maxMip;
    if (curMip < 2) { flatColor = polyFlatColor(); mipMask = MIP_MASKS[curMip]; return curMip; }
    let blk = m.levels[curMip];
    while (!blk) { curMip++; blk = m.levels[curMip]; }
    uvSrc = blk.bits;
    uvOff = blk.uv;                       // element offset into the row page
    mipMask = MIP_MASKS[curMip];
    const wf = MIP_WIDTH[curMip];
    mipScale = fr(256 * 256 * (wf - 2));
    if (curPoly === blk.down) { uoMip = 1.5 * 65536; voMip = 1.5 * 65536; }
    else { uoMip = 0.5 * 65536; voMip = 0.5 * 65536; }
    return curMip;
  }

  function computeMipMap() {              // cache3.c Compute_MipMap (P_Edges!)
    dU1 = fr(pEdges[1].u - pEdges[0].u);
    dU2 = fr(pEdges[2].u - pEdges[1].u);
    dV1 = fr(pEdges[1].v - pEdges[0].v);
    dV2 = fr(pEdges[2].v - pEdges[1].v);
    const r = fr(Math.abs(fr(area * curPoly.u1v2u2v1)));
    return retreiveTexture(r) < 3;
  }

  // ---- Shader_Deal_UVc (sh_uvc.c) ----------------------------------------

  function dealUVc() {
    if (Math.abs(area) > AREA_EPSILON2) {
      flatColor = polyFlatColor();
      return drawFlat;
    }
    if (uvSrc === null && computeMipMap()) return drawFlat;

    const diZ1 = fr(pEdges[1].invZ - pEdges[0].invZ);
    const r1 = fr(diZ1 * pEdges[0].n2);
    const diZ2 = fr(pEdges[2].invZ - pEdges[1].invZ);
    const r2 = fr(diZ2 * pEdges[1].n2);

    let area2 = fr(area * aZ2);
    DxiZ = fr(fr(fr(diZ2 * dx0) - fr(diZ1 * dx1)) * area2);
    DyiZ = fr(fr(fr(diZ1 * dy1) - fr(diZ2 * dy0)) * area2);
    area2 = fr(area2 * mipScale);

    const du1 = fr(fr(fr(r1 * pEdges[1].u) + dU1) * pEdges[0].invZ);
    const du2 = fr(fr(fr(r2 * pEdges[2].u) + dU2) * pEdges[1].invZ);
    DxU = fr(fr(fr(du2 * dx0) - fr(du1 * dx1)) * area2);
    DxU = fr(DxU + fr(DxiZ * uoMip));
    DyU = fr(fr(fr(du1 * dy1) - fr(du2 * dy0)) * area2);
    DyU = fr(DyU + fr(DyiZ * uoMip));

    const dv1 = fr(fr(fr(r1 * pEdges[1].v) + dV1) * pEdges[0].invZ);
    const dv2 = fr(fr(fr(r2 * pEdges[2].v) + dV2) * pEdges[1].invZ);
    DxV = fr(fr(fr(dv2 * dx0) - fr(dv1 * dx1)) * area2);
    DxV = fr(DxV + fr(DxiZ * voMip));
    DyV = fr(fr(fr(dv1 * dy1) - fr(dv2 * dy0)) * area2);
    DyV = fr(DyV + fr(DyiZ * voMip));

    rasterizeUVc();

    if (aZ1 < UVC_LIN) return drawUVcLinear;
    if (aZ1 < UVC_32) return () => drawUVcSpan(5);
    if (aZ1 < UVC_16) return () => drawUVcSpan(4);
    if (aZ1 < UVC_8) return () => drawUVcSpan(3);
    // both the "4-pixel" and "2-pixel" table slots point at the 4-pixel
    // function (Primitives_16 quirk)
    return () => drawUVcSpan(2);
  }

  // ---- Rasterize_UVc (raster.c, the #if 1 version) -----------------------

  function rasterizeUVcSide(edges, nbEdges, isRight) {
    for (let i = nbEdges - 1; i >= 0; --i) {
      const j = edges[i];
      let y = eYo[j];
      const yf = eYf[j];
      const dy = eEy[j];
      const dx = fr(scanPt2[y] - eX[isRight ? j : j + 1]);
      const dxe0 = eDXe[j];
      const slope = dxe0 >= 0 ? fr(dxe0 >> 16) : fr(-((-dxe0) >> 16));
      const ov = outVtx[isRight ? j : j + 1];
      let iz = fr(aZ2 * ov.invZ);
      const dU = fr(DxU + fr(slope * DyU));
      let u = fr(fr(ov.u * mipScale) + uoMip);
      u = fr(u * iz);
      u = fr(u + fr(fr(dy * DxU) + fr(dx * DyU)));
      const dV = fr(DxV + fr(slope * DyV));
      let v = fr(fr(ov.v * mipScale) + voMip);
      v = fr(v * iz);
      v = fr(v + fr(fr(dy * DxV) + fr(dx * DyV)));
      iz = fr(iz + fr(fr(dy * DxiZ) + fr(dx * DyiZ)));
      const diz = fr(DxiZ + fr(slope * DyiZ));
      let xe = eXe[j], dxe = dxe0;
      let incU, incV, incIZ;
      if (dxe < 0) { dxe = -dxe; xe = -xe; incU = fr(-DyU); incV = fr(-DyV); incIZ = fr(-DyiZ); }
      else { incU = DyU; incV = DyV; incIZ = DyiZ; }
      dxe &= 0xffff; xe &= 0xffff;
      while (y < yf) {
        scanUiZ[y] = u; u = fr(u + dU);
        scanViZ[y] = v; v = fr(v + dV);
        scanIZ[y] = iz; iz = fr(iz + diz);
        xe += dxe;
        if (xe & 0x10000) {
          xe &= 0xffff;
          u = fr(u + incU); v = fr(v + incV); iz = fr(iz + incIZ);
        }
        y++;
      }
    }
  }
  function rasterizeUVc() {
    // raster.c's live Rasterize_UVc: only the right-edge pass feeds the
    // Scan_UiZ/ViZ/iZ arrays the span loops read; the left-edge pass fills
    // separate Scan_*_2 arrays consumed by nothing in this draw path.
    // Running it into the same arrays clobbers the texturing (cost a leg
    // of golden divergence) — so it is intentionally absent here.
    rasterizeUVcSide(rightEdges, nbRight, true);
  }

  // ---- span loops (draw16/loops216) --------------------------------------

  function drawFlat() {                   // _Draw_Flat_16
    const color = flatColor & 0xffff;
    let y = scanStart;
    let dy = scanH;
    while (dy > 0) {
      const ptr = (y + 1) * rc.bps - scanPt1[y];   // Dst - Scan_Pt1[y]
      let len = scanPt1[y] - scanPt2[y];      // negative over the span
      while (len < 0) { screen[ptr + len] = color; len++; }
      y++; dy--;
    }
  }

  function drawUVcLinear() {              // _Draw_UVc_16
    let y = scanStart, dy = scanH;
    while (dy > 0) {
      let len = scanPt1[y] - scanPt2[y];
      if (len < 0) {
        const ptr = (y + 1) * rc.bps - scanPt1[y];
        let uiz = scanUiZ[y], viz = scanViZ[y], iz = scanIZ[y];
        let z = fr(1 / iz);
        let u = Math.floor(fr(uiz * z)), v = Math.floor(fr(viz * z));
        iz = fr(iz + fr(len * DyiZ));
        uiz = fr(uiz + fr(len * DyU));
        viz = fr(viz + fr(len * DyV));
        z = fr(1 / iz);
        let du = Math.floor(fr(uiz * z)), dv = Math.floor(fr(viz * z));
        z = fr(1 / len);
        du = Math.trunc(fr((u - du) * z));
        dv = Math.trunc(fr((v - dv) * z));
        while (len < 0) {
          const off = ((u >> 16) & 0x00ff) | ((v >> 8) & 0xff00);
          screen[ptr + len] = uvSrc[uvOff + (off & mipMask)];
          len++;
          u = (u + du) | 0; v = (v + dv) | 0;
        }
      }
      y++; dy--;
    }
  }

  const DIV_TABS = {};                    // tables.c _Div_Tab_N_: i/N
  for (const n of [4, 8, 16, 32, 64]) {
    DIV_TABS[n] = Float32Array.from({ length: n }, (_, i) => fr(i / n));
  }
  const DIV_TAB_1 = Float32Array.from({ length: 64 }, (_, i) => (i === 0 ? 0 : fr(1 / i)));

  function drawUVcSpan(log2) {            // loops216.h template
    const pps = 1 << log2;
    const divTab = DIV_TABS[pps];
    const dyU = fr(DyU * pps), dyV = fr(DyV * pps), dyIZ = fr(DyiZ * pps);
    let y = scanStart, dy = scanH;
    while (dy > 0) {
      const base = (y + 1) * rc.bps;
      let len = scanPt2[y];
      let ptr = base - len;
      len = len - scanPt1[y];
      if (len > 0) {
        let uiz = scanUiZ[y], viz = scanViZ[y], iz = scanIZ[y];
        let z = fr(1 / iz);
        let leftOver = len & (pps - 1);
        let spans = len >> log2;
        let u = Math.trunc(fr(uiz * z)), v = Math.trunc(fr(viz * z));
        if (spans === 0) {
          uiz = fr(uiz - fr(dyU * divTab[leftOver]));
          viz = fr(viz - fr(dyV * divTab[leftOver]));
          iz = fr(iz - fr(dyIZ * divTab[leftOver]));
          z = fr(1 / iz);
        } else {
          iz = fr(iz - dyIZ); uiz = fr(uiz - dyU); viz = fr(viz - dyV);
          z = fr(1 / iz);
          while (--spans >= 0) {
            const u2 = Math.trunc(fr(uiz * z)), v2 = Math.trunc(fr(viz * z));
            const du = (u2 - u) >> log2, dvv = (v2 - v) >> log2;
            if (spans) { uiz = fr(uiz - dyU); viz = fr(viz - dyV); iz = fr(iz - dyIZ); }
            else if (leftOver) {
              uiz = fr(uiz - fr(dyU * divTab[leftOver]));
              viz = fr(viz - fr(dyV * divTab[leftOver]));
              iz = fr(iz - fr(dyIZ * divTab[leftOver]));
            }
            z = fr(1 / iz);
            let span = -pps; ptr -= span;
            while (span < 0) {
              const off = ((u >> 16) & 0x00ff) | ((v >> 8) & 0xff00);
              screen[ptr + span] = uvSrc[uvOff + (off & mipMask)];
              span++;
              u = (u + du) | 0; v = (v + dvv) | 0;
            }
            u = u2; v = v2;
          }
        }
        if (leftOver > 0) {
          const u2 = fr(uiz * z), v2 = fr(viz * z);
          const du = Math.trunc(fr(fr(u2 - u) * DIV_TAB_1[leftOver]));
          const dvv = Math.trunc(fr(fr(v2 - v) * DIV_TAB_1[leftOver]));
          while (--leftOver >= 0) {
            const off = ((u >> 16) & 0x00ff) | ((v >> 8) & 0xff00);
            screen[ptr++] = uvSrc[uvOff + (off & mipMask)];
            u = (u + du) | 0; v = (v + dvv) | 0;
          }
        }
      }
      y++; dy--;
    }
  }

  const stats = { visible: 0, selected: 0, culledBack: 0, culledDot: 0, emitted: 0, drawn: 0, flat: 0 };
  return {
    rc, setupClipping, setupCamScreen, setupTarget, stats, setGeoSink,
    setZBuffer(zb) { zBuffer = zb; },      // part6.c: ZBuffer = ZB_Base / NULL
    setEnvCache,
    renderLightFlare,
    // Default (non-vista) entry point: world3/world4 leave Renderer.Render
    // at Render_Camera instead of installing Render_Vista.
    renderScene(screenU16) {
      screen = screenU16;
      statsRef.culledBack = 0; statsRef.culledDot = 0; statsRef.emitted = 0;
      initRenderNoVista();
      renderCamera(world.nodes);
      Object.assign(stats, statsRef);
      stats.selected = nbObjSorted;
    },
    render(vista, time, screenU16) {
      screen = screenU16;
      stats.visible = stats.selected = stats.culledBack = stats.culledDot = stats.emitted = stats.drawn = stats.flat = 0;
      renderVista(vista, time);
      stats.visible = vista.curNb;
      stats.selected = nbPolySorted;
      Object.assign(stats, statsRef);
    },
  };
}
