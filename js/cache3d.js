// state-of-mind — texture cache III/IV.
//
// Reimplements the mip-page texture cache the tunnel/city worlds use
// (3d_src/cache/ld_mat3.c, cache3.c, mip_cst.c). A cache file is a list of
// records, each one mip image tied to a PAIR of polys (Up/Down) that share
// a square page split along the diagonal. Loading installs each image into
// a shared 256-wide row page per mip level and hangs a per-poly mip chain
// off poly.mips; polys get canonical corner UVs onto their page half.
//
// Cache IV stores small mips (<MIP_INF=4) LZW-packed and larger ones as
// 8x8 blocks gathered from a companion 512-wide RGB565 atlas (the engine's
// own JPEG decode of e.g. nsal.jpg).

const fr = Math.fround;

export const MIP_WIDTH = [1, 2, 4, 8, 16, 32, 64, 128, 256];
export const MIP_SIZE = MIP_WIDTH.map((w) => w * w);
export const MIP_MASKS = [0x0000, 0x0101, 0x0303, 0x0707, 0x0f0f, 0x1f1f, 0x3f3f, 0x7f7f, 0xffff];
const MIP_INF = 4, MIP_WIDTH_IV = 512;

// demo_src/compress.c Compress_Read — headerless variant (the u32-size
// framing belongs to Load_LZW_Block_II, not here). Byte-buffered exactly
// like lzw_dec.c so the stream position lands where the C's FILE* does.
export function compressRead(bytes, pos, out, total) {
  const mode = bytes[pos++];
  if (mode !== 0x02) {
    const size = bytes[pos] | (bytes[pos + 1] << 8) | (bytes[pos + 2] << 16) | (bytes[pos + 3] << 24);
    pos += 4;
    out.set(bytes.subarray(pos, pos + size));
    return pos + size;
  }
  const codeSize = bytes[pos++];
  const clear = 1 << codeSize, end = clear + 1;
  let curSize = codeSize + 1, topSlot = 1 << curSize, slot = clear + 2;
  const prefix = new Uint16Array(4096), suffix = new Uint8Array(4096), stack = new Uint8Array(4097);
  let sp = 0, outPos = 0, oCode = -1, lastByte = 0;
  let buf = null, bufPos = 0, bufLen = 0, bitBuf = 0, bitCnt = 0;
  const nextCode = () => {
    while (bitCnt < curSize) {
      if (bufPos >= bufLen) {
        bufLen = bytes[pos++];
        if (!bufLen) return -1;
        buf = bytes.subarray(pos, pos + bufLen);
        pos += bufLen;
        bufPos = 0;
      }
      bitBuf |= buf[bufPos++] << bitCnt;
      bitCnt += 8;
    }
    const code = bitBuf & ((1 << curSize) - 1);
    bitBuf >>= curSize; bitCnt -= curSize;
    return code;
  };
  for (;;) {
    let code = nextCode();
    if (code < 0 || code === end) break;
    if (code === clear) {
      curSize = codeSize + 1; slot = clear + 2; topSlot = 1 << curSize;
      while (code === clear) code = nextCode();
      if (code === end || code < 0) break;
      if (code >= slot) code = 0;
      oCode = lastByte = code;
      out[outPos++] = code;
      if (outPos >= total) break;
      continue;
    }
    let c = code;
    if (c >= slot) { c = oCode; stack[sp++] = lastByte; }
    while (c >= clear) { stack[sp++] = suffix[c]; c = prefix[c]; }
    stack[sp++] = c;
    lastByte = c;
    if (slot < topSlot) { suffix[slot] = lastByte; prefix[slot++] = oCode; }
    if (slot >= topSlot && curSize < 12) { topSlot <<= 1; curSize++; }
    oCode = code;
    while (sp > 0) {
      out[outPos++] = stack[--sp];
      if (outPos >= total) { sp = 0; return pos; }
    }
  }
  return pos;
}

// cache3.c Search_Free_Mip_Block: rows per level, inserted at list head;
// first free slot wins; block occupies columns [uv, uv+w) of the row page.
function installBlock(cache, level, up, down, maps) {
  const width = MIP_WIDTH[level];
  let row = cache.rows[level].find((r) => r.nbEntries < r.maxEntries &&
    r.slots.some((s) => !s || !s.installed));
  let where;
  if (row) {
    where = row.slots.findIndex((s) => !s || !s.installed);
  } else {
    row = {
      ptr: new Uint16Array(256 * width),
      slots: new Array(Math.trunc(256 / width)).fill(null),
      nbEntries: 0, maxEntries: Math.trunc(256 / width),
    };
    cache.rows[level].unshift(row);       // C inserts new rows first
    where = 0;
  }
  const blk = {
    mipLevel: level, installed: true,
    uv: (where * width) & 0xff,
    up, down, bits: row.ptr,
  };
  let dst = blk.uv;
  for (let i = 0; i < width; i++) {
    row.ptr.set(maps.subarray(i * width, (i + 1) * width), dst);
    dst += 256;
  }
  row.slots[where] = blk;
  row.nbEntries++;
  return blk;
}

// ld_mat3.c Check_Poly_Mip: attach block to the poly's mip chain and stamp
// the canonical page-corner UVs (Up/Down triangle halves).
function checkPolyMip(poly, blk) {
  if (!poly) return;
  if (!poly.mips) poly.mips = { maxMip: 0, levels: new Array(9).fill(null) };
  poly.mips.levels[blk.mipLevel] = blk;
  if (blk.mipLevel > poly.mips.maxMip) poly.mips.maxMip = blk.mipLevel;
  poly.uvI = (poly === blk.up)
    ? [0x0000, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff]
    : [0xffff, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000];
  // Mesh_Store_UV_Area_II — /65536.0 double divisions, FLT result
  const dU1 = fr((poly.uvI[2] - poly.uvI[0]) / 65536);
  const dU2 = fr((poly.uvI[4] - poly.uvI[2]) / 65536);
  const dV1 = fr((poly.uvI[3] - poly.uvI[1]) / 65536);
  const dV2 = fr((poly.uvI[5] - poly.uvI[3]) / 65536);
  poly.u1v2u2v1 = fr(fr(dU1 * dV2) - fr(dU2 * dV1));
}

// ld_mat3.c Load_Txt_Cache_IV: 6-byte magic + u16 record count; per record
// mipLevel u8, Up u32, Down u32, then LZW data (level<4) or an atlas
// offset + 8x8-block gather (level>=4).
export function loadTxtCacheIV(bytes, atlas, world, mipMin) {
  let pos = 6;
  const nbMips = bytes[pos] | (bytes[pos + 1] << 8);
  pos += 2;
  const cache = { rows: Array.from({ length: 9 }, () => []) };
  const byId = new Map(world.nodes.map((n) => [n.id, n]));
  const maps = new Uint16Array(256 * 256);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const atlasU16 = new Uint16Array(atlas.bytes.buffer, atlas.bytes.byteOffset, atlas.bytes.byteLength >> 1);

  for (let rec = 0; rec < nbMips; rec++) {
    const level = bytes[pos]; pos += 1;
    const up = dv.getUint32(pos, true); pos += 4;
    const down = dv.getUint32(pos, true); pos += 4;
    const width = MIP_WIDTH[level];
    if (level < MIP_INF) {
      const raw = new Uint8Array(MIP_SIZE[level] * 2);
      pos = compressRead(bytes, pos, raw, raw.length);
      const m16 = new Uint16Array(raw.buffer);
      maps.set(m16.subarray(0, MIP_SIZE[level]));
    } else {
      let off = dv.getInt32(pos, true); pos += 4;
      let src = off;                     // element index into the atlas
      off &= MIP_WIDTH_IV - 1;
      const nbBlk = Math.trunc(width / 8);
      for (let k = 0; k < nbBlk; k++) {
        let dst = k * 8 * width;
        for (let l = 0; l < nbBlk; l++) {
          for (let j = 0; j < 8; j++) {  // Extract_8x8_Block
            for (let i = 0; i < 8; i++) maps[dst + j * width + i] = atlasU16[src + j * MIP_WIDTH_IV + i];
          }
          src += 8; dst += 8; off += 8;
          if (off === MIP_WIDTH_IV) { src += 7 * MIP_WIDTH_IV; off = 0; }
        }
      }
    }
    // Do_All_Mips: resolve the Up/Down (nodeID<<16 | polyIndex) pairs.
    // NOTE ld_mat3.c reuses Up's poly index when Down's node id is 0xFFFF
    // checked first — replicated via the same order of operations.
    const upNode = byId.get((up >>> 16) & 0xffff);
    const upPoly = upNode.data.polys[up & 0xffff];
    let downPoly = null;
    if (((down >>> 16) & 0xffff) !== 0xffff) {
      const downNode = byId.get((down >>> 16) & 0xffff);
      downPoly = downNode.data.polys[down & 0xffff];
    }
    const blk = installBlock(cache, level, upPoly, downPoly, maps);
    checkPolyMip(upPoly, blk);
    checkPolyMip(downPoly, blk);
    // mipMin=9 in part6 -> no derived lower levels (Do_All_Mips loop body
    // never runs); implement Do_One_Mip only when another world needs it.
    if (level - 1 >= mipMin) throw new Error('derived mip levels not ported');
  }
  return cache;
}

// ld_mat3.c Load_Txt_Cache_III — the same record layout as cache IV, but the
// texels for EVERY level come from the file's own compressed stream; there is
// no companion JPEG atlas. Used by world4 (greets).
export function loadTxtCacheIII(bytes, world, mipMin) {
  let pos = 6;                                   // TXT_MAGIK3
  const nbMips = bytes[pos] | (bytes[pos + 1] << 8);
  pos += 2;
  const cache = { rows: Array.from({ length: 9 }, () => []) };
  const byId = new Map(world.nodes.map((n) => [n.id, n]));
  const maps = new Uint16Array(256 * 256);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  for (let rec = 0; rec < nbMips; rec++) {
    const level = bytes[pos]; pos += 1;
    const up = dv.getUint32(pos, true); pos += 4;
    const down = dv.getUint32(pos, true); pos += 4;
    const raw = new Uint8Array(MIP_SIZE[level] * 2);
    pos = compressRead(bytes, pos, raw, raw.length);
    maps.set(new Uint16Array(raw.buffer).subarray(0, MIP_SIZE[level]));

    const upNode = byId.get((up >>> 16) & 0xffff);
    const upPoly = upNode ? upNode.data.polys[up & 0xffff] : null;
    let downPoly = null;
    if (((down >>> 16) & 0xffff) !== 0xffff) {
      const downNode = byId.get((down >>> 16) & 0xffff);
      if (downNode) downPoly = downNode.data.polys[down & 0xffff];
    }
    const blk = installBlock(cache, level, upPoly, downPoly, maps);
    checkPolyMip(upPoly, blk);
    checkPolyMip(downPoly, blk);
    if (level - 1 >= mipMin) throw new Error('derived mip levels not ported');
  }
  return cache;
}
