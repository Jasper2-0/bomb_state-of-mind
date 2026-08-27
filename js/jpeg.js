// state-of-mind — the engine's own baseline JPEG decoder, ported from
// jpg_src (ld_jpg.c / markers.c / huff.c / adct.c / ycc.c).
//
// WHY A HAND PORT: the pack's JPEGs are decoded by Skal's decoder, whose
// float Chen IDCT and integer YCbCr->RGB rounding differ from any modern
// decoder by a pixel here and there. Those pixels are texture data that the
// 3D parts sample and the mixers blend, so "close enough" drifts the whole
// frame. Reproducing this decoder exactly is what lets the port parse the
// original pack directly instead of consuming oracle-exported pixels
// (see work/re/NOTES.md, and the decoder-parity gate).
//
// FLT DISCIPLINE: the IDCT constants are double literals in the C, so each
// expression evaluates in double over float operands and rounds ONCE at the
// assignment. That is modelled here as fr(<double expression>) — not an
// fround around every operator. The constants are exact binary fractions
// (181/256 etc.), so they survive the trip unchanged.

const fr = Math.fround;

const BLOCK_X = 8, BLOCK_Y = 8, BLOCKSIZE = 64;

// markers.c JPG_Zigzag_Index / JPG_Natural_Order
const ZIGZAG_INDEX = Int32Array.from([
  0, 1, 5, 6, 14, 15, 27, 28,
  2, 4, 7, 13, 16, 26, 29, 42,
  3, 8, 12, 17, 25, 30, 41, 43,
  9, 11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63,
]);
const NATURAL_ORDER = Int32Array.from([
  0, 1, 8, 16, 9, 2, 3, 10,
  17, 24, 32, 25, 18, 11, 4, 5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
  63, 63, 63, 63, 63, 63, 63, 63,
  63, 63, 63, 63, 63, 63, 63, 63,
]);

// tables.h — the default quantization matrices the decoder starts with
const LUMA_Q = Int32Array.from([
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]);
const CHROMA_Q = Int32Array.from([
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
]);

// adct.c — exact binary fractions, kept as doubles like the C macros
const C1D4 = 0.70703125 / 2.0, C1D4P = 0.70703125;
const C1D8 = 0.923828125 / 2.0, C3D8 = 0.3828125 / 2.0;
const C1D16 = 0.98046875 / 2.0, C3D16 = 0.83203125 / 2.0;
const C5D16 = 0.5546875 / 2.0, C7D16 = 0.1953125 / 2.0;

// ycc.c decode-direction constants (note: ycc.c REDEFINES K1..K4 from the
// encoder's values in iycc.c — these are the YCbCr->RGB coefficients)
const K1 = 1.40200, K2 = -0.34414, K3 = -0.71414, K4 = 1.77200;
const KK1 = Math.trunc(0.29900 * 256.0);
const KK2 = Math.trunc(0.58700 * 256.0);
const KK3 = Math.trunc(0.11400 * 256.0);

const M_SOF0 = 0xc0, M_SOF1 = 0xc1, M_SOF3 = 0xc3, M_DHT = 0xc4;
const M_SOI = 0xd8, M_EOI = 0xd9, M_SOS = 0xda, M_DQT = 0xdb, M_DRI = 0xdd;
const M_COM = 0xfe;
const MARKER = 0xff;

const toShort = (x) => (x << 16) >> 16;

// adct.c Do_Chen_IDCT — in place over a 64-float block.
const idctTmp = new Float32Array(64);
function doChenIDCT(In) {
  for (let i = 0; i < 8; i++) {
    const o = i;                            // the C walks In++/PTmp++
    let a1 = fr(C1D4 * (In[o + 0] + In[o + 32]));
    let a2 = fr(C1D8 * In[o + 16] + C3D8 * In[o + 48]);
    { const t = a1; a1 = fr(a1 + a2); a2 = fr(t - a2); }         // FXCHG_REV
    let c3 = fr(C1D16 * In[o + 8] + C7D16 * In[o + 56]);
    let c0 = fr(C3D16 * In[o + 24] + C5D16 * In[o + 40]);
    { const t = c0; c0 = fr(c0 + c3); c3 = fr(c3 - t); }         // FXCHG
    idctTmp[o + 0] = fr(a1 + c0);
    idctTmp[o + 56] = fr(a1 - c0);
    let c1 = fr(C3D16 * In[o + 40] - C5D16 * In[o + 24]);
    c0 = fr(C7D16 * In[o + 8] - C1D16 * In[o + 56]);
    { const t = c0; c0 = fr(c0 + c1); c1 = fr(c1 - t); }         // FXCHG
    { const t = c1; c1 = fr(c1 + c3); c3 = fr(c3 - t); }         // FXCHG
    c1 = fr(C1D4P * c1); c3 = fr(C1D4P * c3);
    idctTmp[o + 24] = fr(a2 + c0);
    idctTmp[o + 32] = fr(a2 - c0);
    a1 = fr(C1D4 * (In[o + 0] - In[o + 32]));
    a2 = fr(C3D8 * In[o + 16] - C1D8 * In[o + 48]);
    { const t = a1; a1 = fr(a1 + a2); a2 = fr(t - a2); }         // FXCHG_REV
    idctTmp[o + 8] = fr(a1 + c3);
    idctTmp[o + 48] = fr(a1 - c3);
    idctTmp[o + 16] = fr(a2 + c1);
    idctTmp[o + 40] = fr(a2 - c1);
  }
  for (let i = 0; i < 8; i++) {
    const p = i * 8, q = i * 8;
    let c0 = fr(C7D16 * idctTmp[p + 1] - C1D16 * idctTmp[p + 7]);
    let c1 = fr(C3D16 * idctTmp[p + 5] - C5D16 * idctTmp[p + 3]);
    { const t = c0; c0 = fr(c0 + c1); c1 = fr(c1 - t); }
    let c2 = fr(C3D16 * idctTmp[p + 3] + C5D16 * idctTmp[p + 5]);
    let c3 = fr(C1D16 * idctTmp[p + 1] + C7D16 * idctTmp[p + 7]);
    { const t = c2; c2 = fr(c2 + c3); c3 = fr(c3 - t); }
    { const t = c1; c1 = fr(c1 + c3); c3 = fr(c3 - t); }
    c1 = fr(C1D4P * c1); c3 = fr(C1D4P * c3);
    let a1 = fr(C1D4 * (idctTmp[p + 0] - idctTmp[p + 4]));
    let a2 = fr(C1D4 * (idctTmp[p + 0] + idctTmp[p + 4]));
    let a3 = fr(C1D8 * idctTmp[p + 2] + C3D8 * idctTmp[p + 6]);
    { const t = a2; a2 = fr(a2 + a3); a3 = fr(t - a3); }
    In[q + 0] = fr(a2 + c2);
    In[q + 7] = fr(a2 - c2);
    In[q + 3] = fr(a3 + c0);
    In[q + 4] = fr(a3 - c0);
    a2 = fr(C3D8 * idctTmp[p + 2] - C1D8 * idctTmp[p + 6]);
    { const t = a1; a1 = fr(a1 + a2); a2 = fr(t - a2); }
    In[q + 2] = fr(a2 + c1);
    In[q + 5] = fr(a2 - c1);
    In[q + 6] = fr(a1 - c3);
    In[q + 1] = fr(a1 + c3);
  }
}

// huff.c bit-stream layer. BSTRM keeps Ptr on the current byte with
// (Bit_Pos & 7) the bit index inside it; 0xFF byte-stuffing is skipped
// while reading entropy-coded data.
class Stream {
  constructor(bytes) { this.b = bytes; this.ptr = 0; this.bitPos = 0; }
  oneByte() {
    const v = this.b[this.ptr];
    this.bitPos = (this.bitPos + 8) & ~0x07;
    this.ptr++;
    return v;
  }
  oneWord() {
    const v = (this.b[this.ptr] << 8) | this.b[this.ptr + 1];
    this.bitPos = (this.bitPos + 16) & ~0x0f;   // C aligns to 16 here
    this.ptr += 2;
    return v;
  }
  sync() {                                  // markers.c JPG_Sync_Stream
    if ((this.bitPos & 0x07) !== 0) {
      this.ptr++;
      this.bitPos = (this.bitPos + 8) & ~0x07;
    }
  }
  oneBitSafe() {                            // huff.c JPG_One_Bit_Safe
    const bitsLeft = (this.bitPos & 0x07) ^ 0x07;
    const ret = this.b[this.ptr] >> bitsLeft;
    if (!bitsLeft) {
      if (this.b[this.ptr] === MARKER) { this.ptr++; this.bitPos += 8; }
      this.ptr++;
    }
    this.bitPos++;
    return ret & 0x01;
  }
  safeBits(howMany) {                       // huff.c JPG_Get_Safe_Bits
    if (!howMany) return 0;
    let ret = 0;
    for (let i = 3, j = 0; i > 0; i--) {
      ret = ((ret << 8) | this.b[this.ptr + j]) >>> 0;
      if (this.b[this.ptr + j] === MARKER) j++;
      j++;
    }
    let bitsLeft = this.bitPos & 0x07;
    ret = (ret << (bitsLeft + 8)) >>> 0;
    ret = ret >>> (32 - howMany);
    this.bitPos += howMany;
    bitsLeft = 8 - bitsLeft - howMany;
    while (bitsLeft <= 0) {
      if (this.b[this.ptr] === MARKER) { this.ptr++; this.bitPos += 8; }
      this.ptr++;
      bitsLeft += 8;
    }
    return ret;
  }
}

// huff.c SizeTable/CodeTable/DecoderTables
function readHuffman(s) {
  const bits = new Int32Array(17);
  let accum = 0;
  for (let i = 1; i <= 16; i++) { bits[i] = s.oneByte(); accum += bits[i]; }
  const huffval = new Int32Array(Math.max(accum, 1));
  for (let i = 0; i < accum; i++) huffval[i] = s.oneByte();

  const size = new Int32Array(257);
  let p = 0;
  for (let i = 1; i < 17; i++) for (let j = 1; j <= bits[i]; j++) size[p++] = i;
  size[p] = 0;

  const code = new Int32Array(257);
  { let pp = 0, c = 0, sz = size[0];
    for (;;) {
      do { code[pp++] = c++; } while (size[pp] === sz && pp < 257);
      if (size[pp] === 0) break;
      do { c <<= 1; sz++; } while (size[pp] !== sz);
    } }

  const maxcode = new Int32Array(18), mincode = new Int32Array(18), valptr = new Int32Array(18);
  let ml = 1;
  { let pp = 0;
    for (let l = 1; l <= 16; ++l) {
      if (bits[l] === 0) maxcode[l] = -1;
      else {
        valptr[l] = pp;
        mincode[l] = code[pp];
        pp += bits[l] - 1;
        maxcode[l] = code[pp];
        ml = l;
        pp++;
      }
    } }
  maxcode[ml]++;
  return { bits, huffval, maxcode, mincode, valptr, ml };
}

function decodeHuffman(s, h) {
  let code = s.oneBitSafe();
  let l = 1;
  for (; code > h.maxcode[l]; l++) code = (code << 1) | s.oneBitSafe();
  return h.huffval[h.valptr[l] + code - h.mincode[l]];
}

function decodeDC(matrix, h, s) {
  const n = decodeHuffman(s, h);
  if (n) {
    let diff = s.safeBits(n);
    if ((diff & (1 << (n - 1))) === 0) {     // bit_set_mask / extend_mask
      diff |= (-1 << n);
      diff++;
    }
    matrix[0] = (matrix[0] + diff) | 0;
  }
  return matrix[0];
}

function decodeAC(matrix, h, s) {
  matrix.fill(0, 1, BLOCKSIZE);
  for (let k = 1; k < BLOCKSIZE;) {
    const r = decodeHuffman(s, h);
    const sz = r & 0x0f, n = (r >> 4) & 0x0f;
    if (sz) {
      k += n;
      if (k >= BLOCKSIZE) break;             // "JPEG mistake" guard in the C
      let v = s.safeBits(sz);
      if ((v & (1 << (sz - 1))) === 0) { v |= (-1 << sz); v++; }
      matrix[k] = v;
      k++;
    } else if (n === 15) k += 16;
    else break;
  }
}

// ycc.c Expand_X / Expand_Y (chroma upsampling inside an 8x8 -> NxN cell)
const MAX_ZOOM = 4, MAX_W = MAX_ZOOM * BLOCK_X, MAX_H = MAX_ZOOM * BLOCK_Y;
function expandX(src) {
  for (let i = MAX_W * MAX_H / 2 - 1; i >= 0; --i) src[2 * i] = src[2 * i + 1] = src[i];
}
function expandY(src, w) {
  w *= BLOCK_X;
  for (let j = (MAX_H / 2 - 1) * w; j >= 0; j -= w) {
    src.copyWithin(2 * j, j, j + w);
    src.copyWithin(2 * j + w, j, j + w);
  }
}

export function decodeJpeg(bytes, format) {
  const s = new Stream(bytes);

  const info = {
    marker: 0, type: 0, dataPrecision: 8, scale: fr(1.0 * (1 << 6)),
    globalWidth: 0, globalHeight: 0, globalNbComp: 0,
    quant: [LUMA_Q, CHROMA_Q, null, null],
    acX: [], acD: [], dcX: [], dcD: [],
    cn: new Int32Array(4), xf: new Int32Array(4), yf: new Int32Array(4),
    tq: new Int32Array(4), width: new Int32Array(4), height: new Int32Array(4),
    maxX: 1, maxY: 1, mduWide: 0, mduHigh: 0,
    nbMDU: 0, curMDU: 0, unit: 0, quantum: 3, size: 0,
    format: format & 0xffff, dst: null, blocks: null, base: null,
  };
  const scan = {
    nbComp: 0, ci: new Int32Array(4), td: new Int32Array(4), ta: new Int32Array(4),
    lastDC: new Int32Array(4), mduWide: 0, mduHigh: 0,
  };

  function makeConsistentFrameSize() {
    let maxX = 1, maxY = 1;
    for (let i = 0; i < 4; i++) {
      if (info.yf[i] > maxY) maxY = info.yf[i];
      if (info.xf[i] > maxX) maxX = info.xf[i];
    }
    for (let i = 0; i < 4; i++) {
      if (info.xf[i]) {
        if (!info.width[i]) info.width[i] = Math.floor((info.globalWidth * info.xf[i] - 1) / maxX) + 1;
        if (!info.height[i]) info.height[i] = Math.floor((info.globalHeight * info.yf[i] - 1) / maxY) + 1;
      }
    }
    info.maxX = maxX; info.maxY = maxY;
    info.mduWide = Math.floor((info.globalWidth - 1) / maxX) + 1;
    info.mduHigh = info.globalHeight ? Math.floor((info.globalHeight - 1) / maxY) + 1 : 0;
    if (info.type !== 3) {
      info.mduWide = Math.floor((info.mduWide - 1) / 8) + 1;
      if (info.mduHigh) info.mduHigh = Math.floor((info.mduHigh - 1) / 8) + 1;
    }
  }

  function getSOF() {
    let len = s.oneWord();
    if (len < 8) return -1;
    info.dataPrecision = s.oneByte();
    info.scale = fr(1.0 * (1 << (14 - info.dataPrecision)));
    info.globalHeight = s.oneWord();
    info.globalWidth = s.oneWord();
    info.globalNbComp = s.oneByte();
    for (let ci = 0; ci < 4; ci++) info.xf[ci] = info.yf[ci] = info.tq[ci] = 0;
    for (let ci = 0; ci < info.globalNbComp; ci++) {
      const j = s.oneByte(), c = s.oneByte();
      info.cn[ci] = j;
      info.xf[j] = (c >> 4) & 0x0f;
      info.yf[j] = c & 0x0f;
      info.tq[j] = s.oneByte();
    }
    makeConsistentFrameSize();
    return 0;
  }

  function getSOS() {
    s.oneWord();
    const n = scan.nbComp = s.oneByte();
    for (let i = 0; i < n; i++) {
      scan.ci[i] = s.oneByte();
      const cb = s.oneByte();
      scan.td[i] = (cb >> 4) & 0x0f;
      scan.ta[i] = cb & 0x0f;
    }
    s.oneByte(); s.oneByte(); s.oneByte();   // SSS / SSE / SAH|SAL
    return 0;
  }

  function getDQT() {
    let len = s.oneWord();
    if (len < 2) return -1;
    len -= 2;
    while (len > 0) {
      const n = s.oneByte();
      if (n === 0xff) break;
      const prec = (n >> 4) & 0x0f, index = n & 0x0f;
      if (index >= 4) return 1;
      if (info.quant[index] === null || info.quant[index] === LUMA_Q || info.quant[index] === CHROMA_Q) {
        info.quant[index] = Int32Array.from(info.quant[index] ?? new Int32Array(64));
      }
      const q = info.quant[index];
      for (let i = 0; i < 64; i++) {
        let tmp = prec ? s.oneWord() : s.oneByte();
        if (!tmp) tmp = i ? q[NATURAL_ORDER[i - 1]] : 16;
        q[NATURAL_ORDER[i]] = tmp;
      }
      len -= 64 + 1;
      if (prec) len -= 64;
    }
    return 0;
  }

  function getDHT() {
    let len = s.oneWord();
    if (len < 2) return -1;
    for (;;) {
      let index = s.oneByte();
      if (index === 0xff) break;
      const where = (index >> 4) & 0x0f;
      index &= 0x0f;
      const h = readHuffman(s);
      if (where) { info.acX[index] = h; info.acD[index] = h; }
      else { info.dcX[index] = h; info.dcD[index] = h; }
    }
    // the C rewinds one byte here (it over-read the 0xFF that ended the loop)
    s.ptr--;
    s.bitPos -= 8;
    return 0;
  }

  function skipMarker() {
    let len = s.oneWord();
    if (len < 2) return -1;
    len -= 2;
    while (--len >= 0) s.oneByte();
    return 0;
  }

  function nextMarker() {
    for (;;) {
      while (s.oneByte() !== 0xff) { if (s.ptr > s.b.length) return 0; }
      do { info.marker = s.oneByte(); } while (info.marker === 0xff);
      if (info.marker === 0x00) continue;
      return info.marker;
    }
  }

  function allocateMDUBlocks() {
    if (info.globalNbComp === 1) { info.quantum = 1; info.format = 0x0000; }
    else {
      info.quantum = (info.format & 0xf000) >> 12;
      if (info.quantum === 0) info.quantum = 1;
    }
    info.unit = info.maxY * info.maxX * BLOCKSIZE * 3;
    info.base = new Int16Array(info.unit);
    info.size = info.globalWidth * info.globalHeight * info.quantum;
    info.dst = new Uint8Array(info.size);
    info.blocks = [];
    let where = 0;
    for (let j = 0; j < info.globalHeight;) {
      const ptr0 = where;
      where += info.maxY * BLOCK_Y * info.globalWidth * info.quantum;
      const dy0 = j;
      j += info.maxY * BLOCK_Y;
      if (j > info.globalHeight) j = info.globalHeight;
      const dy = j - dy0;
      let ptr = ptr0;
      for (let i = 0; i < info.globalWidth;) {
        const dx0 = i;
        i += info.maxX * BLOCK_X;
        if (i > info.globalWidth) i = info.globalWidth;
        info.blocks.push({ ycc: null, where: ptr, comp: 0, dx: i - dx0, dy });
        ptr += info.maxX * BLOCK_X * info.quantum;
      }
    }
  }

  const dstU16 = () => new Uint16Array(info.dst.buffer, 0, info.size >> 1);

  // ycc.c MDU_Block_To_RGB — the SHORT Ycc plane out to the target format
  function mduBlockToRGB(block) {
    const src0 = block.ycc;
    let sp = 0, dp = block.where;
    const dBpS = info.globalWidth * info.quantum;
    const sBpS = (info.maxX * BLOCK_X - block.dx) * 3;
    if (info.globalNbComp === 3) {
      const f = info.format;
      const u16 = f === 0x2565 ? dstU16() : null;
      for (let j = block.dy; j > 0; j--, dp += dBpS, sp += sBpS) {
        for (let i = 0; i < block.dx; i++, sp += 3) {
          const R = src0[sp], G = src0[sp + 1], B = src0[sp + 2];
          if (f === 0x2565) {
            u16[(dp >> 1) + i] = ((R & 0xf8) << 8) | ((G & 0xfc) << 3) | ((B & 0xf8) >> 3);
          } else if (f === 0x3888) {
            info.dst[dp + 3 * i] = B & 0xff;
            info.dst[dp + 3 * i + 1] = G & 0xff;
            info.dst[dp + 3 * i + 2] = R & 0xff;
          } else {
            info.dst[dp + i] = Math.trunc((R + G + B) / 3);
          }
        }
      }
    } else {
      for (let j = block.dy; j > 0; j--, dp += dBpS, sp += sBpS) {
        for (let i = 0; i < block.dx; i++, sp += 3) {
          info.dst[dp + i] = (src0[sp] * KK3 + src0[sp + 1] * KK2 + src0[sp + 2] * KK1) >> 8;
        }
      }
    }
  }

  const wmTmp = new Float32Array(MAX_W * MAX_H);

  // ycc.c JPG_Write_Matrix
  function writeMatrix(matrix, block, x, y, nx, ny, shift) {
    const scX = info.maxX / nx, scY = info.maxY / ny;
    for (let i = 0; i < BLOCKSIZE; i++) wmTmp[i] = fr(matrix[i] + shift);
    for (let i = 1; i < scX; ++i) expandX(wmTmp);
    for (let i = 1; i < scY; ++i) expandY(wmTmp, i - 1 + scX);

    if (block.ycc === null) {
      if (info.base !== null) { block.ycc = info.base; info.base = null; }
      else block.ycc = new Int16Array(info.unit);
    }
    const dstA = block.ycc;
    let dp = x * scX * BLOCK_X * 3 + (y * scY * BLOCK_Y) * info.maxX * BLOCK_X * 3;
    let sp = 0;
    const rowStep = 3 * info.maxX * BLOCK_X;
    const cols = scX * BLOCK_X * 3;

    if (block.comp === 1) {                  // Y
      for (let j = 0; j < scY * BLOCK_Y; ++j, dp += rowStep) {
        for (let i = 0; i < cols; i += 3) {
          const Y = toShort(Math.trunc(fr(wmTmp[sp++] * info.scale)));
          dstA[dp + i] = Y; dstA[dp + i + 1] = Y; dstA[dp + i + 2] = Y;
        }
      }
    } else if (block.comp === 2) {           // Cb
      for (let j = 0; j < scY * BLOCK_Y; ++j, dp += rowStep) {
        for (let i = 0; i < cols; i += 3) {
          const Cb = fr(fr(wmTmp[sp++] - 128) * info.scale);
          dstA[dp + i + 2] += toShort(Math.trunc(fr(Cb * K4)));
          dstA[dp + i + 1] += toShort(Math.trunc(fr(Cb * K2)));
        }
      }
    } else {                                 // Cr
      for (let j = 0; j < scY * BLOCK_Y; ++j, dp += rowStep) {
        for (let i = 0; i < cols; i += 3) {
          const Cr = fr(fr(wmTmp[sp++] - 128) * info.scale);
          dstA[dp + i] += toShort(Math.trunc(fr(Cr * K1)));
          dstA[dp + i + 1] += toShort(Math.trunc(fr(Cr * K3)));
        }
      }
    }

    if (x === nx - 1 && y === ny - 1 && block.comp === info.globalNbComp) {
      const r = 14 - info.dataPrecision;
      const m = (0x100 << r) - 1;
      for (let i = 0; i < info.unit; ++i) {
        let C = block.ycc[i];
        if (C < 0) C = 0; else if (C > m) C = m;
        block.ycc[i] = C >> r;
      }
      mduBlockToRGB(block);
      if (info.base === null) info.base = block.ycc;
      block.ycc = null;
    }
  }

  const inCoef = new Int32Array(BLOCKSIZE);
  const outBlk = new Float32Array(BLOCKSIZE);

  function decodeScan() {
    const idctShift = info.dataPrecision > 8 ? 2048.0 : 128.0;
    let bi = 0;
    for (;;) {
      if (info.nbMDU >= 0 && info.curMDU >= info.nbMDU) break;
      const block = info.blocks[bi];
      if (!block) break;
      for (let j = 0; j < scan.nbComp; j++) {
        const hDC = info.dcD[scan.td[j]];
        const hAC = info.acD[scan.ta[j]];
        const matrix = info.quant[info.tq[scan.ci[j]]];
        const nx = info.xf[scan.ci[j]], ny = info.yf[scan.ci[j]];
        block.comp++;
        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            inCoef[0] = scan.lastDC[j];
            decodeDC(inCoef, hDC, s);
            scan.lastDC[j] = inCoef[0];
            decodeAC(inCoef, hAC, s);
            for (let i = 0; i < BLOCKSIZE; i++)
              outBlk[i] = fr(1.0 * inCoef[ZIGZAG_INDEX[i]] * matrix[i]);
            doChenIDCT(outBlk);
            writeMatrix(outBlk, block, x, y, nx, ny, idctShift);
          }
        }
      }
      bi++;
      info.curMDU++;
    }
    s.sync();
  }

  info.nbMDU = 0;
  for (;;) {
    info.curMDU = 0;
    let type = null;
    // JPG_Read_Markers
    reading: for (;;) {
      const m = nextMarker();
      if (m === 0) { type = 'error'; break; }
      switch (m) {
        case M_SOI: break;
        case M_EOI: type = 'eoi'; break reading;
        case M_SOS:
          getSOS();
          if (scan.nbComp === 1) {
            let i = info.globalWidth * info.xf[scan.ci[0]] - 1;
            i = Math.floor(i / info.maxX) + 1;
            if (info.type !== 3) i = Math.floor((i - 1) / 8) + 1;
            scan.mduWide = i;
            i = info.globalHeight * info.yf[scan.ci[0]] - 1;
            i = Math.floor(i / info.maxY) + 1;
            if (info.type !== 3) i = Math.floor((i - 1) / 8) + 1;
            scan.mduHigh = i;
          } else { scan.mduWide = info.mduWide; scan.mduHigh = info.mduHigh; }
          info.nbMDU = info.globalHeight ? info.mduWide * info.mduHigh : -1;
          scan.lastDC.fill(0);
          type = 'sos';
          break reading;
        case M_DQT: if (getDQT() !== 0) { type = 'error'; break reading; } break;
        case M_DRI: s.oneWord(); s.oneWord(); break;
        case M_DHT: if (getDHT() !== 0) { type = 'error'; break reading; } break;
        case M_SOF0: case M_SOF1: case M_SOF3:
          info.type = m & 0x0f;
          if (getSOF()) { type = 'error'; break reading; }
          break;
        case M_COM: skipMarker(); break;
        default:
          if (m >= 0xe0 && m <= 0xef) { skipMarker(); break; }   // APPn
          if (m >= 0xd0 && m <= 0xd7) { skipMarker(); break; }   // RSTn
          skipMarker();
          type = 'eoi';
          break reading;
      }
    }
    if (type === 'eoi') break;
    if (type === 'error') return null;
    if (type === 'sos') {
      if (info.nbMDU >= 0 && info.curMDU >= info.nbMDU) break;
      if (info.blocks === null) allocateMDUBlocks();
      scan.lastDC.fill(0);
      decodeScan();
    }
  }

  if (info.dst === null) return null;
  return {
    width: info.globalWidth,
    height: info.globalHeight,
    // w/h/fmt aliases keep this a drop-in for the asset-index shape the
    // engine's consumers were written against
    w: info.globalWidth,
    h: info.globalHeight,
    fmt: info.format,
    format: info.format,
    size: info.size,
    bytes: info.dst,
  };
}
