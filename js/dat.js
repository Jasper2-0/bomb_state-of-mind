// state-of-mind — data-layer parsers.
//
// Everything the engine reads from mind.dat, reimplemented from the v1.3
// source release (provenance in comments as file:line into mind_src_v13).
// All formats are little-endian byte streams; no decoder here depends on
// platform layout.

// ---- mind.dat container --------------------------------------------------
// demo_src/io.c F_OPEN: records are `<name>\n<ascii size>\n<size bytes>`,
// concatenated, no index.
export function parseDat(bytes) {
  const entries = new Map();
  let off = 0;
  while (off < bytes.length) {
    let nl = bytes.indexOf(0x0a, off);
    const name = String.fromCharCode(...bytes.subarray(off, nl));
    let nl2 = bytes.indexOf(0x0a, nl + 1);
    const size = parseInt(String.fromCharCode(...bytes.subarray(nl + 1, nl2)), 10);
    if (!/^[\w.-]+$/.test(name) || !Number.isFinite(size)) {
      throw new Error(`mind.dat: bad record at ${off}`);
    }
    entries.set(name, bytes.subarray(nl2 + 1, nl2 + 1 + size));
    off = nl2 + 1 + size;
  }
  return entries;
}

// ---- LZW (GIF-flavor) ----------------------------------------------------
// demo_src/lzw_dec.c LZW_Decoder + Fill_Byte_Buffer: standard GIF LZW with
// count-prefixed sub-blocks, initial code size `size`, clear = 1<<size,
// end = clear+1. Decodes exactly `total` bytes into out.
export function lzwDecode(bytes, pos, out, total, size) {
  const clear = 1 << size;
  const end = clear + 1;
  let curSize = size + 1;
  let topSlot = 1 << curSize;
  let slot = clear + 2;
  const prefix = new Uint16Array(4096);
  const suffix = new Uint8Array(4096);
  const stack = new Uint8Array(4097);
  let sp = 0;
  let bitBuf = 0, bitCnt = 0, blockLeft = 0;
  let outPos = 0;
  let oCode = -1, lastByte = 0;

  const nextCode = () => {
    while (bitCnt < curSize) {
      if (blockLeft === 0) {
        blockLeft = bytes[pos++];
        if (blockLeft === undefined) return -1;
        if (blockLeft === 0) return -1;
      }
      bitBuf |= bytes[pos++] << bitCnt;
      bitCnt += 8;
      blockLeft--;
    }
    const code = bitBuf & (topSlot - 1) & ((1 << curSize) - 1);
    bitBuf >>= curSize;
    bitCnt -= curSize;
    return code;
  };

  for (;;) {
    let code = nextCode();
    if (code < 0 || code === end) break;
    if (code === clear) {
      curSize = size + 1;
      slot = clear + 2;
      topSlot = 1 << curSize;
      while (code === clear) code = nextCode();
      if (code === end || code < 0) break;
      if (code >= slot) code = 0;
      oCode = lastByte = code;
      out[outPos++] = code;
      if (outPos >= total) break;
      continue;
    }
    let c = code;
    if (c >= slot) {              // KwKwK case
      c = oCode;
      stack[sp++] = lastByte;
    }
    while (c >= clear) {
      stack[sp++] = suffix[c];
      c = prefix[c];
    }
    stack[sp++] = c;
    lastByte = c;
    if (slot < topSlot) {
      suffix[slot] = lastByte;
      prefix[slot++] = oCode;
    }
    if (slot >= topSlot && curSize < 12) {
      topSlot <<= 1;
      curSize++;
    }
    oCode = code;
    while (sp > 0) {
      out[outPos++] = stack[--sp];
      if (outPos >= total) return outPos;
    }
  }
  return outPos;
}

// demo_src/compress.c Compress_Read / Load_LZW_Block_II:
// uint32 total size, then mode byte (0x02 = LZW: code-size byte + sub-block
// stream; otherwise: uint32 size + raw bytes).
export function loadLzwBlock(bytes, pos) {
  const dv = { u32: (p) => bytes[p] | (bytes[p + 1] << 8) | (bytes[p + 2] << 16) | (bytes[p + 3] << 24) };
  const total = dv.u32(pos); pos += 4;
  const out = new Uint8Array(total);
  const mode = bytes[pos++];
  if (mode === 0x02) {
    const codeSize = bytes[pos++];
    lzwDecode(bytes, pos, out, total, codeSize);
  } else {
    const rawSize = dv.u32(pos); pos += 4;
    out.set(bytes.subarray(pos, pos + rawSize));
  }
  return out;
}

// ---- DMASK (.msk) --------------------------------------------------------
// demo_src/dmask.c Load_DMask: "d@MsK", then 13 header bytes
// (nbCol:16, w:16, h:16, pakSize:32, alpha:16, rleType:8), palette 3*nbCol,
// H_Index (h * int32, unused by playback), LZW raw block.
export function parseDMask(bytes) {
  const magic = 'd@MsK';
  for (let i = 0; i < 5; i++) {
    if (bytes[i] !== magic.charCodeAt(i)) throw new Error('not a DMASK');
  }
  let p = 5;
  const nbCol = bytes[p] | (bytes[p + 1] << 8);
  const w = bytes[p + 2] | (bytes[p + 3] << 8);
  const h = bytes[p + 4] | (bytes[p + 5] << 8);
  const alpha = bytes[p + 10] | (bytes[p + 11] << 8);
  p += 13;
  const pal = bytes.subarray(p, p + nbCol * 3);
  p += nbCol * 3;
  p += h * 4; // H_Index, playback never uses it
  const raw = loadLzwBlock(bytes, p);
  // MBits table, dmask.c Load_DMask
  let mBits;
  if (nbCol <= 8) mBits = 5;
  else if (nbCol <= 16) mBits = 4;
  else if (nbCol <= 32) mBits = 3;
  else if (nbCol <= 64) mBits = 2;
  else if (nbCol <= 128) mBits = 1;
  else mBits = 0;
  return { w, h, nbCol, pal, alpha, mBits, raw, bits: null };
}

// dmask.c Uncompress_DMask_II + Emit_Raw_Line_II: alpha-filled buffer, then
// per line: EOL=0xFF ends line, FILL=0xFD (n+1 × byte), SKIP=0xFE (n+1
// untouched), BLOCK=0xFC (n+1 literal), SOLO=0xFB (1 literal), any other
// code is itself a literal pixel.
export function uncompressDMask(mask) {
  if (mask.bits) return mask.bits;
  const dst = new Uint8Array(mask.w * mask.h).fill(mask.alpha & 0xff);
  const src = mask.raw;
  let s = 0;
  for (let y = 0; y < mask.h; y++) {
    let d = y * mask.w;
    for (;;) {
      const code = src[s++];
      if (code === 0xff) break;             // EOL
      else if (code === 0xfd) {             // FILL
        const n = src[s++] + 1;
        const c = src[s++];
        dst.fill(c, d, d + n); d += n;
      } else if (code === 0xfe) {           // SKIP
        d += src[s++] + 1;
      } else if (code === 0xfc) {           // BLOCK
        const n = src[s++] + 1;
        dst.set(src.subarray(s, s + n), d); d += n; s += n;
      } else if (code === 0xfb) {           // SOLO
        dst[d++] = src[s++];
      } else dst[d++] = code;
    }
  }
  mask.bits = dst;
  return dst;
}

// ---- GIF (demo_src/ld_gif.c Load_GIF) -----------------------------------
// Minimal parser matching what the engine reads: 13-byte header, global
// color map (2^((flags&0x0F)+1) entries), skip 0x21 extensions, image
// descriptor 0x2C (9 bytes; engine takes W/H from bytes 4-7), pixel-size
// byte, LZW image data. Returns { w, h, nbCol, pal, bits }.
export function parseGif(bytes) {
  const nbCol = 1 << ((bytes[10] & 0x0f) + 1);
  let p = 13;
  const pal = bytes.subarray(p, p + 3 * nbCol);
  p += 3 * nbCol;
  for (;;) {
    const c = bytes[p++];
    if (c === 0x21) {                       // extension: id + sub-blocks
      p++;
      for (;;) { const n = bytes[p++]; if (n === 0) break; p += n; }
    } else if (c === 0x2c) {
      const w = bytes[p + 4] | (bytes[p + 5] << 8);
      const h = bytes[p + 6] | (bytes[p + 7] << 8);
      p += 9;
      const codeSize = bytes[p++];
      const bits = new Uint8Array(w * h);
      lzwDecode(bytes, p, bits, w * h, codeSize);
      return { w, h, nbCol, pal, bits };
    } else if (c === 0x3b || c === undefined) {
      throw new Error('GIF: no image block');
    }
  }
}

// ---- lyric-event table (oracle export "ffx.raw") -------------------------
// Emitted by SOM_Dump_FFx (dem/src/fonts2.c under SOM_ORACLE). Per event:
// i32 start_ms, i32 len_ms, i32 fx, f32 x, f32 y, f32 scale,
// u16 text_len, text bytes.
export function parseFfx(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const events = [];
  let p = 0;
  while (p < bytes.length) {
    const start = dv.getInt32(p, true);
    const len = dv.getInt32(p + 4, true);
    const fx = dv.getInt32(p + 8, true);
    const x = dv.getFloat32(p + 12, true);
    const y = dv.getFloat32(p + 16, true);
    const scale = dv.getFloat32(p + 20, true);
    const tl = dv.getUint16(p + 24, true);
    p += 26;
    const text = String.fromCharCode(...bytes.subarray(p, p + tl));
    p += tl;
    events.push({ start, len, fx, x, y, scale, text });
  }
  return events;
}

// ---- engine-decoded raw assets (work/unpacked/decoded) -------------------
// Emitted by the oracle's SOM_ASSET_DUMP hook (jpg_src/ld_jpg.c): the
// engine's own JPEG decode, byte-exact. assets.idx lines:
// `name w h fmt size`.
export function parseAssetIndex(text) {
  const out = new Map();
  for (const line of text.trim().split('\n')) {
    const [name, w, h, fmt, size] = line.split(' ');
    out.set(name, { w: +w, h: +h, fmt: parseInt(fmt, 16), size: +size });
  }
  return out;
}
