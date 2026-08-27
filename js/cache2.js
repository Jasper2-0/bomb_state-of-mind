// state-of-mind — texture cache II (3d_src/cache/ld_mat2b.c + cache1.c).
//
// A different cache from the tunnel/city's cache IV. Here a single 256x256
// RGB565 image (maps1.jpg) holds THREE mip pyramids packed into one square:
// slot 0 along the top edge, slot 1 down the left edge, slot 2 in the
// remaining quadrant. Mip_Uo/Mip_Vo give each (slot, level) sub-rect origin
// in 16.16, and the level is chosen by projected area (Retreive_Texture).
//
// Used by the blob part through Env_Shader, whose UVs come from the vertex
// normal rather than the mesh, so the mip choice uses the actual UV cross
// product instead of the poly's precomputed U1V2_U2V1.

const fr = Math.fround;

export const MAX_MIP = 9;
export const MIP_WIDTH_II = [1, 2, 4, 8, 16, 32, 64, 128, 256];
export const MIP_MASKS_II = [0x0000, 0x0101, 0x0303, 0x0707, 0x0f0f, 0x1f1f, 0x3f3f, 0x7f7f, 0xffff];
const P16 = 65536.0;
export const MIP_UO = [
  [1 * P16, 2 * P16, 4 * P16, 8 * P16, 16 * P16, 32 * P16, 64 * P16, 128 * P16, 256 * P16],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1 * P16, 2 * P16, 4 * P16, 8 * P16, 16 * P16, 32 * P16, 64 * P16, 128 * P16, 256 * P16],
];
export const MIP_VO = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1 * P16, 2 * P16, 4 * P16, 8 * P16, 16 * P16, 32 * P16, 64 * P16, 128 * P16, 256 * P16],
  [1 * P16, 2 * P16, 4 * P16, 8 * P16, 16 * P16, 32 * P16, 64 * P16, 128 * P16, 256 * P16],
];
export const MIP_SCALES = [
  256.0 * 256.0, 512.0 * 256.0, 1024.0 * 256.0, 2048.0 * 256.0,
  4096.0 * 256.0, 8192.0 * 256.0, 16384.0 * 256.0, 32768.0 * 256.0,
  65536.0 * 256.0,
].map(fr);
export const MIP_SLOT0_OFFSETS = [1, 1 * 256, 1 + 1 * 256];

// raster.h
const EPS_MIP_BLF = fr(1.0 / 256), EPS_MIP_0 = fr(4.0 / 256), EPS_MIP_1 = fr(8.0 / 256);
const EPS_MIP_2 = fr(16.0 / 256), EPS_MIP_3 = fr(32.0 / 256), EPS_MIP_4 = fr(64.0 / 256);
const EPS_MIP_5 = fr(128.0 / 256);

const TEXTURE_MAP_SIZE_16b = 256 * 256 * 2;

// ld_mat2b.c Load_Txt_Cache_IIbis. The .lzw here carries only the texture-map
// RECORDS (16-byte i386 structs); the pixels come from the companion JPEG,
// already decoded to 565 by our own decoder.
export function loadTxtCacheIIbis(recBytes, atlas) {
  let pos = 6;                                   // TXT_MAGIK2
  const nbMaps = recBytes[pos] | (recBytes[pos + 1] << 8);
  pos += 2;
  const dv = new DataView(recBytes.buffer, recBytes.byteOffset, recBytes.byteLength);
  const maps = [];
  for (let i = 0; i < nbMaps; i++) {
    // Read_Texture_Map_Record (3d_src/cache/mat.c): the on-disk record is a
    // 16-byte i386 struct — u32 Type, i16 Users, i16 Quantum, u8 Slot,
    // u8 Base, u16 Offset, then a 4-byte pointer that is NOT data.
    const type = dv.getUint32(pos, true);
    const users = dv.getInt16(pos + 4, true);
    const quantum = dv.getInt16(pos + 6, true);
    const slot = recBytes[pos + 8];
    const base = recBytes[pos + 9];
    const offset = dv.getUint16(pos + 10, true);
    pos += 16;
    maps.push({ type, users, quantum, offset, base, slot, ptr: base * (TEXTURE_MAP_SIZE_16b / 2) });
  }
  const bits = new Uint16Array(atlas.bytes.buffer, atlas.bytes.byteOffset, atlas.bytes.byteLength >> 1);
  return { nbMaps, maps, bits };
}

// cache1.c Retreive_Texture — returns the chosen level; < 3 means "flat".
export function retreiveTextureII(st, cache, texture, R) {
  let mip;
  if (R < EPS_MIP_BLF * EPS_MIP_BLF) mip = 7;    // "bilinear filtering!!" — same slot
  else if (R < EPS_MIP_0 * EPS_MIP_0) mip = 7;
  else if (R < EPS_MIP_1 * EPS_MIP_1) mip = 6;
  else if (R < EPS_MIP_2 * EPS_MIP_2) mip = 5;
  else if (R < EPS_MIP_3 * EPS_MIP_3) mip = 4;
  else if (R < EPS_MIP_4 * EPS_MIP_4) mip = 3;
  else if (R < EPS_MIP_5 * EPS_MIP_5) mip = 2;
  else mip = 1;
  st.curMip = mip;
  const slot = texture.slot;
  if (mip < 3) {
    st.flatColor = cache.bits[texture.ptr + MIP_SLOT0_OFFSETS[slot]];
    st.mipMask = MIP_MASKS_II[mip];
    return mip;
  }
  st.mipScale = MIP_SCALES[mip];
  st.uoMip = MIP_UO[slot][mip];
  st.voMip = MIP_VO[slot][mip];
  st.mipMask = MIP_MASKS_II[mip];
  st.uvSrc = cache.bits;
  st.uvOff = texture.ptr;
  return mip;
}
