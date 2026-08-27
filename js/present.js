// state-of-mind — presentation layer.
//
// The fidelity engine renders 320x200 RGB565 and that never changes: it is
// the verification target, and every gate compares it to the oracle. This
// module is only about getting those pixels onto a modern display honestly.
//
// WHY 1600x1200. The source is 320x200 (8:5) but was displayed at 4:3, so a
// source pixel is not square -- it has a 5:6 pixel aspect. 320*5 = 1600 and
// 200*6 = 1200, so at a 1600x1200 backbuffer every source pixel becomes an
// exact 5x6 block of SQUARE display pixels: period-correct aspect, integer
// scaling, and not one resampled pixel anywhere. The ladder up keeps that
// property at integer multiples -- 3200x2400 (10x/12x), 4800x3600 (15x/18x).
//
// 2880x2160 is deliberately not on the ladder: 2160/200 = 10.8, so it cannot
// be reached without filtering the source.
//
// The final fit to the viewport is left to the browser, which scales the
// 4:3 backbuffer with its own filter. That is the one place smoothing is
// wanted -- it is display fitting, not image reconstruction.

import { W, H, PIXEL_ASPECT_X, PIXEL_ASPECT_Y } from './screen.js';

export const BASE_W = W, BASE_H = H;
export const SCALE_X = PIXEL_ASPECT_X, SCALE_Y = PIXEL_ASPECT_Y;

// Backbuffer ladder. Each is an integer multiple of the 5:6 base, so each
// preserves exact block replication.
export const LADDER = [
  { w: 1600, h: 1200, mult: 1 },
  { w: 3200, h: 2400, mult: 2 },
  { w: 4800, h: 3600, mult: 3 },
];

export function pickBackbuffer(request) {
  if (request === 'native') return LADDER[0];
  const n = Number(request);
  const hit = LADDER.find((l) => l.w === n);
  if (hit) return hit;
  return LADDER[0];
}

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  // a_pos is a full-screen triangle in clip space. The demo's framebuffer is
  // top-down and GL textures are bottom-up, so flip V here rather than
  // paying for UNPACK_FLIP_Y_WEBGL on every upload.
  v_uv = vec2((a_pos.x + 1.0) * 0.5, (1.0 - a_pos.y) * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `
precision mediump float;
uniform sampler2D u_tex;
varying vec2 v_uv;
void main() { gl_FragColor = vec4(texture2D(u_tex, v_uv).rgb, 1.0); }`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(`shader: ${gl.getShaderInfoLog(sh)}`);
  }
  return sh;
}

// `capture` asks the context to KEEP its drawing buffer after compositing.
//
// Off for playback, because preserving the buffer costs a copy every frame and
// nothing in a running show ever reads it back. On for the headless paths,
// where the opposite is true: ?t= renders one frame at load and is then
// screenshotted much later, by which point an unpreserved buffer has been
// discarded and toDataURL returns SOLID BLACK with no error. That is not a
// hypothetical -- it is how this very flag was first "measured" as having no
// effect, because both captures came back black and a correlation of two black
// frames is 1.000.
function createGLPresenter(canvas, backbuffer, capture = false) {
  const attrs = { alpha: false, antialias: false, depth: false, stencil: false,
    preserveDrawingBuffer: capture };
  const gl = canvas.getContext('webgl2', attrs) || canvas.getContext('webgl', attrs);
  if (!gl) return null;

  canvas.width = backbuffer.w;
  canvas.height = backbuffer.h;

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(prog)}`);
  }
  gl.useProgram(prog);

  // One oversized triangle rather than two: no shared edge to crack, one
  // fewer vertex, and the excess is clipped for free.
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'a_pos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // NEAREST is not a stylistic choice here: at an integer scale it is what
  // makes each source pixel an exact block instead of a blend of neighbours.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);

  gl.viewport(0, 0, backbuffer.w, backbuffer.h);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 2);

  let allocW = 0, allocH = 0;

  return {
    kind: gl instanceof WebGL2RenderingContext ? 'webgl2' : 'webgl',
    backbuffer,
    // The engine's RGB565 buffer is uploaded verbatim: GL's
    // UNSIGNED_SHORT_5_6_5 has red in bits 11..15, exactly the demo's own
    // layout. No CPU-side unpacking loop at all, which is also why this path
    // is cheaper than putImageData and not just sharper.
    // `w`/`h` default to the 320x200 framebuffer. A part that rendered
    // natively passes its own size; when that equals the backbuffer the
    // upload is 1:1 and NEAREST is a no-op rather than a block replicate.
    present(frame565, w = BASE_W, h = BASE_H) {
      if (allocW !== w || allocH !== h) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, w, h, 0,
          gl.RGB, gl.UNSIGNED_SHORT_5_6_5, frame565);
        allocW = w; allocH = h;
      } else {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h,
          gl.RGB, gl.UNSIGNED_SHORT_5_6_5, frame565);
      }
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    dispose() { gl.deleteTexture(tex); gl.deleteBuffer(buf); gl.deleteProgram(prog); },
  };
}

function create2DPresenter(canvas) {
  // Fallback: keep the backing store at 320x200 and let CSS do the 4:3
  // stretch, which is what this port did before the GL path existed.
  canvas.width = BASE_W;
  canvas.height = BASE_H;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(BASE_W, BASE_H);
  const image32 = new Uint32Array(image.data.buffer);
  return {
    kind: '2d',
    backbuffer: { w: BASE_W, h: BASE_H, mult: 0 },
    // The 2d fallback has a 320x200 backing store, so it can only present
    // the 320x200 framebuffer; a caller with a native frame must pass the
    // fidelity one alongside.
    present(frame565, w = BASE_W, h = BASE_H) {
      if (w !== BASE_W || h !== BASE_H) return false;
      for (let i = 0; i < BASE_W * BASE_H; i++) {
        const v = frame565[i];
        const r = (v >> 11) & 0x1f, g = (v >> 5) & 0x3f, b = v & 0x1f;
        // 5/6-bit -> 8-bit by bit replication, the standard expansion: it
        // maps 31 -> 255 exactly, where a plain shift would give 248.
        image32[i] = 0xff000000
          | (((b << 3) | (b >> 2)) << 16)
          | (((g << 2) | (g >> 4)) << 8)
          | ((r << 3) | (r >> 2));
      }
      ctx.putImageData(image, 0, 0);
    },
    dispose() {},
  };
}

// Returns a presenter with { kind, backbuffer, present(frame565), dispose() }.
// `mode` is 'webgl' (default, falls back to 2d) or '2d' to force the fallback.
export function createPresenter(canvas, { mode = 'webgl', scale, capture = false } = {}) {
  if (mode !== '2d') {
    try {
      const p = createGLPresenter(canvas, pickBackbuffer(scale), capture);
      if (p) return p;
    } catch {
      // fall through: a context that exists but cannot compile the shader is
      // no more useful than no context at all
    }
  }
  return create2DPresenter(canvas);
}
