// state-of-mind — float32 primitives that model the oracle's CODE GENERATION,
// not just its C source.
//
// The oracle is built `-O2 -ffast-math` (Makefile.oracle:16). That flag lets
// clang rewrite float expressions in two ways that are invisible in the C but
// change the last bit of the result:
//
//   1. CONTRACTION. `a*b + c` becomes a single `fmadd`, which rounds ONCE.
//      Written the obvious way in JS -- fr(fr(a*b) + c) -- it rounds twice and
//      disagrees with the oracle on roughly a quarter of random inputs.
//
//   2. RECIPROCAL FOLDING. `x / C` for a compile-time constant C becomes
//      `x * (1/C)`, with 1/C rounded to float once at compile time. This is
//      NOT the same as dividing, and the difference is again about a quarter
//      of inputs.
//
// So when a float chain refuses to go bit-exact and the C source has already
// been matched character for character, the next move is to compile the
// expression alone with the oracle's flags and read the assembly:
//
//   clang -O2 -ffast-math -std=gnu89 -w -S -o - snippet.c
//
// and port THAT. `fmadd`/`fmsub` in the output mean fmaf() here; a `fmul` by
// a pool constant where the C says `/` means frcp() here.

const fr = Math.fround;

const _b = new ArrayBuffer(8);
const _f64 = new Float64Array(_b);
const _u32 = new Uint32Array(_b);

// One double ULP toward +/-Inf. Used only to break float32 rounding ties.
function nudge(x, up) {
  _f64[0] = x;
  const neg = x < 0;
  let lo = _u32[0], hi = _u32[1];
  if (up !== neg) { // magnitude grows
    lo = (lo + 1) >>> 0;
    if (lo === 0) hi = (hi + 1) >>> 0;
  } else {
    if (lo === 0) hi = (hi - 1) >>> 0;
    lo = (lo - 1) >>> 0;
  }
  _u32[0] = lo; _u32[1] = hi;
  return _f64[0];
}

// Single-rounding a*b + c for float32 operands -- arm64 `fmadd`.
//
// a*b is EXACT in a double (24+24 = 48 significant bits, well inside 53), so
// the only hazard is double rounding on the add: the double sum can land
// exactly on a float32 midpoint and then round to even the wrong way. Dekker's
// two-sum recovers the exact residual; nudging by a single double ULP toward
// it breaks the tie correctly and cannot disturb any non-tie case, since a
// float32 ULP is at least 2^29 double ULPs wide.
export function fmaf(a, b, c) {
  const p = a * b;
  const s = p + c;
  if (!Number.isFinite(s)) return fr(s);
  const bv = s - p;
  const err = (p - (s - bv)) + (c - bv);
  if (err === 0) return fr(s);
  return fr(nudge(s, err > 0));
}

// The float32 reciprocal clang folds for `x / C`.
export function frcp(c) { return fr(1.0 / c); }

export { fr };
