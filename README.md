# state-of-mind (canvas port)

A from-scratch JavaScript reconstruction of *State of Mind* by **Bomb**, which
took 2nd place in the demo competition at The Party 1998. Code by Skal, music
by Senser, graphics by Made and Krafton.

Live: **jasper2-0.github.io/bomb_state-of-mind/**

Nothing is emulated. The original's software renderer is ported routine by
routine to JavaScript, and every asset is read out of the shipped `mind.dat` at
runtime by ported parsers — the pack in `data/` is the 1998 file, byte for byte.

    index.html          the launcher
    js/main.js          shell: clock, controls, presentation
    js/fidelity.js      the engine — the show's parts and their mixers
    js/raster3d.js      the polygon rasteriser
    js/world3ds.js      .3ds parsing, normals, the vertex weld
    js/scene3d.js       camera and world transforms
    js/caustics.js      plasma, caustics, the ray tunnel
    js/bilin.js         the bilinear displacement mapper
    js/mcube.js         marching-cubes blobs
    js/jpeg.js          a ported JPEG decoder, byte-exact with the original's
    js/dat.js           the pack format
    js/present.js       upload and integer upscale
    data/               mind.dat plus three source-derived tables

**controls** — `space` pause · `←`/`→` seek 5s · `f` fullscreen · `esc` stop

## URL parameters

| parameter | effect |
|---|---|
| *(none)* | the 60 Hz remaster, mesh weld off — see below |
| `?original` | the 25 Hz show, as it ran in 1998. `?quality=original` also works |
| `?weld=1` | restore the original's vertex weld |
| `?rate=30` `?rate=50` `?rate=60` | re-time the show to another rate |
| `?present=2d` | force the canvas-2d fallback (default: WebGL) |
| `?scale=1600` | backbuffer width; also `3200` and `4800`, each an exact multiple |
| `?debug` | time / frame / phase readout and a timing overlay |
| `?embedded` | hide the status line, for an iframe |
| `?t=10` | render one deterministic frame at that show time |

## How the original works

The demo is a DOS and Win32 executable rendering at **320x200 in 8-bit**,
presented on a 4:3 monitor — so its pixels are not square, and this port
reproduces that aspect rather than the raw buffer shape.

The show is a **table of 73 phases**. Each names a start time in milliseconds,
an optional `Init_` routine, and a `Loop_` routine run every frame; the timer
advances in fixed **40 ms steps**, so the show is a 25 Hz film rather than
anything frame-rate dependent. Transitions between phases are themselves
phases — the dissolves are `Init_Dissolve` plus `Do_Dissolve_X`/`_Y`, and the
fades and blurs likewise.

Compositing happens in a **packed 777 colour space**: three 7-bit channels held
in one 32-bit word at bits 16-22, 8-14 and 0-6, with the gaps between them
absorbing carries so that a saturating add can be done on all three channels at
once. Everything is table-driven; the framebuffers are 16-bit 565 and the
mixers are lookup tables built at load.

Geometry comes from `.3ds` files in the pack, lit per-vertex and rasterised
with a texture cache. Assets — the meshes, the JPEG textures, the vista tables
and the soundtrack — are all inside `mind.dat`.

Skal later released the source (v1.3), which is what made this port checkable
against something better than screenshots.

## Port notes

**A native oracle, not eyeballing.** The v1.3 source is built as a native
binary that dumps its own framebuffer, and the port is compared against 1097
of its frames spanning the whole 219-second show. **At `?original&weld=1` —
25 Hz with the original's vertex weld — 99.993% of pixels are identical on
average and 99.784% at the worst frame.** That configuration is the one the
oracle describes; the two defaults below deliberately differ from it.

**The 60 Hz remaster is the default, and it is authored.** 25 Hz does not
divide a 60 Hz display evenly (60/25 = 2.4), so the original judders by
construction on modern hardware. The remaster re-times the show and corrects
the nine effects that count frames rather than time, so they keep their
authored speed. No oracle can verify a frame that never existed, so those
corrections are held to *properties* instead of pixels — that each is exactly
the identity at 25 Hz, that no effect changes speed, that nothing decays at a
different rate. `?original` gives the 25 Hz show.

**The vertex weld is off by default, and that is a deviation.** The original
welds nearby mesh vertices together, and its threshold test compares a squared
distance against a linear bound — so the weld radius is far larger than
intended. It deletes 552 of the city's 1974 polygons and empties ten objects
outright. It is a bug, the 1998 executable had it, and `?weld=1` reproduces it.
Turning it off keeps the geometry at the cost of no longer matching the oracle;
measured against a video capture of the original the difference is small and
goes both ways.

**Arithmetic is ported, not approximated.** The original is x87 code compiled
with fast-math, so the C is not the specification — several routines are ported
from the assembly the shipped build actually ran. The JPEG decoder is held
byte-exact against the original's own output rather than swapped for the
browser's.

**The music is the clock.** The soundtrack is the mp3 carried inside
`mind.dat`, and the renderer follows the audio clock rather than counting
frames, so drift shows up as a picture that lags the music instead of
accumulating silently.

## Credits

Original: **Bomb** — code Skal, music Senser, graphics Made and Krafton, 1998.

`LICENSE` covers the reconstruction source in `js/` only. The original
executable, artwork and music remain the property of their authors.
