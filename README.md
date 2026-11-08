# Palette Atlas

Scientific colour palettes as node assets for Blender 5.3, built entirely from code
into an online asset library. Each palette is one node group per editor (Geometry,
Shader and Compositor), returning a `Color`:

- **Smooth palettes**, such as `viridis`, take a `Value` from 0 to 1, and `Reverse`.
- **Discrete palettes**, such as `tab10` and every MetBrewer and LTC palette, also
  take an `Index` and a `Mode`: `Stepped` bands along `Value` (the default), a
  `Smooth` blend of the colours, or the colour at `Index`, wrapping around.

Assets are catalogued as `Palette Atlas/<editor>/<collection>/<category>`,
using each source's own categories (matplotlib's gallery sections, Crameri's
classes, CarbonPlan's types, Panoply's table groups for NASA, and cmocean's
descriptions). MetBrewer, LTC and Tableau have none and stay at the collection;
MetBrewer's colourblind-friendly palettes are tagged `Colorblind-Friendly`.

## Building

1. **Fit** palettes to Color Ramps. The result is cached in
   `src/palette_atlas/data/ramps/`, and only palettes that changed are refitted.

   ```sh
   uv run palette-atlas fit [collection ...]
   ```

2. **Build** the `.blend` files, catalogs and thumbnails into `library/`. This runs
   inside Blender and needs `nodebpy` available to Blender's Python.

   ```sh
   blender -b --factory-startup --python-exit-code 1 \
       -P src/palette_atlas/build.py -- [collection ...]
   ```

3. **Index** the library.

   ```sh
   blender -c asset_listing generate library
   ```

4. **Verify** the built groups in every editor against the source palettes. This
   runs Blender itself, from `--blender` or `$BLENDER`.

   ```sh
   uv run palette-atlas verify [collection ...] --blender /path/to/blender
   ```

## Palettes

Ramps hold sRGB colours and blend them in sRGB, as the palettes' own packages do,
and each group converts its colour to linear at the end, as Blender expects. This
means the colour swatches on a ramp look washed out if you open a group.

Where matplotlib defines a palette exactly, the group follows that definition
rather than samples of it:

- **Formulas** (`gnuplot`, `rainbow`, `cubehelix`, `flag`, `prism`, `afmhot`, `ocean`,
  `gnuplot2`, `gist_heat`) are built from math nodes.
- **Colour lists and segments** (the ColorBrewer maps, `jet`, `hsv` and others) use
  matplotlib's own stops, where they fit in one ramp.

Every other palette is fitted. A Color Ramp holds at most 32 stops. Longer palettes
are fitted to fewer stops, and any that one ramp can't hold within 2/255 are split
across several ramps, chosen by `Value`, until they are within 1/255. Blender bakes
ramps to 256 steps in the Shader and Compositor editors, so there a band edge can
move by up to 1/256 of the range, and the sharpest turns in a palette are slightly
rounded off.

Vendored tables are described in `src/palette_atlas/data/rgb/README.md`.
