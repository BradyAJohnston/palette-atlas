# Scientific Palettes

Scientific colour palettes as node assets for Blender 5.3, built entirely from code
into an online asset library. Each palette is one node group per editor (Geometry,
Shader and Compositor), taking a `Value` from 0 to 1 and returning a `Color`.

## Building

1. **Fit** palettes to Color Ramps. The result is cached in
   `src/node_palettes/data/ramps/`, and only palettes that changed are refitted.

   ```sh
   uv run node-palettes fit [collection ...]
   ```

2. **Build** the `.blend` files, catalogs and thumbnails into `library/`. This runs
   inside Blender and needs `nodebpy` available to Blender's Python.

   ```sh
   blender -b --factory-startup --python-exit-code 1 \
       -P src/node_palettes/build.py -- [collection ...]
   ```

3. **Index** the library.

   ```sh
   blender -c asset_listing generate library
   ```

## Palettes

A Color Ramp holds at most 32 stops. Longer palettes are fitted to fewer stops, and
any that one ramp can't hold within 2/255 are split across several ramps, chosen by
`Value`, until they are within 1/255. Blender bakes ramps to 256 steps in the Shader
and Compositor editors, which limits the very sharpest palettes there.

Vendored tables are described in `src/node_palettes/data/rgb/README.md`.
