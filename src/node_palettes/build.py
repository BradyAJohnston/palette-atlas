"""Build the asset library from the fitted ramps. Run inside Blender 5.3:

    bblender -b --factory-startup --python-exit-code 1 \
        -P src/node_palettes/build.py -- [collection ...]

A node group's name must be unique within a .blend, and an asset is named after
its node group. So that assets keep plain palette names, each editor and
collection gets its own file, library/<editor>/<collection>.blend.

This is a plain script, as Blender has neither matplotlib nor scipy to import
the rest of the package with.
"""

import json
import sys
import uuid
from pathlib import Path

import bpy
import numpy as np
from nodebpy import TreeBuilder, compositor, geometry, shader

RAMPS_DIR = Path(__file__).parent / "data" / "ramps"
LIBRARY_DIR = Path(__file__).parents[2] / "library"
LIBRARY_NAME = "Scientific Palettes"
CONTACT = {"name": "Brady Johnston", "email": "brady.johnston@me.com"}
PREVIEW_SIZE = 128

EDITORS = {
    "Geometry Nodes": ("geometry", TreeBuilder.geometry, geometry),
    "Shader Nodes": ("shader", TreeBuilder.shader, shader),
    "Compositor Nodes": ("compositor", TreeBuilder.compositor, compositor),
}


def catalog_path(editor: str, collection: str) -> str:
    return f"{LIBRARY_NAME}/{editor}/{collection}"


def catalog_id(path: str) -> str:
    """Stable across builds, so an asset keeps its catalog."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


def build_group(ramp: dict, make_tree, nodes) -> bpy.types.NodeTree:
    """A node group mapping Value to Color through the palette's ramps.

    Each segment after the first takes over from the one before once Value
    reaches its start, Mix Color standing in for the Switch shaders don't have.
    """
    with make_tree(ramp["name"]) as tree:
        value = tree.inputs.float(
            "Value",
            0.5,
            description="Position along the palette, 0 to 1",
            min_value=0.0,
            max_value=1.0,
            subtype="FACTOR",
        )
        reverse = tree.inputs.boolean(
            "Reverse", False, description="Run the palette from 1 to 0"
        )
        color = tree.outputs.color("Color", description="Colour at Value")

        position = nodes.Mix.float(reverse, value, 1.0 - value)
        result = None
        for segment in ramp["segments"]:
            start, end = segment["start"], segment["end"]
            local = position if end - start == 1 else (position - start) / (end - start)
            ramp_node = nodes.ColorRamp(
                local,
                items=[(p, (r, g, b, 1.0)) for p, r, g, b in segment["stops"]],
                color_interpolation=segment["interpolation"],
            )
            if result is None:
                result = ramp_node
            else:
                result = nodes.Mix.color(position < start, ramp_node, result)
        result >> color
    return bpy.data.node_groups[ramp["name"]]


def to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def set_preview(group: bpy.types.NodeTree, ramp: dict) -> None:
    """A left to right gradient, sampled from the group's own Color Ramps."""
    color_ramps = [n.color_ramp for n in group.nodes if n.bl_idname == "ShaderNodeValToRGB"]
    x = (np.arange(PREVIEW_SIZE) + 0.5) / PREVIEW_SIZE
    row = np.ones((PREVIEW_SIZE, 4))
    for color_ramp, segment in zip(color_ramps, ramp["segments"], strict=True):
        start, end = segment["start"], segment["end"]
        for i in np.nonzero(x >= start)[0]:
            row[i, :3] = color_ramp.evaluate((x[i] - start) / (end - start))[:3]
    row[:, :3] = to_srgb(row[:, :3])

    preview = group.preview_ensure()
    preview.image_size = (PREVIEW_SIZE, PREVIEW_SIZE)
    pixels = np.tile(row, (PREVIEW_SIZE, 1)).astype(np.float32)
    preview.image_pixels_float.foreach_set(pixels.ravel())


def mark_asset(group, ramp: dict, editor: str, collection: str) -> None:
    group.asset_mark()
    data = group.asset_data
    data.catalog_id = catalog_id(catalog_path(editor, collection))
    data.description = f"{ramp['name']} palette, from {ramp['source']}"
    for tag in (collection, ramp["group"]):
        if tag:
            data.tags.new(tag, skip_if_exists=True)
    set_preview(group, ramp)


def build_file(collection: str, ramps: list[dict], editor: str) -> None:
    directory, make_tree, nodes = EDITORS[editor]
    bpy.ops.wm.read_homefile(use_factory_startup=True, use_empty=True, load_ui=False)
    for ramp in ramps:
        mark_asset(build_group(ramp, make_tree, nodes), ramp, editor, collection)
    path = LIBRARY_DIR / directory / f"{collection}.blend"
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)


def write_catalogs(collections: list[str]) -> None:
    lines = ["VERSION 1"]
    for editor in EDITORS:
        for collection in collections:
            path = catalog_path(editor, collection)
            lines.append(f"{catalog_id(path)}:{path}:{path.replace('/', '-')}")
    (LIBRARY_DIR / "blender_assets.cats.txt").write_text("\n".join(lines) + "\n")


def write_library_meta() -> None:
    """Filled in by `blender -c asset_listing generate`, which keeps these fields."""
    meta = {"api_versions": {}, "name": LIBRARY_NAME, "contact": CONTACT}
    path = LIBRARY_DIR / "_asset-library-meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")


def main(collections: list[str]) -> None:
    every = sorted(p.stem for p in RAMPS_DIR.glob("*.json"))
    for collection in collections or every:
        ramps = json.loads((RAMPS_DIR / f"{collection}.json").read_text())
        for editor in EDITORS:
            build_file(collection, ramps, editor)
            print(f"built {editor} {collection}: {len(ramps)} assets", flush=True)
    LIBRARY_DIR.mkdir(exist_ok=True)
    write_catalogs(every)
    write_library_meta()


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
