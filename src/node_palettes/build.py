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
import math
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
MAX_STOPS = 32  # a Color Ramp's limit, as in palettes.py
EDGE_NUDGE = 1e-6

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


def is_discrete(ramp: dict) -> bool:
    return all(s["interpolation"] == "CONSTANT" for s in ramp["segments"])


def smooth_segments(colors: list) -> list[dict]:
    """Linear ramps through a discrete palette's colours, spread evenly from 0
    to 1, neighbouring segments sharing their boundary colour."""
    n = len(colors)
    parts = math.ceil((n - 1) / (MAX_STOPS - 1))
    bounds = np.linspace(0, n - 1, parts + 1).round().astype(int)
    return [
        {
            "start": a / (n - 1),
            "end": b / (n - 1),
            "interpolation": "LINEAR",
            "stops": [((k - a) / (b - a), *colors[k]) for k in range(a, b + 1)],
        }
        for a, b in zip(bounds, bounds[1:])
    ]


def lookup(t, segments: list[dict], nodes):
    """(colour at t, [(segment, ColorRamp node)]) through one ramp per segment.

    Each segment after the first takes over from the one before once t reaches
    its start.
    """
    result, ramps = None, []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        local = t if end - start == 1 else (t - start) / (end - start)
        constant = segment["interpolation"] == "CONSTANT"
        ramp_node = nodes.ColorRamp(
            local,
            items=[
                # a value exactly on a band edge belongs to the band above, as
                # in matplotlib, which float rounding would otherwise decide
                (max(p - EDGE_NUDGE, 0.0) if constant else p, (r, g, b, 1.0))
                for p, r, g, b in segment["stops"]
            ],
            color_interpolation=segment["interpolation"],
        )
        ramps.append((segment, ramp_node.node))
        result = ramp_node if result is None else nodes.Mix.color(t < start, ramp_node, result)
    return result, ramps


def value_input(tree):
    return tree.inputs.float(
        "Value",
        0.5,
        description="Position along the palette, 0 to 1",
        min_value=0.0,
        max_value=1.0,
        subtype="FACTOR",
    )


def reverse_input(tree):
    return tree.inputs.boolean("Reverse", False, description="Run the palette from 1 to 0")


def build_group(ramp: dict, make_tree, nodes):
    """(node group, the ramps showing its default colours) for the palette.

    A smooth palette maps Value to its fitted ramps. A discrete one has a Mode:
    Stepped bands, a Smooth blend of its colours, or the colour at Index.
    """
    discrete = is_discrete(ramp)
    with make_tree(ramp["name"]) as tree:
        value = value_input(tree)
        if discrete:
            index = tree.inputs.integer(
                "Index", 0, description="Colour to use in Index mode, wrapping around"
            )
        reverse = reverse_input(tree)
        if discrete:
            mode = tree.inputs.menu(
                "Mode", "Stepped", description="Stepped bands, a smooth blend, or by Index"
            )
        color = tree.outputs.color("Color", description="Colour at Value")

        with nodes.Frame("Position"):
            position = nodes.Mix.float(reverse, value, 1.0 - value)
            if ramp.get("cycles", 1.0) != 1.0:
                # a repeating palette holds one repeat, which Value wraps into
                position = nodes.Math.fraction(position * ramp["cycles"])
        if not discrete:
            with nodes.Frame("Palette"):
                result, ramps = lookup(position, ramp["segments"], nodes)
            result >> color
        else:
            colors = [stop[1:] for s in ramp["segments"] for stop in s["stops"]]
            n = len(colors)
            with nodes.Frame("Mode"):
                k = nodes.Math.floored_modulo(index, n)
                k = nodes.Mix.float(reverse, k, (n - 1) - k)
                # one Menu Switch, so the group gets a single Mode menu: it gives
                # (where to read the stepped ramps, whether to use the smooth ones)
                chosen = nodes.SeparateXYZ(
                    nodes.MenuSwitch.vector(
                        mode,
                        {
                            "Stepped": nodes.CombineXYZ(position, 0.0, 0.0),
                            "Smooth": nodes.CombineXYZ(position, 1.0, 0.0),
                            "Index": nodes.CombineXYZ((k + 0.5) / n, 0.0, 0.0),
                        },
                    )
                )
            with nodes.Frame("Stepped"):
                stepped, ramps = lookup(chosen.o.x, ramp["segments"], nodes)
            with nodes.Frame("Smooth"):
                smooth, _ = lookup(position, smooth_segments(colors), nodes)
            nodes.Mix.color(chosen.o.y, stepped, smooth) >> color
    return bpy.data.node_groups[ramp["name"]], ramps


def to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def set_preview(group: bpy.types.NodeTree, ramps: list, cycles: float) -> None:
    """A left to right gradient, sampled from the group's own Color Ramps."""
    x = np.mod((np.arange(PREVIEW_SIZE) + 0.5) / PREVIEW_SIZE * cycles, 1.0)
    row = np.ones((PREVIEW_SIZE, 4))
    for segment, node in ramps:
        start, end = segment["start"], segment["end"]
        for i in np.nonzero(x >= start)[0]:
            row[i, :3] = node.color_ramp.evaluate((x[i] - start) / (end - start))[:3]
    row[:, :3] = to_srgb(row[:, :3])

    preview = group.preview_ensure()
    preview.image_size = (PREVIEW_SIZE, PREVIEW_SIZE)
    pixels = np.tile(row, (PREVIEW_SIZE, 1)).astype(np.float32)
    preview.image_pixels_float.foreach_set(pixels.ravel())


def mark_asset(group, ramps: list, ramp: dict, editor: str, collection: str) -> None:
    group.color_tag = "COLOR"
    group.asset_mark()
    data = group.asset_data
    data.catalog_id = catalog_id(catalog_path(editor, collection))
    data.description = f"{ramp['name']} palette, from {ramp['source']}"
    for tag in (collection, ramp["group"]):
        if tag:
            data.tags.new(tag, skip_if_exists=True)
    set_preview(group, ramps, ramp.get("cycles", 1.0))


def build_file(collection: str, ramps: list[dict], editor: str) -> None:
    directory, make_tree, nodes = EDITORS[editor]
    bpy.ops.wm.read_homefile(use_factory_startup=True, use_empty=True, load_ui=False)
    for ramp in ramps:
        group, ramp_nodes = build_group(ramp, make_tree, nodes)
        mark_asset(group, ramp_nodes, ramp, editor, collection)
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
