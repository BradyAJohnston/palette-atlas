"""Build the asset library from the fitted ramps, through the bpy module:

    uv run palette-atlas build [collection ...]

A node group's name must be unique within a .blend, and an asset is named after
its node group. So that assets keep plain palette names, each editor and
collection gets its own file, library/<editor>/<collection>.blend.

Ramps hold sRGB colours, blended in sRGB as the palettes' packages blend them,
and each group converts its colour to linear at the end, as Blender expects.
matplotlib's formula palettes are built from math nodes instead of ramps.
"""

import html
import json
import math
import uuid
from pathlib import Path

import bpy
import numpy as np
from _bpy_internal.assets.remote_library import asset_listing_main
from nodebpy import TreeBuilder, compositor, geometry, shader

from . import formulas

RAMPS_DIR = Path(__file__).parent / "data" / "ramps"
LIBRARY_DIR = Path(__file__).parents[2] / "library"
LIBRARY_NAME = "Palette Atlas"
REPOSITORY = "https://github.com/BradyAJohnston/palette-atlas"
REMOTE_URL = "https://bradyajohnston.github.io/palette-atlas/"  # as in the manifest
CONTACT = {"name": "Brady Johnston", "email": "brady.johnston@me.com"}
LICENSE = "CC0-1.0"  # SPDX identifier, set on every asset
PREVIEW_SIZE = 128
MAX_STOPS = 32  # a Color Ramp's limit, as in palettes.py
EDGE_NUDGE = 1e-6

EDITORS = {
    "Geometry Nodes": ("geometry", TreeBuilder.geometry, geometry),
    "Shader Nodes": ("shader", TreeBuilder.shader, shader),
    "Compositor Nodes": ("compositor", TreeBuilder.compositor, compositor),
}


def catalog_path(editor: str, collection: str, category: str = "") -> str:
    path = f"{LIBRARY_NAME}/{editor}/{collection}"
    return f"{path}/{category}" if category else path


def catalog_id(path: str) -> str:
    """Stable across builds, so an asset keeps its catalog."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


def is_discrete(ramp: dict) -> bool:
    segments = ramp["segments"]  # none for a formula
    return bool(segments) and all(s["interpolation"] == "CONSTANT" for s in segments)


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


class NodeOps:
    """The operations formulas.py uses, as math nodes."""

    def __init__(self, nodes):
        math_ = nodes.Math
        self.sqrt, self.sin, self.cos = math_.square_root, math_.sine, math_.cosine
        self.abs, self.min, self.max = math_.absolute, math_.minimum, math_.maximum
        self.pow = math_.power


def formula_channels(name: str, position, nodes) -> list:
    """sRGB channels of a formula palette, clipped to 0..1 as matplotlib does."""
    channels = formulas.FORMULAS[name](NodeOps(nodes), position)
    return [nodes.Math.maximum(c, 0.0, use_clamp=True) for c in channels]


def srgb_to_linear(channels: list, nodes):
    """The sRGB colour as Blender's linear colour, per channel."""
    out = []
    for c in channels:
        low = c / 12.92
        high = nodes.Math.power(nodes.Math.multiply_add(c, 1 / 1.055, 0.055 / 1.055), 2.4)
        out.append(nodes.Mix.float(nodes.Math.greater_than(c, 0.04045), low, high))
    return nodes.CombineXYZ(*out)


def channels_of(color, nodes) -> list:
    separate = nodes.SeparateXYZ(color)
    return [separate.o.x, separate.o.y, separate.o.z]


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

    A smooth palette maps Value through its ramps, or its formula. A discrete one
    has a Mode: Stepped bands, a Smooth blend of its colours, or the colour at
    Index. Either way the sRGB result is converted to linear at the end.
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
                "Mode", ramp.get("mode") or "Stepped", description="Stepped bands, a smooth blend, or by Index"
            )
        color = tree.outputs.color("Color", description="Colour at Value")

        with nodes.Frame("Position"):
            position = nodes.Mix.float(reverse, value, 1.0 - value)
        if ramp.get("formula"):
            with nodes.Frame("Formula"):
                channels = formula_channels(ramp["formula"], position, nodes)
            ramps = []
        elif not discrete:
            with nodes.Frame("Palette"):
                result, ramps = lookup(position, ramp["segments"], nodes)
                channels = channels_of(result, nodes)
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
            channels = channels_of(nodes.Mix.color(chosen.o.y, stepped, smooth), nodes)
        with nodes.Frame("sRGB to Linear"):
            srgb_to_linear(channels, nodes) >> color
    return bpy.data.node_groups[ramp["name"]], ramps


def set_preview(group: bpy.types.NodeTree, ramps: list, formula: str) -> None:
    """A left to right gradient in sRGB, sampled from the group's own Color Ramps,
    or its formula."""
    x = (np.arange(PREVIEW_SIZE) + 0.5) / PREVIEW_SIZE
    row = np.ones((PREVIEW_SIZE, 4))
    if formula:
        row[:, :3] = formulas.evaluate(formula, x)
    for segment, node in ramps:
        start, end = segment["start"], segment["end"]
        for i in np.nonzero(x >= start)[0]:
            row[i, :3] = node.color_ramp.evaluate((x[i] - start) / (end - start))[:3]

    preview = group.preview_ensure()
    preview.image_size = (PREVIEW_SIZE, PREVIEW_SIZE)
    pixels = np.tile(row, (PREVIEW_SIZE, 1)).astype(np.float32)
    preview.image_pixels_float.foreach_set(pixels.ravel())


def mark_asset(group, ramps: list, ramp: dict, editor: str, collection: str) -> None:
    group.color_tag = "COLOR"
    group.asset_mark()
    data = group.asset_data
    category = ramp.get("category", "")
    data.catalog_id = catalog_id(catalog_path(editor, collection, category))
    data.description = f"{ramp['name']} palette, from {ramp['source']}"
    data.license = LICENSE
    for tag in (collection, category, ramp["group"]):
        if tag:
            data.tags.new(tag, skip_if_exists=True)
    set_preview(group, ramps, ramp.get("formula", ""))


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
    # parents are listed too, or Blender makes them itself and shows them unsaved
    paths = [LIBRARY_NAME]
    for editor in EDITORS:
        paths.append(f"{LIBRARY_NAME}/{editor}")
        for collection in collections:
            ramps = json.loads((RAMPS_DIR / f"{collection}.json").read_text())
            categories = {r.get("category", "") for r in ramps}  # "" is the collection
            paths += [catalog_path(editor, collection, c) for c in sorted(categories | {""})]
    lines = ["VERSION 1"]
    lines += [f"{catalog_id(p)}:{p}:{p.replace('/', '-')}" for p in paths]
    (LIBRARY_DIR / "blender_assets.cats.txt").write_text("\n".join(lines) + "\n")


def write_library_meta() -> None:
    """Filled in by `blender -c asset_listing generate`, which keeps these fields."""
    meta = {"api_versions": {}, "name": LIBRARY_NAME, "contact": CONTACT}
    path = LIBRARY_DIR / "_asset-library-meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")


def write_index_page(collections: list[str]) -> None:
    """A landing page, as GitHub Pages lists no directories and 404s without it."""
    rows = []
    for collection in collections:
        count = len(json.loads((RAMPS_DIR / f"{collection}.json").read_text()))
        links = " · ".join(
            f'<a href="{d}/{collection}.blend">{e.split()[0]}</a>'
            for e, (d, _, _) in EDITORS.items()
        )
        rows.append(f"<tr><td>{html.escape(collection)}</td><td>{count}</td><td>{links}</td></tr>")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{LIBRARY_NAME}</title>
<style>
  :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
  body {{ max-width: 44rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  code {{ background: color-mix(in srgb, currentColor 10%, transparent); padding: 0.1em 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ text-align: left; padding: 0.3rem 0.6rem 0.3rem 0; }}
</style>
</head>
<body>
<h1>{LIBRARY_NAME}</h1>
<p>Scientific and data visualisation colour palettes as a remote asset library for
Blender 5.3+. Add this page's URL as a remote asset library in Blender:</p>
<p><code>{REMOTE_URL}</code></p>
<p>Source, instructions and licences:
<a href="{REPOSITORY}">{REPOSITORY}</a>. Assets are {LICENSE}.</p>
<table>
<tr><th>Collection</th><th>Palettes</th><th>Download</th></tr>
{chr(10).join(rows)}
</table>
<p><a href="_v1/asset-index.json">asset index</a> ·
<a href="blender_assets.cats.txt">catalogs</a></p>
</body>
</html>
"""
    (LIBRARY_DIR / "index.html").write_text(page)


def index_library() -> None:
    """Write the listing a remote asset library is served with."""
    if asset_listing_main(["generate", str(LIBRARY_DIR)]):
        raise SystemExit("asset_listing generate failed")
    for backup in LIBRARY_DIR.rglob("*.blend1"):
        backup.unlink()


def build(collections: list[str] | None) -> None:
    every = sorted(p.stem for p in RAMPS_DIR.glob("*.json"))
    for collection in collections or every:
        ramps = json.loads((RAMPS_DIR / f"{collection}.json").read_text())
        for editor in EDITORS:
            build_file(collection, ramps, editor)
            print(f"built {editor} {collection}: {len(ramps)} assets", flush=True)
    LIBRARY_DIR.mkdir(exist_ok=True)
    write_catalogs(every)
    write_library_meta()
    write_index_page(every)
    index_library()
