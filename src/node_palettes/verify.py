"""Check the built library against the source palettes, in every editor.

Blender evaluates each node group (verify_blender.py) and this compares its
colours with the palette's own. Geometry Nodes evaluates a Color Ramp exactly,
so it must match the fit. Shader and Compositor read a ramp from a 256 entry
table, so a colour may come from up to 1/255 along the palette instead, and
each sample is allowed however much the palette changes over that distance.

matplotlib palettes built from their own stops or formulas are also checked
between its samples, against matplotlib's lookup table made fine enough to hold
every point checked.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import _create_lookup_table

from .fitting import Ramp, evaluate_ramp, read_ramps
from .palettes import Palette, load_palettes

SCRIPT = Path(__file__).with_name("verify_blender.py")
EDITORS = ("geometry", "shader", "compositor")
MODES = ("Stepped", "Smooth", "Index")
# Value (c + 0.3) / 400 never lands on a band edge, and is k / 4000 for whole k
DISCRETE_COLUMNS = 400
DEFINITION_TABLE = 4001
EXACT = 0.05  # in 1/255, for what is exact up to float precision
TABLE_STEP = 1 / 255
BAKED = 0.5  # in 1/255, rounding through the table, beyond the shift


@dataclass
class Case:
    palette: Palette
    ramp: Ramp
    mode: str  # "" for a smooth palette
    reverse: bool
    count: int
    offset: float
    scale: float
    between: bool = False  # between matplotlib's samples, against its definition

    def to_json(self) -> dict:
        return {
            "collection": self.palette.collection,
            "name": self.palette.name,
            "mode": self.mode,
            "reverse": self.reverse,
            "count": self.count,
            "offset": self.offset,
            "scale": self.scale,
        }

    def position(self) -> np.ndarray:
        x = (np.arange(self.count) + self.offset) / self.scale
        return 1 - x if self.reverse else x

    def colours_at(self, pos: np.ndarray) -> np.ndarray:
        """sRGB the group should give at palette positions pos."""
        srgb = self.palette.srgb
        n = len(srgb)
        if not self.palette.discrete:
            return evaluate_ramp(self.ramp, np.clip(pos, 0, 1))
        if self.mode == "Stepped":
            return srgb[np.clip(np.floor(pos * n), 0, n - 1).astype(int)]
        t = np.clip(pos, 0, 1) * (n - 1)
        i = np.clip(np.floor(t), 0, n - 2).astype(int)
        f = (t - i)[:, None]
        return srgb[i] * (1 - f) + srgb[i + 1] * f  # blended in sRGB

    def expected(self) -> np.ndarray:
        srgb = self.palette.srgb
        if self.mode == "Index":
            k = np.arange(self.count) % len(srgb)
            return srgb[len(srgb) - 1 - k if self.reverse else k]
        if self.between:
            return definition(self.palette.name, self.position())
        if not self.palette.discrete:
            # smooth palettes are read at their own samples, against the source
            return srgb[::-1] if self.reverse else srgb
        return self.colours_at(self.position())

    def shift(self, editor: str) -> np.ndarray:
        """Per sample, how much the colour changes within a table step, in 1/255."""
        if editor == "geometry" or self.mode == "Index":
            return np.zeros(self.count)
        pos = self.position()
        here = self.colours_at(pos)
        moved = [
            np.abs(self.colours_at(pos + d) - here).max(1) for d in (-TABLE_STEP, TABLE_STEP)
        ]
        return np.max(moved, axis=0) * 255

    def tolerance(self, editor: str, shift: np.ndarray) -> np.ndarray:
        """Per sample, in 1/255."""
        base = EXACT + (0.0 if is_exact(self.palette) else self.ramp.error)
        return base + (0.0 if editor == "geometry" else BAKED) + shift


def is_exact(palette: Palette) -> bool:
    """Built from the palette's own colours, stops or formula, without a fit."""
    return palette.discrete or bool(palette.formula) or palette.stops is not None


def definition(name: str, x: np.ndarray) -> np.ndarray:
    """matplotlib's own colours at x, each a whole number of 1/4000ths."""
    data = matplotlib.colormaps[name]._segmentdata
    k = np.round(x * (DEFINITION_TABLE - 1)).astype(int)
    assert np.allclose(k / (DEFINITION_TABLE - 1), x), "x is between table entries"
    channels = [_create_lookup_table(DEFINITION_TABLE, data[c]) for c in ("red", "green", "blue")]
    return np.stack([c[k] for c in channels], 1)


def cases_for(palette: Palette, ramp: Ramp) -> list[Case]:
    if not palette.discrete:
        m = len(palette.srgb)
        cases = [Case(palette, ramp, "", r, m, 0.0, m - 1) for r in (False, True)]
        if palette.collection == "matplotlib" and is_exact(palette):
            cases += [
                Case(palette, ramp, "", r, DISCRETE_COLUMNS, 0.3, DISCRETE_COLUMNS, True)
                for r in (False, True)
            ]
        return cases
    return [
        Case(palette, ramp, mode, r, DISCRETE_COLUMNS, 0.3, DISCRETE_COLUMNS)
        for mode in MODES
        for r in (False, True)
    ]


def run_blender(blender: str, cases: list[Case]) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        cases_path, out_path = Path(tmp) / "cases.json", Path(tmp) / "results.npz"
        cases_path.write_text(json.dumps([c.to_json() for c in cases]))
        subprocess.run(
            [blender, "-b", "--factory-startup", "--python-exit-code", "1",
             "-P", str(SCRIPT), "--", str(cases_path), str(out_path)],
            check=True,
        )
        with np.load(out_path) as results:
            return dict(results)


def verify(collections: list[str] | None, blender: str | None) -> bool:
    """Print how far each editor strays from the palettes, True when all pass."""
    blender = blender or os.environ.get("BLENDER", "blender")
    palettes = load_palettes(collections)
    ramps = {c: read_ramps(c) for c in {p.collection for p in palettes}}
    cases = [c for p in palettes for c in cases_for(p, ramps[p.collection][p.name])]
    results = run_blender(blender, cases)

    # worst error, overall and where a table step changes the colour by < 1/255
    worst: dict[tuple[str, str], list[float]] = {}
    failures = []
    for editor in EDITORS:
        for i, case in enumerate(cases):
            error = np.abs(results[f"{editor}/{i}"] - case.expected()).max(1) * 255
            shift = case.shift(editor)
            steady = error[shift < 1.0]
            cell = worst.setdefault((editor, case.palette.collection), [0.0, 0.0])
            cell[0] = max(cell[0], float(error.max()))
            cell[1] = max(cell[1], float(steady.max()) if len(steady) else 0.0)
            bad = np.nonzero(error > case.tolerance(editor, shift))[0]
            if len(bad):
                failures.append((editor, case, len(bad), float(error[bad].max())))

    print(
        f"\nworst error against the source palette, in 1/255 sRGB ({len(cases)} cases):"
        "\naway from band edges and sharp turns / anywhere"
    )
    print(f"  {'':12}" + "".join(f"{e:>16}" for e in EDITORS))
    for collection in sorted({c for _, c in worst}):
        row = "".join(
            f"{worst[(e, collection)][1]:9.2f} /{worst[(e, collection)][0]:6.1f}" for e in EDITORS
        )
        print(f"  {collection:12}{row}")
    for editor, case, n, error in failures:
        what = f"{case.mode or 'Value'}{' between samples' if case.between else ''}"
        what += " reversed" if case.reverse else ""
        print(
            f"FAIL {editor} {case.palette.collection}/{case.palette.name} {what}: "
            f"{n} of {case.count} samples, up to {error:.2f}/255"
        )
    print("all within tolerance" if not failures else f"{len(failures)} failing cases")
    return not failures
