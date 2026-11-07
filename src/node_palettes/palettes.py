"""Load every palette as sRGB colours, from upstream packages or vendored tables."""

import colorsys
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cmcrameri.cm
import cmocean
import matplotlib
import met_brewer
import numpy as np
from matplotlib import _cm
from matplotlib.colors import Colormap, ListedColormap, to_rgb

RGB_DIR = Path(__file__).parent / "data" / "rgb"
MAX_STOPS = 32  # Blender's Color Ramp limit, extra stops are silently dropped
SAMPLES = 256

BLENDER_COLORMAPS = "Blender-Colormaps"

# matplotlib's own grouping, from its colormap reference gallery
# sine waves in matplotlib/_cm.py, sin(x * 31.5 pi) and sin(x * 20.9 pi): one ramp
# holds a single repeat, which the node group wraps Value into
MATPLOTLIB_CYCLES = {"flag": 31.5 / 2, "prism": 20.9 / 2}
PERIOD_SAMPLES = 1024  # finely enough to catch where the waves are clipped

MATPLOTLIB_GROUPS = {
    "Uniform": ["viridis", "plasma", "inferno", "magma", "cividis"],
    "Sequential": [
        "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr",
        "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu",
        "PuBuGn", "BuGn", "YlGn",
    ],
    "Sequential 2": [
        "binary", "gray", "bone", "pink", "spring", "summer", "autumn",
        "winter", "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper",
    ],
    "Diverging": [
        "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn",
        "Spectral", "coolwarm", "bwr", "seismic", "berlin", "managua", "vanimo",
    ],
    "Cyclic": ["twilight", "twilight_shifted", "hsv"],
    "Qualitative": [
        "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2",
        "Set3", "tab10", "tab20", "tab20b", "tab20c", "okabe_ito",
    ],
    "Miscellaneous": [
        "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot", "gnuplot2",
        "CMRmap", "cubehelix", "brg", "gist_rainbow", "rainbow", "jet",
        "turbo", "nipy_spectral", "gist_ncar", "flag", "prism",
    ],
}  # fmt: skip

CRAMERI_DIVERGING = {
    "bam", "berlin", "broc", "cork", "lisbon", "managua", "roma", "tofino",
    "vanimo", "vik",
}  # fmt: skip
CRAMERI_MULTI_SEQUENTIAL = {"bukavu", "fes", "oleron"}

# GMT_cyclic stores hue, saturation, value rows rather than red, green, blue
HSV_TABLES = {"GMT_cyclic"}

# NASA's tables came from Adobe .act files: 256 colours padded with black, then
# a footer that Blender-Colormaps kept as if it were a final colour
ACT_ROWS = 257


@dataclass(frozen=True, eq=False)
class Palette:
    """One palette as sRGB colours in 0..1, evenly spaced along the map."""

    name: str
    collection: str
    group: str
    source: str
    srgb: np.ndarray
    discrete: bool = False  # hard steps between colours, rather than a gradient
    # a palette that repeats: how many times, and sRGB samples of one repeat
    cycles: float = 1.0
    period: np.ndarray | None = None

    @property
    def digest(self) -> str:
        """Identifies the colours, so a fit is redone only when they change."""
        arrays = [self.srgb] if self.period is None else [self.srgb, self.period]
        digest = hashlib.sha256()
        for a in arrays:
            rounded = np.round(a, 6) + 0.0  # + 0.0 folds -0.0 into 0.0
            digest.update(rounded.astype("<f8").tobytes())
        if self.period is not None:
            digest.update(repr(self.cycles).encode())
        return digest.hexdigest()[:16]


def _sampled(cmap: Colormap) -> np.ndarray:
    return cmap(np.linspace(0.0, 1.0, SAMPLES))[:, :3]


def _period(name: str, cycles: float) -> np.ndarray:
    """One repeat of a matplotlib palette defined by functions of x, as sRGB."""
    functions = _cm.datad[name]
    x = np.linspace(0.0, 1.0, PERIOD_SAMPLES) / cycles
    return np.clip(np.stack([functions[c](x) for c in ("red", "green", "blue")], 1), 0, 1)


def _matplotlib_palette(name: str, group: str) -> Palette:
    cmap = matplotlib.colormaps[name]
    discrete = isinstance(cmap, ListedColormap) and cmap.N <= MAX_STOPS
    cycles = MATPLOTLIB_CYCLES.get(name, 1.0)
    return Palette(
        name=name,
        collection="matplotlib",
        group=group,
        source=f"matplotlib {matplotlib.__version__}",
        srgb=np.array([to_rgb(c) for c in cmap.colors]) if discrete else _sampled(cmap),
        discrete=discrete,
        cycles=cycles,
        period=_period(name, cycles) if name in MATPLOTLIB_CYCLES else None,
    )


def _check_matplotlib_coverage() -> None:
    """Every map is grouped, or an alias of one that is."""
    listed = {name for names in MATPLOTLIB_GROUPS.values() for name in names}
    for name in sorted(matplotlib.colormaps):
        # cmocean and cmcrameri register their maps here as cmo.* and cmc.*
        if name.endswith("_r") or "." in name or name in listed:
            continue
        samples = _sampled(matplotlib.colormaps[name])
        if not any(
            np.array_equal(samples, _sampled(matplotlib.colormaps[other]))
            for other in listed
        ):
            raise ValueError(f"{name!r} is not in MATPLOTLIB_GROUPS, add it")


def load_matplotlib() -> Iterator[Palette]:
    _check_matplotlib_coverage()
    for group, names in MATPLOTLIB_GROUPS.items():
        for name in names:
            yield _matplotlib_palette(name, group)


def load_cmocean() -> Iterator[Palette]:
    source = f"cmocean {version('cmocean')}"
    for name in cmocean.cm.cmapnames:
        yield Palette(
            name, "cmocean", "", source, _sampled(cmocean.cm.cmap_d[name])
        )


def _crameri_group(name: str) -> str:
    if name.endswith("O"):
        return "Cyclic"
    if name in CRAMERI_DIVERGING:
        return "Diverging"
    if name in CRAMERI_MULTI_SEQUENTIAL:
        return "Multi-sequential"
    return "Sequential"


def load_scientific() -> Iterator[Palette]:
    source = f"cmcrameri {version('cmcrameri')}"
    for name, cmap in cmcrameri.cm.cmaps.items():
        if name.endswith("_r"):
            continue
        if name.endswith("S"):  # categorical sets of 100 colours
            srgb = np.array([to_rgb(c) for c in cmap.colors])
            yield Palette(name, "scientific", "Categorical", source, srgb, True)
            continue
        yield Palette(name, "scientific", _crameri_group(name), source, _sampled(cmap))
    yield from _load_tables("scientific", _crameri_group)  # missing from cmcrameri


def load_metbrewer() -> Iterator[Palette]:
    source = f"met-brewer {version('met-brewer')}"
    for name, palette in met_brewer.MET_PALETTES.items():
        srgb = np.array([to_rgb(c) for c in palette["colors"]])
        # hand-picked colour sets: discrete by default in MetBrewer itself
        yield Palette(name, "metbrewer", "", source, srgb, discrete=True)


def read_rgb_table(path: Path) -> np.ndarray:
    """Rows of a Blender-Colormaps .rgb file, as sRGB in 0..1."""
    rows = [
        [float(value) for value in line.split()[:3]]
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(("#", "ncolor"))
    ]
    if len(rows) == ACT_ROWS:
        # the footer's first two bytes are the big-endian count of used colours
        high, low, _ = rows[-1]
        rows = rows[: int(high) * 256 + int(low)]
    if path.stem in HSV_TABLES:
        return np.array([colorsys.hsv_to_rgb(h / 360, s, v) for h, s, v in rows])
    return np.array(rows) / 255


def _load_tables(
    collection: str, group_of=lambda name: "", discrete: bool = False
) -> Iterator[Palette]:
    for path in sorted((RGB_DIR / collection).glob("*.rgb")):
        yield Palette(
            path.stem,
            collection,
            group_of(path.stem),
            BLENDER_COLORMAPS,
            read_rgb_table(path),
            discrete,
        )


def _carbonplan_mode(name: str) -> str:
    """Light or Dark; a map that is the same in both has no suffix."""
    mode = name.rpartition("_")[2]
    return mode.capitalize() if mode in ("light", "dark") else ""


def load_carbonplan() -> Iterator[Palette]:
    return _load_tables("carbonplan", _carbonplan_mode)


def load_nasa() -> Iterator[Palette]:
    return _load_tables("nasa", lambda name: name.partition("_")[0])


def load_ltc() -> Iterator[Palette]:
    return _load_tables("ltc", discrete=True)


def load_tableau() -> Iterator[Palette]:
    return _load_tables("tableau", discrete=True)


LOADERS = {
    "matplotlib": load_matplotlib,
    "cmocean": load_cmocean,
    "scientific": load_scientific,
    "metbrewer": load_metbrewer,
    "carbonplan": load_carbonplan,
    "nasa": load_nasa,
    "ltc": load_ltc,
    "tableau": load_tableau,
}


def load_palettes(collections: list[str] | None = None) -> list[Palette]:
    """Every palette, or only those in the named collections."""
    return [
        palette
        for collection in collections or LOADERS
        for palette in LOADERS[collection]()
    ]
