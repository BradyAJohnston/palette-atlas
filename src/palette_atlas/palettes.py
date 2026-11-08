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
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_rgb

from .formulas import FORMULAS

RGB_DIR = Path(__file__).parent / "data" / "rgb"
MAX_STOPS = 32  # Blender's Color Ramp limit, extra stops are silently dropped
SAMPLES = 256

BLENDER_COLORMAPS = "Blender-Colormaps"

# where a matplotlib palette jumps from one colour to another, the stop for the
# colour before sits this far ahead of the one after
JUMP = 1e-4

# matplotlib's own grouping, from its colormap reference gallery
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

# cmocean's docs call each map sequential, diverging or circular; topo, which
# joins sea depth to land height, is given none
CMOCEAN_CATEGORIES = {
    "Sequential": [
        "thermal", "haline", "solar", "ice", "gray", "oxy", "deep", "dense",
        "algae", "matter", "turbid", "speed", "amp", "tempo", "rain",
    ],
    "Diverging": ["balance", "delta", "curl", "diff", "tarn"],
    "Cyclic": ["phase"],
    "Topographic": ["topo"],
}  # fmt: skip

# the type field of @carbonplan/colormaps 4.0.0
CARBONPLAN_CATEGORIES = {
    "Single Hue": [
        "reds", "oranges", "yellows", "greens", "teals", "blues", "purples",
        "pinks", "greys",
    ],
    "Multi Hue": ["fire", "earth", "water", "heart", "wind", "warm", "cool"],
    "Diverging": [
        "pinkgreen", "redteal", "orangeblue", "yellowpurple", "redgrey",
        "orangegrey", "yellowgrey", "greengrey", "tealgrey", "bluegrey",
        "purplegrey", "pinkgrey",
    ],
    "Cyclical": ["rainbow", "sinebow"],
}  # fmt: skip

# Panoply's colour table page (giss.nasa.gov/tools/panoply/colorbars), which
# lists NEO_gebco_bathymetry and NEO_srtm_topography as both Sequential and
# Topographic; they go in Topographic
NASA_CATEGORIES = {
    "Sequential": [
        "EO_aura_omi_formal", "EO_carbon_density", "EO_lightning_lis_otd",
        "EO_sargassum_tamo", "GIST_heat", "GSFC_landsat_udf_density",
        "MRO_ice_freq", "NEO_amsre_sst", "NEO_aquarius_sss", "NEO_cdom",
        "NEO_ceres_insol", "NEO_ceres_lw", "NEO_ceres_sw", "NEO_ggmcf",
        "NEO_imperv_surf", "NEO_modis_aer_od", "NEO_modis_bs_albedo",
        "NEO_modis_chlor", "NEO_modis_cld_ci", "NEO_modis_cld_fr",
        "NEO_modis_cld_ot", "NEO_modis_cld_rd", "NEO_modis_cld_wp",
        "NEO_modis_lst", "NEO_modis_ndvi", "NEO_modis_sky_wv",
        "NEO_modis_sst_45", "NEO_mopitt_co", "NEO_omi_no2", "NEO_omi_ozone_to3",
        "NEO_omi_uvi", "NEO_pollution_conc_9", "NEO_sedac_pop",
        "NEO_snow_water", "NEO_soil_moisture", "NEO_tree_cover",
        "NEO_trmm_rainfall", "SVS_soilmoisture",
    ],
    "Diverging": [
        "EO_soil_moist_div", "EO_temp_anom_4", "EVL_wind_anom", "GMT_split",
        "NCDC_snow_anom", "NCDC_temp_anom_f", "NEO_albedo_change",
        "NEO_amsre_sst_anom", "NEO_aod_diff", "NEO_bright_temp",
        "NEO_carb_emit_anom", "NEO_ceres_net", "NEO_chlorophyll",
        "NEO_chlorophyll_oc", "NEO_div_vegetation_a", "NEO_div_vegetation_b",
        "NEO_div_vegetation_c", "NEO_evapstress", "NEO_giss_temp_anom",
        "NEO_grace_lwe_anom", "NEO_grav_anom", "NEO_meltseason_anom",
        "NEO_modis_lst_anom", "NEO_ns_airtemp", "NEO_pollution_mort",
        "NEO_rainfall_anom_9", "NEO_seasurf_hgt_anom", "NEO_soil_moisture_anom",
        "NEO_soil_moisture_anom_nl", "NEO_soil_moisture_pale",
        "NEO_wind_spd_anom", "NEO_wind_spd_anomaly_pwg", "NYT_drought",
        "SVS_tempanomaly", "UKM_hadcrut_10",
    ],
    "Topographic": [
        "GIST_earth", "GMT_globe", "GMT_relief", "NEO_gebco_bathymetry",
        "NEO_srtm_topography",
    ],
    "Rainbow": [
        "GISS_isccp_rainbow", "GISS_isccp_rainbow_20", "GMT_cyclic",
        "GMT_haxby", "GMT_seis", "GMT_seis_20",
    ],
}  # fmt: skip

COLORBLIND_TAG = "Colorblind-Friendly"

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
    group: str  # an extra tag, such as Light or Dark
    source: str
    srgb: np.ndarray
    discrete: bool = False  # hard steps between colours, rather than a gradient
    # how the package defines it, where that is exact in nodes: a formula in
    # formulas.py, or the sRGB stops (position, r, g, b) it blends between
    formula: str = ""
    stops: np.ndarray | None = None
    category: str = ""  # the source's own, as a catalog within the collection

    @property
    def digest(self) -> str:
        """Identifies the colours, so a fit is redone only when they change."""
        digest = hashlib.sha256()
        for a in (self.srgb, self.stops):
            if a is not None:
                rounded = np.round(a, 6) + 0.0  # + 0.0 folds -0.0 into 0.0
                digest.update(rounded.astype("<f8").tobytes())
        digest.update(self.formula.encode())
        return digest.hexdigest()[:16]


def _category(categories: dict[str, list[str]], name: str) -> str:
    """The category listing name; raises for a name none of them list."""
    return next(c for c, names in categories.items() if name in names)


def _sampled(cmap: Colormap) -> np.ndarray:
    return cmap(np.linspace(0.0, 1.0, SAMPLES))[:, :3]


def segment_stops(segmentdata: dict) -> np.ndarray:
    """(position, r, g, b) stops matplotlib's segment data blends between in sRGB.

    Each channel is given as (x, value below x, value above x); a stop goes at
    every channel's breakpoints, two where a value jumps.
    """
    channels = [np.asarray(segmentdata[c], float) for c in ("red", "green", "blue")]

    def at(points, x, side):
        i = np.searchsorted(points[:, 0], x)
        if i < len(points) and points[i, 0] == x:
            return points[i, 1 if side == "below" else 2]
        f = (x - points[i - 1, 0]) / (points[i, 0] - points[i - 1, 0])
        return points[i - 1, 2] * (1 - f) + points[i, 1] * f

    stops = []
    for x in sorted({float(p[0]) for points in channels for p in points}):
        below = [at(points, x, "below") for points in channels]
        above = [at(points, x, "above") for points in channels]
        if below != above:
            # the colour before, on its own line just short of the jump
            stops.append(
                [x - JUMP, *[at(points, x - JUMP, "below") for points in channels]]
            )
        stops.append([x, *above])
    return np.array(stops)


def _matplotlib_palette(name: str, category: str) -> Palette:
    cmap = matplotlib.colormaps[name]
    discrete = isinstance(cmap, ListedColormap) and cmap.N <= MAX_STOPS
    formula, stops = "", None
    if name in FORMULAS:
        formula = name
    elif isinstance(cmap, LinearSegmentedColormap):
        stops = segment_stops(cmap._segmentdata)
        if len(stops) > MAX_STOPS:
            stops = None  # fitted from samples instead
    return Palette(
        name=name,
        collection="matplotlib",
        group="",
        source=f"matplotlib {matplotlib.__version__}",
        srgb=np.array([to_rgb(c) for c in cmap.colors]) if discrete else _sampled(cmap),
        discrete=discrete,
        formula=formula,
        stops=stops,
        category=category,
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
    for category, names in MATPLOTLIB_GROUPS.items():
        for name in names:
            yield _matplotlib_palette(name, category)


def load_cmocean() -> Iterator[Palette]:
    source = f"cmocean {version('cmocean')}"
    for name in cmocean.cm.cmapnames:
        yield Palette(
            name,
            "cmocean",
            "",
            source,
            _sampled(cmocean.cm.cmap_d[name]),
            category=_category(CMOCEAN_CATEGORIES, name),
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
            yield Palette(
                name, "scientific", "", source, srgb, True, category="Categorical"
            )
            continue
        yield Palette(
            name,
            "scientific",
            "",
            source,
            _sampled(cmap),
            category=_crameri_group(name),
        )
    # missing from cmcrameri
    yield from _load_tables("scientific", category_of=_crameri_group)


def load_metbrewer() -> Iterator[Palette]:
    source = f"met-brewer {version('met-brewer')}"
    for name, palette in met_brewer.MET_PALETTES.items():
        srgb = np.array([to_rgb(c) for c in palette["colors"]])
        # hand-picked colour sets: discrete by default in MetBrewer itself
        tag = COLORBLIND_TAG if name in met_brewer.COLORBLIND_PALETTES_NAMES else ""
        yield Palette(name, "metbrewer", tag, source, srgb, discrete=True)


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
    collection: str,
    group_of=lambda name: "",
    discrete: bool = False,
    category_of=lambda name: "",
) -> Iterator[Palette]:
    for path in sorted((RGB_DIR / collection).glob("*.rgb")):
        yield Palette(
            path.stem,
            collection,
            group_of(path.stem),
            BLENDER_COLORMAPS,
            read_rgb_table(path),
            discrete,
            category=category_of(path.stem),
        )


def _carbonplan_mode(name: str) -> str:
    """Light or Dark; a map that is the same in both has no suffix."""
    mode = name.rpartition("_")[2]
    return mode.capitalize() if mode in ("light", "dark") else ""


def load_carbonplan() -> Iterator[Palette]:
    return _load_tables(
        "carbonplan",
        _carbonplan_mode,
        category_of=lambda name: _category(
            CARBONPLAN_CATEGORIES, name.removesuffix("_light").removesuffix("_dark")
        ),
    )


def load_nasa() -> Iterator[Palette]:
    return _load_tables(
        "nasa",
        lambda name: name.partition("_")[0],  # the agency
        category_of=lambda name: _category(NASA_CATEGORIES, name),
    )


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
