"""Render a swatch image per collection for the README, into docs/images/swatches/.

    uv run scripts/swatches.py

Swatches show each palette's source colours: smooth palettes as a gradient,
discrete ones as equal blocks.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from palette_atlas.palettes import load_palettes

OUT = Path(__file__).parents[1] / "docs" / "images" / "swatches"
WIDTH = 512  # samples across a smooth palette
LABEL = "#888888"  # readable on GitHub's light and dark themes

SHOWN = {
    "matplotlib": ["viridis", "magma", "cividis", "turbo", "coolwarm", "twilight", "tab10", "Set2"],
    "cmocean": ["thermal", "haline", "deep", "algae", "balance", "curl", "phase", "topo"],
    "scientific": ["batlow", "hawaii", "lajolla", "oslo", "roma", "vik", "romaO", "batlowK"],
    "metbrewer": ["Hokusai1", "Archambault", "Cassatt2", "Demuth", "Egypt", "Hiroshige", "Juarez", "Klimt"],
    "carbonplan": [
        "fire_light", "earth_light", "water_light", "heart_light", "cool_light",
        "orangeblue_light", "pinkgreen_light", "sinebow",
    ],
    "nasa": [
        "GMT_haxby", "GMT_relief", "GMT_globe", "NEO_modis_ndvi", "NEO_modis_sst_45",
        "NEO_trmm_rainfall", "NEO_giss_temp_anom", "NCDC_temp_anom_f",
    ],
    "ltc": ["paloma", "alger", "minou", "olga", "shuggie", "sylvie", "trio1", "heatmap1"],
    "tableau": [
        "tableau_10", "tableau_20", "colorblind_10", "tableaumedium_10",
        "bluered_12", "greenorange_12", "purplegray_12", "trafficlight_9",
    ],
}  # fmt: skip


def strip(palette) -> np.ndarray:
    """One row of sRGB colours, shape (1, n, 3)."""
    srgb = palette.srgb
    if palette.discrete:
        return srgb[None]
    x = np.linspace(0.0, 1.0, WIDTH)
    xp = np.linspace(0.0, 1.0, len(srgb))
    return np.stack([np.interp(x, xp, srgb[:, c]) for c in range(3)], 1)[None]


def render(collection: str, names: list[str], palettes: dict) -> Path:
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 0.42 * len(names)), dpi=150)
    for ax, name in zip(axes, names):
        palette = palettes[collection, name]
        ax.imshow(strip(palette), aspect="auto", interpolation="nearest")
        ax.set_axis_off()
        ax.text(-0.01, 0.5, name, transform=ax.transAxes, ha="right", va="center",
                fontsize=9, color=LABEL)  # fmt: skip
    fig.subplots_adjust(left=0.2, right=0.99, top=0.98, bottom=0.02, hspace=0.35)
    path = OUT / f"{collection}.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    palettes = {(p.collection, p.name): p for p in load_palettes(list(SHOWN))}
    for collection, names in SHOWN.items():
        print(render(collection, names, palettes))


if __name__ == "__main__":
    main()
