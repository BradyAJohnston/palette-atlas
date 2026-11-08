import numpy as np
import pytest

from matplotlib import _cm

from node_palettes import formulas
from node_palettes.palettes import LOADERS, RGB_DIR, load_palettes, read_rgb_table

EXPECTED = {
    "matplotlib": 85,
    "cmocean": 22,
    "scientific": 61,
    "metbrewer": 58,
    "carbonplan": 58,
    "nasa": 84,
    "ltc": 32,
    "tableau": 13,
}


@pytest.fixture(scope="module")
def palettes():
    return load_palettes()


def test_collection_sizes(palettes):
    counts = {c: sum(p.collection == c for p in palettes) for c in LOADERS}
    assert counts == EXPECTED


@pytest.mark.parametrize("collection", ["metbrewer", "ltc", "tableau"])
def test_hand_picked_sets_are_discrete(palettes, collection):
    members = [p for p in palettes if p.collection == collection]
    assert members and all(p.discrete for p in members)


def test_carbonplan_modes_differ(palettes):
    by_name = {p.name: p for p in palettes if p.collection == "carbonplan"}
    for name, light in by_name.items():
        if name.endswith("_light"):
            dark = by_name[name.removesuffix("_light") + "_dark"]
            assert not np.allclose(light.srgb, dark.srgb), name
            assert (light.group, dark.group) == ("Light", "Dark"), name
    assert by_name["rainbow"].group == ""  # the same in both modes


@pytest.mark.parametrize("name", sorted(formulas.FORMULAS))
def test_formulas_match_matplotlib(name):
    x = np.linspace(0, 1, 10001)
    functions = _cm.datad[name]
    expected = np.clip(np.stack([functions[c](x) for c in ("red", "green", "blue")], 1), 0, 1)
    assert np.abs(formulas.evaluate(name, x) - expected).max() < 1e-12


def test_matplotlib_package_stops_reproduce_its_samples(palettes):
    with_stops = [p for p in palettes if p.stops is not None]
    assert len(with_stops) == 49
    for p in with_stops:
        x = np.linspace(0, 1, len(p.srgb))
        out = np.stack([np.interp(x, p.stops[:, 0], p.stops[:, k]) for k in (1, 2, 3)], 1)
        assert np.abs(out - p.srgb).max() < 1e-9, p.name
        assert len(p.stops) <= 32 and np.all(np.diff(p.stops[:, 0]) > 0), p.name


def test_names_unique_within_collection(palettes):
    keys = [(p.collection, p.name) for p in palettes]
    assert len(keys) == len(set(keys))


def test_colours_are_valid_srgb(palettes):
    for p in palettes:
        assert p.srgb.ndim == 2 and p.srgb.shape[1] == 3, p.name
        assert len(p.srgb) >= 2, p.name
        assert p.srgb.min() >= 0 and p.srgb.max() <= 1, p.name


def test_hsv_table_is_converted():
    srgb = read_rgb_table(RGB_DIR / "nasa" / "GMT_cyclic.rgb")
    assert np.allclose(srgb[0], (1, 0, 0))  # hue 0
    assert np.allclose(srgb[1], (0, 1, 1))  # hue 180
    assert np.allclose(srgb[-1], (1, 0, 0))  # hue 360


@pytest.mark.parametrize("header", ["ncolors=2\n# r g b\n", "ncolor=2\n", ""])
def test_table_headers_are_optional(tmp_path, header):
    path = tmp_path / "table.rgb"
    path.write_text(header + "0 0 0\n255 51 102\n")
    assert np.allclose(read_rgb_table(path), [[0, 0, 0], [1, 0.2, 0.4]])


@pytest.mark.parametrize("used, footer", [(256, "1 0 1"), (255, "0 255 1"), (20, "0 20 1")])
def test_act_footer_trims_padding_and_itself(tmp_path, used, footer):
    path = tmp_path / "table.rgb"
    body = [f"{i} {i} {i}" for i in range(used)] + ["0 0 0"] * (256 - used)
    path.write_text("\n".join([*body, footer]) + "\n")
    srgb = read_rgb_table(path)
    assert len(srgb) == used
    assert np.allclose(srgb[-1], (used - 1) / 255)


def test_no_nasa_table_ends_in_a_footer(palettes):
    for p in palettes:
        if p.collection == "nasa":
            last = np.round(p.srgb[-1] * 255)
            assert not (last[2] == 1 and last[0] <= 1), p.name


@pytest.mark.parametrize("collection", ["matplotlib", "cmocean", "scientific", "carbonplan", "nasa"])
def test_categorised_collections_have_categories(palettes, collection):
    members = [p for p in palettes if p.collection == collection]
    assert members and all(p.category for p in members)


@pytest.mark.parametrize("collection", ["metbrewer", "ltc", "tableau"])
def test_uncategorised_collections_stay_flat(palettes, collection):
    assert not any(p.category for p in palettes if p.collection == collection)


def test_category_lists_name_real_palettes():
    """A name dropped or renamed upstream would otherwise linger in the lists."""
    from node_palettes import palettes as module

    for collection, lists in [
        ("cmocean", module.CMOCEAN_CATEGORIES),
        ("nasa", module.NASA_CATEGORIES),
    ]:
        names = [n for members in lists.values() for n in members]
        loaded = {p.name for p in load_palettes([collection])}
        assert len(names) == len(set(names)) and set(names) == loaded, collection
    names = [n for members in module.CARBONPLAN_CATEGORIES.values() for n in members]
    loaded = {p.name.removesuffix("_light").removesuffix("_dark") for p in load_palettes(["carbonplan"])}
    assert len(names) == len(set(names)) and set(names) == loaded


def test_metbrewer_colorblind_tag(palettes):
    tagged = [p for p in palettes if p.collection == "metbrewer" and p.group]
    assert len(tagged) == 24 and {p.group for p in tagged} == {"Colorblind-Friendly"}
