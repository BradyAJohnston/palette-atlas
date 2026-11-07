import numpy as np
import pytest

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
