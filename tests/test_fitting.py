import numpy as np
import pytest

from node_palettes import fitting
from node_palettes.fitting import (
    copy_stops,
    evaluate,
    evaluate_ramp,
    fit_palettes,
    fit_split,
    fit_stops,
    read_ramps,
)
from node_palettes.palettes import MAX_STOPS, Palette, load_palettes


def palette(srgb, name="test", discrete=False):
    return Palette(name, "test", "", "test", np.asarray(srgb, float), discrete)


def gradient(n=256):
    x = np.linspace(0, 1, n)[:, None]
    return x * np.array([1.0, 0.5, 0.2])


def test_discrete_matches_matplotlib_indexing():
    colours = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], float)
    ramp = copy_stops(palette(colours, discrete=True))
    assert ramp.segments[0].interpolation == "CONSTANT"
    x = np.array([0.0, 0.24, 0.25, 0.6, 0.99, 1.0])
    index = np.minimum((x * 4).astype(int), 3)
    assert np.allclose(evaluate_ramp(ramp, x), colours[index], atol=1e-4)


def test_long_discrete_palette_splits_exactly():
    colours = np.random.default_rng(0).random((100, 3))
    ramp = copy_stops(palette(colours, discrete=True))
    assert len(ramp.segments) == 4
    assert all(len(s.stops) == 25 for s in ramp.segments)
    x = (np.arange(1000) + 0.5) / 1000
    index = (x * 100).astype(int)
    assert np.allclose(evaluate_ramp(ramp, x), colours[index], atol=1e-4)
    # the first colour of a segment starts exactly at its boundary
    assert np.allclose(evaluate_ramp(ramp, np.array([0.25])), colours[25], atol=1e-4)


def test_small_gradient_hits_every_colour():
    colours = np.array([[0.1, 0.2, 0.3], [0.9, 0.5, 0.1], [0.2, 0.8, 0.6]])
    ramp = copy_stops(palette(colours))
    assert np.allclose(evaluate_ramp(ramp, np.linspace(0, 1, 3)), colours, atol=1e-4)


def test_splitting_recovers_what_one_ramp_cannot_hold():
    x = np.linspace(0, 1, 256)[:, None]
    stripes = 0.5 + 0.5 * np.sin(x * 2 * np.pi * 40 + np.array([0, 2, 4]))
    p = palette(stripes)
    assert fit_split(p, 1).error > 2.0
    exact = fit_split(p, fitting.most_parts(p))
    assert exact.error < 0.1
    assert all(len(s.stops) <= MAX_STOPS for s in exact.segments)


def samples_and_midpoints(srgb):
    """The points fit_stops holds a ramp to: samples, and the straight line
    between neighbouring samples at the midpoint."""
    x = np.linspace(0, 1, 2 * len(srgb) - 1)
    ref = np.stack([np.interp(x, np.linspace(0, 1, len(srgb)), srgb[:, k]) for k in range(3)], 1)
    return x, ref


@pytest.mark.parametrize("mode", ["LINEAR", "B_SPLINE"])
def test_fit_reports_its_true_error(mode):
    srgb = gradient()
    error, pos, cols = fit_stops(srgb, 12, mode)
    stops = np.column_stack([pos, cols])
    x, ref = samples_and_midpoints(srgb)
    assert np.abs(evaluate(stops, x, mode) - ref).max() * 255 == pytest.approx(error, abs=1e-6)
    if mode == "LINEAR":
        assert error < 0.01  # a straight line in sRGB, which ramps now blend in


def test_stops_stay_apart_at_a_hard_final_step():
    srgb = gradient()
    srgb[-1] = (0, 1, 0)  # tempts the fit to stack stops at 1.0
    error, pos, cols = fit_stops(srgb, 16, "B_SPLINE")
    assert np.diff(pos).min() >= fitting.MIN_GAP * 0.99
    stops = np.round(np.column_stack([pos, cols]), 6)  # as stored
    x, ref = samples_and_midpoints(srgb)
    out = evaluate(stops, x, "B_SPLINE")
    assert np.abs(out - ref).max() * 255 == pytest.approx(error, abs=0.05)


def test_fit_cannot_hide_a_spike_between_samples():
    # samples alone would let a stop between two of them take any colour
    srgb = gradient(64)
    error, pos, cols = fit_stops(srgb, 20, "LINEAR")
    assert np.diff(pos).min() >= 1 / 63 * 0.99
    fine = np.linspace(0, 1, 63 * 16 + 1)
    ref = np.stack([np.interp(fine, np.linspace(0, 1, 64), srgb[:, k]) for k in range(3)], 1)
    out = evaluate(np.column_stack([pos, cols]), fine, "LINEAR")
    assert np.abs(out - ref).max() * 255 < 1.0


def test_cache_skips_unchanged_and_refits_changed(tmp_path):
    a = palette(gradient(), "a")
    (first,) = fit_palettes([a], workers=1, ramps_dir=tmp_path)["test"]
    assert len(first.segments) == 1 and first.error < 1.0

    seen = []
    fit_palettes([a], 1, tmp_path, on_done=lambda p, r: seen.append(p.name))
    assert seen == []

    changed = palette(gradient()[::-1], "a")
    fit_palettes([changed], 1, tmp_path, on_done=lambda p, r: seen.append(p.name))
    assert seen == ["a"]
    assert read_ramps("test", tmp_path)["a"].digest == changed.digest


def test_cache_refits_when_a_palette_becomes_discrete(tmp_path):
    colours = np.random.default_rng(1).random((7, 3))
    fit_palettes([palette(colours, "a")], 1, tmp_path)
    (ramp,) = fit_palettes([palette(colours, "a", discrete=True)], 1, tmp_path)["test"]
    assert [s.interpolation for s in ramp.segments] == ["CONSTANT"]


def test_formula_and_package_stops_are_copied_not_fitted():
    by_name = {p.name: p for p in load_palettes(["matplotlib"])}
    jet = copy_stops(by_name["jet"])
    assert len(jet.segments) == 1 and len(jet.segments[0].stops) == 12
    assert jet.error < 1e-3
    gnuplot = copy_stops(by_name["gnuplot"])
    assert gnuplot.formula == "gnuplot" and gnuplot.segments == [] and gnuplot.error < 1e-9


def test_worker_count_does_not_change_results(tmp_path):
    items = [palette(gradient(), "a"), palette(gradient()[::-1], "b")]
    one = fit_palettes(items, 1, tmp_path / "one")["test"]
    many = fit_palettes(items, 4, tmp_path / "many")["test"]
    assert one == many


def stored_ramps():
    palettes = load_palettes()
    ramps = {c: read_ramps(c) for c in {p.collection for p in palettes}}
    return [
        pytest.param(p, ramps[p.collection].get(p.name), id=f"{p.collection}-{p.name}")
        for p in palettes
    ]


@pytest.mark.parametrize("source, ramp", stored_ramps())
def test_stored_ramp_matches_source(source, ramp):
    assert ramp is not None, "run `node-palettes fit`"
    assert ramp.digest == source.digest and ramp.fit_version == fitting.FIT_VERSION
    assert ramp.error <= fitting.LOSSY
    if ramp.formula:
        assert not ramp.segments and ramp.error < 1e-9
        return
    assert ramp.segments[0].start == 0.0 and ramp.segments[-1].end == 1.0
    if len(ramp.segments) > 1:
        assert ramp.error <= fitting.TOLERANCE

    for before, after in zip(ramp.segments, ramp.segments[1:]):
        assert before.end == after.start
    for segment in ramp.segments:
        stops = np.array(segment.stops)
        assert 2 <= len(stops) <= MAX_STOPS
        assert stops[0, 0] == 0.0 and np.all(np.diff(stops[:, 0]) > 0)
        assert stops[:, 1:].min() >= 0 and stops[:, 1:].max() <= 1
        if segment.interpolation != "CONSTANT":
            assert np.diff(stops[:, 0]).min() >= fitting.MIN_GAP * 0.99

    n = len(source.srgb)
    x = (np.arange(n) + 0.5) / n if source.discrete else np.linspace(0, 1, n)
    error = np.abs(evaluate_ramp(ramp, x) - source.srgb).max() * 255
    assert error == pytest.approx(ramp.error, abs=0.002)

    if not source.discrete and source.stops is None:
        # between neighbouring samples, a fitted ramp stays within their range;
        # matplotlib's own stops are checked between samples by `verify`
        steps = 16
        fine = evaluate_ramp(ramp, np.linspace(0, 1, (n - 1) * steps + 1))[:-1]
        fine = fine.reshape(n - 1, steps, 3)
        low = np.minimum(source.srgb[:-1], source.srgb[1:])[:, None]
        high = np.maximum(source.srgb[:-1], source.srgb[1:])[:, None]
        outside = np.maximum(low - fine, fine - high).max() * 255
        assert outside <= max(ramp.error, 1.0) + 0.5
