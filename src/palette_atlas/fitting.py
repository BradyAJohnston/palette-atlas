"""Fit palettes to Blender Color Ramps of at most 32 stops, and cache the result.

Ramps hold sRGB colours, blended in sRGB as the packages blend them, and the node
group converts the result to linear afterwards. Stop colours and positions are
fitted by least squares against a numpy model of Blender's ramp
(BKE_colorband_evaluate). Linear and B-spline interpolation are each tried and
the one needing fewer stops is kept. Error is measured in 8-bit sRGB against the
palette's samples. Palettes of 32 colours or fewer need no fit, their colours
are copied exactly, as are matplotlib's own stops and formulas.

A palette that one ramp can't hold is split into even segments, one ramp each.
A segment of 32 samples or fewer is copied exactly, so splitting always succeeds.

Ported from MolecularNodes' docs/dev/colormap_node.py.
"""

import json
import math
import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

from . import formulas
from .palettes import MAX_STOPS, Palette

RAMPS_DIR = Path(__file__).parent / "data" / "ramps"
TOLERANCE = 1.0  # fewest stops that keep every sample within this, in 1/255 sRGB
LOSSY = 2.0  # a single ramp worse than this is split into segments
# stops closer than this collapse in Blender's 32-bit positions, wrecking the
# colour at the end of a ramp; fitted stops also keep a sample apart (fit_stops)
MIN_GAP = 1e-4
MODES = ("LINEAR", "B_SPLINE")
FIT_VERSION = 7  # raise when a change here should invalidate cached fits


@dataclass
class Segment:
    """One Color Ramp covering start..end of a palette.

    Stops are (position, r, g, b) in sRGB, positions local to the segment.
    """

    start: float
    end: float
    interpolation: str
    stops: list[list[float]]


@dataclass
class Ramp:
    """A palette as one or more Color Ramps, or a formula from formulas.py, for
    the build to turn into nodes."""

    name: str
    group: str
    source: str
    digest: str
    error: float
    segments: list[Segment]
    fit_version: int = FIT_VERSION
    formula: str = ""
    category: str = ""
    mode: str = ""


def ramp_basis(pos: np.ndarray, x: np.ndarray, mode: str) -> np.ndarray:
    """Matrix B with Blender's ramp colours at x = clip(B @ stop colours).

    Follows BKE_colorband_evaluate in RGB mode, for sorted stops from 0 to 1.
    """
    n = len(pos)
    rows = np.arange(len(x))
    B = np.zeros((len(x), n))
    if mode == "B_SPLINE":
        # at x == 1 Blender stays on the last segment rather than snapping
        x = np.minimum(x, np.nextafter(pos[-1], 0))
    a = np.searchsorted(pos, x, side="right")  # first stop past x
    end = a == n
    B[rows[end], n - 1] = 1.0
    r, a = rows[~end], a[~end]
    span = pos[a - 1] - pos[a]
    fac = np.where(span != 0, (x[r] - pos[a]) / np.where(span != 0, span, 1), 0.0)
    if mode == "LINEAR":
        np.add.at(B, (r, a), 1 - fac)
        np.add.at(B, (r, a - 1), fac)
        return B
    t = np.clip(fac, 0, 1)
    weights = [
        -(t**3) / 6 + t**2 / 2 - t / 2 + 1 / 6,
        t**3 / 2 - t**2 + 2 / 3,
        -(t**3) / 2 + t**2 / 2 + t / 2 + 1 / 6,
        t**3 / 6,
    ]
    neighbours = [
        np.where(a >= n - 1, a, a + 1),
        a,
        a - 1,
        np.where(a < 2, a - 1, a - 2),
    ]
    for w, i in zip(weights, neighbours):
        np.add.at(B, (r, i), w)
    return B


def evaluate(stops: np.ndarray, x: np.ndarray, mode: str) -> np.ndarray:
    """sRGB colours of a ramp at x, as Blender would give them."""
    pos, cols = stops[:, 0], stops[:, 1:4]
    if mode == "CONSTANT":
        return cols[np.searchsorted(pos, x, side="right") - 1]
    return np.clip(ramp_basis(pos, x, mode) @ cols, 0, 1)


def fit_stops(srgb: np.ndarray, n: int, mode: str):
    """(error, positions, colours) of an n-stop ramp fitted to srgb.

    Fitted to the samples alone, a ramp can spike between them unseen: stops can
    zigzag about a sample and still land on it. Shader and Compositor show such
    spikes once they bake the ramp to a table. So the fit also holds the ramp to
    the straight line between neighbouring samples, at the midpoint, and keeps
    stops at least a sample apart, so no stop slips between those points.
    """
    x = np.linspace(0.0, 1.0, len(srgb))
    gap = max(MIN_GAP, 1.0 / (len(srgb) - 1))
    fit_x = np.linspace(0.0, 1.0, 2 * len(srgb) - 1)  # samples and midpoints
    target = np.stack([np.interp(fit_x, x, srgb[:, k]) for k in range(3)], 1)

    def unpack(p):
        gaps = np.exp(p[: n - 1] - p[: n - 1].max())
        gaps = gap + gaps / gaps.sum() * (1 - (n - 1) * gap)
        pos = np.concatenate([[0.0], np.cumsum(gaps)])
        pos[-1] = 1.0
        return pos, p[n - 1 :].reshape(n, 3)

    def error(pos, cols):
        return (np.clip(ramp_basis(pos, fit_x, mode) @ cols, 0, 1) - target) * 255

    # start from the knots a greedy linear fit through the samples would pick
    chosen = [0, len(x) - 1]
    while len(chosen) < n:
        s = sorted(chosen)
        approx = np.stack([np.interp(x, x[s], srgb[s, k]) for k in range(3)], 1)
        e = np.abs(approx - srgb).max(1)
        e[s] = -1
        chosen.append(int(e.argmax()))
    pos = x[sorted(chosen)]
    # QR rather than numpy's SVD solver, which fails to converge on some
    # well-conditioned bases here (cork, B-spline, 26 stops)
    cols = lstsq(ramp_basis(pos, fit_x, mode), target, lapack_driver="gelsy")[0]
    cols = np.clip(cols, 1e-9, 1 - 1e-9)

    best = (np.abs(error(pos, cols)).max(), pos, cols)
    lo = np.r_[np.full(n - 1, -np.inf), np.zeros(3 * n)]
    hi = np.r_[np.full(n - 1, np.inf), np.ones(3 * n)]
    for power in (2, 8):  # least squares, then close to the worst-case error
        p0 = np.r_[np.log(np.maximum(np.diff(pos), 1e-4)), cols.ravel()]
        try:
            sol = least_squares(
                lambda p: (
                    np.sign(r := error(*unpack(p)).ravel()) * np.abs(r) ** (power / 2)
                ),
                np.clip(p0, lo + 1e-9, hi - 1e-9),
                bounds=(lo, hi),
                max_nfev=200,
                x_scale="jac",
            )
        except np.linalg.LinAlgError:
            break  # an occasional ill-conditioned step, keep the best fit so far
        pos, cols = unpack(sol.x)
        e = np.abs(error(pos, cols)).max()
        if e < best[0]:
            best = (e, pos, cols)
    return best


def fewest_stops(srgb: np.ndarray, mode: str):
    """(error, positions, colours) with the fewest stops within TOLERANCE, or
    the best MAX_STOPS fit when none is."""
    fits = {}

    def at(n):
        if n not in fits:
            fits[n] = fit_stops(srgb, n, mode)
        return fits[n]

    lo, hi = 2, MAX_STOPS
    if at(hi)[0] >= TOLERANCE:
        return at(hi)
    while lo < hi:  # bisect, taking error to fall as stops are added
        mid = (lo + hi) // 2
        if at(mid)[0] < TOLERANCE:
            hi = mid
        else:
            lo = mid + 1
    return at(hi)


def evaluate_ramp(ramp: Ramp, x: np.ndarray) -> np.ndarray:
    """sRGB colours of a whole palette at x, as its nodes would give them."""
    if ramp.formula:
        return formulas.evaluate(ramp.formula, x)
    out = np.zeros((len(x), 3))
    for i, segment in enumerate(ramp.segments):
        chosen = x >= segment.start if i else np.ones(len(x), bool)
        local = (x[chosen] - segment.start) / (segment.end - segment.start)
        stops = np.array(segment.stops)
        out[chosen] = evaluate(stops, np.clip(local, 0, 1), segment.interpolation)
    return out


def _segment(start, end, interpolation, pos, cols) -> Segment:
    stops = [
        [round(float(v), 6) for v in (p, *c)] for p, c in zip(pos, cols, strict=True)
    ]
    return Segment(round(float(start), 6), round(float(end), 6), interpolation, stops)


def _fit_segment(srgb: np.ndarray, start: float, end: float) -> Segment:
    n = len(srgb)
    if n <= MAX_STOPS:
        return _segment(start, end, "LINEAR", np.linspace(0, 1, n), srgb)
    fits = {mode: fewest_stops(srgb, mode) for mode in MODES}
    # fewest stops, then lowest error; MODES order settles any tie
    mode = min(MODES, key=lambda m: (len(fits[m][1]), fits[m][0]))
    return _segment(start, end, mode, fits[mode][1], fits[mode][2])


def _ramp(palette: Palette, segments: list[Segment], formula: str = "") -> Ramp:
    """A ramp as stored, with its error measured on the rounded stops."""
    n = len(palette.srgb)
    x = (np.arange(n) + 0.5) / n if palette.discrete else np.linspace(0, 1, n)
    ramp = Ramp(
        palette.name,
        palette.group,
        palette.source,
        palette.digest,
        0.0,
        segments,
        category=palette.category,
        mode=palette.mode,
    )
    ramp.formula = formula
    ramp.error = round(float(np.abs(evaluate_ramp(ramp, x) - palette.srgb).max() * 255), 3)
    return ramp


def copy_stops(palette: Palette) -> Ramp:
    """A ramp holding the palette's colours exactly: its formula, its package's
    own stops, or its colours."""
    if palette.formula:
        return _ramp(palette, [], palette.formula)
    if palette.stops is not None:
        pos, cols = palette.stops[:, 0], palette.stops[:, 1:]
        return _ramp(palette, [_segment(0, 1, "LINEAR", pos, cols)])
    n = len(palette.srgb)
    if not palette.discrete:
        cols = palette.srgb
        return _ramp(palette, [_segment(0, 1, "LINEAR", np.linspace(0, 1, n), cols)])
    # matplotlib maps x to colour floor(x * N), a constant ramp with stop i at
    # i / N does the same
    bounds = np.linspace(0, n, math.ceil(n / MAX_STOPS) + 1).round().astype(int)
    segments = []
    for a, b in zip(bounds, bounds[1:]):
        cols = palette.srgb[a:b]
        pos = np.arange(b - a) / (b - a)
        segments.append(_segment(a / n, b / n, "CONSTANT", pos, cols))
    return _ramp(palette, segments)


def fit_split(palette: Palette, parts: int) -> Ramp:
    """Fit a palette as even segments, sharing the sample where they meet."""
    n = len(palette.srgb)
    bounds = np.linspace(0, n - 1, parts + 1).round().astype(int)
    return _ramp(
        palette,
        [
            _fit_segment(palette.srgb[a : b + 1], a / (n - 1), b / (n - 1))
            for a, b in zip(bounds, bounds[1:])
        ],
    )


def most_parts(palette: Palette) -> int:
    """Segments at which every segment is short enough to copy exactly."""
    return math.ceil((len(palette.srgb) - 1) / (MAX_STOPS - 1))


def is_stepped(ramp: Ramp) -> bool:
    return bool(ramp.segments) and all(
        s.interpolation == "CONSTANT" for s in ramp.segments
    )


def needs_fit(palette: Palette) -> bool:
    exact = palette.discrete or palette.formula or palette.stops is not None
    return not exact and len(palette.srgb) > MAX_STOPS


def read_ramps(collection: str, ramps_dir: Path = RAMPS_DIR) -> dict[str, Ramp]:
    path = ramps_dir / f"{collection}.json"
    if not path.exists():
        return {}
    ramps = {}
    known = set(Ramp.__dataclass_fields__)
    for fields in json.loads(path.read_text()):
        segments = [Segment(**s) for s in fields.pop("segments")]
        # fields from older fits are dropped; FIT_VERSION marks those for a refit
        fields = {k: v for k, v in fields.items() if k in known}
        ramps[fields["name"]] = Ramp(**fields, segments=segments)
    return ramps


def write_ramps(
    collection: str, ramps: list[Ramp], ramps_dir: Path = RAMPS_DIR
) -> None:
    ramps_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps([asdict(r) for r in ramps], indent=1)
    (ramps_dir / f"{collection}.json").write_text(text + "\n")


def _limit_blas_threads() -> None:
    """Workers inherit this, so one fit per core doesn't oversubscribe the CPU."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def fit_palettes(
    palettes: list[Palette],
    workers: int | None = None,
    ramps_dir: Path = RAMPS_DIR,
    on_done: Callable[[Palette, Ramp], None] = lambda palette, ramp: None,
) -> dict[str, list[Ramp]]:
    """Ramps for each palette by collection, reusing cached fits and saving them.

    Each round fits the pending palettes across processes, as the solver is
    largely Python and would hold the GIL in threads. A palette stays pending,
    with one more segment, until it is close enough.
    """
    by_collection = defaultdict(list)
    for palette in palettes:
        by_collection[palette.collection].append(palette)
    cached = {c: read_ramps(c, ramps_dir) for c in by_collection}

    done: dict[tuple[str, str], Ramp] = {}
    pending = []
    for palette in palettes:
        key = (palette.collection, palette.name)
        hit = cached[palette.collection].get(palette.name)
        if (
            hit
            and hit.digest == palette.digest
            and hit.fit_version == FIT_VERSION
            and is_stepped(hit) == palette.discrete
        ):
            # these can change without the colours changing
            hit.group, hit.source = palette.group, palette.source
            hit.category = palette.category
            done[key] = hit
        elif needs_fit(palette):
            pending.append(palette)
        else:
            done[key] = copy_stops(palette)
            on_done(palette, done[key])

    if pending:
        _limit_blas_threads()
        with ProcessPoolExecutor(workers) as pool:
            parts, limit = 1, LOSSY
            while pending:
                futures = {
                    pool.submit(fit_split, palette, parts): palette
                    for palette in pending
                }
                pending = []
                for future in as_completed(futures):
                    palette, ramp = futures[future], future.result()
                    if ramp.error > limit and parts < most_parts(palette):
                        pending.append(palette)
                        continue
                    done[(palette.collection, palette.name)] = ramp
                    on_done(palette, ramp)
                pending.sort(key=lambda p: (p.collection, p.name))
                parts, limit = parts + 1, TOLERANCE

    ramps = {}
    for collection, members in by_collection.items():
        ramps[collection] = [done[(collection, p.name)] for p in members]
        write_ramps(collection, ramps[collection], ramps_dir)
    return ramps
