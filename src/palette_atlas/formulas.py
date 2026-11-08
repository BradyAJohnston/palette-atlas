"""matplotlib's palettes that are formulas of x, from matplotlib/_cm.py.

Each is written once against a few operations, so the same code gives numpy
arrays (for checks and thumbnails) and math nodes (for the build). The channels
are sRGB, clipped to 0..1 as matplotlib clips its lookup table.

This module needs only numpy, so build.py can import it inside Blender.
"""

import math

import numpy as np

PI = math.pi


class NumpyOps:
    sqrt = staticmethod(np.sqrt)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    abs = staticmethod(np.abs)
    min = staticmethod(np.minimum)
    max = staticmethod(np.maximum)
    pow = staticmethod(np.power)


def _cubehelix(m, x, p0, p1, s=0.5, r=-1.5, h=1.0):
    """matplotlib's default cubehelix, gamma 1."""
    a = h * x * (1.0 - x) / 2
    phi = 2 * PI * (s / 3 + r * x)
    return x + a * (p0 * m.cos(phi) + p1 * m.sin(phi))


FORMULAS = {
    "afmhot": lambda m, x: (2 * x, 2 * x - 0.5, 2 * x - 1),
    "gist_heat": lambda m, x: (1.5 * x, 2 * x - 1, 4 * x - 3),
    "ocean": lambda m, x: (3 * x - 2, m.abs((3 * x - 1) / 2), x),
    "gnuplot": lambda m, x: (m.sqrt(x), m.pow(x, 3), m.sin(x * 2 * PI)),
    "gnuplot2": lambda m, x: (
        x / 0.32 - 0.78125,
        2 * x - 0.84,
        # 4x below 0.25, 1.84 - 2x to 0.92, then x / 0.08 - 11.5; equal once clipped
        m.min(4 * x, m.max(1.84 - 2 * x, x / 0.08 - 11.5)),
    ),
    "rainbow": lambda m, x: (m.abs(2 * x - 0.5), m.sin(x * PI), m.cos(x * PI / 2)),
    "cubehelix": lambda m, x: (
        _cubehelix(m, x, -0.14861, 1.78277),
        _cubehelix(m, x, -0.29227, -0.90649),
        _cubehelix(m, x, 1.97294, 0.0),
    ),
    "flag": lambda m, x: (
        0.75 * m.sin((x * 31.5 + 0.25) * PI) + 0.5,
        m.sin(x * 31.5 * PI),
        0.75 * m.sin((x * 31.5 - 0.25) * PI) + 0.5,
    ),
    "prism": lambda m, x: (
        0.75 * m.sin((x * 20.9 + 0.25) * PI) + 0.67,
        0.75 * m.sin((x * 20.9 - 0.25) * PI) + 0.33,
        -1.1 * m.sin(x * 20.9 * PI),
    ),
}


def evaluate(name: str, x: np.ndarray) -> np.ndarray:
    """sRGB colours of a formula palette at x, shape (len(x), 3)."""
    x = np.asarray(x, float)
    channels = [np.broadcast_to(c, x.shape) for c in FORMULAS[name](NumpyOps, x)]
    return np.clip(np.stack(channels, 1), 0.0, 1.0)
