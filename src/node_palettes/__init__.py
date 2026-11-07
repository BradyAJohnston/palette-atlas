import argparse
import time

from .fitting import fit_palettes
from .palettes import LOADERS, load_palettes


def fit(collections: list[str] | None, workers: int | None) -> None:
    palettes = load_palettes(collections)
    print(f"{len(palettes)} palettes")
    start = time.perf_counter()
    ramps = fit_palettes(
        palettes,
        workers,
        on_done=lambda p, r: print(
            f"  {p.collection}/{p.name}: {len(r.segments)} ramps, "
            f"{sum(len(s.stops) for s in r.segments)} stops, error {r.error:.2f}",
            flush=True,
        ),
    )
    print(f"finished in {time.perf_counter() - start:.0f}s")
    for collection, members in ramps.items():
        split = [r for r in members if len(r.segments) > 1]
        print(
            f"{collection}: {len(members)} palettes, {len(split)} split, "
            f"worst error {max(r.error for r in members):.2f}/255"
        )
        for ramp in sorted(split, key=lambda r: -len(r.segments)):
            print(f"  {ramp.name}: {len(ramp.segments)} ramps, {ramp.error:.2f}/255")


def main() -> None:
    parser = argparse.ArgumentParser(prog="node-palettes")
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="fit palettes to Color Ramps")
    fit_parser.add_argument(
        "collections", nargs="*", help=f"default: all of {', '.join(LOADERS)}"
    )
    fit_parser.add_argument("--workers", type=int, help="default: one per core")
    args = parser.parse_args()
    fit(args.collections, args.workers)
