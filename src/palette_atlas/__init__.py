import argparse
import time
from pathlib import Path

from .fitting import fit_palettes
from .palettes import LOADERS, load_palettes

EXTENSION_DIR = Path(__file__).parents[2] / "extension"
DIST_DIR = Path(__file__).parents[2] / "dist"


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


def package() -> None:
    """Build the extension, which only points Blender at the hosted library."""
    import bpy  # noqa: F401, sets up the path bl_pkg is imported from
    from bl_pkg.bl_extension_cli import cli_extension_handler

    DIST_DIR.mkdir(exist_ok=True)
    args = ["build", "--source-dir", str(EXTENSION_DIR), "--output-dir", str(DIST_DIR)]
    if cli_extension_handler(args):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="palette-atlas")
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="fit palettes to Color Ramps")
    fit_parser.add_argument(
        "collections", nargs="*", help=f"default: all of {', '.join(LOADERS)}"
    )
    fit_parser.add_argument("--workers", type=int, help="default: one per core")
    build_parser = commands.add_parser(
        "build", help="build and index the asset library into library/"
    )
    build_parser.add_argument(
        "collections", nargs="*", help=f"default: all of {', '.join(LOADERS)}"
    )
    verify_parser = commands.add_parser(
        "verify", help="check the built library in every editor against the palettes"
    )
    verify_parser.add_argument(
        "collections", nargs="*", help=f"default: all of {', '.join(LOADERS)}"
    )
    commands.add_parser("package", help="build the extension into dist/")
    args = parser.parse_args()
    # bpy is imported only by the commands that need it
    if args.command == "fit":
        fit(args.collections, args.workers)
    elif args.command == "build":
        from .build import build

        build(args.collections)
    elif args.command == "verify":
        from .verify import verify

        if not verify(args.collections):
            raise SystemExit(1)
    else:
        package()
