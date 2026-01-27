import argparse
import sys
from pathlib import Path

import numpy as np


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def _find_repo_root(start: Path) -> Path | None:
    start = start.resolve()
    candidates = [start, *list(start.parents)]
    for p in candidates:
        if (p / "pyproject.toml").exists():
            return p
    return None


def _default_data_root() -> Path:
    repo_root = _find_repo_root(Path.cwd())
    if repo_root is not None:
        return repo_root / "data"
    return Path.cwd() / "data"


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    default_data_root = _default_data_root()

    parser = argparse.ArgumentParser(
        prog="shapes",
        description="Print shapes/dtypes/sizes of all .npz arrays under a directory.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help=(
            "Path to a directory to scan OR a folder name to match under ./data "
            "(e.g. `spd_covariance`)."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--folder",
        help="Exact folder name to match anywhere under ./data (can match multiple directories).",
    )
    group.add_argument(
        "--contains",
        help="Substring to match in folder names anywhere under ./data (can match multiple).",
    )

    args = parser.parse_args(argv[1:])

    search_dirs: list[Path] = []

    if args.folder is not None:
        name = str(args.folder)
        search_dirs = sorted(
            [p for p in default_data_root.rglob("*") if p.is_dir() and p.name == name]
        )
        if not search_dirs:
            print(
                f"No directories named {name!r} found under {default_data_root}",
                file=sys.stderr,
            )
            return 1
    elif args.contains is not None:
        needle = str(args.contains)
        search_dirs = sorted(
            [p for p in default_data_root.rglob("*") if p.is_dir() and needle in p.name]
        )
        if not search_dirs:
            print(
                f"No directories with names containing {needle!r} found under {default_data_root}",
                file=sys.stderr,
            )
            return 1
    elif args.target is not None:
        target_str = str(args.target)
        candidate_path = Path(target_str).expanduser()
        if candidate_path.exists() and candidate_path.is_dir():
            search_dirs = [candidate_path.resolve()]
        else:
            # Treat as folder name under ./data
            search_dirs = sorted(
                [p for p in default_data_root.rglob("*") if p.is_dir() and p.name == target_str]
            )
            if not search_dirs:
                # Also try direct child under ./data for convenience
                direct = default_data_root / target_str
                if direct.exists() and direct.is_dir():
                    search_dirs = [direct.resolve()]
            if not search_dirs:
                print(
                    f"Target directory not found (path or folder name): {target_str!r}",
                    file=sys.stderr,
                )
                return 1
    else:
        search_dirs = [default_data_root]

    npz_files: list[Path] = []
    for d in search_dirs:
        npz_files.extend(d.rglob("*.npz"))
    npz_files = sorted(set(npz_files))
    if not npz_files:
        if len(search_dirs) == 1:
            print(f"No .npz files found under {search_dirs[0]}")
        else:
            joined = "\n".join(f"- {d}" for d in search_dirs)
            print(f"No .npz files found under any of:\n{joined}")
        return 0

    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                archive_keys = list(archive.keys())
                file_size = npz_path.stat().st_size
                print(f"\nFile: {npz_path} ({_format_bytes(file_size)})")
                if not archive_keys:
                    print("  (empty archive)")
                for key in archive_keys:
                    arr = archive[key]
                    shape = tuple(arr.shape)
                    dtype = str(arr.dtype)
                    nbytes = int(arr.nbytes)
                    print(
                        f"  {key}: shape={shape}, dtype={dtype}, size={_format_bytes(nbytes)}"
                    )
        except Exception as exc:
            print(f"\nFile: {npz_path}")
            print(f"  Error reading file: {exc}")

    return 0

