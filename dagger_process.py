from __future__ import annotations

import argparse
import shutil
from pathlib import Path

input_path = [
    "/mnt/oss_data/arx1/peel/peel_huanggua_0629",
    "/mnt/oss_data/arx1/peel/peel_huanggua_0630",
    "/mnt/oss_data/arx1/peel/peel_huanggua_0701",
    "/mnt/oss_data/arx1/peel/peel_huanggua_0709_11"
]
output_path = "/mnt/oss_data/arx1/peel/peel_huanggua_06"

def _sort_episode_dirs(corrections_dir: Path) -> list[Path]:
    episodes = [p for p in corrections_dir.iterdir() if p.is_dir()]
    return sorted(episodes, key=lambda p: int(p.name))

def merge_corrections(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    start_index: int = 0,
) -> list[tuple[Path, Path]]:
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []
    next_idx = start_index

    for corrections_dir in input_paths:
        corrections_dir = Path(corrections_dir)
        if not corrections_dir.is_dir():
            raise FileNotFoundError(f"corrections dir not found: {corrections_dir}")

        for src in _sort_episode_dirs(corrections_dir):
            dst = output_dir / f"{next_idx:04d}"
            if dst.exists():
                raise FileExistsError(f"destination already exists: {dst}")

            print(f"{'[dry-run] ' if dry_run else ''}{src} -> {dst}")
            if not dry_run:
                shutil.copytree(src, dst)
            copied.append((src, dst))
            next_idx += 1

    print(f"total episodes: {len(copied)}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge correction episodes with sequential xxxx naming.")
    parser.add_argument("--dry-run", action="store_true", help="Print copy plan without writing files.")
    parser.add_argument("--output", default=output_path, help="Output dataset directory.")
    args = parser.parse_args()
    merge_corrections(input_path, args.output, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
