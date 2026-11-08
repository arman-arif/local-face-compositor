#!/usr/bin/env python3
"""Run the same source/target pairs through each installed swap model.

Usage:
  python scripts/benchmark-models.py --sources samples/sources --targets samples/targets
Each target filename is paired with every source image. Outputs and CSV are written
under benchmark-results/. This reports identity similarity and runtime, not a
single composite quality score; inspect images side by side.
"""
import argparse
import csv
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import run_swap  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["inswapper_128", "ghost_1_256"],
                        choices=["inswapper_128", "ghost_1_256"])
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results")
    args = parser.parse_args()
    sources, targets = images(args.sources), images(args.targets)
    if not sources or not targets:
        parser.error("Both folders must contain at least one supported image.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in targets:
        for model in args.models:
            out = args.output / f"{target.stem}-{model}.png"
            started = time.perf_counter()
            try:
                device, source_note, similarity = run_swap(
                    sources, target, out, color_match_strength=0.35,
                    blend_strength=0.95, blend_mode="feather",
                    restore_face=False, merge_source_references=True,
                    swap_model=model,
                )
                elapsed = time.perf_counter() - started
                rows.append({
                    "target": target.name, "model": model, "status": "ok",
                    "seconds": round(elapsed, 3), "device": device,
                    "arcface_cosine": "" if similarity is None else round(similarity, 5),
                    "output": str(out.relative_to(ROOT)), "error": "",
                })
                print(f"[ok] {target.name} | {model} | {elapsed:.2f}s | ArcFace={similarity}")
            except Exception as exc:
                rows.append({
                    "target": target.name, "model": model, "status": "failed",
                    "seconds": round(time.perf_counter() - started, 3),
                    "device": "", "arcface_cosine": "", "output": "",
                    "error": str(exc),
                })
                print(f"[failed] {target.name} | {model}: {exc}")
    report = args.output / "results.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {report}")


if __name__ == "__main__":
    main()
