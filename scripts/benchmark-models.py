#!/usr/bin/env python3
"""Compare installed swap models on a fixed set of target images and source references.

All source images are fused as same-person references for each target. Each model
is warmed up first; measured runs exclude model initialization. The CSV records
median/min/max full-pipeline runtime and ArcFace cosine diagnostics. Visual review
is still required; this is not an automated perceptual-quality score.
"""
import argparse
import csv
import re
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import run_swap  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def images(folder: Path):
    if not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def run_once(sources, target, output, model):
    return run_swap(
        sources, target, output, color_match_strength=0.35,
        blend_strength=0.95, blend_mode="feather",
        restore_face=False, merge_source_references=True,
        swap_model=model,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["inswapper_128", "ghost_1_256"],
                        choices=["inswapper_128", "ghost_1_256"])
    parser.add_argument("--repeat", type=int, default=3,
                        help="Measured repetitions per target/model (default: 3).")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1.")
    try:
        sources, targets = images(args.sources), images(args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    if not sources or not targets:
        parser.error("Both directories must contain at least one supported image.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in args.models:
        warmup_output = args.output / f".warmup-{model}.png"
        warmup_error = ""
        try:
            run_once(sources, targets[0], warmup_output, model)
        except Exception as exc:
            warmup_error = str(exc)
        finally:
            warmup_output.unlink(missing_ok=True)

        for target in targets:
            safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", target.name)
            output = args.output / f"{safe_target}.{model}.png"
            durations, similarities = [], []
            device, error = "", warmup_error
            if not error:
                for _ in range(args.repeat):
                    started = time.perf_counter()
                    try:
                        device, _source_note, similarity = run_once(sources, target, output, model)
                        durations.append(time.perf_counter() - started)
                        if similarity is not None:
                            similarities.append(float(similarity))
                    except Exception as exc:
                        error = str(exc)
                        break

            ok = not error and len(durations) == args.repeat
            rows.append({
                "target": target.name,
                "source_files": " ; ".join(p.name for p in sources),
                "model": model,
                "status": "ok" if ok else "failed",
                "runs": len(durations),
                "median_seconds": round(statistics.median(durations), 3) if durations else "",
                "min_seconds": round(min(durations), 3) if durations else "",
                "max_seconds": round(max(durations), 3) if durations else "",
                "device": device,
                "arcface_cosine_median": round(statistics.median(similarities), 5) if similarities else "",
                "output": str(output) if ok else "",
                "error": error,
            })
            if ok:
                print(f"[ok] {target.name} | {model} | median {statistics.median(durations):.2f}s | ArcFace={statistics.median(similarities) if similarities else 'n/a'}")
            else:
                print(f"[failed] {target.name} | {model}: {error}")

    report = args.output / "results.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {report}")


if __name__ == "__main__":
    main()
