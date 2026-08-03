"""
package_results.py — run via `make package` (automatically chained after
`make train-reconstructor` in `make all`).

Zips everything under results/ into results.zip so you only need to send
back a single file for me to analyze -- the router's confusion matrix and
history, the reconstructor's loss curve and qualitative samples, and both
model checkpoints (small enough to include; skip that by passing
--no-checkpoints if bandwidth is a concern).
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="./results")
    ap.add_argument("--out", type=str, default="./results.zip")
    ap.add_argument("--no-checkpoints", action="store_true",
                     help="exclude the (larger) .pt model files, keep only logs/metrics/samples")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not any(results_dir.iterdir()):
        raise SystemExit(
            f"{results_dir} is empty or missing -- run the training steps first "
            "(make train-router && make train-reconstructor)."
        )

    if args.no_checkpoints:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_results = Path(tmp) / "results"
            shutil.copytree(results_dir, tmp_results, ignore=shutil.ignore_patterns("*.pt"))
            archive = shutil.make_archive(str(Path(args.out).with_suffix("")), "zip", root_dir=tmp, base_dir="results")
    else:
        archive = shutil.make_archive(
            str(Path(args.out).with_suffix("")), "zip",
            root_dir=str(results_dir.parent), base_dir=results_dir.name,
        )

    size_mb = Path(archive).stat().st_size / (1024 * 1024)
    print(f"Packaged: {archive} ({size_mb:.1f} MB)")
    print("Send this results.zip back -- I'll read the metrics/samples and tell you what to do next.")


if __name__ == "__main__":
    main()
