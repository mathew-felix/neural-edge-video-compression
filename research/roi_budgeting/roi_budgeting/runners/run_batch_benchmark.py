from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from roi_budgeting.eval.batch_report import (
    BATCH_AGGREGATE_COLUMNS,
    BATCH_DETAIL_COLUMNS,
    aggregate_batch_rows,
    collect_batch_rows,
    write_csv_report,
    write_markdown_report,
)


def _parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Run ROI-budgeting experiments across a folder of benchmark videos")
    parser.add_argument("--repo-root", type=str, default=str(repo_root_default), help="Repo root")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="research/roi_budgeting/test",
        help="Directory containing benchmark videos",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="research/roi_budgeting/results/benchmark",
        help="Root directory for per-clip benchmark outputs",
    )
    parser.add_argument(
        "--roi-target-kbps",
        type=float,
        default=150.0,
        help="Target ROI bitrate budget for each clip",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Optional limit for a smaller pilot run")
    parser.add_argument("--skip-v1", action="store_true", help="Skip the motion-only experiment")
    parser.add_argument("--skip-v2", action="store_true", help="Skip the motion+uncertainty experiment")
    parser.add_argument("--skip-v3", action="store_true", help="Skip the motion+uncertainty+AMT experiment")
    parser.add_argument("--skip-v4", action="store_true", help="Skip the segment-aware budget-DP AMT experiment")
    parser.add_argument(
        "--allow-proxy",
        action="store_true",
        help="Allow proxy AMT manifests instead of requiring the full AMT probe path",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue benchmarking the remaining clips when one clip fails",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose required output manifests already exist",
    )
    return parser.parse_args()


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _discover_videos(video_dir: Path) -> List[Path]:
    suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(
        path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    )


def _run(cmd: List[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _required_manifest_paths(*, clip_results: Path, args: argparse.Namespace) -> List[Path]:
    manifests_dir = clip_results / "manifests"
    required = [manifests_dir / "offline_eval_summary.json", manifests_dir / "amt_probe_manifest.json"]
    if not args.skip_v1:
        required.append(manifests_dir / "v1_motion_only_summary.json")
    if not args.skip_v2:
        required.append(manifests_dir / "v2_motion_uncertainty_summary.json")
    if not args.skip_v3:
        required.append(manifests_dir / "v3_motion_uncertainty_amt_summary.json")
    if not args.skip_v4:
        required.append(manifests_dir / "v4_segment_dp_amt_summary.json")
    return required


def _clip_is_complete(*, clip_results: Path, args: argparse.Namespace) -> bool:
    return all(path.exists() for path in _required_manifest_paths(clip_results=clip_results, args=args))


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    research_dir = repo_root / "research/roi_budgeting"
    if not research_dir.exists():
        raise FileNotFoundError(f"Research workspace is missing: {research_dir}")

    video_dir = _resolve(repo_root, args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Benchmark video directory does not exist: {video_dir}")

    results_root = _resolve(repo_root, args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    videos = _discover_videos(video_dir)
    if args.max_videos and int(args.max_videos) > 0:
        videos = videos[: int(args.max_videos)]
    if not videos:
        raise RuntimeError(f"No benchmark videos were found in: {video_dir}")

    completed: List[str] = []
    failed: List[Dict[str, str]] = []
    skipped: List[str] = []

    for video_path in videos:
        clip_name = video_path.stem
        clip_results = results_root / clip_name
        runtime_dir = clip_results / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and _clip_is_complete(clip_results=clip_results, args=args):
            skipped.append(str(video_path))
            continue

        cmd = [
            sys.executable,
            "-m",
            "roi_budgeting.runners.run_local_pipeline",
            "--repo-root",
            str(repo_root),
            "--video",
            str(video_path),
            "--results-dir",
            str(clip_results),
            "--runtime-research-config",
            str((runtime_dir / "local_runtime.yaml").resolve()),
            "--runtime-compression-config",
            str((runtime_dir / "local_runtime_compression.yaml").resolve()),
            "--roi-target-kbps",
            str(float(args.roi_target_kbps)),
        ]
        if args.skip_v1:
            cmd.append("--skip-v1")
        if args.skip_v2:
            cmd.append("--skip-v2")
        if args.skip_v3:
            cmd.append("--skip-v3")
        if args.skip_v4:
            cmd.append("--skip-v4")
        if not args.allow_proxy:
            cmd.append("--strict-amt")

        try:
            _run(cmd, cwd=research_dir)
            completed.append(str(video_path))
        except Exception as exc:
            failed.append({"video_path": str(video_path), "error": f"{type(exc).__name__}: {exc}"})
            if not args.continue_on_error:
                raise

    detail_rows = collect_batch_rows(benchmark_root=results_root)
    aggregate_rows = aggregate_batch_rows(rows=detail_rows)

    tables_dir = results_root / "_aggregate"
    detail_csv = write_csv_report(
        rows=detail_rows,
        columns=BATCH_DETAIL_COLUMNS,
        output_path=tables_dir / "clip_experiment_rows.csv",
    )
    detail_md = write_markdown_report(
        rows=detail_rows,
        columns=BATCH_DETAIL_COLUMNS,
        output_path=tables_dir / "clip_experiment_rows.md",
    )
    aggregate_csv = write_csv_report(
        rows=aggregate_rows,
        columns=BATCH_AGGREGATE_COLUMNS,
        output_path=tables_dir / "experiment_aggregate.csv",
    )
    aggregate_md = write_markdown_report(
        rows=aggregate_rows,
        columns=BATCH_AGGREGATE_COLUMNS,
        output_path=tables_dir / "experiment_aggregate.md",
    )

    print(
        json.dumps(
            {
                "video_dir": str(video_dir),
                "results_root": str(results_root),
                "videos_discovered": [str(path) for path in videos],
                "completed_count": int(len(completed)),
                "skipped_count": int(len(skipped)),
                "skipped": skipped,
                "failed": failed,
                "detail_csv": detail_csv,
                "detail_markdown": detail_md,
                "aggregate_csv": aggregate_csv,
                "aggregate_markdown": aggregate_md,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
