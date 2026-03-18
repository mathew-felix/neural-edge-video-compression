from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the research configs. Install requirements.txt first.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the research configs. Install requirements.txt first.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Run the isolated ROI-budgeting flow locally on a non-Colab machine")
    parser.add_argument("--repo-root", type=str, default=str(repo_root_default), help="Repo root")
    parser.add_argument(
        "--video",
        type=str,
        default="research/roi_budgeting/video.mp4",
        help="Absolute or repo-relative path to the input test clip",
    )
    parser.add_argument(
        "--roi-target-kbps",
        type=float,
        default=150.0,
        help="Target ROI bitrate budget for offline experiments",
    )
    parser.add_argument(
        "--research-config-template",
        type=str,
        default="research/roi_budgeting/configs/local.yaml",
        help="Template research config to materialize into a runtime config",
    )
    parser.add_argument(
        "--compression-config-template",
        type=str,
        default="research/roi_budgeting/configs/compression_local_cpu.yaml",
        help="Template compression config used for local baseline frame-removal artifacts",
    )
    parser.add_argument(
        "--runtime-research-config",
        type=str,
        default="research/roi_budgeting/configs/local_runtime.yaml",
        help="Output path for the generated runtime research config",
    )
    parser.add_argument(
        "--runtime-compression-config",
        type=str,
        default="research/roi_budgeting/configs/local_runtime_compression.yaml",
        help="Output path for the generated runtime compression config",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="research/roi_budgeting/results",
        help="Root results directory for baseline artifacts and experiment manifests",
    )
    parser.add_argument("--skip-v1", action="store_true", help="Skip the motion-only experiment")
    parser.add_argument("--skip-v2", action="store_true", help="Skip the motion+uncertainty experiment")
    parser.add_argument("--skip-v3", action="store_true", help="Skip the motion+uncertainty+AMT experiment")
    parser.add_argument(
        "--strict-amt",
        action="store_true",
        help="Require AMT probing to succeed instead of falling back to the labeled local proxy",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root does not exist: {repo_root}")

    research_dir = repo_root / "research/roi_budgeting"
    if not research_dir.exists():
        raise FileNotFoundError(f"Research workspace is missing: {research_dir}")

    video_path = _resolve(repo_root, args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    results_dir = _resolve(repo_root, args.results_dir)
    baseline_dir = results_dir / "baseline/frame_removal"
    manifests_dir = results_dir / "manifests"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    compression_template_path = _resolve(repo_root, args.compression_config_template)
    research_template_path = _resolve(repo_root, args.research_config_template)
    runtime_compression_path = _resolve(repo_root, args.runtime_compression_config)
    runtime_research_path = _resolve(repo_root, args.runtime_research_config)

    compression_cfg = _load_yaml(compression_template_path)
    compression_cfg.setdefault("output", {})
    compression_cfg["output"]["out_dir"] = str((results_dir / "baseline/compression").resolve())
    _write_yaml(runtime_compression_path, compression_cfg)

    research_cfg = _load_yaml(research_template_path)
    research_cfg.setdefault("paths", {})
    research_cfg["paths"]["repo_root"] = str(repo_root)
    research_cfg["paths"]["output_dir"] = str(results_dir)
    research_cfg["paths"]["video_path"] = str(video_path)
    research_cfg["paths"]["roi_detections_json"] = str((baseline_dir / "roi_detections.json").resolve())
    research_cfg["paths"]["frame_drop_json"] = str((baseline_dir / "frame_drop.json").resolve())
    research_cfg["paths"]["amt_probe_manifest"] = str((manifests_dir / "amt_probe_manifest.json").resolve())
    research_cfg.setdefault("budget", {})
    research_cfg["budget"]["roi_target_kbps"] = float(args.roi_target_kbps)
    _write_yaml(runtime_research_path, research_cfg)

    _run(
        [
            sys.executable,
            "scripts/test_frame_removal.py",
            "--config",
            str(runtime_compression_path),
            "--video",
            str(video_path),
            "--out-dir",
            str(baseline_dir),
            "--no-viz",
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "roi_budgeting.runners.run_offline_eval",
            "--config",
            str(runtime_research_path),
        ],
        cwd=research_dir,
    )

    if not args.skip_v1:
        _run(
            [
                sys.executable,
                "-m",
                "roi_budgeting.runners.run_offline_eval",
                "--config",
                str(runtime_research_path),
                "--experiment-config",
                "configs/experiments/v1_motion_only.yaml",
            ],
            cwd=research_dir,
        )

    if not args.skip_v2:
        _run(
            [
                sys.executable,
                "-m",
                "roi_budgeting.runners.run_offline_eval",
                "--config",
                str(runtime_research_path),
                "--experiment-config",
                "configs/experiments/v2_motion_uncertainty.yaml",
            ],
            cwd=research_dir,
        )

    amt_probe_cmd = [
        sys.executable,
        "-m",
        "roi_budgeting.runners.run_amt_probes",
        "--config",
        str(runtime_research_path),
    ]
    if not args.strict_amt:
        amt_probe_cmd.append("--allow-proxy")
    _run(amt_probe_cmd, cwd=research_dir)

    if not args.skip_v3:
        _run(
            [
                sys.executable,
                "-m",
                "roi_budgeting.runners.run_offline_eval",
                "--config",
                str(runtime_research_path),
                "--experiment-config",
                "configs/experiments/v3_motion_uncertainty_amt.yaml",
            ],
            cwd=research_dir,
        )

    summary = {
        "repo_root": str(repo_root),
        "video_path": str(video_path),
        "results_dir": str(results_dir),
        "runtime_research_config": str(runtime_research_path),
        "runtime_compression_config": str(runtime_compression_path),
        "baseline_dir": str(baseline_dir),
        "amt_probe_manifest": str((manifests_dir / "amt_probe_manifest.json").resolve()),
        "manifests": {
            "baseline": str((manifests_dir / "offline_eval_summary.json").resolve()),
            "v1": str((manifests_dir / "v1_motion_only_summary.json").resolve()),
            "v2": str((manifests_dir / "v2_motion_uncertainty_summary.json").resolve()),
            "v3": str((manifests_dir / "v3_motion_uncertainty_amt_summary.json").resolve()),
        },
        "amt_mode": "strict" if args.strict_amt else "proxy_fallback_allowed",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
