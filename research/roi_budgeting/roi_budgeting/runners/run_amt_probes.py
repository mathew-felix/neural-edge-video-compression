from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from roi_budgeting.data.load_frame_drop import load_frame_drop
from roi_budgeting.data.load_roi_detections import load_roi_detections
from roi_budgeting.signals.amt_risk import (
    build_amt_risk_proxy_features,
    generate_amt_probe_manifest,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the research configs. Install requirements.txt first.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _resolve_path(raw_path: str | None, *, cfg_path: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidates = [
        (Path.cwd() / p).resolve(),
        (cfg_path.parent / p).resolve(),
        (cfg_path.parent.parent / p).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROI AMT probe manifest")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Path to a research config file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output manifest path override",
    )
    parser.add_argument(
        "--allow-proxy",
        action="store_true",
        help="Allow a non-CUDA local proxy manifest when GPU AMT probing is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    amt_cfg = (signals_cfg.get("amt_risk", {}) or {})

    repo_root = _resolve_path(paths_cfg.get("repo_root", "../.."), cfg_path=cfg_path)
    video_path = _resolve_path(paths_cfg.get("video_path", None), cfg_path=cfg_path)
    roi_path = _resolve_path(paths_cfg.get("roi_detections_json", None), cfg_path=cfg_path)
    frame_drop_path = _resolve_path(paths_cfg.get("frame_drop_json", None), cfg_path=cfg_path)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _resolve_path(paths_cfg.get("amt_probe_manifest", "results/manifests/amt_probe_manifest.json"), cfg_path=cfg_path)
    )

    if repo_root is None:
        raise RuntimeError("Could not resolve repo_root from config.")
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Configured video_path is missing.")
    if roi_path is None or not roi_path.exists():
        raise FileNotFoundError("Configured roi_detections_json is missing.")
    if frame_drop_path is None or not frame_drop_path.exists():
        raise FileNotFoundError("Configured frame_drop_json is missing.")
    if output_path is None:
        raise RuntimeError("Could not resolve amt_probe_manifest output path.")

    roi_payload = load_roi_detections(roi_path)
    frame_drop_json = load_frame_drop(frame_drop_path)

    try:
        manifest = generate_amt_probe_manifest(
            video_path=video_path,
            roi_payload=roi_payload,
            frame_drop_json=frame_drop_json,
            repo_root=repo_root,
            amt_cfg=amt_cfg,
        )
    except Exception as exc:
        if not args.allow_proxy:
            raise
        proxy_features, proxy_meta = build_amt_risk_proxy_features(
            video_path=video_path,
            roi_payload=roi_payload,
            crop_margin_px=int(amt_cfg.get("crop_margin_px", 8) or 8),
            max_crop_side=int(amt_cfg.get("max_crop_side", 256) or 256),
        )
        manifest = {
            "meta": {
                **proxy_meta,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            },
            "per_frame": {str(k): v for k, v in proxy_features.items()},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "meta": manifest.get("meta", {})}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
