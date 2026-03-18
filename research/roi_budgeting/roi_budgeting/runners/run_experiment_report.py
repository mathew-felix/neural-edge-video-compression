from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from roi_budgeting.eval.plots import save_experiment_comparison_plot
from roi_budgeting.eval.report import build_report_rows, write_csv_report, write_markdown_report


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
    parser = argparse.ArgumentParser(description="Generate a compact ROI budgeting comparison table and plot")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Path to a research config file")
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default=None,
        help="Optional manifests directory override",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional CSV output path override",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        help="Optional Markdown output path override",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Optional plot output path override",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}

    output_dir = _resolve_path(paths_cfg.get("output_dir", "results"), cfg_path=cfg_path)
    if output_dir is None:
        raise RuntimeError("Could not resolve output_dir from config.")

    manifests_dir = (
        Path(args.manifests_dir).expanduser().resolve()
        if args.manifests_dir
        else (output_dir / "manifests").resolve()
    )
    csv_output = (
        Path(args.csv_output).expanduser().resolve()
        if args.csv_output
        else (output_dir / "tables" / "experiment_comparison.csv").resolve()
    )
    markdown_output = (
        Path(args.markdown_output).expanduser().resolve()
        if args.markdown_output
        else (output_dir / "tables" / "experiment_comparison.md").resolve()
    )
    plot_output = (
        Path(args.plot_output).expanduser().resolve()
        if args.plot_output
        else (output_dir / "plots" / "experiment_comparison.png").resolve()
    )

    rows = build_report_rows(manifests_dir=manifests_dir)
    csv_path = write_csv_report(rows=rows, output_path=csv_output)
    markdown_path = write_markdown_report(rows=rows, output_path=markdown_output)
    plot_path = save_experiment_comparison_plot(
        rows=rows,
        title="ROI Budgeting Experiment Comparison",
        output_path=plot_output,
    )

    print(
        json.dumps(
            {
                "manifests_dir": str(manifests_dir),
                "csv_output": csv_path,
                "markdown_output": markdown_path,
                "plot_output": plot_path,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
