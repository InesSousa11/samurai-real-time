#!/usr/bin/env python3
"""
Generate a LaTeX table for the temporal sampling / deployment-rate sensitivity
experiment from TrackEval MOTChallenge outputs.

It reads TrackEval pedestrian_summary.txt files for different stride runs and
creates a thesis-ready table.

Example:
python demo/make_temporal_sampling_table.py `
  --trackeval_root "C:\\Users\\inesg\\OneDrive\\Desktop\\Thesis\\code\\TrackEval" `
  --split train `
  --run "1|30.0|KTP-30Hz|reid_samurai_stride_s1" `
  --run "2|15.0|KTP-30Hz|reid_samurai_stride_s2" `
  --run "3|10.0|KTP-30Hz|reid_samurai_stride_s3" `
  --run "6|5.0|KTP-30Hz|reid_samurai_stride_s6" `
  --run "10|3.0|KTP-30Hz|reid_samurai_stride_s10" `
  --out_tex "C:\\tmp\\temporal_sampling_results.tex"
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TRACKER_SUBDIR = Path("data") / "trackers" / "mot_challenge"

METRIC_ALIASES = {
    "HOTA": ["HOTA___AUC", "HOTA"],
    "DetA": ["DetA___AUC", "DetA"],
    "AssA": ["AssA___AUC", "AssA"],
    "MOTA": ["MOTA"],
    "IDF1": ["IDF1"],
    "IDR": ["IDR"],
    "IDSW": ["IDSW"],
    "CLR_FP": ["CLR_FP"],
    "CLR_FN": ["CLR_FN"],
}

PERCENT_METRICS = {
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDR",
}

INTEGER_METRICS = {
    "IDSW", "CLR_FP", "CLR_FN",
}

LOWER_IS_BETTER = {
    "IDSW", "CLR_FP", "CLR_FN",
}

TABLE_METRICS = [
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDR", "IDSW",
]


@dataclass
class StrideRun:
    stride: int
    effective_fps: float
    benchmark: str
    tracker_name: str
    metrics: Dict[str, float] = field(default_factory=dict)


def normalize_header_name(text: str) -> str:
    text = str(text).strip().replace("\ufeff", "")
    text = text.replace(" ", "")
    text = text.replace("-", "_")
    return text.lower()


def safe_float(value: str) -> Optional[float]:
    if value is None:
        return None

    value = str(value).strip()
    if value == "":
        return None

    try:
        x = float(value)
    except Exception:
        return None

    if not math.isfinite(x):
        return None

    return x


def normalize_metric_value(metric: str, value: Optional[float]) -> Optional[float]:
    """
    TrackEval usually stores percentage metrics in [0, 100].
    If values are in [0, 1], convert them to [0, 100].
    """
    if value is None:
        return None

    if metric in PERCENT_METRICS and abs(value) <= 1.0000001:
        return value * 100.0

    return value


def metric_from_header(header: str) -> Optional[str]:
    header_norm = normalize_header_name(header)

    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if header_norm == normalize_header_name(alias):
                return metric

    return None


def read_summary_txt(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing TrackEval summary file: {path}")

    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    if len(lines) < 2:
        raise RuntimeError(f"TrackEval summary file does not contain enough lines: {path}")

    headers = re.split(r"\s+", lines[0])
    values = re.split(r"\s+", lines[1])

    metrics: Dict[str, float] = {}

    for header, value_text in zip(headers, values):
        metric = metric_from_header(header)
        if metric is None:
            continue

        value = safe_float(value_text)
        value = normalize_metric_value(metric, value)

        if value is not None:
            metrics[metric] = value

    return metrics


def tracker_summary_path(
    trackeval_root: Path,
    benchmark: str,
    split: str,
    tracker_name: str,
) -> Path:
    return (
        trackeval_root
        / TRACKER_SUBDIR
        / f"{benchmark}-{split}"
        / tracker_name
        / "pedestrian_summary.txt"
    )


def parse_run_spec(text: str, default_original_fps: float) -> StrideRun:
    """
    Accepted formats:

      stride|effective_fps|benchmark|tracker_name
      stride|benchmark|tracker_name

    If effective_fps is omitted, it is computed as original_fps / stride.
    """
    parts = [p.strip() for p in text.split("|")]

    if len(parts) == 4:
        stride_text, fps_text, benchmark, tracker_name = parts
        stride = int(stride_text)
        effective_fps = float(fps_text)
    elif len(parts) == 3:
        stride_text, benchmark, tracker_name = parts
        stride = int(stride_text)
        effective_fps = default_original_fps / float(stride)
    else:
        raise ValueError(
            "--run must have one of these forms:\n"
            "  'stride|effective_fps|benchmark|tracker_name'\n"
            "  'stride|benchmark|tracker_name'\n"
            "Examples:\n"
            "  --run \"6|5.0|KTP-30Hz|reid_samurai_stride_s6\"\n"
            "  --run \"6|KTP-30Hz|reid_samurai_stride_s6\""
        )

    if not benchmark:
        raise ValueError("Benchmark name cannot be empty.")

    if not tracker_name:
        raise ValueError("Tracker name cannot be empty.")

    return StrideRun(
        stride=stride,
        effective_fps=effective_fps,
        benchmark=benchmark,
        tracker_name=tracker_name,
    )


def fmt_metric(metric: str, value: Optional[float], decimals: int) -> str:
    if value is None:
        return "--"

    if metric in INTEGER_METRICS:
        return str(int(round(value)))

    return f"{float(value):.{decimals}f}"


def fmt_fps(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{value:.1f}"
    return f"{value:.2f}"


def metric_header(metric: str) -> str:
    headers = {
        "HOTA": r"HOTA $\uparrow$",
        "DetA": r"DetA $\uparrow$",
        "AssA": r"AssA $\uparrow$",
        "MOTA": r"MOTA $\uparrow$",
        "IDF1": r"IDF1 $\uparrow$",
        "IDR": r"IDR $\uparrow$",
        "IDSW": r"IDSW $\downarrow$",
        "CLR_FP": r"FP $\downarrow$",
        "CLR_FN": r"FN $\downarrow$",
    }
    return headers.get(metric, metric)


def is_lower_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER


def best_values(runs: Sequence[StrideRun], metrics: Sequence[str]) -> Dict[str, float]:
    best: Dict[str, float] = {}

    for metric in metrics:
        vals = [
            run.metrics.get(metric)
            for run in runs
            if run.metrics.get(metric) is not None
        ]

        if not vals:
            continue

        best[metric] = min(vals) if is_lower_better(metric) else max(vals)

    return best


def is_best(value: Optional[float], best: Optional[float]) -> bool:
    if value is None or best is None:
        return False
    return abs(float(value) - float(best)) <= 1e-9


def maybe_best(text: str, value: Optional[float], best: Optional[float], enabled: bool) -> str:
    if enabled and is_best(value, best):
        return rf"\best{{{text}}}"
    return text


def validate_runs(runs: Sequence[StrideRun], metrics: Sequence[str]) -> None:
    missing: List[str] = []

    for run in runs:
        for metric in metrics:
            if metric not in run.metrics:
                missing.append(
                    f"stride {run.stride}, tracker {run.tracker_name}: missing {metric}"
                )

    if missing:
        msg = [
            "Some required metrics are missing from the TrackEval summaries.",
            "",
            *[f"  - {m}" for m in missing],
            "",
            "Check that TrackEval completed successfully and that pedestrian_summary.txt exists.",
        ]
        raise RuntimeError("\n".join(msg))


def make_latex_table(
    runs: Sequence[StrideRun],
    metrics: Sequence[str],
    decimals: int,
    highlight_best: bool,
) -> str:
    ordered_runs = sorted(runs, key=lambda r: r.stride)
    best = best_values(ordered_runs, metrics)

    col_spec = "cc" + ("c" * len(metrics))

    lines: List[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Effect of temporal sampling on ReID-SAMURAI performance. The effective frame rate is estimated from the original 30~Hz KTP recordings. Higher values are better for HOTA, DetA, AssA, MOTA, IDF1, and IDR, while lower values are better for IDSW.}",
        r"\label{tab:temporal_sampling_results}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Stride & Effective FPS & " + " & ".join(metric_header(m) for m in metrics) + r" \\",
        r"\midrule",
    ]

    for run in ordered_runs:
        row = [
            str(run.stride),
            fmt_fps(run.effective_fps),
        ]

        for metric in metrics:
            value = run.metrics.get(metric)
            cell = fmt_metric(metric, value, decimals)
            cell = maybe_best(cell, value, best.get(metric), highlight_best)
            row.append(cell)

        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate temporal sampling LaTeX table from TrackEval outputs."
    )

    parser.add_argument(
        "--trackeval_root",
        required=True,
        help="Path to the TrackEval repository.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="TrackEval split name.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Stride run specification. Accepted forms: "
            "'stride|effective_fps|benchmark|tracker_name' or "
            "'stride|benchmark|tracker_name'."
        ),
    )
    parser.add_argument(
        "--out_tex",
        required=True,
        help="Output .tex file.",
    )
    parser.add_argument(
        "--original_fps",
        type=float,
        default=30.0,
        help="Original dataset FPS, used only if effective FPS is omitted from --run.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for non-integer metrics.",
    )
    parser.add_argument(
        "--no_highlight_best",
        action="store_true",
        help="Disable \\best{} highlighting.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Do not fail if some metrics are missing; missing cells become '--'.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print less information.",
    )

    args = parser.parse_args()

    trackeval_root = Path(args.trackeval_root)
    runs = [
        parse_run_spec(text, default_original_fps=args.original_fps)
        for text in args.run
    ]

    for run in runs:
        summary_path = tracker_summary_path(
            trackeval_root=trackeval_root,
            benchmark=run.benchmark,
            split=args.split,
            tracker_name=run.tracker_name,
        )

        if not args.quiet:
            print(f"[load] stride={run.stride}, fps={run.effective_fps}")
            print(f"       benchmark: {run.benchmark}-{args.split}")
            print(f"       tracker  : {run.tracker_name}")
            print(f"       summary  : {summary_path}")

        run.metrics = read_summary_txt(summary_path)

    if not args.allow_missing:
        validate_runs(runs, TABLE_METRICS)

    latex = make_latex_table(
        runs=runs,
        metrics=TABLE_METRICS,
        decimals=args.decimals,
        highlight_best=not args.no_highlight_best,
    )

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex, encoding="utf-8")

    print(f"[ok] wrote LaTeX table to: {out_tex}")
    print("")
    print("Use in the thesis with:")
    print(f"  \\input{{{out_tex.as_posix()}}}")
    print("")
    print("Make sure the thesis preamble includes:")
    print(r"  \usepackage{booktabs}")
    print(r"  \usepackage{graphicx}")
    print(r"  \usepackage{xcolor}")
    print(r"  \newcommand{\best}[1]{\textcolor{blue}{#1}}")


if __name__ == "__main__":
    main()