#!/usr/bin/env python3
"""
Generate thesis-ready LaTeX tables from TrackEval MOTChallenge output.

This version is intended for the KTP final comparison section. It generates:

  1) Overall tracking/detection quality table.
  2) Overall identity-preservation table.
  3) Per-sequence key-results table grouped by sequence.
  4) Optional manual-review table placeholder or filled rows.

Example:
python demo/make_ktp_latex_tables.py `
  --trackeval_root "C:\\Users\\inesg\\OneDrive\\Desktop\\Thesis\\code\\TrackEval" `
  --benchmark KTP-5Hz `
  --split train `
  --system "SAMURAI-only|samurai_baseline_default|1" `
  --system "TransReID-only|transreid_first_prompt|0" `
  --system "ReID-SAMURAI|reid_samurai_final_selected|1" `
  --out_tex "C:\\tmp\\ktp_final_comparison_tables.tex" `
  --include_manual_placeholder
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRACKER_SUBDIR = Path("data") / "trackers" / "mot_challenge"

DEFAULT_SEQUENCE_ORDER = ["Arc", "Rotation", "Still", "Translation"]

# TrackEval sometimes stores HOTA-family metrics with the ___AUC suffix in
# pedestrian_detailed.csv. The summary TXT usually stores the simpler names.
METRIC_ALIASES = {
    "HOTA": ["HOTA___AUC", "HOTA"],
    "DetA": ["DetA___AUC", "DetA"],
    "AssA": ["AssA___AUC", "AssA"],
    "DetRe": ["DetRe___AUC", "DetRe"],
    "DetPr": ["DetPr___AUC", "DetPr"],
    "AssRe": ["AssRe___AUC", "AssRe"],
    "AssPr": ["AssPr___AUC", "AssPr"],
    "LocA": ["LocA___AUC", "LocA"],
    "MOTA": ["MOTA"],
    "MOTP": ["MOTP"],
    "MODA": ["MODA"],
    "CLR_Re": ["CLR_Re", "CLR_Re___AUC"],
    "CLR_Pr": ["CLR_Pr", "CLR_Pr___AUC"],
    "CLR_TP": ["CLR_TP"],
    "CLR_FP": ["CLR_FP"],
    "CLR_FN": ["CLR_FN"],
    "IDSW": ["IDSW"],
    "Frag": ["Frag"],
    "IDF1": ["IDF1"],
    "IDR": ["IDR"],
    "IDP": ["IDP"],
    "IDTP": ["IDTP"],
    "IDFN": ["IDFN"],
    "IDFP": ["IDFP"],
}

PERCENT_METRICS = {
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA",
    "MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "IDF1", "IDR", "IDP",
}

INTEGER_METRICS = {
    "CLR_TP", "CLR_FP", "CLR_FN", "IDSW", "Frag", "IDTP", "IDFN", "IDFP",
}

TRACKING_QUALITY_METRICS = [
    "HOTA", "DetA", "MOTA", "DetRe", "DetPr", "CLR_FP", "CLR_FN",
]

IDENTITY_METRICS = [
    "AssA", "IDF1", "IDP", "IDR", "IDSW",
]

PER_SEQUENCE_KEY_METRICS = [
    "HOTA", "MOTA", "IDF1", "IDSW",
]

LOWER_IS_BETTER = {
    "IDSW", "CLR_FP", "CLR_FN", "Frag", "IDFN", "IDFP",
}


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class SystemSpec:
    display_name: str
    tracker_name: str
    has_masks: bool
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ManualReviewRow:
    system_name: str
    valid_unannotated_target: Optional[int]
    wrong_identity: Optional[int]
    background_or_bad_mask: Optional[int]

    @property
    def total(self) -> Optional[int]:
        vals = [
            self.valid_unannotated_target,
            self.wrong_identity,
            self.background_or_bad_mask,
        ]
        if any(v is None for v in vals):
            return None
        return int(sum(vals))


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def parse_bool_mask_flag(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "mask", "masks", "cmark"}:
        return True
    if v in {"0", "false", "no", "n", "nomask", "no_mask", "xmark"}:
        return False
    raise ValueError(
        f"Invalid masks flag {value!r}. Use 1/0, true/false, yes/no, masks/nomask."
    )


def parse_system_spec(text: str) -> SystemSpec:
    parts = text.split("|")
    if len(parts) != 3:
        raise ValueError(
            "--system must have the form 'Display name|tracker_folder_name|masks_flag'.\n"
            "Example: --system \"ReID-SAMURAI|reid_samurai_final_selected|1\""
        )

    display_name, tracker_name, masks_flag = [p.strip() for p in parts]
    if not display_name:
        raise ValueError("System display name cannot be empty.")
    if not tracker_name:
        raise ValueError("Tracker folder name cannot be empty.")

    return SystemSpec(
        display_name=display_name,
        tracker_name=tracker_name,
        has_masks=parse_bool_mask_flag(masks_flag),
    )


def parse_optional_int(value: str) -> Optional[int]:
    value = str(value).strip()
    if value in {"", "--", "NA", "N/A", "none", "None"}:
        return None
    return int(value)


def parse_manual_review_row(text: str) -> ManualReviewRow:
    parts = text.split("|")
    if len(parts) != 4:
        raise ValueError(
            "--manual_row must have the form "
            "'System name|valid_unannotated_target|wrong_identity|background_bad_mask'.\n"
            "Example: --manual_row \"ReID-SAMURAI|108|10|0\""
        )

    system_name, valid, wrong, bad = [p.strip() for p in parts]
    return ManualReviewRow(
        system_name=system_name,
        valid_unannotated_target=parse_optional_int(valid),
        wrong_identity=parse_optional_int(wrong),
        background_or_bad_mask=parse_optional_int(bad),
    )


def safe_float(value) -> Optional[float]:
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
    TrackEval normally writes percentage metrics already in [0,100].
    If a value is in [0,1], convert it to [0,100] to make the script robust.
    """
    if value is None:
        return None
    if metric in PERCENT_METRICS and abs(value) <= 1.0000001:
        return value * 100.0
    return value


def normalize_header_name(text: str) -> str:
    text = str(text).strip().replace("\ufeff", "")
    text = text.replace(" ", "")
    text = text.replace("-", "_")
    return text.lower()


def row_get_any_metric(row: Dict[str, str], metric: str) -> Optional[float]:
    aliases = METRIC_ALIASES.get(metric, [metric])

    # 1) Exact raw match.
    for alias in aliases:
        if alias in row:
            val = safe_float(row.get(alias))
            if val is not None:
                return normalize_metric_value(metric, val)

    # 2) Normalized exact match.
    norm_row = {normalize_header_name(k): v for k, v in row.items()}
    for alias in aliases:
        alias_norm = normalize_header_name(alias)
        if alias_norm in norm_row:
            val = safe_float(norm_row.get(alias_norm))
            if val is not None:
                return normalize_metric_value(metric, val)

    # 3) Normalized suffix match.
    for alias in aliases:
        alias_norm = normalize_header_name(alias)
        candidates = [
            (k, v) for k, v in norm_row.items()
            if k.endswith(alias_norm) or k.endswith("__" + alias_norm)
        ]
        for _, v in candidates:
            val = safe_float(v)
            if val is not None:
                return normalize_metric_value(metric, val)

    return None


def get_sequence_name(row: Dict[str, str]) -> str:
    for key in ["seq", "sequence", "Sequence", "SEQ", "seq_name"]:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()

    norm_to_raw = {normalize_header_name(k): k for k in row.keys()}
    for key in ["seq", "sequence", "seq_name"]:
        if key in norm_to_raw:
            value = str(row[norm_to_raw[key]]).strip()
            if value:
                return value

    return ""


def read_detailed_csv(path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing TrackEval detailed CSV: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    results: Dict[str, Dict[str, float]] = {}

    all_metrics = sorted(set(
        TRACKING_QUALITY_METRICS
        + IDENTITY_METRICS
        + PER_SEQUENCE_KEY_METRICS
        + ["CLR_TP", "IDTP", "IDFN", "IDFP", "MOTP", "MODA", "CLR_Re", "CLR_Pr"]
    ))

    for row in rows:
        seq = get_sequence_name(row)
        if not seq:
            continue

        seq_metrics: Dict[str, float] = {}
        for metric in all_metrics:
            value = row_get_any_metric(row, metric)
            if value is not None:
                seq_metrics[metric] = value

        results[seq] = seq_metrics

    return results, fieldnames


def read_summary_txt(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}

    headers = re.split(r"\s+", lines[0])
    values = re.split(r"\s+", lines[1])

    out: Dict[str, float] = {}
    for header, value in zip(headers, values):
        metric_name = None
        for metric, aliases in METRIC_ALIASES.items():
            if normalize_header_name(header) in {normalize_header_name(a) for a in aliases}:
                metric_name = metric
                break

        if metric_name is None:
            metric_name = header

        x = safe_float(value)
        if x is not None:
            out[metric_name] = normalize_metric_value(metric_name, x)

    return out


def tracker_output_dir(
    trackeval_root: Path,
    benchmark: str,
    split: str,
    tracker_name: str,
) -> Path:
    return (
        Path(trackeval_root)
        / TRACKER_SUBDIR
        / f"{benchmark}-{split}"
        / tracker_name
    )


def load_tracker_results(
    trackeval_root: Path,
    benchmark: str,
    split: str,
    tracker_name: str,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    folder = tracker_output_dir(trackeval_root, benchmark, split, tracker_name)
    detailed_path = folder / "pedestrian_detailed.csv"
    summary_path = folder / "pedestrian_summary.txt"

    detailed_results, fieldnames = read_detailed_csv(detailed_path)
    summary_results = read_summary_txt(summary_path)

    # Use pedestrian_summary.txt as the safest source for COMBINED values.
    combined = detailed_results.setdefault("COMBINED", {})
    for metric, value in summary_results.items():
        if value is not None:
            combined[metric] = value

    if verbose:
        print(f"[load] tracker: {tracker_name}")
        print(f"       folder  : {folder}")
        print(f"       detailed: {detailed_path}")
        print(f"       summary : {summary_path}")
        print(f"       sequences in detailed: {list(detailed_results.keys())}")

        has_hota_auc = any(normalize_header_name(c) == "hota___auc" for c in fieldnames)
        has_hota = any(normalize_header_name(c) == "hota" for c in fieldnames)
        if has_hota_auc:
            print("       HOTA-family source: HOTA___AUC / DetA___AUC / AssA___AUC")
        elif has_hota:
            print("       HOTA-family source: HOTA / DetA / AssA")
        else:
            print("       warning: HOTA columns were not obvious from the header")

    return detailed_results


def validate_results(
    systems: Sequence[SystemSpec],
    sequences: Sequence[str],
    need_per_sequence: bool = True,
) -> None:
    missing: List[str] = []

    for system in systems:
        if "COMBINED" not in system.results:
            missing.append(f"{system.display_name}: missing COMBINED row")
            continue

        for metric in TRACKING_QUALITY_METRICS + IDENTITY_METRICS:
            if metric not in system.results.get("COMBINED", {}):
                missing.append(f"{system.display_name}: COMBINED missing {metric}")

        if need_per_sequence:
            for seq in sequences:
                if seq == "COMBINED":
                    continue
                if seq not in system.results:
                    missing.append(f"{system.display_name}: missing sequence {seq}")
                    continue
                for metric in PER_SEQUENCE_KEY_METRICS:
                    if metric not in system.results.get(seq, {}):
                        missing.append(f"{system.display_name}: {seq} missing {metric}")

    if missing:
        msg = [
            "Some required metrics are missing.",
            "",
            "Missing values:",
            *[f"  - {m}" for m in missing],
            "",
            "Check that TrackEval was run with:",
            "  --METRICS HOTA CLEAR Identity",
            "and that pedestrian_detailed.csv and pedestrian_summary.txt exist for each tracker.",
        ]
        raise RuntimeError("\n".join(msg))


# ---------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------

def latex_escape(text: str) -> str:
    if "\\" in text or "$" in text:
        return text

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_mask(has_masks: bool) -> str:
    return r"\cmark" if has_masks else r"\xmark"


def fmt_value(metric: str, value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "--"
    if metric in INTEGER_METRICS:
        return str(int(round(value)))
    return f"{float(value):.{decimals}f}"


def fmt_manual_value(value: Optional[int]) -> str:
    return "--" if value is None else str(int(value))


def metric_header(metric: str) -> str:
    arrows = {
        "HOTA": r"HOTA $\uparrow$",
        "DetA": r"DetA $\uparrow$",
        "AssA": r"AssA $\uparrow$",
        "DetRe": r"DetRe $\uparrow$",
        "DetPr": r"DetPr $\uparrow$",
        "MOTA": r"MOTA $\uparrow$",
        "IDF1": r"IDF1 $\uparrow$",
        "IDP": r"IDP $\uparrow$",
        "IDR": r"IDR $\uparrow$",
        "IDSW": r"IDSW $\downarrow$",
        "CLR_FP": r"FP $\downarrow$",
        "CLR_FN": r"FN $\downarrow$",
        "CLR_TP": r"TP $\uparrow$",
    }
    return arrows.get(metric, latex_escape(metric))


def is_lower_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER


def best_values(
    systems: Sequence[SystemSpec],
    sequences: Sequence[str],
    metrics: Sequence[str],
) -> Dict[Tuple[str, str], float]:
    best: Dict[Tuple[str, str], float] = {}

    for seq in sequences:
        for metric in metrics:
            vals = [
                system.results.get(seq, {}).get(metric)
                for system in systems
            ]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue

            best[(seq, metric)] = min(vals) if is_lower_better(metric) else max(vals)

    return best


def is_best(value: Optional[float], best: Optional[float]) -> bool:
    return value is not None and best is not None and abs(float(value) - float(best)) <= 1e-9


def maybe_best(
    text: str,
    value: Optional[float],
    best: Optional[float],
    enabled: bool = True,
) -> str:
    if enabled and is_best(value, best):
        return rf"\best{{{text}}}"
    return text


def table_column_spec(n_metrics: int, first_cols: str = "lc") -> str:
    return first_cols + ("c" * n_metrics)


# ---------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------

def make_combined_table(
    systems: Sequence[SystemSpec],
    metrics: Sequence[str],
    caption: str,
    label: str,
    decimals: int = 2,
    highlight_best: bool = True,
) -> str:
    best = best_values(systems, ["COMBINED"], metrics)

    lines: List[str] = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{table_column_spec(len(metrics), first_cols='lc')}}}",
        r"\toprule",
        "System & Masks & " + " & ".join(metric_header(m) for m in metrics) + r" \\",
        r"\midrule",
    ]

    for system in systems:
        row = [latex_escape(system.display_name), latex_mask(system.has_masks)]

        for metric in metrics:
            value = system.results.get("COMBINED", {}).get(metric)
            cell = fmt_value(metric, value, decimals)
            cell = maybe_best(cell, value, best.get(("COMBINED", metric)), highlight_best)
            row.append(cell)

        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def make_per_sequence_key_table(
    systems: Sequence[SystemSpec],
    sequences: Sequence[str],
    metrics: Sequence[str] = PER_SEQUENCE_KEY_METRICS,
    decimals: int = 2,
    highlight_best: bool = True,
) -> str:
    """
    Per-sequence table grouped by sequence first, then system.

    This is easier to read for the results chapter because each KTP sequence
    becomes a small local comparison between the subsystem variants and the
    proposed method.
    """
    ordered_sequences = [s for s in sequences if s != "COMBINED"]
    best = best_values(systems, ordered_sequences, metrics)

    lines: List[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Per-sequence KTP results for the key comparison metrics. The best value for each sequence and metric is highlighted in blue.}",
        r"\label{tab:ktp_per_sequence_key_results}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{table_column_spec(len(metrics), first_cols='llc')}}}",
        r"\toprule",
        "Sequence & System & Masks & " + " & ".join(metric_header(m) for m in metrics) + r" \\",
        r"\midrule",
    ]

    for seq_idx, seq in enumerate(ordered_sequences):
        nrows = len(systems)

        for sys_idx, system in enumerate(systems):
            row: List[str] = [
                rf"\multirow{{{nrows}}}{{*}}{{{latex_escape(seq)}}}" if sys_idx == 0 else "",
                latex_escape(system.display_name),
                latex_mask(system.has_masks),
            ]

            for metric in metrics:
                value = system.results.get(seq, {}).get(metric)
                cell = fmt_value(metric, value, decimals)
                cell = maybe_best(cell, value, best.get((seq, metric)), highlight_best)
                row.append(cell)

            lines.append(" & ".join(row) + r" \\")

        if seq_idx != len(ordered_sequences) - 1:
            lines += ["", r"\midrule"]

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def make_manual_review_table(
    systems: Sequence[SystemSpec],
    manual_rows: Sequence[ManualReviewRow],
) -> str:
    """
    Builds the manual-review table.

    If manual_rows is empty, creates a placeholder row for each system.
    If manual_rows is provided, it uses those values.
    """
    row_by_name = {row.system_name: row for row in manual_rows}

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Manual review of predictions excluded from the standard TrackEval evaluation because no sufficiently overlapping same-identity ground-truth box was available.}",
        r"\label{tab:ktp_manual_unannotated_review}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"System & Valid unannotated target & Wrong identity & Background / bad mask & Total reviewed \\",
        r"\midrule",
    ]

    for system in systems:
        row = row_by_name.get(system.display_name)
        if row is None:
            cells = ["--", "--", "--", "--"]
        else:
            cells = [
                fmt_manual_value(row.valid_unannotated_target),
                fmt_manual_value(row.wrong_identity),
                fmt_manual_value(row.background_or_bad_mask),
                fmt_manual_value(row.total),
            ]

        lines.append(
            latex_escape(system.display_name)
            + " & "
            + " & ".join(cells)
            + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def make_all_tables(
    systems: Sequence[SystemSpec],
    sequences: Sequence[str],
    decimals: int = 2,
    highlight_best: bool = True,
    include_manual_placeholder: bool = False,
    manual_rows: Sequence[ManualReviewRow] = (),
) -> str:
    tracking_caption = (
        "Overall KTP tracking and detection quality results for the proposed system "
        "and subsystem variants. Higher values are better for HOTA, DetA, MOTA, "
        "DetRe, and DetPr, while lower values are better for false positives and "
        "false negatives."
    )

    identity_caption = (
        "Overall KTP identity-preservation results for the proposed system and "
        "subsystem variants. Higher values are better for AssA, IDF1, IDP, and IDR, "
        "while lower values are better for IDSW."
    )

    chunks = [
        "% Required packages in the thesis preamble:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{multirow}",
        "% \\usepackage{graphicx}",
        "% \\usepackage{xcolor}",
        "% \\usepackage{amssymb}",
        "%",
        "% Suggested commands in the thesis preamble:",
        "% \\newcommand{\\best}[1]{\\textcolor{blue}{#1}}",
        "% \\newcommand{\\cmark}{\\checkmark}",
        "% \\newcommand{\\xmark}{$\\times$}",
        "",
        make_combined_table(
            systems=systems,
            metrics=TRACKING_QUALITY_METRICS,
            caption=tracking_caption,
            label="tab:ktp_tracking_quality_results",
            decimals=decimals,
            highlight_best=highlight_best,
        ),
        "",
        make_combined_table(
            systems=systems,
            metrics=IDENTITY_METRICS,
            caption=identity_caption,
            label="tab:ktp_identity_preservation_results",
            decimals=decimals,
            highlight_best=highlight_best,
        ),
        "",
        make_per_sequence_key_table(
            systems=systems,
            sequences=sequences,
            metrics=PER_SEQUENCE_KEY_METRICS,
            decimals=decimals,
            highlight_best=highlight_best,
        ),
    ]

    if include_manual_placeholder or manual_rows:
        chunks += [
            "",
            make_manual_review_table(
                systems=systems,
                manual_rows=manual_rows,
            ),
        ]

    return "\n".join(chunks)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_sequence_list(text: str) -> List[str]:
    seqs = [s.strip() for s in text.split(",") if s.strip()]
    return seqs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis LaTeX tables from TrackEval KTP outputs."
    )

    parser.add_argument(
        "--trackeval_root",
        required=True,
        help="Path to the TrackEval repository.",
    )
    parser.add_argument(
        "--benchmark",
        default="KTP-5Hz",
        help="TrackEval benchmark name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="TrackEval split name.",
    )

    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help=(
            "System specification in the form 'Display name|tracker_folder_name|masks_flag'. "
            "Example: --system \"ReID-SAMURAI|reid_samurai_final_selected|1\""
        ),
    )

    parser.add_argument(
        "--sequences",
        default="Arc,Rotation,Still,Translation",
        help="Comma-separated sequence order to include in the per-sequence table.",
    )

    parser.add_argument(
        "--out_tex",
        required=True,
        help="Output .tex file.",
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
        "--include_manual_placeholder",
        action="store_true",
        help="Also add a manual-review table. Missing values become '--'.",
    )
    parser.add_argument(
        "--manual_row",
        action="append",
        default=[],
        help=(
            "Optional manual-review row in the form "
            "'System name|valid_unannotated_target|wrong_identity|background_bad_mask'. "
            "Example: --manual_row \"ReID-SAMURAI|108|10|0\""
        ),
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
    systems = [parse_system_spec(s) for s in args.system]
    sequences = parse_sequence_list(args.sequences)
    manual_rows = [parse_manual_review_row(r) for r in args.manual_row]

    for system in systems:
        system.results = load_tracker_results(
            trackeval_root=trackeval_root,
            benchmark=args.benchmark,
            split=args.split,
            tracker_name=system.tracker_name,
            verbose=not args.quiet,
        )

    if not args.allow_missing:
        validate_results(
            systems=systems,
            sequences=sequences,
            need_per_sequence=True,
        )

    latex = make_all_tables(
        systems=systems,
        sequences=sequences,
        decimals=args.decimals,
        highlight_best=not args.no_highlight_best,
        include_manual_placeholder=args.include_manual_placeholder,
        manual_rows=manual_rows,
    )

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex, encoding="utf-8")

    print(f"[ok] wrote LaTeX tables to: {out_tex}")
    print("")
    print("Use in the thesis with:")
    print(f"  \\input{{{out_tex.as_posix()}}}")
    print("")
    print("Make sure the thesis preamble includes:")
    print(r"  \usepackage{booktabs}")
    print(r"  \usepackage{multirow}")
    print(r"  \usepackage{graphicx}")
    print(r"  \usepackage{xcolor}")
    print(r"  \usepackage{amssymb}")
    print(r"  \newcommand{\best}[1]{\textcolor{blue}{#1}}")
    print(r"  \newcommand{\cmark}{\checkmark}")
    print(r"  \newcommand{\xmark}{$\times$}")


if __name__ == "__main__":
    main()