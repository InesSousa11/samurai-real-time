#!/usr/bin/env python3
# make_thesis_results.py
#
# Reads multiple KTP full-system evaluation JSON files and generates:
#   - combined CSV
#   - best-per-backend summary
#   - threshold sweep summary
#   - per-sequence summary for best configs
#   - LaTeX tables
#   - thesis-ready plots
#
# Expected input files:
#   *_summary.json
#
# Example:
# python .\demo\make_thesis_results.py ^
#   --input_dir "C:\Users\inesg\OneDrive\Desktop\Thesis\datasets\full_model_eval_final3" ^
#   --output_dir "C:\Users\inesg\OneDrive\Desktop\Thesis\datasets\thesis_results"

import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------------- IO helpers ----------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_summary_jsons(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*_summary.json"))
    return [p for p in files if p.is_file()]


# ---------------- parsing ----------------
def extract_rows_from_json(js: Dict[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    rows = js.get("rows_flat", [])
    out = []
    for r in rows:
        rr = dict(r)
        rr["_source_json"] = str(source_path)
        rr["_run_created_at"] = js.get("created_at", None)
        rr["_run_name"] = js.get("run", None)
        rr["_label"] = js.get("label", None)
        out.append(rr)
    return out


def build_dataframe(json_files: List[Path]) -> pd.DataFrame:
    rows = []
    for fp in json_files:
        try:
            js = load_json(fp)
            rows.extend(extract_rows_from_json(js, fp))
        except Exception as e:
            print(f"[warn] failed to parse {fp.name}: {repr(e)}")

    if not rows:
        raise RuntimeError("No valid rows found in JSON files.")

    df = pd.DataFrame(rows)

    numeric_cols = [
        "stable_frames_threshold",
        "stable_ious_threshold",
        "min_obj_score_logits",
        "kf_score_weight",
        "memory_bank_iou_threshold",
        "memory_bank_obj_score_threshold",
        "memory_bank_kf_score_threshold",
        "reid_thr",
        "frames",
        "gt_boxes",
        "matches",
        "misses",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "mota",
        "match_rate",
        "miss_rate",
        "id_switches",
        "id_switches_per_match",
        "id_switches_per_gt",
        "reacq_events",
        "reacq_rate_per_gt",
        "reacq_mean_frames",
        "reacq_median_frames",
        "reacq_max_frames",
        "mean_iou_when_matched",
        "total_unique_gt_ids",
        "seeded_ids_count",
        "seed_coverage",
        "seed_skipped_small",
        "seed_skipped_overlap",
        "seed_failed",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------- selection logic ----------------
def choose_best_runs(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Choose the best ALL-row per backend.
    Priority:
      1) highest MOTA
      2) lowest ID switches
      3) highest recall
      4) highest precision
    """
    df = df_all[df_all["seq"] == "ALL"].copy()

    if df.empty:
        raise RuntimeError("No rows with seq == 'ALL' found.")

    sort_cols = ["reid_backend", "mota", "id_switches", "recall", "precision"]
    ascending = [True, False, True, False, False]
    df = df.sort_values(sort_cols, ascending=ascending)

    best = df.groupby("reid_backend", as_index=False).first()
    return best


def build_threshold_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all[df_all["seq"] == "ALL"].copy()
    keep_cols = [
        "run",
        "reid_backend",
        "reid_thr",
        "mota",
        "id_switches",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "reacq_events",
        "reacq_mean_frames",
        "mean_iou_when_matched",
        "_source_json",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].sort_values(["reid_backend", "reid_thr"], ascending=[True, True])


def build_best_per_backend_table(best_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "reid_backend",
        "reid_thr",
        "mota",
        "id_switches",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "reacq_events",
        "reacq_mean_frames",
        "mean_iou_when_matched",
        "seed_coverage",
        "run",
    ]
    keep_cols = [c for c in keep_cols if c in best_df.columns]
    return best_df[keep_cols].copy()


def build_per_sequence_best_table(df_all: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    runs = set(best_df["run"].tolist())
    df = df_all[df_all["run"].isin(runs)].copy()
    df = df[df["seq"] != "ALL"].copy()

    keep_cols = [
        "run",
        "reid_backend",
        "seq",
        "reid_thr",
        "mota",
        "id_switches",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "reacq_events",
        "reacq_mean_frames",
        "mean_iou_when_matched",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].sort_values(["reid_backend", "seq"])


# ---------------- latex helpers ----------------
def _fmt(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if math.isfinite(x):
            return f"{x:.{digits}f}"
        return "-"
    return str(x)


def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    digits_map: Optional[Dict[str, int]] = None,
) -> str:
    if digits_map is None:
        digits_map = {}

    cols = list(df.columns)

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    latex.append("\\hline")
    latex.append(" & ".join(cols) + " \\\\")
    latex.append("\\hline")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            digits = digits_map.get(c, 3)
            vals.append(_fmt(row[c], digits))
        latex.append(" & ".join(vals) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table}")
    return "\n".join(latex)


# ---------------- plots ----------------
def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
):
    plt.figure(figsize=(6.2, 4.2))

    for backend in sorted(df["reid_backend"].dropna().unique()):
        sub = df[df["reid_backend"] == backend].sort_values(x_col)
        plt.plot(sub[x_col], sub[y_col], marker="o", label=backend)

    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_barplot_per_sequence(
    df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
):
    plt.figure(figsize=(7.0, 4.4))

    sequences = ["Arc", "Rotation", "Still", "Translation"]
    backends = sorted(df["reid_backend"].dropna().unique())

    width = 0.35
    x = list(range(len(sequences)))

    for bi, backend in enumerate(backends):
        sub = df[df["reid_backend"] == backend].copy()
        vals = []
        for seq in sequences:
            row = sub[sub["seq"] == seq]
            vals.append(float(row.iloc[0][metric_col]) if not row.empty else float("nan"))

        offset = [-width / 2, width / 2][bi] if len(backends) == 2 else (bi - (len(backends)-1)/2) * width
        xpos = [xx + offset for xx in x]
        plt.bar(xpos, vals, width=width, label=backend)

    plt.xticks(x, sequences)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ---------------- main generation ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing *_summary.json files")
    ap.add_argument("--output_dir", type=str, required=True, help="Folder where tables/plots will be saved")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    safe_mkdir(output_dir)
    plots_dir = output_dir / "plots"
    latex_dir = output_dir / "latex"
    tables_dir = output_dir / "tables"
    safe_mkdir(plots_dir)
    safe_mkdir(latex_dir)
    safe_mkdir(tables_dir)

    json_files = find_summary_jsons(input_dir)
    if not json_files:
        raise RuntimeError(f"No *_summary.json files found in {input_dir}")

    print(f"[info] found {len(json_files)} summary JSON files")

    df_all = build_dataframe(json_files)

    combined_csv = tables_dir / "all_runs_combined.csv"
    df_all.to_csv(combined_csv, index=False)

    threshold_df = build_threshold_summary(df_all)
    threshold_csv = tables_dir / "threshold_sweep_summary.csv"
    threshold_df.to_csv(threshold_csv, index=False)

    best_df = choose_best_runs(df_all)
    best_table = build_best_per_backend_table(best_df)
    best_csv = tables_dir / "best_per_backend.csv"
    best_table.to_csv(best_csv, index=False)

    per_seq_best_df = build_per_sequence_best_table(df_all, best_df)
    per_seq_csv = tables_dir / "best_per_backend_per_sequence.csv"
    per_seq_best_df.to_csv(per_seq_csv, index=False)

    # -------- LaTeX tables --------
    digits_map = {
        "reid_thr": 2,
        "mota": 3,
        "precision": 3,
        "recall": 3,
        "reacq_mean_frames": 2,
        "mean_iou_when_matched": 3,
        "seed_coverage": 2,
    }

    # Table 1: best backend comparison
    latex_best = dataframe_to_latex_table(
        best_table,
        caption="Comparison of the best full-system configuration obtained for each ReID backend.",
        label="tab:fullsystem_best_backend",
        digits_map=digits_map,
    )
    (latex_dir / "table_best_backend.tex").write_text(latex_best, encoding="utf-8")

    # Table 2: threshold sweep
    threshold_table_for_latex = threshold_df.copy()
    threshold_table_for_latex = threshold_table_for_latex[
        [c for c in [
            "reid_backend", "reid_thr", "mota", "id_switches", "precision",
            "recall", "false_positives", "false_negatives",
            "reacq_events", "reacq_mean_frames", "mean_iou_when_matched"
        ] if c in threshold_table_for_latex.columns]
    ]
    latex_thr = dataframe_to_latex_table(
        threshold_table_for_latex,
        caption="Threshold sweep results for the evaluated ReID backends in the full system.",
        label="tab:threshold_sweep_fullsystem",
        digits_map=digits_map,
    )
    (latex_dir / "table_threshold_sweep.tex").write_text(latex_thr, encoding="utf-8")

    # Table 3: per-sequence results for best configs
    latex_perseq = dataframe_to_latex_table(
        per_seq_best_df,
        caption="Per-sequence results for the best configuration of each ReID backend.",
        label="tab:per_sequence_best_backends",
        digits_map=digits_map,
    )
    (latex_dir / "table_per_sequence_best.tex").write_text(latex_perseq, encoding="utf-8")

    # -------- Plots --------
    df_all_allseq = df_all[df_all["seq"] == "ALL"].copy()

    if "mota" in df_all_allseq.columns:
        save_line_plot(
            df_all_allseq,
            x_col="reid_thr",
            y_col="mota",
            out_path=plots_dir / "threshold_vs_mota.png",
            title="Threshold vs MOTA",
            ylabel="MOTA",
        )

    if "id_switches" in df_all_allseq.columns:
        save_line_plot(
            df_all_allseq,
            x_col="reid_thr",
            y_col="id_switches",
            out_path=plots_dir / "threshold_vs_id_switches.png",
            title="Threshold vs ID Switches",
            ylabel="ID Switches",
        )

    if "precision" in df_all_allseq.columns:
        save_line_plot(
            df_all_allseq,
            x_col="reid_thr",
            y_col="precision",
            out_path=plots_dir / "threshold_vs_precision.png",
            title="Threshold vs Precision",
            ylabel="Precision",
        )

    if "recall" in df_all_allseq.columns:
        save_line_plot(
            df_all_allseq,
            x_col="reid_thr",
            y_col="recall",
            out_path=plots_dir / "threshold_vs_recall.png",
            title="Threshold vs Recall",
            ylabel="Recall",
        )

    if "reacq_mean_frames" in df_all_allseq.columns:
        save_line_plot(
            df_all_allseq,
            x_col="reid_thr",
            y_col="reacq_mean_frames",
            out_path=plots_dir / "threshold_vs_reacq_gap.png",
            title="Threshold vs Mean Reacquisition Gap",
            ylabel="Mean Reacquisition Gap (frames)",
        )

    # Per-sequence plots only for best runs
    if "mota" in per_seq_best_df.columns:
        save_barplot_per_sequence(
            per_seq_best_df,
            metric_col="mota",
            out_path=plots_dir / "per_sequence_mota_best.png",
            title="Per-sequence MOTA for the best configuration of each backend",
            ylabel="MOTA",
        )

    if "id_switches" in per_seq_best_df.columns:
        save_barplot_per_sequence(
            per_seq_best_df,
            metric_col="id_switches",
            out_path=plots_dir / "per_sequence_ids_best.png",
            title="Per-sequence ID Switches for the best configuration of each backend",
            ylabel="ID Switches",
        )

    if "reacq_mean_frames" in per_seq_best_df.columns:
        save_barplot_per_sequence(
            per_seq_best_df,
            metric_col="reacq_mean_frames",
            out_path=plots_dir / "per_sequence_reacq_best.png",
            title="Per-sequence mean reacquisition gap for the best configuration of each backend",
            ylabel="Mean reacquisition gap (frames)",
        )

    # -------- Small text summary --------
    summary_txt = []
    summary_txt.append("Best configuration per backend\n")
    for _, row in best_table.iterrows():
        summary_txt.append(
            f"- {row['reid_backend']}: "
            f"thr={row['reid_thr']}, "
            f"MOTA={_fmt(row.get('mota', None), 3)}, "
            f"IDS={_fmt(row.get('id_switches', None), 0)}, "
            f"Precision={_fmt(row.get('precision', None), 3)}, "
            f"Recall={_fmt(row.get('recall', None), 3)}"
        )

    (output_dir / "summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    print("[done]")
    print("Combined CSV:        ", combined_csv)
    print("Threshold summary:   ", threshold_csv)
    print("Best per backend:    ", best_csv)
    print("Per-sequence best:   ", per_seq_csv)
    print("LaTeX tables dir:    ", latex_dir)
    print("Plots dir:           ", plots_dir)


if __name__ == "__main__":
    main()