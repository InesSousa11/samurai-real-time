#!/usr/bin/env python3
"""
ktp_make_thesis_plots.py

Usage examples:

# (1) Point to a folder that contains your sweep outputs
python ktp_make_thesis_plots.py --in_dir "C:\\Users\\inesg\\OneDrive\\Desktop\\Thesis\\datasets\\KTP\\results_r"

# (2) Point directly to a sweep summary csv
python ktp_make_thesis_plots.py --summary_csv "C:\\...\\arc_internal_sweep_v1_20260216_123456__sweep_summary.csv"

# Optional:
# - Only use seq==ALL rows (default)
# - Compute reacq seconds by opening each out_csv (default ON)
# - Choose top_k configs for the “curves” plot
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------- IO helpers ----------------------------
def find_latest_summary(in_dir: Path) -> Path:
    cands = list(in_dir.rglob("*__sweep_summary.csv"))
    if not cands:
        raise FileNotFoundError(f"No '*__sweep_summary.csv' found under: {in_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_summary_csv(p: Path) -> pd.DataFrame:
    # This file should be a clean CSV (no leading '#'), but we handle either way.
    df = pd.read_csv(p, comment="#")
    return df


def read_per_seq_csv(p: Path) -> pd.DataFrame:
    # per-seq csv has comment/header lines starting with '#'
    df = pd.read_csv(p, comment="#")
    return df


# ---------------------------- reacq seconds ----------------------------
def compute_reacq_seconds_from_outcsv(out_csv: Path) -> Dict[str, float]:
    """
    Computes reacquisition gap durations in SECONDS using t_sec differences.

    Logic:
      For each gt_id (independently):
        - If pred_id is empty -> "in gap"
        - Gap starts at the first frame where pred_id becomes empty
        - Gap ends at the first subsequent frame where pred_id becomes non-empty
        - duration = t_sec(end) - t_sec(start)
    """
    if (out_csv is None) or (str(out_csv).strip() == ""):
        return {
            "reacq_events_seconds": 0.0,
            "reacq_mean_seconds": 0.0,
            "reacq_median_seconds": 0.0,
            "reacq_max_seconds": 0.0,
        }

    out_csv = Path(out_csv)
    if not out_csv.exists():
        # Don’t hard fail: sometimes you moved files.
        return {
            "reacq_events_seconds": 0.0,
            "reacq_mean_seconds": 0.0,
            "reacq_median_seconds": 0.0,
            "reacq_max_seconds": 0.0,
        }

    df = read_per_seq_csv(out_csv)

    # Required columns
    # pred_id might be numeric or empty string
    needed = {"gt_id", "frame_idx", "t_sec", "pred_id"}
    missing = needed - set(df.columns)
    if missing:
        return {
            "reacq_events_seconds": 0.0,
            "reacq_mean_seconds": 0.0,
            "reacq_median_seconds": 0.0,
            "reacq_max_seconds": 0.0,
        }

    # Normalize
    df = df.copy()
    df["gt_id"] = pd.to_numeric(df["gt_id"], errors="coerce")
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")

    # pred_id: empty => NaN
    df["pred_id"] = df["pred_id"].replace("", np.nan)
    df["pred_id"] = pd.to_numeric(df["pred_id"], errors="coerce")

    df = df.dropna(subset=["gt_id", "frame_idx", "t_sec"])
    df = df.sort_values(["gt_id", "frame_idx"])

    durations: List[float] = []

    for gt_id, g in df.groupby("gt_id", sort=False):
        gap_start_t: Optional[float] = None

        for _, row in g.iterrows():
            t = float(row["t_sec"])
            has_pred = not np.isnan(row["pred_id"])

            if not has_pred:
                if gap_start_t is None:
                    gap_start_t = t
            else:
                if gap_start_t is not None:
                    dur = max(0.0, t - gap_start_t)
                    durations.append(dur)
                    gap_start_t = None

    if not durations:
        return {
            "reacq_events_seconds": 0.0,
            "reacq_mean_seconds": 0.0,
            "reacq_median_seconds": 0.0,
            "reacq_max_seconds": 0.0,
        }

    durations = [float(x) for x in durations if np.isfinite(x)]
    durations.sort()

    return {
        "reacq_events_seconds": float(len(durations)),
        "reacq_mean_seconds": float(np.mean(durations)),
        "reacq_median_seconds": float(np.median(durations)),
        "reacq_max_seconds": float(np.max(durations)),
    }


def add_reacq_seconds_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds reacq_*_seconds for seq rows that have out_csv.
    For seq == ALL rows, computes a weighted mean from the per-seq rows
    (weighted by reacq_events_seconds).
    """
    df = summary_df.copy()

    for c in ["out_csv", "seq", "label", "run"]:
        if c not in df.columns:
            return df

    # Compute per-seq seconds (seq != ALL)
    sec_cols = ["reacq_events_seconds", "reacq_mean_seconds", "reacq_median_seconds", "reacq_max_seconds"]
    for c in sec_cols:
        df[c] = 0.0

    mask_seq = (df["seq"].astype(str) != "ALL") & df["out_csv"].notna() & (df["out_csv"].astype(str).str.len() > 0)

    # Cache per out_csv to avoid re-reading duplicates
    cache: Dict[str, Dict[str, float]] = {}

    for idx in df[mask_seq].index:
        out_csv = str(df.at[idx, "out_csv"])
        if out_csv not in cache:
            cache[out_csv] = compute_reacq_seconds_from_outcsv(Path(out_csv))
        for c in sec_cols:
            df.at[idx, c] = cache[out_csv][c]

    # Fill ALL rows by aggregating their per-seq rows
    # Group key: (run, label)
    mask_all = df["seq"].astype(str) == "ALL"
    if mask_all.any():
        for (run, label), g_all in df[mask_all].groupby(["run", "label"], sort=False):
            g_seq = df[(df["run"] == run) & (df["label"] == label) & (df["seq"].astype(str) != "ALL")]
            if g_seq.empty:
                continue

            w = g_seq["reacq_events_seconds"].astype(float).to_numpy()
            wsum = float(np.sum(w))
            if wsum <= 0:
                mean_sec = float(np.mean(g_seq["reacq_mean_seconds"].astype(float).to_numpy())) if len(g_seq) else 0.0
                med_sec = float(np.mean(g_seq["reacq_median_seconds"].astype(float).to_numpy())) if len(g_seq) else 0.0
                max_sec = float(np.max(g_seq["reacq_max_seconds"].astype(float).to_numpy())) if len(g_seq) else 0.0
                events = 0.0
            else:
                mean_sec = float(np.sum(g_seq["reacq_mean_seconds"].astype(float).to_numpy() * w) / wsum)
                med_sec = float(np.sum(g_seq["reacq_median_seconds"].astype(float).to_numpy() * w) / wsum)
                max_sec = float(np.max(g_seq["reacq_max_seconds"].astype(float).to_numpy()))
                events = float(wsum)

            df.loc[(df["run"] == run) & (df["label"] == label) & (df["seq"].astype(str) == "ALL"), "reacq_events_seconds"] = events
            df.loc[(df["run"] == run) & (df["label"] == label) & (df["seq"].astype(str) == "ALL"), "reacq_mean_seconds"] = mean_sec
            df.loc[(df["run"] == run) & (df["label"] == label) & (df["seq"].astype(str) == "ALL"), "reacq_median_seconds"] = med_sec
            df.loc[(df["run"] == run) & (df["label"] == label) & (df["seq"].astype(str) == "ALL"), "reacq_max_seconds"] = max_sec

    return df


# ---------------------------- ranking ----------------------------
def rank_configs(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking that prioritizes:
      1) id_switches ascending (most important)
      2) match_rate descending
      3) reacq_mean_seconds ascending (if present; else reacq_mean_frames)
    """
    df = df_all.copy()

    for col in ["id_switches", "match_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "reacq_mean_seconds" in df.columns:
        df["reacq_mean_seconds"] = pd.to_numeric(df["reacq_mean_seconds"], errors="coerce").fillna(0)
        sort_cols = ["id_switches", "match_rate", "reacq_mean_seconds"]
        ascending = [True, False, True]
    else:
        if "reacq_mean_frames" in df.columns:
            df["reacq_mean_frames"] = pd.to_numeric(df["reacq_mean_frames"], errors="coerce").fillna(0)
            sort_cols = ["id_switches", "match_rate", "reacq_mean_frames"]
            ascending = [True, False, True]
        else:
            sort_cols = ["id_switches", "match_rate"]
            ascending = [True, False]

    # If fps exists, rank within each fps so comparisons are fair
    if "fps_sim" in df.columns:
        df["fps_sim"] = pd.to_numeric(df["fps_sim"], errors="coerce")
        df = df.sort_values(["fps_sim"] + sort_cols, ascending=[True] + ascending)
        df["rank_within_fps"] = df.groupby("fps_sim").cumcount() + 1
    else:
        df = df.sort_values(sort_cols, ascending=ascending)
        df["rank_overall"] = np.arange(1, len(df) + 1)

    return df


# ---------------------------- plotting style ----------------------------
def set_thesis_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def save_fig(fig, out_path_no_ext: Path):
    fig.tight_layout()
    fig.savefig(str(out_path_no_ext.with_suffix(".png")))
    fig.savefig(str(out_path_no_ext.with_suffix(".pdf")))
    plt.close(fig)


# ---------------------------- plots ----------------------------
def plot_pareto(df: pd.DataFrame, out_dir: Path, title: str, annotate_top: int = 5):
    if not {"id_switches", "match_rate"}.issubset(df.columns):
        return

    set_thesis_style()

    # Color by fps_sim if available
    has_fps = "fps_sim" in df.columns and df["fps_sim"].notna().any()

    fig = plt.figure(figsize=(8.5, 6))
    ax = plt.gca()

    if has_fps:
        fps_vals = pd.to_numeric(df["fps_sim"], errors="coerce")
        sc = ax.scatter(df["id_switches"], df["match_rate"], c=fps_vals, s=28, alpha=0.85)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("fps_sim")
    else:
        ax.scatter(df["id_switches"], df["match_rate"], s=28, alpha=0.85)

    ax.set_xlabel("ID switches (lower is better)")
    ax.set_ylabel("match_rate (higher is better)")
    ax.set_title(title)

    # Annotate top configs (by ranking logic)
    df_ranked = rank_configs(df)
    if has_fps and "rank_within_fps" in df_ranked.columns:
        # annotate best in each fps
        for fps, g in df_ranked.groupby("fps_sim"):
            gtop = g.head(annotate_top)
            for _, r in gtop.iterrows():
                ax.annotate(str(r.get("label", "")), (r["id_switches"], r["match_rate"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.85)
    else:
        gtop = df_ranked.head(annotate_top)
        for _, r in gtop.iterrows():
            ax.annotate(str(r.get("label", "")), (r["id_switches"], r["match_rate"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.85)

    save_fig(fig, out_dir / "pareto_idsw_vs_matchrate")


def plot_idsw_box_by_fps(df: pd.DataFrame, out_dir: Path, title: str):
    if "fps_sim" not in df.columns or "id_switches" not in df.columns:
        return

    set_thesis_style()
    d = df.copy()
    d["fps_sim"] = pd.to_numeric(d["fps_sim"], errors="coerce")
    d = d.dropna(subset=["fps_sim"])
    if d.empty:
        return

    fps_sorted = sorted(d["fps_sim"].unique().tolist())
    data = [pd.to_numeric(d.loc[d["fps_sim"] == f, "id_switches"], errors="coerce").dropna().to_numpy()
            for f in fps_sorted]

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()
    ax.boxplot(data, labels=[str(int(f)) if float(f).is_integer() else str(f) for f in fps_sorted], showfliers=False)
    ax.set_xlabel("fps_sim")
    ax.set_ylabel("ID switches")
    ax.set_title(title)
    save_fig(fig, out_dir / "id_switches_boxplot_by_fps")


def plot_reacq_seconds_by_fps(df: pd.DataFrame, out_dir: Path, title: str):
    if "fps_sim" not in df.columns:
        return

    col = "reacq_mean_seconds" if "reacq_mean_seconds" in df.columns else None
    if col is None:
        return

    set_thesis_style()
    d = df.copy()
    d["fps_sim"] = pd.to_numeric(d["fps_sim"], errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["fps_sim", col])
    if d.empty:
        return

    # Plot mean +/- std across configs
    fps_sorted = sorted(d["fps_sim"].unique().tolist())
    means = []
    stds = []
    for f in fps_sorted:
        vals = d.loc[d["fps_sim"] == f, col].to_numpy()
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()
    ax.errorbar(fps_sorted, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("fps_sim")
    ax.set_ylabel("reacq_mean_seconds (mean ± std across configs)")
    ax.set_title(title)
    ax.set_xscale("linear")
    save_fig(fig, out_dir / "reacq_mean_seconds_by_fps")


def plot_topk_curves(df: pd.DataFrame, out_dir: Path, title: str, top_k: int = 5):
    """
    Plots ID switches vs fps_sim for the best K configs (by average rank across fps).
    Requires fps_sim.
    """
    if "fps_sim" not in df.columns or "label" not in df.columns or "id_switches" not in df.columns:
        return

    set_thesis_style()
    d = df.copy()
    d["fps_sim"] = pd.to_numeric(d["fps_sim"], errors="coerce")
    d["id_switches"] = pd.to_numeric(d["id_switches"], errors="coerce")
    d = d.dropna(subset=["fps_sim", "id_switches"])
    if d.empty:
        return

    # Rank within each fps
    ranked = rank_configs(d)

    if "rank_within_fps" in ranked.columns:
        # Average rank across fps for each label
        agg = ranked.groupby("label")["rank_within_fps"].mean().sort_values()
        top_labels = agg.head(top_k).index.tolist()
    else:
        # fallback: global best
        top_labels = ranked.sort_values("id_switches", ascending=True).head(top_k)["label"].tolist()

    fig = plt.figure(figsize=(8.5, 6))
    ax = plt.gca()

    for lab in top_labels:
        g = ranked[ranked["label"] == lab].sort_values("fps_sim")
        ax.plot(g["fps_sim"], g["id_switches"], marker="o", label=lab)

    ax.set_xlabel("fps_sim")
    ax.set_ylabel("ID switches")
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, out_dir / f"top{top_k}_id_switches_vs_fps")


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="", help="Folder containing sweep outputs (summary + per-seq CSVs)")
    ap.add_argument("--summary_csv", type=str, default="", help="Path to *sweep_summary.csv (overrides in_dir search)")
    ap.add_argument("--out_dir", type=str, default="", help="Where to save plots (default: alongside summary)")
    ap.add_argument("--only_all", action="store_true", help="Use only seq == ALL rows (recommended for reporting)")
    ap.add_argument("--compute_reacq_seconds", action="store_true", help="Parse out_csv files to compute reacq seconds")
    ap.add_argument("--top_k", type=int, default=6, help="Top-K configs to plot curves for")
    args = ap.parse_args()

    if args.summary_csv:
        summary_path = Path(args.summary_csv).resolve()
        if not summary_path.exists():
            raise FileNotFoundError(f"summary_csv not found: {summary_path}")
    else:
        if not args.in_dir:
            raise ValueError("Provide either --summary_csv or --in_dir")
        summary_path = find_latest_summary(Path(args.in_dir).resolve())

    df = read_summary_csv(summary_path)

    # Optional filter
    if args.only_all:
        if "seq" in df.columns:
            df = df[df["seq"].astype(str) == "ALL"].copy()

    # Try to detect a nice run name
    run_name = None
    if "run" in df.columns and df["run"].notna().any():
        run_name = str(df["run"].iloc[0])
    else:
        run_name = summary_path.stem.replace("__sweep_summary", "")

    base_out = Path(args.out_dir).resolve() if args.out_dir else summary_path.parent
    plots_out = base_out / "plots_thesis" / run_name
    safe_mkdir(plots_out)

    # Add reacq seconds if requested
    if args.compute_reacq_seconds:
        df = add_reacq_seconds_columns(df)

    # Save an enriched summary for convenience
    enriched_path = plots_out / "sweep_summary_enriched.csv"
    df.to_csv(enriched_path, index=False)

    # Ranking + top table
    ranked = rank_configs(df)
    top_table = ranked.copy()

    # If fps exists, keep top per fps (so you can compare fairly)
    if "fps_sim" in top_table.columns and "rank_within_fps" in top_table.columns:
        top_table = top_table.sort_values(["fps_sim", "rank_within_fps"])
        top_table = top_table.groupby("fps_sim").head(25)
    else:
        top_table = top_table.head(50)

    top_cols = [c for c in [
        "fps_sim", "label", "seq",
        "id_switches", "match_rate",
        "reacq_mean_seconds", "reacq_median_seconds", "reacq_max_seconds",
        "reacq_mean_frames", "reacq_median_frames", "reacq_max_frames",
        "stable_frames_threshold", "stable_ious_threshold", "min_obj_score_logits", "kf_score_weight",
        "memory_bank_iou_threshold", "memory_bank_obj_score_threshold", "memory_bank_kf_score_threshold",
        "frames", "gt_boxes", "matches"
    ] if c in top_table.columns]

    top_out = plots_out / "top_configs.csv"
    top_table[top_cols].to_csv(top_out, index=False)

    # Make plots
    title_prefix = f"{run_name}"
    plot_pareto(df, plots_out, title=f"{title_prefix}: ID switches vs match_rate", annotate_top=4)
    plot_idsw_box_by_fps(df, plots_out, title=f"{title_prefix}: ID switches distribution vs fps_sim")
    plot_reacq_seconds_by_fps(df, plots_out, title=f"{title_prefix}: reacq mean seconds vs fps_sim")
    plot_topk_curves(df, plots_out, title=f"{title_prefix}: Top-{args.top_k} configs ID switches vs fps_sim", top_k=args.top_k)

    # Console summary
    print("\n[done]")
    print("  summary used:", summary_path)
    print("  plots out   :", plots_out)
    print("  enriched    :", enriched_path)
    print("  top table   :", top_out)
    print("\nTip: For thesis, use the .pdf versions of plots.\n")


if __name__ == "__main__":
    main()