#!/usr/bin/env python3
"""
ktp_make_sweep_plots.py

Thesis-friendly plots from KTP sweep *_sweep_summary.csv files
with a paper-like aesthetic (serif font, clean spines, light grid, nice legend).

This version uses ID switch percentage:
    id_switch_pct = 100 * id_switches / <denominator>
Default denominator: matches
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


# -------------------------
# Paper style
# -------------------------

PAPER_COLORS = [
    "#2F6BDE",  # blue
    "#F39C12",  # orange
    "#2AA198",  # teal
    "#D81B60",  # magenta
    "#7E57C2",  # purple (extra)
    "#43A047",  # green  (extra)
]

PAPER_LINESTYLES = ["-", "--", "-.", ":"]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",

        "figure.dpi": 140,
        "savefig.dpi": 300,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
        "axes.labelsize": 12,
        "axes.titlesize": 13,

        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "grid.linewidth": 0.9,

        "legend.frameon": False,
        "legend.fontsize": 10,

        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,

        "axes.prop_cycle": cycler(color=PAPER_COLORS),
    })


def _savefig(out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _auto_seq_from_filename(p: Path) -> Optional[str]:
    name = p.name.lower()
    if "arc" in name:
        return "Arc"
    if "rotation" in name:
        return "Rotation"
    return None


def _sort_fps_unique(vals: pd.Series) -> List[float]:
    return sorted([v for v in vals.dropna().unique()])


# -------------------------
# stable_kf_time_sec filtering + snapping
# -------------------------

def _parse_keep_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def filter_and_snap_stable_kf_time_sec(
    df: pd.DataFrame,
    col: str = "stable_kf_time_sec",
    keep: Sequence[float] = (0.5, 1.0, 1.5),
    tol: float = 1e-6,
) -> pd.DataFrame:
    if col not in df.columns:
        return df

    out = df.copy()
    vals = pd.to_numeric(out[col], errors="coerce").to_numpy()
    keep_arr = np.array(list(keep), dtype=float)

    diffs = np.abs(vals.reshape(-1, 1) - keep_arr.reshape(1, -1))
    ok = np.isfinite(vals) & (diffs.min(axis=1) <= tol)
    out = out.loc[ok].copy()

    if out.empty:
        return out

    vals_ok = pd.to_numeric(out[col], errors="coerce").to_numpy()
    diffs_ok = np.abs(vals_ok.reshape(-1, 1) - keep_arr.reshape(1, -1))
    nearest = keep_arr[diffs_ok.argmin(axis=1)]
    out[col] = nearest

    return out


# -------------------------
# NEW: ID switch percentage
# -------------------------

def add_id_switch_percentage(
    df: pd.DataFrame,
    denom_col: str = "matches",
    switches_col: str = "id_switches",
    out_col: str = "id_switch_pct",
) -> pd.DataFrame:
    """
    Adds:
        id_switch_pct = 100 * id_switches / denom_col
    Interpreting denom_col as "how many ID assignments happened" (e.g., matches).
    """
    out = df.copy()

    if switches_col not in out.columns:
        raise ValueError(f"Missing required column '{switches_col}' for ID switch percentage.")

    if denom_col not in out.columns:
        raise ValueError(
            f"Missing denominator column '{denom_col}'. "
            f"Choose an existing one (common: 'matches' or 'gt_boxes')."
        )

    _ensure_numeric(out, [switches_col, denom_col])

    denom = out[denom_col].to_numpy(dtype=float)
    sw = out[switches_col].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (sw / denom)

    pct[~np.isfinite(pct)] = np.nan  # handles denom=0 or NaNs

    out[out_col] = pct
    return out


# -------------------------
# Plot helpers
# -------------------------

def plot_box_by_fps(
    df: pd.DataFrame,
    metric: str,
    fps_col: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    logy: bool = False,
) -> None:
    if metric not in df.columns:
        print(f"[skip] Missing metric '{metric}'")
        return

    fps_vals = _sort_fps_unique(df[fps_col])
    data, labels = [], []
    for f in fps_vals:
        vals = df.loc[df[fps_col] == f, metric].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(str(f))

    if not data:
        print(f"[skip] No data for {metric}")
        return

    plt.figure(figsize=(7.2, 4.6))

    bp = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
        boxprops=dict(linewidth=1.6),
        flierprops=dict(marker="o", markersize=3.5, alpha=0.4),
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PAPER_COLORS[i % len(PAPER_COLORS)])
        box.set_alpha(0.18)

    plt.title(title)
    plt.xlabel("FPS")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")

    _savefig(out_dir, stem)


def plot_mean_lines_by_group(
    df: pd.DataFrame,
    metric: str,
    fps_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    max_groups: int = 4,
) -> None:
    if metric not in df.columns or group_col not in df.columns:
        print(f"[skip] Missing {metric} or {group_col}")
        return

    top_groups = (
        df[group_col]
        .dropna()
        .astype(str)
        .value_counts()
        .head(max_groups)
        .index.tolist()
    )

    plt.figure(figsize=(7.2, 4.6))

    markers = ["o", "o", "o", "o", "s", "^", "D", "v"]
    for i, g in enumerate(top_groups):
        sub = df[df[group_col].astype(str) == g]
        means = sub.groupby(fps_col)[metric].mean().sort_index()
        if means.empty:
            continue

        xs = means.index.values
        ys = means.values

        plt.plot(
            xs, ys,
            marker=markers[i % len(markers)],
            linestyle=PAPER_LINESTYLES[i % len(PAPER_LINESTYLES)],
            label=str(g),
        )

    plt.title(title)
    plt.xlabel("FPS")
    plt.ylabel(ylabel)
    plt.legend(loc="best")

    _savefig(out_dir, stem)


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
) -> None:
    if x not in df.columns or y not in df.columns or color_by not in df.columns:
        print(f"[skip] Missing {x}/{y}/{color_by}")
        return

    sub = df[[x, y, color_by]].dropna()
    if sub.empty:
        print(f"[skip] No data for scatter {x} vs {y}")
        return

    plt.figure(figsize=(7.2, 4.8))

    uniq = sub[color_by].nunique()
    if uniq <= 10:
        for _, (v, g) in enumerate(sub.groupby(color_by)):
            plt.scatter(g[x], g[y], s=45, alpha=0.85, label=str(v), edgecolors="none")
        plt.legend(title=color_by)
    else:
        c = plt.scatter(sub[x], sub[y], c=sub[color_by], s=45, alpha=0.85, edgecolors="none")
        plt.colorbar(c, label=color_by)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    _savefig(out_dir, stem)


# -------------------------
# Data pipeline
# -------------------------

def load_sweep_summaries(in_path: Path) -> pd.DataFrame:
    if in_path.is_file():
        paths = [in_path]
    else:
        paths = sorted(in_path.glob("*_sweep_summary*.csv"))
        if not paths:
            paths = sorted(in_path.glob("*summary*.csv"))

    if not paths:
        raise FileNotFoundError(f"No sweep summary CSVs found in: {in_path}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df = _normalize_cols(df)

        if "seq" not in df.columns:
            seq_guess = _auto_seq_from_filename(p)
            if seq_guess is not None:
                df["seq"] = seq_guess

        df["__source_file"] = p.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_config_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "fps_sim",
        "stable_kf_time_sec",
        "stable_frames_threshold",
        "id_switches",
        "match_rate",
        "matches",
        "gt_boxes",
        "reacq_mean_seconds",
        "reacq_median_seconds",
        "reacq_max_seconds",
    ]
    _ensure_numeric(df, [c for c in numeric_cols if c in df.columns])

    # Only derive stable_kf_time_sec if it is missing
    if "stable_kf_time_sec" not in df.columns:
        if "stable_frames_threshold" in df.columns and "fps_sim" in df.columns:
            df["stable_kf_time_sec"] = df["stable_frames_threshold"] / df["fps_sim"]

    existing = _find_first_col(df, ["config_id", "label", "config"])
    if existing:
        df["config_id"] = df[existing].astype(str)
    else:
        params = [c for c in [
            "stable_kf_time_sec",
            "stable_frames_threshold",
            "stable_ious_threshold",
            "min_obj_score_logits",
            "kf_score_weight",
            "memory_bank_iou_threshold",
            "memory_bank_obj_score_threshold",
            "memory_bank_kf_score_threshold",
        ] if c in df.columns]

        def mk(row: pd.Series) -> str:
            parts = []
            for k in params:
                v = row.get(k, None)
                if pd.isna(v):
                    continue
                if isinstance(v, float):
                    v = round(float(v), 4)
                parts.append(f"{k}={v}")
            return ",".join(parts) if parts else "config=unknown"

        df["config_id"] = df.apply(mk, axis=1)

    return df


def rank_best_configs(df: pd.DataFrame, out_dir: Path) -> None:
    # Rank using percentage (lower is better) + match_rate (higher is better) + reacq (lower is better)
    required = ["seq", "fps_sim", "config_id", "id_switch_pct"]
    if any(c not in df.columns for c in required):
        print("[skip] ranking: missing required columns for percentage ranking")
        return

    agg = {"id_switch_pct": "mean"}
    if "match_rate" in df.columns:
        agg["match_rate"] = "mean"
    if "reacq_mean_seconds" in df.columns:
        agg["reacq_mean_seconds"] = "mean"

    summary = (
        df.groupby(["seq", "fps_sim", "config_id"], dropna=False)
        .agg(agg)
        .reset_index()
    )

    sort_cols = ["id_switch_pct"]
    ascending = [True]
    if "match_rate" in summary.columns:
        sort_cols.append("match_rate")
        ascending.append(False)
    if "reacq_mean_seconds" in summary.columns:
        sort_cols.append("reacq_mean_seconds")
        ascending.append(True)

    summary = summary.sort_values(sort_cols, ascending=ascending)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "best_configs.csv", index=False)


def make_plots(
    df: pd.DataFrame,
    out_dir: Path,
    seq_filter: Optional[str] = None,
    stable_kf_keep: Sequence[float] = (0.5, 1.0, 1.5),
    stable_kf_tol: float = 1e-6,
    id_switch_denom: str = "matches",
) -> None:
    df = df.copy()

    fps_col = _find_first_col(df, ["fps_sim", "fps"])
    if fps_col is None:
        raise ValueError("Could not find fps column (expected 'fps_sim' or 'fps').")

    if seq_filter:
        df = df[df["seq"].astype(str).str.lower() == seq_filter.lower()]
    if df.empty:
        raise ValueError("No rows left after filtering (seq_filter?).")

    # Add percentage column globally so every plot can use it
    df = add_id_switch_percentage(df, denom_col=id_switch_denom)

    seqs = sorted(df["seq"].dropna().astype(str).unique().tolist())
    for seq in seqs:
        dseq = df[df["seq"].astype(str) == seq].copy()
        seq_dir = out_dir / seq

        # 1) ID switch % vs FPS (boxplot across runs/configs)
        if "id_switch_pct" in dseq.columns:
            plot_box_by_fps(
                dseq, "id_switch_pct", fps_col,
                title=f"{seq}: ID switch percentage vs FPS",
                ylabel=f"ID switches (%)  [100 * id_switches / {id_switch_denom}]",
                out_dir=seq_dir,
                stem="id_switch_pct_box_by_fps",
            )

        # 2) Match rate vs FPS
        if "match_rate" in dseq.columns:
            plot_box_by_fps(
                dseq, "match_rate", fps_col,
                title=f"{seq}: Match rate vs FPS",
                ylabel="Match rate",
                out_dir=seq_dir,
                stem="match_rate_box_by_fps",
            )

        # 3) Reacquisition mean seconds vs FPS
        if "reacq_mean_seconds" in dseq.columns:
            plot_box_by_fps(
                dseq, "reacq_mean_seconds", fps_col,
                title=f"{seq}: Reacquisition mean time vs FPS",
                ylabel="Reacq mean (s)",
                out_dir=seq_dir,
                stem="reacq_mean_seconds_box_by_fps",
            )

        # 4) Mean ID switch % vs FPS: one line per stable_kf_time_sec (filtered to keep-list)
        if "stable_kf_time_sec" in dseq.columns and "id_switch_pct" in dseq.columns:
            dclean = filter_and_snap_stable_kf_time_sec(
                dseq,
                col="stable_kf_time_sec",
                keep=stable_kf_keep,
                tol=stable_kf_tol,
            )

            if dclean.empty:
                print(f"[skip] {seq}: no rows left after stable_kf_time_sec filtering")
            else:
                plot_mean_lines_by_group(
                    dclean,
                    metric="id_switch_pct",
                    fps_col=fps_col,
                    group_col="stable_kf_time_sec",
                    title=f"{seq}: Mean ID switch percentage vs FPS (stable_kf_time_sec)",
                    ylabel=f"Mean ID switches (%)  [100 * id_switches / {id_switch_denom}]",
                    out_dir=seq_dir,
                    stem="id_switch_pct_lines_by_stable_kf_time_sec",
                    max_groups=4,
                )

        # 5) Trade-off scatter: match rate vs ID switch %
        if "match_rate" in dseq.columns and "id_switch_pct" in dseq.columns:
            plot_tradeoff_scatter(
                dseq,
                x="id_switch_pct",
                y="match_rate",
                color_by=fps_col,
                title=f"{seq}: Trade-off (match rate vs ID switch %)",
                xlabel=f"ID switches (%) (lower is better)  [100 * id_switches / {id_switch_denom}]",
                ylabel="Match rate (higher is better)",
                out_dir=seq_dir,
                stem="tradeoff_match_rate_vs_id_switch_pct_colored_by_fps",
            )

    rank_best_configs(df, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="*_sweep_summary.csv OR directory containing them")
    ap.add_argument("--out_dir", required=True, help="Output directory for plots")
    ap.add_argument("--seq", default=None, help="Optional: filter seq (Arc/Rotation)")

    ap.add_argument(
        "--stable_kf_keep",
        default="0.5,1.0,1.5",
        help="Comma-separated stable_kf_time_sec values to keep (e.g., '0.5,1.0,1.5')",
    )
    ap.add_argument(
        "--stable_kf_tol",
        type=float,
        default=1e-6,
        help="Tolerance for matching stable_kf_time_sec values",
    )

    ap.add_argument(
        "--id_switch_denom",
        default="matches",
        help="Denominator for ID switch percentage (common: 'matches' or 'gt_boxes')",
    )

    args = ap.parse_args()

    apply_paper_style()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    df = load_sweep_summaries(in_path)
    df = build_config_id(df)

    keep_vals = _parse_keep_list(args.stable_kf_keep)

    make_plots(
        df,
        out_dir,
        seq_filter=args.seq,
        stable_kf_keep=keep_vals,
        stable_kf_tol=args.stable_kf_tol,
        id_switch_denom=args.id_switch_denom,
    )

    print(f"[done] Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()