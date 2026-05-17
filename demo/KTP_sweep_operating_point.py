#!/usr/bin/env python3
# KTP_sweep_operating_point.py
#
# First-stage operating-point sweep for ReID-SAMURAI on KTP.
#
# This script:
#   - reuses the KTP evaluation procedure implemented in KTP_eval_run.py
#   - evaluates multiple threshold configurations
#   - samples KTP at approximately robot deployment frequency using stride=6 by default
#   - exports per-config MOT-style GT/pred files for later TrackEval
#   - writes one global sweep CSV and JSON summary
#
# Recommended first-stage sweep:
#   reid_thr:                       0.75,0.80,0.85
#   memory_bank_reid_threshold:     0.55,0.65,0.75
#   min_obj_score_logits:           0.0,0.5,1.0
#   reid_gallery_add_sim_threshold: 0.75,0.80,0.85

import sys
import csv
import json
import time
import argparse
import traceback
from pathlib import Path
from contextlib import nullcontext
from itertools import product
from typing import List, Optional, Dict, Any

import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

# ---------------------------------------------------------------------
# Locate repo and import the existing single-run evaluation script
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "demo" else Path.cwd()
DEMO_DIR = REPO_ROOT / "demo"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEMO_DIR))

try:
    import KTP_eval_run as base
except ImportError as e:
    raise ImportError(
        "Could not import demo/KTP_eval_run.py.\n"
        "Make sure this sweep script is saved inside demo/ and that "
        "KTP_eval_run.py is also inside demo/."
    ) from e

from sam2.build_sam import build_sam2_camera_predictor


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_float_for_label(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def make_config_label(
    config_id: int,
    reid_thr: float,
    memory_bank_reid_threshold: float,
    min_obj_score_logits: float,
    reid_gallery_add_sim_threshold: float,
) -> str:
    """
    Human-readable configuration label.

    This is intentionally NOT used for folder/file names because it can make
    Windows paths too long. It is stored in CSV/JSON instead.
    """
    return (
        f"cfg{config_id:04d}"
        f"_rthr{fmt_float_for_label(reid_thr)}"
        f"_mbreid{fmt_float_for_label(memory_bank_reid_threshold)}"
        f"_obj{fmt_float_for_label(min_obj_score_logits)}"
        f"_gadd{fmt_float_for_label(reid_gallery_add_sim_threshold)}"
    )


# ---------------------------------------------------------------------
# Extra threshold setter
# ---------------------------------------------------------------------
def set_extra_reid_samurai_thresholds(
    predictor,
    memory_bank_reid_threshold: Optional[float] = None,
    reid_gallery_add_sim_threshold: Optional[float] = None,
    reid_gallery_max_size: Optional[int] = None,
    reid_gallery_add_cooldown: Optional[int] = None,
    reid_gallery_random_replace_prob: Optional[float] = None,
    reid_gallery_random_replace_if_diverse_prob: Optional[float] = None,
):
    """
    Sets the ReID-SAMURAI thresholds that are not covered by the original
    KTP_eval_run.py set_predictor_thresholds() helper.
    """

    def _set_attr_and_state(name: str, value):
        if value is None:
            return

        try:
            setattr(predictor, name, value)
        except Exception:
            pass

        try:
            cs = getattr(predictor, "condition_state", None)
            if isinstance(cs, dict):
                cs[name] = value
        except Exception:
            pass

    _set_attr_and_state(
        "memory_bank_reid_threshold",
        float(memory_bank_reid_threshold) if memory_bank_reid_threshold is not None else None,
    )
    _set_attr_and_state(
        "reid_gallery_add_sim_threshold",
        float(reid_gallery_add_sim_threshold) if reid_gallery_add_sim_threshold is not None else None,
    )
    _set_attr_and_state(
        "reid_gallery_max_size",
        int(reid_gallery_max_size) if reid_gallery_max_size is not None else None,
    )
    _set_attr_and_state(
        "reid_gallery_add_cooldown",
        int(reid_gallery_add_cooldown) if reid_gallery_add_cooldown is not None else None,
    )
    _set_attr_and_state(
        "reid_gallery_random_replace_prob",
        float(reid_gallery_random_replace_prob) if reid_gallery_random_replace_prob is not None else None,
    )
    _set_attr_and_state(
        "reid_gallery_random_replace_if_diverse_prob",
        float(reid_gallery_random_replace_if_diverse_prob)
        if reid_gallery_random_replace_if_diverse_prob is not None
        else None,
    )


def print_sweep_thresholds(predictor):
    keys = [
        "stable_frames_threshold",
        "stable_ious_threshold",
        "min_obj_score_logits",
        "kf_score_weight",
        "memory_bank_iou_threshold",
        "memory_bank_obj_score_threshold",
        "memory_bank_kf_score_threshold",
        "memory_bank_reid_threshold",
        "reid_thr",
        "reid_gallery_add_sim_threshold",
        "reid_gallery_max_size",
        "reid_gallery_add_cooldown",
        "reid_gallery_random_replace_prob",
        "reid_gallery_random_replace_if_diverse_prob",
        "samurai_mode",
    ]

    print("[predictor thresholds]")
    for key in keys:
        print(f"  {key}: {getattr(predictor, key, None)}")
    print("")


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def metrics_to_sweep_row(
    run_prefix: str,
    config_id: int,
    label: str,
    seq: str,
    reid_backend: str,
    cfg_path: str,
    ckpt_path: str,
    settings: Dict[str, Any],
    met,
    out_csv: str,
    gt_mot_path: str,
    pred_mot_path: str,
) -> Dict[str, Any]:
    reacq_n, reacq_mean, reacq_med, reacq_max = base.summarize_reacq(met.reacq_gaps)

    denom_gt = met.eligible_gt_boxes
    misses = met.false_negatives

    match_rate = met.matches / denom_gt if denom_gt > 0 else 0.0
    miss_rate = misses / denom_gt if denom_gt > 0 else 0.0
    mean_iou = met.iou_sum / met.iou_count if met.iou_count > 0 else 0.0

    precision = (
        met.matches / (met.matches + met.false_positives)
        if (met.matches + met.false_positives) > 0
        else 0.0
    )
    recall = (
        met.matches / (met.matches + met.false_negatives)
        if (met.matches + met.false_negatives) > 0
        else 0.0
    )
    mota = (
        1.0 - ((met.false_negatives + met.false_positives + met.id_switches) / denom_gt)
        if denom_gt > 0
        else 0.0
    )

    id_switches_per_match = met.id_switches / met.matches if met.matches > 0 else 0.0
    id_switches_per_gt = met.id_switches / denom_gt if denom_gt > 0 else 0.0
    reacq_rate_per_gt = reacq_n / denom_gt if denom_gt > 0 else 0.0
    seed_coverage = (
        met.seeded_ids_count / met.total_unique_gt_ids
        if met.total_unique_gt_ids > 0
        else 0.0
    )

    return {
        "run": run_prefix,
        "config_id": config_id,
        "label": label,
        "seq": seq,
        "reid_backend": reid_backend,
        "config": cfg_path,
        "checkpoint": ckpt_path,

        "stable_frames_threshold": settings["stable_frames_threshold"],
        "stable_ious_threshold": settings["stable_ious_threshold"],
        "min_obj_score_logits": settings["min_obj_score_logits"],
        "kf_score_weight": settings["kf_score_weight"],
        "memory_bank_iou_threshold": settings["memory_bank_iou_threshold"],
        "memory_bank_obj_score_threshold": settings["memory_bank_obj_score_threshold"],
        "memory_bank_kf_score_threshold": settings["memory_bank_kf_score_threshold"],
        "memory_bank_reid_threshold": settings["memory_bank_reid_threshold"],
        "reid_thr": settings["reid_thr"],
        "reid_gallery_add_sim_threshold": settings["reid_gallery_add_sim_threshold"],
        "reid_gallery_max_size": settings["reid_gallery_max_size"],
        "reid_gallery_add_cooldown": settings["reid_gallery_add_cooldown"],
        "reid_gallery_random_replace_prob": settings["reid_gallery_random_replace_prob"],
        "reid_gallery_random_replace_if_diverse_prob": settings["reid_gallery_random_replace_if_diverse_prob"],

        "stride": settings["stride"],
        "approx_eval_fps": settings["approx_eval_fps"],

        "frames": met.frames,
        "total_gt_boxes": met.total_gt_boxes,
        "eligible_gt_boxes": met.eligible_gt_boxes,
        "matches": met.matches,
        "misses": misses,

        "false_positives": met.false_positives,
        "false_negatives": met.false_negatives,
        "precision": precision,
        "recall": recall,
        "mota": mota,
        "match_rate": match_rate,
        "miss_rate": miss_rate,

        "id_switches": met.id_switches,
        "id_switches_per_match": id_switches_per_match,
        "id_switches_per_gt": id_switches_per_gt,

        "reacq_events": reacq_n,
        "reacq_rate_per_gt": reacq_rate_per_gt,
        "reacq_mean_frames": reacq_mean,
        "reacq_median_frames": reacq_med,
        "reacq_max_frames": reacq_max,

        "mean_iou_when_matched": mean_iou,

        "total_unique_gt_ids": met.total_unique_gt_ids,
        "seeded_ids_count": met.seeded_ids_count,
        "seed_coverage": seed_coverage,
        "seed_skipped_small": met.seed_skipped_small,
        "seed_skipped_overlap": met.seed_skipped_overlap,
        "seed_failed": met.seed_failed,

        "out_csv": out_csv,
        "gt_mot_path": gt_mot_path,
        "pred_mot_path": pred_mot_path,
    }


def accumulate_metrics(total, met):
    total.frames += met.frames
    total.total_gt_boxes += met.total_gt_boxes
    total.eligible_gt_boxes += met.eligible_gt_boxes
    total.matches += met.matches
    total.false_positives += met.false_positives
    total.false_negatives += met.false_negatives
    total.id_switches += met.id_switches
    total.reacq_events += met.reacq_events
    total.reacq_gaps.extend(met.reacq_gaps)
    total.iou_sum += met.iou_sum
    total.iou_count += met.iou_count
    total.seed_skipped_small += met.seed_skipped_small
    total.seed_skipped_overlap += met.seed_skipped_overlap
    total.seed_failed += met.seed_failed
    total.total_unique_gt_ids += met.total_unique_gt_ids
    total.seeded_ids_count += met.seeded_ids_count


def sort_rows_for_operating_point(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorting used only to help inspect the sweep.

    Main final evaluation should still be computed with TrackEval using the
    exported MOT files.
    """
    all_rows = [r for r in rows if r.get("seq") == "ALL"]

    return sorted(
        all_rows,
        key=lambda r: (
            -float(r["mota"]),
            int(r["id_switches"]),
            -float(r["match_rate"]),
            -float(r["mean_iou_when_matched"]),
        ),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ktp_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="reid_samurai_sweep_stage1")

    ap.add_argument("--sequences", type=str, default="Arc,Rotation,Still,Translation")

    ap.add_argument(
        "--cfg_path",
        type=str,
        default=str((REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()),
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=str((REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()),
    )

    ap.add_argument(
        "--reid_backend",
        type=str,
        default="transreid",
        choices=["osnet_x1_0", "osnet_ain_x1_0", "transreid"],
    )

    # Sweep values
    ap.add_argument("--reid_thr_values", type=str, default="0.75,0.80,0.85")
    ap.add_argument("--memory_bank_reid_threshold_values", type=str, default="0.55,0.65,0.75")
    ap.add_argument("--min_obj_score_logits_values", type=str, default="0.0,0.5,1.0")
    ap.add_argument("--reid_gallery_add_sim_threshold_values", type=str, default="0.75,0.80,0.85")

    # Fixed thresholds for stage 1
    ap.add_argument("--stable_frames_threshold", type=int, default=15)
    ap.add_argument("--stable_ious_threshold", type=float, default=0.30)
    ap.add_argument("--kf_score_weight", type=float, default=0.25)
    ap.add_argument("--memory_bank_iou_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_obj_score_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_kf_score_threshold", type=float, default=0.0)

    ap.add_argument("--reid_gallery_max_size", type=int, default=10)
    ap.add_argument("--reid_gallery_add_cooldown", type=int, default=10)
    ap.add_argument("--reid_gallery_random_replace_prob", type=float, default=0.15)
    ap.add_argument("--reid_gallery_random_replace_if_diverse_prob", type=float, default=0.30)

    # KTP evaluation procedure parameters
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument(
        "--stride",
        type=int,
        default=6,
        help="Default 6 approximates 5 Hz from KTP 30 Hz.",
    )
    ap.add_argument("--approx_eval_fps", type=float, default=5.0)
    ap.add_argument("--max_frames", type=int, default=-1)

    ap.add_argument("--visible_area_frac", type=float, default=0.02)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--visible_min_w", type=int, default=0)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.10)
    ap.add_argument("--iou_match_thr", type=float, default=0.30)
    ap.add_argument("--eval_seed_frame", action="store_true")

    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--display_scale", type=float, default=1.0)

    ap.add_argument(
        "--save_video",
        action="store_true",
        help="Usually keep this off for full sweeps because videos take space/time.",
    )
    ap.add_argument("--save_video_fps", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument(
        "--max_configs",
        type=int,
        default=-1,
        help="Optional limit for debugging. Use -1 for all configurations.",
    )
    ap.add_argument(
        "--start_config",
        type=int,
        default=0,
        help="Optional start index for resuming/splitting a sweep.",
    )

    args = ap.parse_args()

    ktp_root = Path(args.ktp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    cfg_path = Path(args.cfg_path).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()

    safe_mkdir(out_dir)

    if not ktp_root.exists():
        raise FileNotFoundError(f"KTP root not found: {ktp_root}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    reid_thr_values = parse_float_list(args.reid_thr_values)
    memory_bank_reid_threshold_values = parse_float_list(args.memory_bank_reid_threshold_values)
    min_obj_score_logits_values = parse_float_list(args.min_obj_score_logits_values)
    reid_gallery_add_sim_threshold_values = parse_float_list(args.reid_gallery_add_sim_threshold_values)

    full_configs = list(product(
        reid_thr_values,
        memory_bank_reid_threshold_values,
        min_obj_score_logits_values,
        reid_gallery_add_sim_threshold_values,
    ))

    indexed_configs = list(enumerate(full_configs))

    if args.start_config > 0:
        indexed_configs = indexed_configs[args.start_config:]

    if args.max_configs > 0:
        indexed_configs = indexed_configs[:args.max_configs]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    sweep_prefix = f"{args.run_name}_{run_id}"

    sweep_dir = out_dir / sweep_prefix
    safe_mkdir(sweep_dir)

    mot_root = sweep_dir / "mot_exports"
    safe_mkdir(mot_root)

    per_config_dir = sweep_dir / "per_config"
    safe_mkdir(per_config_dir)

    global_csv_path = sweep_dir / f"{sweep_prefix}__summary_all_rows.csv"
    sorted_csv_path = sweep_dir / f"{sweep_prefix}__summary_ALL_sorted.csv"
    global_json_path = sweep_dir / f"{sweep_prefix}__summary.json"

    print("[setup]")
    print("  repo root:", REPO_ROOT)
    print("  ktp root :", ktp_root)
    print("  out dir  :", sweep_dir)
    print("  cfg      :", cfg_path)
    print("  ckpt     :", ckpt_path)
    print("  backend  :", args.reid_backend)
    print("  sequences:", seqs)
    print("  stride   :", args.stride, "(approximately 5 Hz if KTP is 30 Hz)")
    print("  configs  :", len(indexed_configs), "of", len(full_configs))
    print("  cuda     :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  gpu      :", torch.cuda.get_device_name(0))
    print("")

    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    all_rows: List[Dict[str, Any]] = []
    failed_configs: List[Dict[str, Any]] = []

    global_start = time.time()

    for config_id, combo in indexed_configs:
        reid_thr, memory_bank_reid_threshold, min_obj_score_logits, reid_gallery_add_sim_threshold = combo

        label = make_config_label(
            config_id=config_id,
            reid_thr=reid_thr,
            memory_bank_reid_threshold=memory_bank_reid_threshold,
            min_obj_score_logits=min_obj_score_logits,
            reid_gallery_add_sim_threshold=reid_gallery_add_sim_threshold,
        )

        short_id = f"cfg{config_id:04d}"
        run_prefix = f"{sweep_prefix}_{short_id}"

        print("=" * 80)
        print(f"[config {config_id}] {label}")
        print("=" * 80)

        # IMPORTANT:
        # Use short directory names to avoid Windows/OneDrive long-path errors.
        config_mot_dir = mot_root / short_id
        safe_mkdir(config_mot_dir)

        config_out_dir = per_config_dir / short_id
        safe_mkdir(config_out_dir)

        settings = {
            "stable_frames_threshold": args.stable_frames_threshold,
            "stable_ious_threshold": args.stable_ious_threshold,
            "min_obj_score_logits": float(min_obj_score_logits),
            "kf_score_weight": args.kf_score_weight,
            "memory_bank_iou_threshold": args.memory_bank_iou_threshold,
            "memory_bank_obj_score_threshold": args.memory_bank_obj_score_threshold,
            "memory_bank_kf_score_threshold": args.memory_bank_kf_score_threshold,
            "memory_bank_reid_threshold": float(memory_bank_reid_threshold),
            "reid_thr": float(reid_thr),
            "reid_gallery_add_sim_threshold": float(reid_gallery_add_sim_threshold),
            "reid_gallery_max_size": args.reid_gallery_max_size,
            "reid_gallery_add_cooldown": args.reid_gallery_add_cooldown,
            "reid_gallery_random_replace_prob": args.reid_gallery_random_replace_prob,
            "reid_gallery_random_replace_if_diverse_prob": args.reid_gallery_random_replace_if_diverse_prob,
            "stride": args.stride,
            "approx_eval_fps": args.approx_eval_fps,
        }

        config_metrics_total = base.SeqMetrics()
        config_rows = []

        try:
            for seq_idx, seq in enumerate(seqs):
                print(f"[config {config_id}] sequence: {seq}")

                with autocast_cm:
                    predictor = build_sam2_camera_predictor(
                        str(cfg_path),
                        str(ckpt_path),
                        reid_backend_name=args.reid_backend,
                    )

                base.set_predictor_thresholds(
                    predictor,
                    stable_frames_threshold=args.stable_frames_threshold,
                    stable_ious_threshold=args.stable_ious_threshold,
                    min_obj_score_logits=float(min_obj_score_logits),
                    kf_score_weight=args.kf_score_weight,
                    memory_bank_iou_threshold=args.memory_bank_iou_threshold,
                    memory_bank_obj_score_threshold=args.memory_bank_obj_score_threshold,
                    memory_bank_kf_score_threshold=args.memory_bank_kf_score_threshold,
                    reid_thr=float(reid_thr),
                )

                set_extra_reid_samurai_thresholds(
                    predictor,
                    memory_bank_reid_threshold=float(memory_bank_reid_threshold),
                    reid_gallery_add_sim_threshold=float(reid_gallery_add_sim_threshold),
                    reid_gallery_max_size=args.reid_gallery_max_size,
                    reid_gallery_add_cooldown=args.reid_gallery_add_cooldown,
                    reid_gallery_random_replace_prob=args.reid_gallery_random_replace_prob,
                    reid_gallery_random_replace_if_diverse_prob=args.reid_gallery_random_replace_if_diverse_prob,
                )

                if seq_idx == 0:
                    print_sweep_thresholds(predictor)

                # Short filenames to avoid Windows long-path errors.
                out_csv = config_out_dir / f"{short_id}_{seq}.csv"
                gt_mot_path = config_mot_dir / f"{seq}_gt.txt"
                pred_mot_path = config_mot_dir / f"{seq}_pred.txt"

                # Extra safety: create parents immediately before writing.
                safe_mkdir(out_csv.parent)
                safe_mkdir(gt_mot_path.parent)
                safe_mkdir(pred_mot_path.parent)

                with autocast_cm:
                    met = base.run_sequence(
                        seq_name=seq,
                        ktp_root=ktp_root,
                        predictor=predictor,
                        out_csv_path=out_csv,
                        reid_backend_name=args.reid_backend,
                        mot_gt_path=gt_mot_path,
                        mot_pred_path=pred_mot_path,
                        rotate_deg=args.rotate,
                        stride=args.stride,
                        max_frames=args.max_frames,
                        visible_area_frac=args.visible_area_frac,
                        visible_min_h=args.visible_min_h,
                        visible_min_w=args.visible_min_w,
                        seed_overlap_iou_max=args.seed_overlap_iou_max,
                        iou_match_thr=args.iou_match_thr,
                        eval_seed_frame=args.eval_seed_frame,
                        no_display=args.no_display,
                        display_scale=args.display_scale,
                        save_video=args.save_video,
                        save_video_fps=args.save_video_fps,
                        alpha=args.alpha,
                    )

                row = metrics_to_sweep_row(
                    run_prefix=run_prefix,
                    config_id=config_id,
                    label=label,
                    seq=seq,
                    reid_backend=args.reid_backend,
                    cfg_path=str(cfg_path),
                    ckpt_path=str(ckpt_path),
                    settings=settings,
                    met=met,
                    out_csv=str(out_csv),
                    gt_mot_path=str(gt_mot_path),
                    pred_mot_path=str(pred_mot_path),
                )

                config_rows.append(row)
                all_rows.append(row)

                accumulate_metrics(config_metrics_total, met)

                print(
                    f"  [{seq}] "
                    f"mota={row['mota']:.3f} "
                    f"match_rate={row['match_rate']:.3f} "
                    f"idsw={row['id_switches']} "
                    f"fp={row['false_positives']} "
                    f"fn={row['false_negatives']} "
                    f"mean_iou={row['mean_iou_when_matched']:.3f}"
                )

            row_all = metrics_to_sweep_row(
                run_prefix=run_prefix,
                config_id=config_id,
                label=label,
                seq="ALL",
                reid_backend=args.reid_backend,
                cfg_path=str(cfg_path),
                ckpt_path=str(ckpt_path),
                settings=settings,
                met=config_metrics_total,
                out_csv="",
                gt_mot_path="",
                pred_mot_path="",
            )

            config_rows.append(row_all)
            all_rows.append(row_all)

            print(
                f"  [ALL] "
                f"mota={row_all['mota']:.3f} "
                f"match_rate={row_all['match_rate']:.3f} "
                f"idsw={row_all['id_switches']} "
                f"fp={row_all['false_positives']} "
                f"fn={row_all['false_negatives']} "
                f"mean_iou={row_all['mean_iou_when_matched']:.3f}"
            )

            config_json_path = config_out_dir / f"{short_id}_summary.json"
            with config_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run": run_prefix,
                        "config_id": config_id,
                        "label": label,
                        "short_id": short_id,
                        "settings": settings,
                        "sequences": seqs,
                        "mot_exports_dir": str(config_mot_dir),
                        "per_sequence": [r for r in config_rows if r["seq"] != "ALL"],
                        "overall": row_all,
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            print(f"[FAILED config {config_id}] {repr(e)}")
            traceback.print_exc()
            failed_configs.append(
                {
                    "config_id": config_id,
                    "label": label,
                    "short_id": short_id,
                    "settings": settings,
                    "error": repr(e),
                }
            )

        # Write global summaries after every config, so progress is not lost.
        sorted_rows = sort_rows_for_operating_point(all_rows) if all_rows else []

        if all_rows:
            fieldnames = list(all_rows[0].keys())

            with global_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            if sorted_rows:
                with sorted_csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(sorted_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(sorted_rows)

        with global_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run": sweep_prefix,
                    "created_at": run_id,
                    "repo_root": str(REPO_ROOT),
                    "ktp_root": str(ktp_root),
                    "out_dir": str(sweep_dir),
                    "config": str(cfg_path),
                    "checkpoint": str(ckpt_path),
                    "reid_backend": args.reid_backend,
                    "sequences": seqs,
                    "num_configs_total_grid": len(full_configs),
                    "num_configs_requested": len(indexed_configs),
                    "num_rows": len(all_rows),
                    "settings_fixed": {
                        "stable_frames_threshold": args.stable_frames_threshold,
                        "stable_ious_threshold": args.stable_ious_threshold,
                        "kf_score_weight": args.kf_score_weight,
                        "memory_bank_iou_threshold": args.memory_bank_iou_threshold,
                        "memory_bank_obj_score_threshold": args.memory_bank_obj_score_threshold,
                        "memory_bank_kf_score_threshold": args.memory_bank_kf_score_threshold,
                        "reid_gallery_max_size": args.reid_gallery_max_size,
                        "reid_gallery_add_cooldown": args.reid_gallery_add_cooldown,
                        "reid_gallery_random_replace_prob": args.reid_gallery_random_replace_prob,
                        "reid_gallery_random_replace_if_diverse_prob": args.reid_gallery_random_replace_if_diverse_prob,
                        "stride": args.stride,
                        "approx_eval_fps": args.approx_eval_fps,
                        "visible_area_frac": args.visible_area_frac,
                        "visible_min_h": args.visible_min_h,
                        "visible_min_w": args.visible_min_w,
                        "seed_overlap_iou_max": args.seed_overlap_iou_max,
                        "iou_match_thr": args.iou_match_thr,
                        "eval_seed_frame": args.eval_seed_frame,
                    },
                    "sweep_values": {
                        "reid_thr_values": reid_thr_values,
                        "memory_bank_reid_threshold_values": memory_bank_reid_threshold_values,
                        "min_obj_score_logits_values": min_obj_score_logits_values,
                        "reid_gallery_add_sim_threshold_values": reid_gallery_add_sim_threshold_values,
                    },
                    "environment": {
                        "cuda_available": torch.cuda.is_available(),
                        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    },
                    "failed_configs": failed_configs,
                    "top_rows_by_internal_sort": sorted_rows[:20] if sorted_rows else [],
                },
                f,
                indent=2,
            )

    elapsed = time.time() - global_start

    print("\n[done]")
    print(f"  elapsed sec       : {elapsed:.1f}")
    print("  all rows csv      :", global_csv_path)
    print("  sorted ALL csv    :", sorted_csv_path)
    print("  summary json      :", global_json_path)
    print("  MOT exports root  :", mot_root)

    if failed_configs:
        print("\n[warning] Some configs failed:")
        for item in failed_configs:
            print(f"  config {item['config_id']}: {item['error']}")

    print("\nNext step:")
    print("  Use the exported MOT files for the best candidate configurations with TrackEval")
    print("  to compute the same final metrics used in the main 3-system comparison.")


if __name__ == "__main__":
    main()