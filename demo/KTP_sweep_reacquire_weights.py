#!/usr/bin/env python3
# KTP_sweep_reacquire_weights.py
#
# Sweep reacquisition-score fusion weights for ReID-SAMURAI on KTP.
#
# This script keeps the selected operating point fixed, e.g. config 44:
#   reid_thr                       = 0.80
#   memory_bank_reid_threshold     = 0.65
#   min_obj_score_logits           = 1.0
#   reid_gallery_add_sim_threshold = 0.85
#
# It then sweeps the weights used inside:
#   _combined_reacquire_score()
#
# The goal is to test whether giving more or less importance to ReID,
# objectness, Kalman/filter motion score, and IoU improves identity
# preservation and tracking quality.
#
# Example:
# python demo/KTP_sweep_reacquire_weights.py `
#   --ktp_root "C:\Users\inesg\OneDrive\Desktop\Thesis\datasets\KTP" `
#   --out_dir "C:\tmp\reid_samurai_reacq_weight_sweep" `
#   --run_name reacq_weight_sweep_config44_5hz `
#   --sequences Arc,Rotation,Still,Translation `
#   --stride 6 `
#   --no_display

import sys
import csv
import json
import time
import argparse
import traceback
from pathlib import Path
from contextlib import nullcontext
from typing import List, Dict, Any, Optional, Tuple

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
# Helpers
# ---------------------------------------------------------------------
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_float_for_label(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_weight_sets(s: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse weight sets of the form:

        "0.50,0.35,0.10,0.05;0.70,0.20,0.10,0.00;1.00,0.00,0.00,0.00"

    Each set corresponds to:
        reacquire_w_reid, reacquire_w_obj, reacquire_w_kf, reacquire_w_iou
    """
    weight_sets = []

    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = [float(x.strip()) for x in item.split(",") if x.strip()]

        if len(parts) != 4:
            raise ValueError(
                f"Invalid weight set '{item}'. "
                "Each weight set must have exactly four values: "
                "w_reid,w_obj,w_kf,w_iou"
            )

        weight_sets.append(tuple(parts))

    if not weight_sets:
        raise ValueError("No valid weight sets were provided.")

    return weight_sets


def make_weight_label(config_id: int, w_reid: float, w_obj: float, w_kf: float, w_iou: float) -> str:
    return (
        f"cfg{config_id:04d}"
        f"_wr{fmt_float_for_label(w_reid)}"
        f"_wo{fmt_float_for_label(w_obj)}"
        f"_wk{fmt_float_for_label(w_kf)}"
        f"_wi{fmt_float_for_label(w_iou)}"
    )


# ---------------------------------------------------------------------
# Predictor parameter setters
# ---------------------------------------------------------------------
def set_attr_and_state(predictor, name: str, value) -> None:
    """
    Set a parameter both as a predictor attribute and, if available, inside
    condition_state. The predictor attribute is the important part for the
    reacquisition score function if it uses getattr(self, ...).
    """
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


def set_extra_reid_samurai_thresholds(
    predictor,
    memory_bank_reid_threshold: Optional[float] = None,
    reid_gallery_add_sim_threshold: Optional[float] = None,
    reid_gallery_max_size: Optional[int] = None,
    reid_gallery_add_cooldown: Optional[int] = None,
    reid_gallery_random_replace_prob: Optional[float] = None,
    reid_gallery_random_replace_if_diverse_prob: Optional[float] = None,
) -> None:
    if memory_bank_reid_threshold is not None:
        set_attr_and_state(predictor, "memory_bank_reid_threshold", float(memory_bank_reid_threshold))

    if reid_gallery_add_sim_threshold is not None:
        set_attr_and_state(predictor, "reid_gallery_add_sim_threshold", float(reid_gallery_add_sim_threshold))

    if reid_gallery_max_size is not None:
        set_attr_and_state(predictor, "reid_gallery_max_size", int(reid_gallery_max_size))

    if reid_gallery_add_cooldown is not None:
        set_attr_and_state(predictor, "reid_gallery_add_cooldown", int(reid_gallery_add_cooldown))

    if reid_gallery_random_replace_prob is not None:
        set_attr_and_state(predictor, "reid_gallery_random_replace_prob", float(reid_gallery_random_replace_prob))

    if reid_gallery_random_replace_if_diverse_prob is not None:
        set_attr_and_state(
            predictor,
            "reid_gallery_random_replace_if_diverse_prob",
            float(reid_gallery_random_replace_if_diverse_prob),
        )


def set_reacquire_weights(
    predictor,
    w_reid: float,
    w_obj: float,
    w_kf: float,
    w_iou: float,
) -> None:
    set_attr_and_state(predictor, "reacquire_w_reid", float(w_reid))
    set_attr_and_state(predictor, "reacquire_w_obj", float(w_obj))
    set_attr_and_state(predictor, "reacquire_w_kf", float(w_kf))
    set_attr_and_state(predictor, "reacquire_w_iou", float(w_iou))


def print_predictor_settings(predictor) -> None:
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
        "reacquire_w_reid",
        "reacquire_w_obj",
        "reacquire_w_kf",
        "reacquire_w_iou",
        "samurai_mode",
    ]

    print("[predictor settings]")
    for key in keys:
        print(f"  {key}: {getattr(predictor, key, None)}")
    print("")


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def metrics_to_row(
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

        "reacquire_w_reid": settings["reacquire_w_reid"],
        "reacquire_w_obj": settings["reacquire_w_obj"],
        "reacquire_w_kf": settings["reacquire_w_kf"],
        "reacquire_w_iou": settings["reacquire_w_iou"],

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


def accumulate_metrics(total, met) -> None:
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
    ap.add_argument("--run_name", type=str, default="reid_samurai_reacq_weight_sweep")

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

    # Fixed selected operating point, defaulting to config 44.
    ap.add_argument("--reid_thr", type=float, default=0.80)
    ap.add_argument("--memory_bank_reid_threshold", type=float, default=0.65)
    ap.add_argument("--min_obj_score_logits", type=float, default=1.0)
    ap.add_argument("--reid_gallery_add_sim_threshold", type=float, default=0.85)

    # Other fixed thresholds.
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

    # Weight sets.
    ap.add_argument(
        "--weight_sets",
        type=str,
        default=(
            "0.50,0.35,0.10,0.05;"
            "0.60,0.25,0.10,0.05;"
            "0.70,0.20,0.07,0.03;"
            "0.80,0.15,0.05,0.00;"
            "0.90,0.10,0.00,0.00;"
            "1.00,0.00,0.00,0.00;"
            "0.50,0.50,0.00,0.00;"
            "0.65,0.35,0.00,0.00;"
            "0.75,0.25,0.00,0.00;"
            "0.50,0.25,0.20,0.05;"
            "0.50,0.25,0.10,0.15;"
            "0.40,0.40,0.15,0.05"
        ),
        help=(
            "Semicolon-separated list of weight sets. "
            "Each set is w_reid,w_obj,w_kf,w_iou. "
            "Example: '0.5,0.35,0.1,0.05;1,0,0,0'"
        ),
    )

    # KTP evaluation procedure parameters.
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6)
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

    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_video_fps", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--max_configs", type=int, default=-1)
    ap.add_argument("--start_config", type=int, default=0)

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
    full_weight_sets = parse_weight_sets(args.weight_sets)

    indexed_configs = list(enumerate(full_weight_sets))

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
    print("  configs  :", len(indexed_configs), "of", len(full_weight_sets))
    print("  fixed operating point:")
    print("    reid_thr                      :", args.reid_thr)
    print("    memory_bank_reid_threshold    :", args.memory_bank_reid_threshold)
    print("    min_obj_score_logits          :", args.min_obj_score_logits)
    print("    reid_gallery_add_sim_threshold:", args.reid_gallery_add_sim_threshold)
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

    for config_id, weights in indexed_configs:
        w_reid, w_obj, w_kf, w_iou = weights

        label = make_weight_label(config_id, w_reid, w_obj, w_kf, w_iou)
        short_id = f"cfg{config_id:04d}"
        run_prefix = f"{sweep_prefix}_{short_id}"

        print("=" * 80)
        print(f"[config {config_id}] {label}")
        print("=" * 80)

        config_mot_dir = mot_root / short_id
        safe_mkdir(config_mot_dir)

        config_out_dir = per_config_dir / short_id
        safe_mkdir(config_out_dir)

        settings = {
            "reacquire_w_reid": float(w_reid),
            "reacquire_w_obj": float(w_obj),
            "reacquire_w_kf": float(w_kf),
            "reacquire_w_iou": float(w_iou),

            "stable_frames_threshold": args.stable_frames_threshold,
            "stable_ious_threshold": args.stable_ious_threshold,
            "min_obj_score_logits": args.min_obj_score_logits,
            "kf_score_weight": args.kf_score_weight,
            "memory_bank_iou_threshold": args.memory_bank_iou_threshold,
            "memory_bank_obj_score_threshold": args.memory_bank_obj_score_threshold,
            "memory_bank_kf_score_threshold": args.memory_bank_kf_score_threshold,
            "memory_bank_reid_threshold": args.memory_bank_reid_threshold,
            "reid_thr": args.reid_thr,
            "reid_gallery_add_sim_threshold": args.reid_gallery_add_sim_threshold,
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
                    min_obj_score_logits=float(args.min_obj_score_logits),
                    kf_score_weight=float(args.kf_score_weight),
                    memory_bank_iou_threshold=float(args.memory_bank_iou_threshold),
                    memory_bank_obj_score_threshold=float(args.memory_bank_obj_score_threshold),
                    memory_bank_kf_score_threshold=float(args.memory_bank_kf_score_threshold),
                    reid_thr=float(args.reid_thr),
                )

                set_extra_reid_samurai_thresholds(
                    predictor,
                    memory_bank_reid_threshold=float(args.memory_bank_reid_threshold),
                    reid_gallery_add_sim_threshold=float(args.reid_gallery_add_sim_threshold),
                    reid_gallery_max_size=int(args.reid_gallery_max_size),
                    reid_gallery_add_cooldown=int(args.reid_gallery_add_cooldown),
                    reid_gallery_random_replace_prob=float(args.reid_gallery_random_replace_prob),
                    reid_gallery_random_replace_if_diverse_prob=float(
                        args.reid_gallery_random_replace_if_diverse_prob
                    ),
                )

                set_reacquire_weights(
                    predictor,
                    w_reid=float(w_reid),
                    w_obj=float(w_obj),
                    w_kf=float(w_kf),
                    w_iou=float(w_iou),
                )

                if seq_idx == 0:
                    print_predictor_settings(predictor)

                out_csv = config_out_dir / f"{short_id}_{seq}.csv"
                gt_mot_path = config_mot_dir / f"{seq}_gt.txt"
                pred_mot_path = config_mot_dir / f"{seq}_pred.txt"

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

                row = metrics_to_row(
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

            row_all = metrics_to_row(
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

        # Write global summaries after every config.
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
                    "num_configs_total_grid": len(full_weight_sets),
                    "num_configs_requested": len(indexed_configs),
                    "num_rows": len(all_rows),
                    "fixed_operating_point": {
                        "reid_thr": args.reid_thr,
                        "memory_bank_reid_threshold": args.memory_bank_reid_threshold,
                        "min_obj_score_logits": args.min_obj_score_logits,
                        "reid_gallery_add_sim_threshold": args.reid_gallery_add_sim_threshold,
                    },
                    "weight_sets": [
                        {
                            "config_id": i,
                            "reacquire_w_reid": ws[0],
                            "reacquire_w_obj": ws[1],
                            "reacquire_w_kf": ws[2],
                            "reacquire_w_iou": ws[3],
                        }
                        for i, ws in enumerate(full_weight_sets)
                    ],
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
    print("  Inspect the sorted ALL CSV first.")
    print("  Then run TrackEval only for the best one or two weight configurations.")


if __name__ == "__main__":
    main()