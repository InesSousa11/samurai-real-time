#!/usr/bin/env python3
"""
Runtime benchmark for the final ReID-SAMURAI system on KTP.

This script measures runtime only. It does not run TrackEval, save videos,
write MOT files, or create review frames.

It imports helper functions and constants from:
    demo/KTP_reid_samurai_eval_run_with_trackeval.py

Measured times:
  - preprocess_ms: image loading + rotation + RGB conversion
  - model_ms: seeding operations + predictor.track + mask-to-box postprocessing
  - track_call_ms: predictor.track only

For the thesis table, use model_ms/FPS as the main runtime number.
"""

from __future__ import annotations

import contextlib
import io
import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------
# Import the existing KTP evaluation script as a module
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "demo" else Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import KTP_reid_samurai_eval_run_with_trackeval as ktp_eval  # noqa: E402
from sam2.build_sam import build_sam2_camera_predictor  # noqa: E402


# ---------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------

def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_now() -> float:
    sync_cuda()
    return time.perf_counter()


def summarize_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "frames": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    mean_ms = float(np.mean(arr))

    return {
        "frames": int(arr.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def fmt2(x: float) -> str:
    return f"{float(x):.2f}"


# ---------------------------------------------------------------------
# KTP runtime sequence
# ---------------------------------------------------------------------

@torch.inference_mode()
def benchmark_sequence(
    seq_name: str,
    ktp_root: Path,
    predictor,
    stride: int,
    max_frames: int,
    rotate_deg: int,
    visible_area_frac: float,
    visible_min_h: int,
    visible_min_w: int,
    seed_overlap_iou_max: float,
    warmup_frames: int,
    suppress_model_output: bool = True,
) -> Tuple[List[dict], Dict[str, float]]:
    img_dir = ktp_root / "images" / seq_name / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq_name}_gt2D.txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_map = ktp_eval.parse_gt2d_file(gt_path)

    frames_all = list(img_dir.glob("*.jpg"))
    if not frames_all:
        raise RuntimeError(f"No .jpg frames found in {img_dir}")

    items = []
    for p in frames_all:
        ts_str = ktp_eval.ts_from_filename_robust(p)
        if ts_str is None:
            continue
        try:
            ts_f = float(ts_str)
        except Exception:
            continue
        items.append((ts_f, ts_str, p))

    if not items:
        raise RuntimeError(f"No parseable timestamped frames found in {img_dir}")

    items.sort(key=lambda t: t[0])

    frames: List[Path] = []
    ts_by_path: Dict[Path, str] = {}
    seen_ts = set()

    for _, ts_str, p in items:
        if ts_str in seen_ts:
            continue
        seen_ts.add(ts_str)
        frames.append(p)
        ts_by_path[p] = ts_str

    if stride > 1:
        frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]

    if not frames:
        raise RuntimeError(f"No frames left after stride/max_frames in {img_dir}")

    bgr0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")

    bgr0 = ktp_eval.rotate_frame(bgr0, rotate_deg)
    H, W = bgr0.shape[:2]
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)

    init_t0 = timed_now()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        predictor.load_first_frame(rgb0)
        ktp_eval.sync_reid_threshold(predictor, getattr(predictor, "reid_thr", None))
        ktp_eval.sync_runtime_thresholds_to_state(predictor)
    init_t1 = timed_now()
    init_ms = 1000.0 * (init_t1 - init_t0)

    seeded = set()
    rows: List[dict] = []

    def seed_bbox(gt_id: int, bbox_xyxy, rgb_frame: np.ndarray, late: bool) -> bool:
        bbox = np.array(
            [[bbox_xyxy[0], bbox_xyxy[1]], [bbox_xyxy[2], bbox_xyxy[3]]],
            dtype=np.float32,
        )

        if not late:
            predictor.add_new_prompt(
                frame_idx=0,
                obj_id=int(gt_id),
                bbox=bbox,
            )
            ktp_eval.sync_reid_threshold(predictor, getattr(predictor, "reid_thr", None))
            ktp_eval.sync_runtime_thresholds_to_state(predictor)
        else:
            predictor.add_conditioning_frame(rgb_frame)
            ktp_eval.sync_reid_threshold(predictor, getattr(predictor, "reid_thr", None))
            ktp_eval.sync_runtime_thresholds_to_state(predictor)

            predictor.add_new_prompt_during_track(
                bbox=bbox,
                if_new_target=True,
                obj_id=int(gt_id),
                labels=None,
                clear_old_points=True,
            )
            ktp_eval.sync_reid_threshold(predictor, getattr(predictor, "reid_thr", None))
            ktp_eval.sync_runtime_thresholds_to_state(predictor)

        return True

    def get_obj_id_to_idx() -> Dict[int, int]:
        m = None

        if hasattr(predictor, "condition_state"):
            m = predictor.condition_state.get("obj_id_to_idx", None)

        if m is None and hasattr(predictor, "obj_id_to_idx"):
            m = getattr(predictor, "obj_id_to_idx", None)

        if m is None:
            return {}

        try:
            return {int(k): int(v) for k, v in dict(m).items()}
        except Exception:
            return {}

    def logits_for_obj_id(out_mask_logits, obj_id: int) -> Optional[torch.Tensor]:
        obj_id_to_idx = get_obj_id_to_idx()

        if obj_id not in obj_id_to_idx:
            return None

        obj_idx = obj_id_to_idx[obj_id]

        if out_mask_logits is None:
            return None

        if torch.is_tensor(out_mask_logits):
            if out_mask_logits.ndim < 3:
                return None
            if not (0 <= obj_idx < int(out_mask_logits.shape[0])):
                return None
            return out_mask_logits[obj_idx]

        if isinstance(out_mask_logits, (list, tuple)):
            if not (0 <= obj_idx < len(out_mask_logits)):
                return None
            return out_mask_logits[obj_idx] if torch.is_tensor(out_mask_logits[obj_idx]) else None

        return None

    for fidx, fp in enumerate(frames):
        ts = ts_by_path.get(fp, None)
        if ts is None:
            continue

        prep_t0 = time.perf_counter()

        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        bgr = ktp_eval.rotate_frame(bgr, rotate_deg)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        prep_t1 = time.perf_counter()
        preprocess_ms = 1000.0 * (prep_t1 - prep_t0)

        gt_dets = gt_map.get(ts, [])

        gt_bb_by_id_all: Dict[int, Tuple[int, int, int, int]] = {}
        for gid, x, y, w, h in gt_dets:
            gt_bb_by_id_all[int(gid)] = ktp_eval.clamp_bbox_xyxy(
                ktp_eval.bbox_xywh_to_xyxy(x, y, w, h),
                W,
                H,
            )

        seeded_now = 0
        pred_count = 0

        model_t0 = timed_now()

        if suppress_model_output:
            output_cm = contextlib.redirect_stdout(io.StringIO())
            error_cm = contextlib.redirect_stderr(io.StringIO())
        else:
            output_cm = contextlib.nullcontext()
            error_cm = contextlib.nullcontext()

        with output_cm, error_cm:
            for gid, x, y, w, h in gt_dets:
                gid = int(gid)

                if gid in seeded:
                    continue

                bb = gt_bb_by_id_all[gid]
                bw = max(0, bb[2] - bb[0])
                bh = max(0, bb[3] - bb[1])
                area = bw * bh
                area_frac = area / float(W * H + 1e-9)

                visible_ok = (
                    area_frac >= float(visible_area_frac)
                    and bh >= int(visible_min_h)
                )

                if visible_min_w and int(visible_min_w) > 0:
                    visible_ok = visible_ok and (bw >= int(visible_min_w))

                if not visible_ok:
                    continue

                max_iou_other = 0.0
                for ogid, obb in gt_bb_by_id_all.items():
                    if int(ogid) == gid:
                        continue
                    max_iou_other = max(max_iou_other, ktp_eval.iou_xyxy(bb, obb))

                if max_iou_other > float(seed_overlap_iou_max):
                    continue

                late = fidx != 0

                try:
                    ok = seed_bbox(gid, bb, rgb_frame=rgb, late=late)
                except Exception:
                    ok = False

                if ok:
                    seeded.add(gid)
                    seeded_now += 1

            track_t0 = timed_now()

            try:
                out_obj_ids, out_mask_logits = predictor.track(rgb)
            except Exception:
                out_obj_ids, out_mask_logits = [], None

            track_t1 = timed_now()
            track_call_ms = 1000.0 * (track_t1 - track_t0)

            if out_obj_ids is None:
                out_obj_ids = []
            if torch.is_tensor(out_obj_ids):
                out_obj_ids = [
                    int(x) for x in out_obj_ids.detach().reshape(-1).tolist()
                ]
            elif isinstance(out_obj_ids, (list, tuple)):
                out_obj_ids = [int(x) for x in out_obj_ids]
            else:
                out_obj_ids = [int(out_obj_ids)]

            for oid in out_obj_ids:
                logits = logits_for_obj_id(out_mask_logits, int(oid))
                if logits is None:
                    continue

                res = ktp_eval.logits_to_mask_bbox(logits)
                if res is None:
                    continue

                _mask_bool, _bb = res
                pred_count += 1

        model_t1 = timed_now()
        model_ms = 1000.0 * (model_t1 - model_t0)

        included = int(fidx >= int(warmup_frames))

        rows.append(
            {
                "seq": seq_name,
                "frame_idx": int(fidx),
                "timestamp": ts,
                "included": included,
                "preprocess_ms": preprocess_ms,
                "model_ms": model_ms,
                "track_call_ms": track_call_ms,
                "seeded_now": int(seeded_now),
                "num_seeded_total": int(len(seeded)),
                "num_pred_objects": int(pred_count),
            }
        )

    model_values = [r["model_ms"] for r in rows if r["included"]]
    track_values = [r["track_call_ms"] for r in rows if r["included"]]
    preprocess_values = [r["preprocess_ms"] for r in rows if r["included"]]

    summary = {
        "seq": seq_name,
        "processed_frames": len(rows),
        "timed_frames": len(model_values),
        "warmup_frames": int(warmup_frames),
        "init_ms": float(init_ms),
        "model": summarize_ms(model_values),
        "track_call": summarize_ms(track_values),
        "preprocess": summarize_ms(preprocess_values),
    }

    return rows, summary


# ---------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------

def make_latex_runtime_table(per_sequence: List[dict], overall: dict) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Runtime performance of the final ReID-SAMURAI system on the KTP sequences. FPS and frame times are computed from the model-processing time, excluding disk writing, TrackEval export, plotting, and visualization.}",
        r"\label{tab:reid_samurai_runtime}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Sequence & Timed frames & FPS $\uparrow$ & Mean ms/frame $\downarrow$ & Median ms/frame $\downarrow$ & P90 ms/frame $\downarrow$ \\",
        r"\midrule",
    ]

    for s in per_sequence:
        m = s["model"]
        lines.append(
            f"{s['seq']} & "
            f"{int(s['timed_frames'])} & "
            f"{fmt2(m['fps'])} & "
            f"{fmt2(m['mean_ms'])} & "
            f"{fmt2(m['median_ms'])} & "
            f"{fmt2(m['p90_ms'])} \\\\"
        )

    m = overall["model"]
    lines += [
        r"\midrule",
        f"\\textbf{{Overall}} & "
        f"\\textbf{{{int(overall['timed_frames'])}}} & "
        f"\\textbf{{{fmt2(m['fps'])}}} & "
        f"\\textbf{{{fmt2(m['mean_ms'])}}} & "
        f"\\textbf{{{fmt2(m['median_ms'])}}} & "
        f"\\textbf{{{fmt2(m['p90_ms'])}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ktp_root", required=True)
    ap.add_argument("--sequences", default="Arc,Rotation,Still,Translation")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", default="reid_samurai_runtime")
    ap.add_argument(
        "--suppress_model_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress stdout/stderr during timed model calls.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--warmup_frames", type=int, default=10)
    ap.add_argument("--rotate", type=int, default=0)

    ap.add_argument("--visible_area_frac", type=float, default=0.05)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--visible_min_w", type=int, default=50)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.05)

    ap.add_argument("--reid_backend", default="transreid",
                    choices=["osnet_x1_0", "osnet_ain_x1_0", "transreid"])

    ap.add_argument("--stable_frames_threshold", type=int, default=15)
    ap.add_argument("--stable_ious_threshold", type=float, default=0.30)
    ap.add_argument("--min_obj_score_logits", type=float, default=1.0)
    ap.add_argument("--kf_score_weight", type=float, default=0.25)
    ap.add_argument("--memory_bank_iou_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_obj_score_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_kf_score_threshold", type=float, default=0.0)
    ap.add_argument("--memory_bank_reid_threshold", type=float, default=0.75)
    ap.add_argument("--reid_thr", type=float, default=0.80)

    ap.add_argument("--reid_gallery_max_size", type=int, default=10)
    ap.add_argument("--reid_gallery_add_sim_threshold", type=float, default=0.85)
    ap.add_argument("--reid_gallery_add_cooldown", type=int, default=10)
    ap.add_argument("--reid_gallery_random_replace_prob", type=float, default=0.15)
    ap.add_argument("--reid_gallery_random_replace_if_diverse_prob", type=float, default=0.30)
    ap.add_argument("--reid_gallery_anchor_protect", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    ktp_eval.set_global_seed(args.seed, deterministic=args.deterministic)

    ktp_root = Path(args.ktp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ktp_eval.CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ktp_eval.CKPT_PATH}")
    if not ktp_eval.CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {ktp_eval.CFG_PATH}")

    print("[runtime benchmark]")
    print("  REPO_ROOT:", REPO_ROOT)
    print("  CKPT     :", ktp_eval.CKPT_PATH)
    print("  CFG      :", ktp_eval.CFG_PATH)
    print("  KTP_ROOT :", ktp_root)
    print("  OUT_DIR  :", out_dir)
    print("  RUN_NAME :", args.run_name)
    print("  stride   :", args.stride)
    print("  seed     :", args.seed)

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{args.run_name}_{run_id}"

    per_frame_csv = out_dir / f"{run_prefix}__per_frame_runtime.csv"
    summary_json = out_dir / f"{run_prefix}__runtime_summary.json"
    table_tex = out_dir / "reid_samurai_runtime.tex"

    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    all_rows: List[dict] = []
    per_sequence_summaries: List[dict] = []

    for seq in sequences:
        print(f"\n[sequence] {seq}")

        with autocast_cm:
            predictor = build_sam2_camera_predictor(
                str(ktp_eval.CFG_PATH),
                str(ktp_eval.CKPT_PATH),
                reid_backend_name=args.reid_backend,
            )

        ktp_eval.set_predictor_thresholds(
            predictor,
            stable_frames_threshold=args.stable_frames_threshold,
            stable_ious_threshold=args.stable_ious_threshold,
            min_obj_score_logits=args.min_obj_score_logits,
            kf_score_weight=args.kf_score_weight,
            memory_bank_iou_threshold=args.memory_bank_iou_threshold,
            memory_bank_obj_score_threshold=args.memory_bank_obj_score_threshold,
            memory_bank_kf_score_threshold=args.memory_bank_kf_score_threshold,
            memory_bank_reid_threshold=args.memory_bank_reid_threshold,
            reid_thr=args.reid_thr,
            reid_gallery_max_size=args.reid_gallery_max_size,
            reid_gallery_add_sim_threshold=args.reid_gallery_add_sim_threshold,
            reid_gallery_add_cooldown=args.reid_gallery_add_cooldown,
            reid_gallery_random_replace_prob=args.reid_gallery_random_replace_prob,
            reid_gallery_random_replace_if_diverse_prob=args.reid_gallery_random_replace_if_diverse_prob,
            reid_gallery_anchor_protect=args.reid_gallery_anchor_protect,
        )

        with autocast_cm:
            rows, summary = benchmark_sequence(
                seq_name=seq,
                ktp_root=ktp_root,
                predictor=predictor,
                stride=args.stride,
                max_frames=args.max_frames,
                rotate_deg=args.rotate,
                visible_area_frac=args.visible_area_frac,
                visible_min_h=args.visible_min_h,
                visible_min_w=args.visible_min_w,
                seed_overlap_iou_max=args.seed_overlap_iou_max,
                warmup_frames=args.warmup_frames,
            )

        all_rows.extend(rows)
        per_sequence_summaries.append(summary)

        m = summary["model"]
        print(
            f"  timed_frames={summary['timed_frames']} "
            f"fps={m['fps']:.2f} "
            f"mean={m['mean_ms']:.2f}ms "
            f"median={m['median_ms']:.2f}ms "
            f"p90={m['p90_ms']:.2f}ms"
        )

    model_values = [r["model_ms"] for r in all_rows if r["included"]]
    track_values = [r["track_call_ms"] for r in all_rows if r["included"]]
    preprocess_values = [r["preprocess_ms"] for r in all_rows if r["included"]]

    overall = {
        "seq": "Overall",
        "processed_frames": len(all_rows),
        "timed_frames": len(model_values),
        "warmup_frames_per_sequence": args.warmup_frames,
        "model": summarize_ms(model_values),
        "track_call": summarize_ms(track_values),
        "preprocess": summarize_ms(preprocess_values),
    }

    # Per-frame CSV
    fieldnames = [
        "seq",
        "frame_idx",
        "timestamp",
        "included",
        "preprocess_ms",
        "model_ms",
        "track_call_ms",
        "seeded_now",
        "num_seeded_total",
        "num_pred_objects",
    ]

    with per_frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "run": run_prefix,
        "created_at": run_id,
        "checkpoint": str(ktp_eval.CKPT_PATH),
        "config": str(ktp_eval.CFG_PATH),
        "ktp_root": str(ktp_root),
        "sequences": sequences,
        "settings": vars(args),
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "notes": {
            "main_runtime_metric": "model_ms",
            "model_ms_includes": [
                "KTP prompt/seeding operations",
                "predictor.track",
                "mask-logit to bounding-box postprocessing",
            ],
            "model_ms_excludes": [
                "disk writing",
                "TrackEval export",
                "plotting",
                "video saving",
                "visualization/display",
            ],
            "preprocess_ms": "image loading + rotation + RGB conversion, reported separately",
            "warmup_policy": f"first {args.warmup_frames} processed frames per sequence excluded from timing statistics",
        },
        "per_sequence": per_sequence_summaries,
        "overall": overall,
        "per_frame_csv": str(per_frame_csv),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    table_tex.write_text(
        make_latex_runtime_table(per_sequence_summaries, overall),
        encoding="utf-8",
    )

    m = overall["model"]
    print("\n[overall]")
    print(f"  timed_frames : {overall['timed_frames']}")
    print(f"  FPS          : {m['fps']:.2f}")
    print(f"  mean ms/frame: {m['mean_ms']:.2f}")
    print(f"  median ms    : {m['median_ms']:.2f}")
    print(f"  p90 ms       : {m['p90_ms']:.2f}")
    print("")
    print("[saved]")
    print("  per-frame CSV:", per_frame_csv)
    print("  summary JSON :", summary_json)
    print("  LaTeX table  :", table_tex)


if __name__ == "__main__":
    main()