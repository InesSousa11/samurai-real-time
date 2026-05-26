import argparse
import csv
import itertools
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_str_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def safe_name(x):
    x = str(x)
    x = x.replace(".", "p")
    x = x.replace("-", "m")
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    return x.strip("_")


def read_trackeval_summary(summary_path):
    """
    Reads TrackEval pedestrian_summary.txt.

    TrackEval usually writes:
        header line
        values line
    separated by whitespace.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return {}

    lines = [
        line.strip()
        for line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return {}

    headers = lines[0].split()
    values = lines[1].split()

    out = {}
    for h, v in zip(headers, values):
        try:
            if "." in v:
                out[h] = float(v)
            else:
                out[h] = int(v)
        except Exception:
            out[h] = v

    return out


def build_config_grid(args):
    min_obj_values = parse_float_list(args.min_obj_score_logits_values)
    mbrid_values = parse_float_list(args.memory_bank_reid_threshold_values)
    stable_iou_values = parse_float_list(args.stable_ious_threshold_values)
    kf_values = parse_float_list(args.kf_score_weight_values)
    reid_thr_values = parse_float_list(args.reid_thr_values)
    gadd_values = parse_float_list(args.reid_gallery_add_sim_threshold_values)
    mbo_values = parse_float_list(args.memory_bank_obj_score_threshold_values)
    mbi_values = parse_float_list(args.memory_bank_iou_threshold_values)
    mbkf_values = parse_float_list(args.memory_bank_kf_score_threshold_values)
    stable_frames_values = parse_str_list(args.stable_frames_threshold_values)

    # Defaults: if the user does not provide a list, keep the single value.
    if not min_obj_values:
        min_obj_values = [args.min_obj_score_logits]
    if not mbrid_values:
        mbrid_values = [args.memory_bank_reid_threshold]
    if not stable_iou_values:
        stable_iou_values = [args.stable_ious_threshold]
    if not kf_values:
        kf_values = [args.kf_score_weight]
    if not reid_thr_values:
        reid_thr_values = [args.reid_thr]
    if not gadd_values:
        gadd_values = [args.reid_gallery_add_sim_threshold]
    if not mbo_values:
        mbo_values = [args.memory_bank_obj_score_threshold]
    if not mbi_values:
        mbi_values = [args.memory_bank_iou_threshold]
    if not mbkf_values:
        mbkf_values = [args.memory_bank_kf_score_threshold]
    if not stable_frames_values:
        stable_frames_values = [str(args.stable_frames_threshold)]

    configs = []

    for (
        min_obj,
        mbrid,
        stable_iou,
        kf,
        reid_thr,
        gadd,
        mbo,
        mbi,
        mbkf,
        stable_frames,
    ) in itertools.product(
        min_obj_values,
        mbrid_values,
        stable_iou_values,
        kf_values,
        reid_thr_values,
        gadd_values,
        mbo_values,
        mbi_values,
        mbkf_values,
        stable_frames_values,
    ):
        cfg = {
            "min_obj_score_logits": float(min_obj),
            "memory_bank_reid_threshold": float(mbrid),
            "stable_ious_threshold": float(stable_iou),
            "kf_score_weight": float(kf),
            "reid_thr": float(reid_thr),
            "reid_gallery_add_sim_threshold": float(gadd),
            "memory_bank_obj_score_threshold": float(mbo),
            "memory_bank_iou_threshold": float(mbi),
            "memory_bank_kf_score_threshold": float(mbkf),
            "stable_frames_threshold": int(float(stable_frames)),
        }
        configs.append(cfg)

    return configs


def make_run_name(base_name, cfg, trackeval_threshold):
    parts = [
        base_name,
        f"obj{safe_name(cfg['min_obj_score_logits'])}",
        f"mbrid{safe_name(cfg['memory_bank_reid_threshold'])}",
        f"si{safe_name(cfg['stable_ious_threshold'])}",
        f"kf{safe_name(cfg['kf_score_weight'])}",
        f"rthr{safe_name(cfg['reid_thr'])}",
        f"gadd{safe_name(cfg['reid_gallery_add_sim_threshold'])}",
        f"te{safe_name(trackeval_threshold)}",
    ]
    return "_".join(parts)


def run_one_config(args, cfg, run_name, tracker_name):
    evaluator_script = Path(args.evaluator_script)

    if not evaluator_script.exists():
        raise FileNotFoundError(f"Evaluator script not found: {evaluator_script}")

    cmd = [
        sys.executable,
        str(evaluator_script),

        "--ktp_root", str(args.ktp_root),
        "--out_dir", str(args.out_dir),
        "--run_name", str(run_name),
        "--sequences", str(args.sequences),
        "--stride", str(args.stride),

        "--visible_area_frac", str(args.visible_area_frac),
        "--visible_min_h", str(args.visible_min_h),
        "--visible_min_w", str(args.visible_min_w),
        "--seed_overlap_iou_max", str(args.seed_overlap_iou_max),

        "--reid_backend", str(args.reid_backend),
        "--reid_thr", str(cfg["reid_thr"]),
        "--memory_bank_reid_threshold", str(cfg["memory_bank_reid_threshold"]),
        "--min_obj_score_logits", str(cfg["min_obj_score_logits"]),
        "--reid_gallery_add_sim_threshold", str(cfg["reid_gallery_add_sim_threshold"]),
        "--stable_frames_threshold", str(cfg["stable_frames_threshold"]),
        "--stable_ious_threshold", str(cfg["stable_ious_threshold"]),
        "--kf_score_weight", str(cfg["kf_score_weight"]),
        "--memory_bank_iou_threshold", str(cfg["memory_bank_iou_threshold"]),
        "--memory_bank_obj_score_threshold", str(cfg["memory_bank_obj_score_threshold"]),
        "--memory_bank_kf_score_threshold", str(cfg["memory_bank_kf_score_threshold"]),

        "--iou_match_thr", str(args.iou_match_thr),

        "--run_trackeval",
        "--trackeval_root", str(args.trackeval_root),
        "--trackeval_tracker_name", str(tracker_name),
        "--trackeval_benchmark", str(args.trackeval_benchmark),
        "--trackeval_split", str(args.trackeval_split),
        "--trackeval_threshold", str(args.trackeval_threshold),
    ]

    if args.ignore_predictions_without_gt:
        cmd.append("--ignore_predictions_without_gt")
        cmd.extend(["--ignore_predictions_without_gt_iou", str(args.ignore_predictions_without_gt_iou)])

    if args.save_video:
        cmd.append("--save_video")

    if args.no_display:
        cmd.append("--no_display")

    if args.trackeval_overwrite:
        cmd.append("--trackeval_overwrite")

    print("\n" + "=" * 100)
    print(f"[sweep] Running config: {run_name}")
    print("=" * 100)
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd)
    return completed.returncode


def main():
    ap = argparse.ArgumentParser()

    # Script paths
    ap.add_argument(
        "--evaluator_script",
        type=str,
        default="demo/KTP_reid_samurai_eval_run_with_trackeval.py",
        help="Single-run evaluator script to call.",
    )

    # Dataset / output
    ap.add_argument("--ktp_root", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--base_run_name", default="reid_samurai_sweep", type=str)
    ap.add_argument("--sequences", default="Arc,Rotation,Still,Translation", type=str)
    ap.add_argument("--stride", default=6, type=int)

    # Seeding/evaluation options
    ap.add_argument("--visible_area_frac", default=0.05, type=float)
    ap.add_argument("--visible_min_h", default=120, type=int)
    ap.add_argument("--visible_min_w", default=50, type=int)
    ap.add_argument("--seed_overlap_iou_max", default=0.05, type=float)
    ap.add_argument("--iou_match_thr", default=0.50, type=float)

    ap.add_argument("--ignore_predictions_without_gt", action="store_true")
    ap.add_argument("--ignore_predictions_without_gt_iou", default=0.01, type=float)

    # Fixed model options
    ap.add_argument("--reid_backend", default="transreid", type=str)

    # Single default values
    ap.add_argument("--reid_thr", default=0.80, type=float)
    ap.add_argument("--memory_bank_reid_threshold", default=0.65, type=float)
    ap.add_argument("--min_obj_score_logits", default=1.0, type=float)
    ap.add_argument("--reid_gallery_add_sim_threshold", default=0.85, type=float)
    ap.add_argument("--stable_frames_threshold", default=15, type=int)
    ap.add_argument("--stable_ious_threshold", default=0.30, type=float)
    ap.add_argument("--kf_score_weight", default=0.25, type=float)
    ap.add_argument("--memory_bank_iou_threshold", default=0.5, type=float)
    ap.add_argument("--memory_bank_obj_score_threshold", default=0.5, type=float)
    ap.add_argument("--memory_bank_kf_score_threshold", default=0.0, type=float)

    # Sweep value lists. Comma-separated.
    ap.add_argument("--reid_thr_values", default="", type=str)
    ap.add_argument("--memory_bank_reid_threshold_values", default="", type=str)
    ap.add_argument("--min_obj_score_logits_values", default="", type=str)
    ap.add_argument("--reid_gallery_add_sim_threshold_values", default="", type=str)
    ap.add_argument("--stable_frames_threshold_values", default="", type=str)
    ap.add_argument("--stable_ious_threshold_values", default="", type=str)
    ap.add_argument("--kf_score_weight_values", default="", type=str)
    ap.add_argument("--memory_bank_iou_threshold_values", default="", type=str)
    ap.add_argument("--memory_bank_obj_score_threshold_values", default="", type=str)
    ap.add_argument("--memory_bank_kf_score_threshold_values", default="", type=str)

    # TrackEval
    ap.add_argument("--trackeval_root", required=True, type=str)
    ap.add_argument("--trackeval_benchmark", default="KTP-5Hz", type=str)
    ap.add_argument("--trackeval_split", default="train", type=str)
    ap.add_argument("--trackeval_threshold", default=0.5, type=float)
    ap.add_argument("--trackeval_overwrite", action="store_true")

    # Runtime
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--max_configs", default=None, type=int)
    ap.add_argument("--start_config_idx", default=1, type=int)
    ap.add_argument("--end_config_idx", default=None, type=int)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_configs = build_config_grid(args)

    indexed_configs = list(enumerate(all_configs, start=1))

    indexed_configs = [
        (idx, cfg)
        for idx, cfg in indexed_configs
        if idx >= int(args.start_config_idx)
    ]

    if args.end_config_idx is not None:
        indexed_configs = [
            (idx, cfg)
            for idx, cfg in indexed_configs
            if idx <= int(args.end_config_idx)
        ]

    if args.max_configs is not None:
        indexed_configs = indexed_configs[: int(args.max_configs)]

    print(f"[sweep] Total configs in full grid: {len(all_configs)}")
    print(f"[sweep] Configs to run now: {len(indexed_configs)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = out_dir / f"{args.base_run_name}_trackeval_sweep_{timestamp}.csv"

    rows = []

    for idx, cfg in indexed_configs:
        run_name = make_run_name(args.base_run_name, cfg, args.trackeval_threshold)
        tracker_name = run_name

        print(f"\n[sweep] Config {idx}/{len(all_configs)}: {run_name}")

        returncode = run_one_config(args, cfg, run_name, tracker_name)

        summary_path = (
            Path(args.trackeval_root)
            / "data"
            / "trackers"
            / "mot_challenge"
            / f"{args.trackeval_benchmark}-{args.trackeval_split}"
            / tracker_name
            / "pedestrian_summary.txt"
        )

        metrics = read_trackeval_summary(summary_path)

        row = {
            "config_idx": idx,
            "run_name": run_name,
            "tracker_name": tracker_name,
            "returncode": returncode,
            "summary_path": str(summary_path),
            **cfg,
        }

        for key in [
            "HOTA",
            "DetA",
            "AssA",
            "MOTA",
            "IDF1",
            "IDP",
            "IDR",
            "IDSW",
            "CLR_FP",
            "CLR_FN",
            "CLR_TP",
        ]:
            row[key] = metrics.get(key, "")

        rows.append(row)

        fieldnames = list(rows[0].keys())
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[sweep] Updated results CSV: {results_csv}")

        if returncode != 0:
            print(f"[sweep] Config failed with return code {returncode}: {run_name}")
            if not args.continue_on_error:
                raise SystemExit(returncode)

    print("\n[sweep] Done.")
    print(f"[sweep] Results CSV: {results_csv}")

    if rows:
        sortable = []
        for r in rows:
            try:
                hota = float(r.get("HOTA", -1))
            except Exception:
                hota = -1
            try:
                idf1 = float(r.get("IDF1", -1))
            except Exception:
                idf1 = -1
            try:
                mota = float(r.get("MOTA", -1))
            except Exception:
                mota = -1

            sortable.append((hota, idf1, mota, r))

        sortable.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        print("\n[sweep] Top configs by HOTA, then IDF1, then MOTA:")
        for rank, (hota, idf1, mota, r) in enumerate(sortable[:10], start=1):
            print(
                f"{rank:02d}. "
                f"HOTA={r.get('HOTA', '')} "
                f"IDF1={r.get('IDF1', '')} "
                f"MOTA={r.get('MOTA', '')} "
                f"IDSW={r.get('IDSW', '')} "
                f"name={r.get('run_name', '')}"
            )


if __name__ == "__main__":
    main()