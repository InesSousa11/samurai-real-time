#!/usr/bin/env python3
# prepare_ktp_for_trackeval.py
#
# Converts the MOT exports produced by KTP_eval_run.py into the folder
# structure expected by TrackEval's MOTChallenge evaluator.
#
# Input folder expected:
#   Arc_gt.txt
#   Arc_pred.txt
#   Rotation_gt.txt
#   Rotation_pred.txt
#   Still_gt.txt
#   Still_pred.txt
#   Translation_gt.txt
#   Translation_pred.txt
#
# Output structure created:
#   <trackeval_root>/data/gt/mot_challenge/<benchmark>-<split>/
#       seqmaps/<benchmark>-<split>.txt
#       Arc/gt/gt.txt
#       Arc/seqinfo.ini
#       ...
#
#   <trackeval_root>/data/trackers/mot_challenge/<benchmark>-<split>/<tracker_name>/data/
#       Arc.txt
#       Rotation.txt
#       Still.txt
#       Translation.txt
#
# Example:
# python demo/prepare_ktp_for_trackeval.py ^
#   --mot_export_dir "C:\tmp\final_eval_reid_samurai_config44\mot_exports\..." ^
#   --trackeval_root "C:\Users\inesg\OneDrive\Desktop\Thesis\code\TrackEval" ^
#   --tracker_name ReID-SAMURAI-config44 ^
#   --benchmark KTP-5Hz ^
#   --split train

import argparse
import shutil
from pathlib import Path
from typing import List


DEFAULT_SEQUENCES = ["Arc", "Rotation", "Still", "Translation"]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_max_frame(mot_file: Path) -> int:
    """
    Returns the maximum frame index found in a MOT txt file.
    MOT format:
      frame,id,x,y,w,h,conf,...
    """
    max_frame = 0

    if not mot_file.exists():
        return max_frame

    with mot_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 1:
                continue

            try:
                frame_id = int(float(parts[0]))
                max_frame = max(max_frame, frame_id)
            except ValueError:
                continue

    return max_frame


def write_seqinfo(
    seq_dir: Path,
    seq_name: str,
    seq_length: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    """
    Writes seqinfo.ini for MOTChallenge-style datasets.
    """
    seqinfo_path = seq_dir / "seqinfo.ini"

    content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate={fps:g}
seqLength={seq_length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""

    seqinfo_path.write_text(content, encoding="utf-8")


def write_seqmap(seqmaps_dir: Path, benchmark_split_name: str, sequences: List[str]) -> Path:
    """
    Writes TrackEval sequence map file.
    """
    safe_mkdir(seqmaps_dir)

    seqmap_path = seqmaps_dir / f"{benchmark_split_name}.txt"

    lines = ["name"]
    lines.extend(sequences)

    seqmap_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return seqmap_path


def validate_input_files(mot_export_dir: Path, sequences: List[str]) -> None:
    missing = []

    for seq in sequences:
        gt_file = mot_export_dir / f"{seq}_gt.txt"
        pred_file = mot_export_dir / f"{seq}_pred.txt"

        if not gt_file.exists():
            missing.append(str(gt_file))
        if not pred_file.exists():
            missing.append(str(pred_file))

    if missing:
        msg = "Missing required MOT export files:\n" + "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mot_export_dir",
        type=str,
        required=True,
        help="Folder produced by KTP_eval_run.py containing *_gt.txt and *_pred.txt files.",
    )

    parser.add_argument(
        "--trackeval_root",
        type=str,
        required=True,
        help="Path to the TrackEval repository root.",
    )

    parser.add_argument(
        "--tracker_name",
        type=str,
        required=True,
        help="Name to give this tracker/run inside TrackEval.",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="KTP-5Hz",
        help="Benchmark name used by TrackEval.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name used by TrackEval. Usually train for custom MOTChallenge-style data.",
    )

    parser.add_argument(
        "--sequences",
        type=str,
        default="Arc,Rotation,Still,Translation",
        help="Comma-separated sequence names.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Sequence FPS written to seqinfo.ini.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width written to seqinfo.ini.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height written to seqinfo.ini.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing TrackEval files.",
    )

    args = parser.parse_args()

    mot_export_dir = Path(args.mot_export_dir).resolve()
    trackeval_root = Path(args.trackeval_root).resolve()

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    benchmark_split_name = f"{args.benchmark}-{args.split}"

    if not mot_export_dir.exists():
        raise FileNotFoundError(f"MOT export directory not found: {mot_export_dir}")

    if not trackeval_root.exists():
        raise FileNotFoundError(f"TrackEval root not found: {trackeval_root}")

    validate_input_files(mot_export_dir, sequences)

    gt_root = trackeval_root / "data" / "gt" / "mot_challenge" / benchmark_split_name
    tracker_root = (
        trackeval_root
        / "data"
        / "trackers"
        / "mot_challenge"
        / benchmark_split_name
        / args.tracker_name
    )

    tracker_data_dir = tracker_root / "data"
    seqmaps_dir = gt_root / "seqmaps"

    safe_mkdir(gt_root)
    safe_mkdir(tracker_data_dir)
    safe_mkdir(seqmaps_dir)

    seqmap_path = write_seqmap(seqmaps_dir, benchmark_split_name, sequences)

    print("[setup]")
    print(f"  MOT export dir : {mot_export_dir}")
    print(f"  TrackEval root : {trackeval_root}")
    print(f"  Benchmark split: {benchmark_split_name}")
    print(f"  Tracker name   : {args.tracker_name}")
    print(f"  Sequences      : {sequences}")
    print("")

    for seq in sequences:
        src_gt = mot_export_dir / f"{seq}_gt.txt"
        src_pred = mot_export_dir / f"{seq}_pred.txt"

        seq_gt_dir = gt_root / seq / "gt"
        seq_dir = gt_root / seq
        dst_gt = seq_gt_dir / "gt.txt"
        dst_pred = tracker_data_dir / f"{seq}.txt"

        safe_mkdir(seq_gt_dir)

        if not args.overwrite:
            if dst_gt.exists():
                raise FileExistsError(f"GT file already exists: {dst_gt}\nUse --overwrite to replace it.")
            if dst_pred.exists():
                raise FileExistsError(f"Prediction file already exists: {dst_pred}\nUse --overwrite to replace it.")

        shutil.copyfile(src_gt, dst_gt)
        shutil.copyfile(src_pred, dst_pred)

        seq_len = max(count_max_frame(src_gt), count_max_frame(src_pred))
        write_seqinfo(
            seq_dir=seq_dir,
            seq_name=seq,
            seq_length=seq_len,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )

        print(f"[ok] {seq}")
        print(f"  GT   -> {dst_gt}")
        print(f"  Pred -> {dst_pred}")
        print(f"  len  -> {seq_len} frames")
        print("")

    print("[done]")
    print(f"  seqmap       : {seqmap_path}")
    print(f"  gt root      : {gt_root}")
    print(f"  tracker root : {tracker_root}")
    print("")
    print("Next command example:")
    print(
        f'python scripts/run_mot_challenge.py '
        f'--BENCHMARK {args.benchmark} '
        f'--SPLIT_TO_EVAL {args.split} '
        f'--TRACKERS_TO_EVAL {args.tracker_name} '
        f'--METRICS HOTA CLEAR Identity '
        f'--USE_PARALLEL False '
        f'--NUM_PARALLEL_CORES 1'
    )


if __name__ == "__main__":
    main()