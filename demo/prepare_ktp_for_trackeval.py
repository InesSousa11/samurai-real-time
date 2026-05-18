#!/usr/bin/env python3
# prepare_ktp_for_trackeval.py
#
# Converts KTP MOT exports produced by KTP_eval_run.py into the folder layout
# expected by TrackEval's MotChallenge2DBox dataset.
#
# Expected input:
#   mot_export_dir/
#       Arc_gt.txt
#       Arc_pred.txt
#       Rotation_gt.txt
#       Rotation_pred.txt
#       Still_gt.txt
#       Still_pred.txt
#       Translation_gt.txt
#       Translation_pred.txt
#
# Output layout:
#   TrackEval/data/gt/mot_challenge/KTP-5Hz-train/<SEQ>/gt/gt.txt
#   TrackEval/data/trackers/mot_challenge/KTP-5Hz-train/<TRACKER>/data/<SEQ>.txt
#
# Seqmap is written to both:
#   TrackEval/data/gt/mot_challenge/seqmaps/KTP-5Hz-train.txt
#   TrackEval/data/gt/mot_challenge/KTP-5Hz-train/seqmaps/KTP-5Hz-train.txt
#
# After this, TrackEval can usually be run with:
#
# python scripts/run_mot_challenge.py `
#   --BENCHMARK KTP-5Hz `
#   --SPLIT_TO_EVAL train `
#   --TRACKERS_TO_EVAL ReID-SAMURAI-config44 `
#   --METRICS HOTA CLEAR Identity `
#   --USE_PARALLEL False `
#   --NUM_PARALLEL_CORES 1

import argparse
import shutil
from pathlib import Path


DEFAULT_SEQUENCES = ["Arc", "Rotation", "Still", "Translation"]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing input file: {src}")

    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {dst}\n"
            f"Use --overwrite if you want to replace it."
        )

    safe_mkdir(dst.parent)
    shutil.copy2(src, dst)


def infer_sequence_length_from_mot(gt_file: Path, pred_file: Path) -> int:
    """
    TrackEval needs sequence lengths in the seqmap. We infer the sequence
    length as the largest frame index appearing in either GT or prediction file.
    """
    max_frame = 0

    for path in [gt_file, pred_file]:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8", errors="ignore") as f:
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

    if max_frame <= 0:
        raise RuntimeError(
            f"Could not infer sequence length from:\n"
            f"  GT:   {gt_file}\n"
            f"  Pred: {pred_file}"
        )

    return max_frame


def write_seqinfo_ini(seq_dir: Path, seq_name: str, seq_length: int, width: int = 640, height: int = 480) -> None:
    """
    Writes a minimal seqinfo.ini file. TrackEval can often run without using all
    fields, but MOTChallenge-style folders normally include it.
    """
    seqinfo_path = seq_dir / "seqinfo.ini"

    content = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=img1\n"
        "frameRate=5\n"
        f"seqLength={seq_length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )

    with seqinfo_path.open("w", encoding="utf-8") as f:
        f.write(content)


def write_seqmap(seqmap_path: Path, seq_lengths: dict) -> None:
    """
    TrackEval's MOTChallenge seqmap format:
        name
        Arc 353
        Rotation 366
        ...
    """
    safe_mkdir(seqmap_path.parent)

    with seqmap_path.open("w", encoding="utf-8") as f:
        f.write("name\n")
        for seq_name, seq_len in seq_lengths.items():
            f.write(f"{seq_name} {seq_len}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mot_export_dir",
        type=str,
        required=True,
        help="Folder containing <SEQ>_gt.txt and <SEQ>_pred.txt exported by KTP_eval_run.py.",
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
        help="Name to use for this tracker inside TrackEval.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="KTP-5Hz",
        help="Benchmark name used by TrackEval. Default: KTP-5Hz.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name used by TrackEval. Default: train.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="Arc,Rotation,Still,Translation",
        help="Comma-separated sequence names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TrackEval files.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=640,
        help="Image width for seqinfo.ini. Default: 640.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=480,
        help="Image height for seqinfo.ini. Default: 480.",
    )

    args = parser.parse_args()

    mot_export_dir = Path(args.mot_export_dir).resolve()
    trackeval_root = Path(args.trackeval_root).resolve()

    if not mot_export_dir.exists():
        raise FileNotFoundError(f"MOT export directory not found: {mot_export_dir}")

    if not trackeval_root.exists():
        raise FileNotFoundError(f"TrackEval root not found: {trackeval_root}")

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]

    benchmark_split = f"{args.benchmark}-{args.split}"

    gt_root = trackeval_root / "data" / "gt" / "mot_challenge" / benchmark_split
    tracker_root = (
        trackeval_root
        / "data"
        / "trackers"
        / "mot_challenge"
        / benchmark_split
        / args.tracker_name
    )
    tracker_data_dir = tracker_root / "data"

    # TrackEval default seqmap location for MOTChallenge.
    default_seqmap_dir = trackeval_root / "data" / "gt" / "mot_challenge" / "seqmaps"
    default_seqmap_path = default_seqmap_dir / f"{benchmark_split}.txt"

    # Extra local copy, useful for inspection/debugging.
    local_seqmap_dir = gt_root / "seqmaps"
    local_seqmap_path = local_seqmap_dir / f"{benchmark_split}.txt"

    print("[setup]")
    print("  MOT export dir :", mot_export_dir)
    print("  TrackEval root :", trackeval_root)
    print("  Benchmark split:", benchmark_split)
    print("  Tracker name   :", args.tracker_name)
    print("  Sequences      :", sequences)
    print("")

    safe_mkdir(gt_root)
    safe_mkdir(tracker_data_dir)
    safe_mkdir(default_seqmap_dir)
    safe_mkdir(local_seqmap_dir)

    seq_lengths = {}

    for seq in sequences:
        src_gt = mot_export_dir / f"{seq}_gt.txt"
        src_pred = mot_export_dir / f"{seq}_pred.txt"

        dst_gt = gt_root / seq / "gt" / "gt.txt"
        dst_pred = tracker_data_dir / f"{seq}.txt"

        seq_len = infer_sequence_length_from_mot(src_gt, src_pred)
        seq_lengths[seq] = seq_len

        copy_file(src_gt, dst_gt, overwrite=args.overwrite)
        copy_file(src_pred, dst_pred, overwrite=args.overwrite)

        write_seqinfo_ini(
            seq_dir=gt_root / seq,
            seq_name=seq,
            seq_length=seq_len,
            width=args.image_width,
            height=args.image_height,
        )

        print(f"[ok] {seq}")
        print("  GT   ->", dst_gt)
        print("  Pred ->", dst_pred)
        print("  len  ->", seq_len, "frames")
        print("")

    write_seqmap(default_seqmap_path, seq_lengths)
    write_seqmap(local_seqmap_path, seq_lengths)

    print("[done]")
    print("  default seqmap :", default_seqmap_path)
    print("  local seqmap   :", local_seqmap_path)
    print("  gt root        :", gt_root)
    print("  tracker root   :", tracker_root)
    print("")
    print("Next command:")
    print(
        "python scripts/run_mot_challenge.py "
        f"--BENCHMARK {args.benchmark} "
        f"--SPLIT_TO_EVAL {args.split} "
        f"--TRACKERS_TO_EVAL {args.tracker_name} "
        "--METRICS HOTA CLEAR Identity "
        "--USE_PARALLEL False "
        "--NUM_PARALLEL_CORES 1"
    )


if __name__ == "__main__":
    main()