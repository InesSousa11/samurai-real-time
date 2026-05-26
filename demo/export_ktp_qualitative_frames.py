import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2


SEQ_NAMES = ["Arc", "Rotation", "Still", "Translation"]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_videos(root: Path) -> Dict[str, Path]:
    """
    Find the newest .mp4 per KTP sequence under root.
    """
    candidates = {seq: [] for seq in SEQ_NAMES}

    for p in root.rglob("*.mp4"):
        name = p.name.lower()
        for seq in SEQ_NAMES:
            if seq.lower() in name:
                candidates[seq].append(p)

    selected = {}
    for seq, paths in candidates.items():
        if not paths:
            continue
        paths = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
        selected[seq] = paths[0]

    return selected


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[WARN] Could not read frame {frame_idx} from {video_path}")
        return None

    return frame


def load_review_rows(review_root: Path) -> List[dict]:
    rows = []
    for csv_path in review_root.rglob("ignored_predictions_no_gt.csv"):
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_csv_path"] = str(csv_path)
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--review_root", required=True, help="ReID-SAMURAI _review root")
    ap.add_argument("--reid_video_root", required=True, help="Root folder containing ReID-SAMURAI videos")
    ap.add_argument("--transreid_video_root", required=True, help="Root folder containing TransReID videos")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--categories",
        default="missing_same_id_no_other_gt,same_id_gt_exists_low_iou,missing_same_id_overlaps_other_gt",
        help="Comma-separated categories to export",
    )
    ap.add_argument("--max_per_category", type=int, default=100, help="Max frames per sequence/category")
    args = ap.parse_args()

    review_root = Path(args.review_root)
    reid_video_root = Path(args.reid_video_root)
    transreid_video_root = Path(args.transreid_video_root)
    out_dir = Path(args.out_dir)

    safe_mkdir(out_dir)

    wanted_categories = {x.strip() for x in args.categories.split(",") if x.strip()}

    print("[info] Finding videos...")
    reid_videos = find_videos(reid_video_root)
    trans_videos = find_videos(transreid_video_root)

    print("[ReID-SAMURAI videos]")
    for seq, p in reid_videos.items():
        print(f"  {seq}: {p}")

    print("[TransReID videos]")
    for seq, p in trans_videos.items():
        print(f"  {seq}: {p}")

    rows = load_review_rows(review_root)
    print(f"[info] Loaded {len(rows)} review rows")

    exported_counts = {}

    index_csv = out_dir / "qualitative_frames_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq",
            "frame_idx",
            "pred_id",
            "category",
            "reid_samurai_frame",
            "transreid_frame",
        ])

        for row in rows:
            seq = row.get("seq", "")
            category = row.get("category", "")
            if seq not in SEQ_NAMES:
                continue
            if category not in wanted_categories:
                continue

            try:
                frame_idx = int(float(row["frame_idx"]))
                pred_id = int(float(row["pred_id"]))
            except Exception:
                continue

            if seq not in reid_videos or seq not in trans_videos:
                continue

            key = (seq, category)
            exported_counts[key] = exported_counts.get(key, 0)
            if exported_counts[key] >= args.max_per_category:
                continue
            exported_counts[key] += 1

            frame_reid = read_frame(reid_videos[seq], frame_idx)
            frame_trans = read_frame(trans_videos[seq], frame_idx)

            if frame_reid is None or frame_trans is None:
                continue

            reid_dir = out_dir / "ReID-SAMURAI" / seq / category
            trans_dir = out_dir / "TransReID" / seq / category
            safe_mkdir(reid_dir)
            safe_mkdir(trans_dir)

            name = f"{seq}_f{frame_idx:06d}_pid{pred_id}.jpg"

            reid_path = reid_dir / name
            trans_path = trans_dir / name

            cv2.imwrite(str(reid_path), frame_reid)
            cv2.imwrite(str(trans_path), frame_trans)

            writer.writerow([
                seq,
                frame_idx,
                pred_id,
                category,
                str(reid_path),
                str(trans_path),
            ])

    print(f"[done] Exported frames")
    print(f"[done] Index CSV: {index_csv}")


if __name__ == "__main__":
    main()