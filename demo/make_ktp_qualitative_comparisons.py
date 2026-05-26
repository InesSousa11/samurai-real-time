import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SEQ_NAMES = ["Arc", "Rotation", "Still", "Translation"]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_videos(root: Path) -> Dict[str, Path]:
    """
    Finds one video per KTP sequence under root.
    Chooses the newest matching .mp4 if multiple exist.
    """
    videos_by_seq: Dict[str, List[Path]] = {s: [] for s in SEQ_NAMES}

    for p in root.rglob("*.mp4"):
        name = p.name.lower()
        for seq in SEQ_NAMES:
            if seq.lower() in name:
                videos_by_seq[seq].append(p)

    selected: Dict[str, Path] = {}
    for seq, paths in videos_by_seq.items():
        if not paths:
            continue
        paths = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
        selected[seq] = paths[0]

    return selected


def read_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
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


def draw_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    bar_h = 34
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    cv2.putText(
        out,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def make_side_by_side(
    transreid_img: np.ndarray,
    reid_samurai_img: np.ndarray,
    seq: str,
    frame_idx: int,
    category: str,
) -> np.ndarray:
    target_h = min(transreid_img.shape[0], reid_samurai_img.shape[0])
    transreid_img = resize_to_height(transreid_img, target_h)
    reid_samurai_img = resize_to_height(reid_samurai_img, target_h)

    transreid_img = draw_label(transreid_img, f"TransReID baseline | {seq} frame {frame_idx}")
    reid_samurai_img = draw_label(reid_samurai_img, f"ReID-SAMURAI | {category}")

    gap = np.ones((target_h, 8, 3), dtype=np.uint8) * 255
    return np.concatenate([transreid_img, gap, reid_samurai_img], axis=1)


def load_review_rows(review_root: Path) -> List[dict]:
    rows: List[dict] = []

    for csv_path in review_root.rglob("ignored_predictions_no_gt.csv"):
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_csv_path"] = str(csv_path)
                rows.append(row)

    return rows


def row_to_int(row: dict, key: str) -> Optional[int]:
    try:
        value = row.get(key, "")
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--reid_review_root",
        required=True,
        help="Path to the ReID-SAMURAI _review folder.",
    )
    ap.add_argument(
        "--reid_video_root",
        required=True,
        help="Folder containing ReID-SAMURAI saved videos.",
    )
    ap.add_argument(
        "--transreid_video_root",
        required=True,
        help="Folder containing TransReID baseline saved videos.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output folder for side-by-side candidate images.",
    )
    ap.add_argument(
        "--categories",
        default="missing_same_id_no_other_gt,same_id_gt_exists_low_iou,missing_same_id_overlaps_other_gt",
        help="Comma-separated categories to export.",
    )
    ap.add_argument(
        "--max_per_category",
        type=int,
        default=40,
        help="Maximum number of examples to export per category and sequence.",
    )

    args = ap.parse_args()

    review_root = Path(args.reid_review_root)
    reid_video_root = Path(args.reid_video_root)
    transreid_video_root = Path(args.transreid_video_root)
    out_dir = Path(args.out_dir)

    safe_mkdir(out_dir)

    wanted_categories = {c.strip() for c in args.categories.split(",") if c.strip()}

    print("[info] Finding videos...")
    reid_videos = find_videos(reid_video_root)
    transreid_videos = find_videos(transreid_video_root)

    print("[ReID-SAMURAI videos]")
    for seq, path in reid_videos.items():
        print(f"  {seq}: {path}")

    print("[TransReID videos]")
    for seq, path in transreid_videos.items():
        print(f"  {seq}: {path}")

    rows = load_review_rows(review_root)
    print(f"[info] Loaded {len(rows)} review rows")

    exported = 0
    counts: Dict[Tuple[str, str], int] = {}

    index_csv_path = out_dir / "qualitative_candidates_index.csv"
    with index_csv_path.open("w", newline="", encoding="utf-8") as f_index:
        writer = csv.writer(f_index)
        writer.writerow([
            "seq",
            "frame_idx",
            "pred_id",
            "category",
            "output_image",
            "source_csv",
            "transreid_video",
            "reid_samurai_video",
        ])

        for row in rows:
            seq = row.get("seq", "")
            category = row.get("category", "")
            frame_idx = row_to_int(row, "frame_idx")
            pred_id = row.get("pred_id", "")

            if seq not in SEQ_NAMES:
                continue
            if category not in wanted_categories:
                continue
            if frame_idx is None:
                continue
            if seq not in reid_videos:
                print(f"[WARN] No ReID-SAMURAI video for {seq}")
                continue
            if seq not in transreid_videos:
                print(f"[WARN] No TransReID video for {seq}")
                continue

            key = (seq, category)
            counts[key] = counts.get(key, 0)
            if counts[key] >= args.max_per_category:
                continue
            counts[key] += 1

            trans_img = read_frame(transreid_videos[seq], frame_idx)
            reid_img = read_frame(reid_videos[seq], frame_idx)

            if trans_img is None or reid_img is None:
                continue

            comp = make_side_by_side(
                transreid_img=trans_img,
                reid_samurai_img=reid_img,
                seq=seq,
                frame_idx=frame_idx,
                category=category,
            )

            seq_out_dir = out_dir / seq / category
            safe_mkdir(seq_out_dir)

            out_name = f"{seq}_f{frame_idx:06d}_pid{pred_id}_{category}.jpg"
            out_path = seq_out_dir / out_name

            cv2.imwrite(str(out_path), comp)
            exported += 1

            writer.writerow([
                seq,
                frame_idx,
                pred_id,
                category,
                str(out_path),
                row.get("_csv_path", ""),
                str(transreid_videos[seq]),
                str(reid_videos[seq]),
            ])

    print(f"[done] Exported {exported} comparison images")
    print(f"[done] Index CSV: {index_csv_path}")


if __name__ == "__main__":
    main()