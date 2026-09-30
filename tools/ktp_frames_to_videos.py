#!/usr/bin/env python3
"""
ktp_frames_to_videos.py

Convert KTP image sequences to MP4 videos (no overlays).
- Reads:  KTP/images/<Seq>/rgb/*.jpg
- Writes: <out_dir>/<Seq>.mp4

Run from repo root or from demo/; paths are resolved via --ktp_root.

Example (PowerShell):
  python .\demo\ktp_frames_to_videos.py --ktp_root "C:\...\KTP" --out_dir "C:\...\KTP_videos" --fps 15
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2

_TS_LEAD_NUM = re.compile(r"^(\d+(?:\.\d+)?)")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ts_from_filename_robust(p: Path) -> Optional[float]:
    m = _TS_LEAD_NUM.match(p.stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def list_sequences(ktp_root: Path) -> List[str]:
    images_root = ktp_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing: {images_root}")
    seqs = []
    for d in images_root.iterdir():
        if d.is_dir() and (d / "rgb").is_dir():
            seqs.append(d.name)
    seqs.sort()
    return seqs

def collect_frames(rgb_dir: Path) -> List[Path]:
    frames = []
    for p in rgb_dir.glob("*.jpg"):
        t = ts_from_filename_robust(p)
        if t is None:
            continue
        frames.append((t, p))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]

def write_video(frames: List[Path], out_path: Path, fps: float, codec: str = "mp4v", verbose: bool = True):
    if not frames:
        if verbose:
            print(f"[skip] no frames for {out_path.stem}")
        return

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    H, W = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path} (codec={codec}, size={W}x{H}, fps={fps})")

    n = 0
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(img)
        n += 1

    vw.release()
    if verbose:
        print(f"[ok] {out_path.name}  ({n} frames, {fps} fps, {W}x{H})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", type=str, required=True, help="Path to KTP root folder")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write the mp4 files")
    ap.add_argument("--fps", type=float, default=15.0, help="Output video FPS")
    ap.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (default mp4v). Try avc1 if available.")
    ap.add_argument("--seq", type=str, default=None, help="Optional: only export this sequence name")
    ap.add_argument("--max_frames", type=int, default=-1, help="Optional: limit frames per video (for quick tests)")
    ap.add_argument("--stride", type=int, default=1, help="Optional: take every Nth frame")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ktp_root = Path(args.ktp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    seqs = [args.seq] if args.seq else list_sequences(ktp_root)
    if not seqs:
        raise RuntimeError("No sequences found under KTP/images/<Seq>/rgb")

    for seq in seqs:
        rgb_dir = ktp_root / "images" / seq / "rgb"
        if not rgb_dir.exists():
            print(f"[skip] missing {rgb_dir}")
            continue

        frames = collect_frames(rgb_dir)
        if args.stride > 1:
            frames = frames[::args.stride]
        if args.max_frames > 0:
            frames = frames[:args.max_frames]

        out_path = out_dir / f"{seq}.mp4"
        write_video(frames, out_path, fps=args.fps, codec=args.codec, verbose=True)

if __name__ == "__main__":
    main()