#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def stamp_to_sec(msg, fallback_time_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_time_ns * 1e-9


def image_to_cv(msg):
    h, w = int(msg.height), int(msg.width)
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ["bgr8", "rgb8"]:
        img = data.reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if enc in ["mono8", "8uc1"]:
        return data.reshape(h, w)

    if enc in ["bgra8", "rgba8"]:
        img = data.reshape(h, w, 4)
        if enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img

    if enc in ["16uc1", "mono16"]:
        data16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        return cv2.convertScaleAbs(data16, alpha=255.0 / max(1, int(data16.max())))

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def colorize_mask(mask):
    if mask.ndim == 3:
        return mask

    mask_u8 = mask.astype(np.uint8)
    color = np.zeros((*mask_u8.shape, 3), dtype=np.uint8)

    ids = [int(x) for x in np.unique(mask_u8) if int(x) != 0]
    palette = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 128, 0),
    ]

    for i, obj_id in enumerate(ids):
        color[mask_u8 == obj_id] = palette[i % len(palette)]

    return color


def make_overlay(rgb_bgr, mask_img, alpha):
    if mask_img.shape[:2] != rgb_bgr.shape[:2]:
        mask_img = cv2.resize(mask_img, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    if mask_img.ndim == 2:
        mask_color = colorize_mask(mask_img)
        active = mask_img > 0
    else:
        mask_color = mask_img[:, :, :3]
        active = np.any(mask_color > 10, axis=2)

    out = rgb_bgr.copy()
    blended = cv2.addWeighted(rgb_bgr, 1.0 - alpha, mask_color, alpha, 0.0)
    out[active] = blended[active]
    return out


def nearest_rgb(rgb_buffer, mask_t, max_dt):
    best = None
    best_dt = None

    for t, img in rgb_buffer:
        dt = abs(t - mask_t)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = img

    if best is None or best_dt is None or best_dt > max_dt:
        return None, best_dt

    return best, best_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to rosbag folder")
    ap.add_argument("--out_dir", required=True, help="Output folder for extracted frames")
    ap.add_argument("--rgb_topic", default="/k4a/rgb/image_raw")
    ap.add_argument("--mask_topic", default="/samurai/img_masks")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--max_dt", type=float, default=0.10, help="Max RGB/mask sync difference in seconds")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--use_mask_topic_directly", action="store_true",
                    help="Use /samurai/img_masks as the saved image directly, if it is already an overlay.")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "frames_index.csv"

    rgb_buffer = []
    max_buffer = 200
    saved = 0
    seen_masks = 0

    with AnyReader([bag_path]) as reader, csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["saved_idx", "mask_time", "rgb_time", "dt_sec", "path"])

        conns = [
            c for c in reader.connections
            if c.topic in {args.rgb_topic, args.mask_topic}
        ]

        print("[topics found]")
        for c in conns:
            print(f"  {c.topic}: {c.msgtype}")

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            t = stamp_to_sec(msg, timestamp)

            if conn.topic == args.rgb_topic:
                try:
                    rgb_img = image_to_cv(msg)
                except Exception as e:
                    print(f"[warn] failed RGB image at {t:.3f}: {e}")
                    continue

                rgb_buffer.append((t, rgb_img))
                if len(rgb_buffer) > max_buffer:
                    rgb_buffer = rgb_buffer[-max_buffer:]

            elif conn.topic == args.mask_topic:
                seen_masks += 1

                if seen_masks % max(1, args.save_every) != 0:
                    continue

                try:
                    mask_img = image_to_cv(msg)
                except Exception as e:
                    print(f"[warn] failed mask image at {t:.3f}: {e}")
                    continue

                if args.use_mask_topic_directly:
                    out_img = mask_img
                    rgb_t = ""
                    dt = ""
                else:
                    rgb_img, dt_val = nearest_rgb(rgb_buffer, t, args.max_dt)
                    if rgb_img is None:
                        print(f"[warn] no RGB match for mask at {t:.3f}, nearest dt={dt_val}")
                        continue

                    out_img = make_overlay(rgb_img, mask_img, args.alpha)
                    rgb_t = t - dt_val if dt_val is not None else ""
                    dt = dt_val if dt_val is not None else ""

                out_path = frames_dir / f"frame_{saved:06d}_t{t:.3f}.jpg"
                cv2.imwrite(str(out_path), out_img)

                writer.writerow([saved, f"{t:.6f}", rgb_t, dt, str(out_path)])
                saved += 1

                if args.max_frames > 0 and saved >= args.max_frames:
                    break

    print(f"[done] saved {saved} frames to {frames_dir}")
    print(f"[done] index CSV: {csv_path}")


if __name__ == "__main__":
    main()