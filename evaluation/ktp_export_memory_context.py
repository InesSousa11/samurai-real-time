#!/usr/bin/env python3
"""
ktp_export_memory_context.py

Export a "memory context case" for a chosen KTP frame:
- Target prediction frame (overlay)
- Conditioning frame + prompt boxes used to seed IDs
- Cached non-conditioning outputs currently stored in condition_state["output_dict"]["non_cond_frame_outputs"]
- ACTUAL frames used for memory attention (requires sam2_base.py patch that records
  condition_state["output_dict"]["debug_memory_attn"])

KTP expected:
  KTP/images/<Seq>/rgb/*.jpg
  KTP/ground_truth/<Seq>_gt2D.txt
"""

import sys, re, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'", category=UserWarning)

# --- repo root ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "demo") else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_camera_predictor

CKPT_PATH = (REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH  = (REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()

_TS_LEAD_NUM = re.compile(r"^(\d+(?:\.\d+)?)")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ts_from_filename_robust(p: Path) -> Optional[str]:
    m = _TS_LEAD_NUM.match(p.stem)
    return m.group(1) if m else None

def parse_gt2d_file(gt_path: Path) -> Dict[str, List[Tuple[int,float,float,float,float]]]:
    d: Dict[str, List[Tuple[int,float,float,float,float]]] = {}
    with gt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            ts_part, rest = line.split(":", 1)
            ts = ts_part.strip()
            rest = rest.strip()
            dets_raw = [r.strip() for r in rest.split(",") if r.strip()]
            dets = []
            for dr in dets_raw:
                parts = dr.split()
                if len(parts) < 5:
                    continue
                try:
                    gid = int(parts[0])
                    x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                    dets.append((gid,x,y,w,h))
                except Exception:
                    pass
            d[ts] = dets
    return d

def bbox_xywh_to_xyxy(x,y,w,h):
    x1=int(round(x)); y1=int(round(y))
    x2=int(round(x+w)); y2=int(round(y+h))
    return (x1,y1,x2,y2)

def clamp_bbox_xyxy(bb, W, H):
    x1,y1,x2,y2 = bb
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2))
    y2 = max(0, min(H,   y2))
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return (x1,y1,x2,y2)

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0, ix2-ix1); ih=max(0, iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b=max(0,bx2-bx1)*max(0,by2-by1)
    denom=area_a+area_b-inter
    return float(inter/denom) if denom>0 else 0.0

def id_to_hue(obj_id: int) -> int:
    return int((37 * int(obj_id) + 61) % 180)

def overlay_masks_from_logits(rgb_frame: np.ndarray, obj_ids: List[int], logits_tensor: torch.Tensor, alpha=0.5):
    """logits_tensor: (N,1,H,W) or (N,H,W). obj_ids length N (same order)."""
    if rgb_frame is None or logits_tensor is None or not torch.is_tensor(logits_tensor):
        return rgb_frame
    if logits_tensor.ndim == 4:
        lg = logits_tensor[:,0]
    elif logits_tensor.ndim == 3:
        lg = logits_tensor
    else:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,1] = 255

    for i in range(min(len(obj_ids), int(lg.shape[0]))):
        m = (lg[i] > 0).detach().cpu().numpy().astype(bool)
        hsv[m,0] = id_to_hue(obj_ids[i])
        hsv[m,2] = 255

    overlay = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, overlay, float(alpha), 0.0)

def draw_boxes_and_labels(bgr, boxes: List[Tuple[int,int,int,int]], labels: List[str], color, thick=2):
    for bb, lab in zip(boxes, labels):
        x1,y1,x2,y2 = bb
        cv2.rectangle(bgr, (x1,y1), (x2,y2), color, thick)
        cv2.putText(bgr, lab, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return bgr

def get_latest_noncond_entry(predictor):
    cs = predictor.condition_state
    od = cs.get("output_dict", {})
    ncfo = od.get("non_cond_frame_outputs", {})
    if not isinstance(ncfo, dict) or len(ncfo) == 0:
        return None, None, None
    f_last = max(ncfo.keys())
    return f_last, ncfo[f_last], list(ncfo.keys())

def ptr_sim(ptr: torch.Tensor) -> np.ndarray:
    P = F.normalize(ptr.detach().float().cpu(), dim=1)
    return (P @ P.t()).cpu().numpy()

def masks_iou_from_pred_masks(pm_1hw: torch.Tensor, max_n: int = 3) -> List[Tuple[int,int,float]]:
    if (pm_1hw is None) or (not torch.is_tensor(pm_1hw)) or pm_1hw.ndim != 4:
        return []
    N = int(pm_1hw.shape[0])
    K = min(N, int(max_n))
    if K < 2:
        return []

    pm = pm_1hw[:K, 0]  # (K,H,W)
    masks = [(pm[i] > 0).detach().cpu().numpy().astype(np.uint8) for i in range(K)]

    def iou(a, b) -> float:
        inter = int((a & b).sum())
        uni = int((a | b).sum())
        return float(inter / uni) if uni > 0 else 0.0

    out = []
    for i in range(K):
        for j in range(i + 1, K):
            out.append((i, j, iou(masks[i], masks[j])))
    return out

def _closest_debug_key(dbg: Dict[int, Any], target: int) -> Optional[int]:
    if not isinstance(dbg, dict) or len(dbg) == 0:
        return None
    keys = sorted([int(k) for k in dbg.keys()])
    if target in dbg:
        return target
    best = None
    best_dist = 10**18
    for k in keys:
        dist = abs(k - target)
        if dist < best_dist:
            best_dist = dist
            best = k
    return best


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", type=str, required=True)
    ap.add_argument("--seq", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--export_fidx", type=int, required=True,
                    help="KTP loop index (after stride) at which to export the memory context case.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=-1)

    ap.add_argument("--visible_area_frac", type=float, default=0.02)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.10)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--memory_limit", type=int, default=12,
                    help="Limit how many cached non-cond frames to export (most recent).")

    ap.add_argument("--debug", action="store_true", help="Print debug info.")
    args = ap.parse_args()

    ktp_root = Path(args.ktp_root).resolve()
    seq = args.seq
    out_root = Path(args.out_dir).resolve() / f"{seq}_case_f{args.export_fidx:05d}"
    safe_mkdir(out_root)
    mem_dir = out_root / "memory_frames"
    safe_mkdir(mem_dir)
    attn_dir = out_root / "attn_memory_frames"
    safe_mkdir(attn_dir)

    img_dir = ktp_root / "images" / seq / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq}_gt2D.txt"
    if not img_dir.exists(): raise FileNotFoundError(img_dir)
    if not gt_path.exists(): raise FileNotFoundError(gt_path)
    if not CKPT_PATH.exists(): raise FileNotFoundError(CKPT_PATH)
    if not CFG_PATH.exists(): raise FileNotFoundError(CFG_PATH)

    gt_map = parse_gt2d_file(gt_path)

    frames_all = list(img_dir.glob("*.jpg"))
    items = []
    for p in frames_all:
        ts = ts_from_filename_robust(p)
        if ts is None: continue
        try:
            tf = float(ts)
        except Exception:
            continue
        items.append((tf, ts, p))
    items.sort(key=lambda t: t[0])

    if args.stride > 1:
        items = items[::args.stride]
    if args.max_frames > 0:
        items = items[:args.max_frames]
    if args.export_fidx < 0 or args.export_fidx >= len(items):
        raise ValueError(f"--export_fidx out of range: 0..{len(items)-1}")

    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
    print("SAMURAI mode:", getattr(predictor, "samurai_mode", None))

    # Load first frame
    _, _, p0 = items[0]
    bgr0 = cv2.imread(str(p0))
    if bgr0 is None: raise RuntimeError("Failed to read first frame")
    H, W = bgr0.shape[:2]
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
    predictor.load_first_frame(rgb0)

    # ------------------------------------------------------------------
    # TWO different RGB maps:
    #  - noncond/global frame_idx -> rgb (from predictor.track timeline)
    #  - cond slot idx -> rgb (from predictor.add_conditioning_frame timeline)
    # ------------------------------------------------------------------
    predframe_to_rgb: Dict[int, np.ndarray] = {}
    fidx_to_predframe: Dict[int, int] = {}

    condframe_to_rgb: Dict[int, np.ndarray] = {}
    condframe_to_rgb[0] = rgb0.copy()  # conditioning slot 0 is first conditioning image

    # Also keep global frame 0 if needed (some codepaths might reference it)
    predframe_to_rgb[0] = rgb0.copy()

    prompt_boxes_xyxy: Dict[int, Tuple[int,int,int,int]] = {}
    seeded: set = set()

    for fidx, (ts_f, ts_s, fp) in enumerate(items[:args.export_fidx + 1]):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # oracle seed GT ids
        gt_dets = gt_map.get(ts_s, [])
        gt_bb_by_id = {}
        for (gid, x, y, w, h) in gt_dets:
            gt_bb_by_id[gid] = clamp_bbox_xyxy(bbox_xywh_to_xyxy(x,y,w,h), W, H)

        # ---- seed new ids (if any) ----
        for (gid, x, y, w, h) in gt_dets:
            if gid in seeded:
                continue

            bb = gt_bb_by_id[gid]
            bw = max(0, bb[2]-bb[0]); bh = max(0, bb[3]-bb[1])
            area_frac = (bw*bh) / float(W*H + 1e-9)
            if not (area_frac >= args.visible_area_frac and bh >= args.visible_min_h):
                continue

            max_iou_other = 0.0
            for ogid, obb in gt_bb_by_id.items():
                if ogid == gid: continue
                max_iou_other = max(max_iou_other, iou_xyxy(bb, obb))
            if max_iou_other > args.seed_overlap_iou_max:
                continue

            bbox = np.array([[bb[0], bb[1]], [bb[2], bb[3]]], dtype=np.float32)

            # IMPORTANT: late seeding creates a NEW conditioning slot.
            late = (fidx != 0)
            try:
                if not late:
                    predictor.add_new_prompt(frame_idx=0, obj_id=int(gid), bbox=bbox)
                else:
                    # next conditioning slot index is current length of images
                    cond_idx = int(len(predictor.condition_state.get("images", [])))
                    predictor.add_conditioning_frame(rgb)      # appends a new conditioning image
                    condframe_to_rgb[cond_idx] = rgb.copy()   # <-- map cond slot -> actual RGB
                    predictor.add_new_prompt_during_track(
                        bbox=bbox,
                        if_new_target=True,
                        obj_id=int(gid),
                        labels=None,
                        clear_old_points=True,
                    )

                seeded.add(gid)
                prompt_boxes_xyxy[int(gid)] = bb
            except Exception:
                pass

        # ---- track ----
        try:
            predictor.track(rgb)
        except Exception:
            pass

        pred_frame, _, _ = get_latest_noncond_entry(predictor)
        if pred_frame is not None:
            predframe_to_rgb[int(pred_frame)] = rgb.copy()
            fidx_to_predframe[fidx] = int(pred_frame)

    # target
    target_fidx = args.export_fidx
    target_pred_frame = fidx_to_predframe.get(target_fidx, None)

    cs = predictor.condition_state
    od = cs.get("output_dict", {})
    ncfo = od.get("non_cond_frame_outputs", {})

    mem_frames = sorted(list(ncfo.keys())) if isinstance(ncfo, dict) else []
    if args.memory_limit > 0 and len(mem_frames) > args.memory_limit:
        mem_frames = mem_frames[-args.memory_limit:]

    print("\n[check] noncond score stats (last cached frames):")
    for k in mem_frames:
        e = ncfo[int(k)]
        bi = e.get("best_iou_score", None)
        os = e.get("object_score_logits", None)
        ks = e.get("kf_score", None)

        def stat(x):
            if not torch.is_tensor(x):
                return str(type(x))
            xcpu = x.detach().float().cpu().reshape(-1)
            if xcpu.numel() == 0:
                return "empty"
            mask = ~torch.isnan(xcpu)
            n_nan = int((~mask).sum().item())
            if int(mask.sum().item()) == 0:
                return f"shape={tuple(x.shape)} nan={n_nan} all_nan"
            vals = xcpu[mask]
            mn = float(vals.min().item())
            mx = float(vals.max().item())
            return f"shape={tuple(x.shape)} nan={n_nan} min={mn:.3f} max={mx:.3f}"

        print(f"  frame {k}: best_iou_score {stat(bi)} | obj_score {stat(os)} | kf_score {stat(ks)}")

    if target_pred_frame is None:
        target_pred_frame = max(ncfo.keys()) if isinstance(ncfo, dict) and len(ncfo) else None
    if target_pred_frame is None:
        raise RuntimeError("No non_cond_frame_outputs available to export.")

    target_entry = ncfo[int(target_pred_frame)]
    obj_ids = cs.get("obj_ids", [])
    obj_id_to_idx = cs.get("obj_id_to_idx", {})

    # --- 00: target overlay ---
    tgt_rgb = predframe_to_rgb.get(int(target_pred_frame), rgb0.copy())
    stored_pred_masks = target_entry.get("pred_masks", None)  # (N,1,256,256)

    overlay_rgb = tgt_rgb.copy()
    if torch.is_tensor(stored_pred_masks):
        pm = stored_pred_masks.detach().float().cpu()
        pm_up = torch.nn.functional.interpolate(
            pm, size=(overlay_rgb.shape[0], overlay_rgb.shape[1]),
            mode="bilinear", align_corners=False
        )
        idx2id = cs.get("obj_idx_to_id", {})
        ids_by_idx = []
        for i in range(int(pm_up.shape[0])):
            oid = idx2id.get(i, None)
            if oid is None and isinstance(obj_ids, list) and i < len(obj_ids):
                oid = obj_ids[i]
            ids_by_idx.append(int(oid) if oid is not None else int(i))
        overlay_rgb = overlay_masks_from_logits(
            overlay_rgb, ids_by_idx, pm_up[:,0].unsqueeze(1), alpha=args.alpha
        )

    disp = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        disp, f"TARGET fidx={target_fidx} pred_frame={int(target_pred_frame)}",
        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA
    )

    if torch.is_tensor(stored_pred_masks):
        pairs = masks_iou_from_pred_masks(stored_pred_masks, max_n=3)
        y = 52
        for (i, j, v) in pairs:
            cv2.putText(
                disp, f"IoU(idx{i},idx{j})={v:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )
            y += 22

    cv2.imwrite(str(out_root / "00_target_frame_overlay.png"), disp)

    # --- 01: conditioning frame with prompt boxes (first conditioning frame only) ---
    cond_bgr = cv2.cvtColor(rgb0.copy(), cv2.COLOR_RGB2BGR)
    if prompt_boxes_xyxy:
        boxes = list(prompt_boxes_xyxy.values())
        labels = [f"prompt id={oid}" for oid in prompt_boxes_xyxy.keys()]
        cond_bgr = draw_boxes_and_labels(cond_bgr, boxes, labels, color=(255,255,255), thick=2)
    cv2.imwrite(str(out_root / "01_condition_frame_prompts.png"), cond_bgr)

    # --- cached non-cond frames export ---
    for pf in mem_frames:
        rgb_mem = predframe_to_rgb.get(int(pf), None)
        if rgb_mem is None:
            continue
        entry = ncfo[int(pf)]
        pm = entry.get("pred_masks", None)
        sc = entry.get("object_score_logits", None)

        out_rgb = rgb_mem.copy()
        if torch.is_tensor(pm):
            pm_cpu = pm.detach().float().cpu()
            pm_up = torch.nn.functional.interpolate(
                pm_cpu, size=(out_rgb.shape[0], out_rgb.shape[1]),
                mode="bilinear", align_corners=False
            )
            idx2id = cs.get("obj_idx_to_id", {})
            ids_by_idx = []
            for i in range(int(pm_up.shape[0])):
                oid = idx2id.get(i, None)
                if oid is None and isinstance(obj_ids, list) and i < len(obj_ids):
                    oid = obj_ids[i]
                ids_by_idx.append(int(oid) if oid is not None else int(i))
            out_rgb = overlay_masks_from_logits(out_rgb, ids_by_idx, pm_up[:,0].unsqueeze(1), alpha=args.alpha)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            out_bgr, f"cached noncond pred_frame={int(pf)}",
            (10, out_bgr.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA
        )

        if torch.is_tensor(sc):
            scv = sc.detach().reshape(-1).cpu().numpy().tolist()
            y = 25
            for i, s in enumerate(scv[:8]):
                oid = cs.get("obj_idx_to_id", {}).get(i, i)
                cv2.putText(
                    out_bgr, f"id={oid} score_logit={s:.3f}", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA
                )
                y += 22

        cv2.imwrite(str(mem_dir / f"mem_{int(pf):05d}_overlay.png"), out_bgr)

    # --- export ACTUAL attention memory frames ---
    dbg = od.get("debug_memory_attn", {})
    if args.debug:
        print("[debug] debug_memory_attn keys head:", (sorted(list(dbg.keys()))[:10] if isinstance(dbg, dict) else type(dbg)))

    dbg_key = _closest_debug_key(dbg, int(target_pred_frame))
    attn_info = dbg.get(dbg_key, None) if dbg_key is not None else None

    selected_cond = []
    selected_noncond = []
    if isinstance(attn_info, dict):
        selected_cond = [int(x) for x in attn_info.get("selected_cond_frames", [])]
        selected_noncond = [int(x) for x in attn_info.get("selected_noncond_frames", [])]

    def _save_frame_simple(tag: str, frame_idx: int):
        if tag == "cond":
            rgb_img = condframe_to_rgb.get(int(frame_idx), None)
        else:
            rgb_img = predframe_to_rgb.get(int(frame_idx), None)

        if rgb_img is None:
            if args.debug:
                print(f"[warn] missing rgb for {tag} frame_idx={frame_idx}")
            return False

        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr_img, f"{tag} frame_idx={int(frame_idx)}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA
        )
        cv2.imwrite(str(attn_dir / f"{tag}_{int(frame_idx):05d}.png"), bgr_img)
        return True

    for t in selected_cond:
        _save_frame_simple("cond", t)
    for t in selected_noncond:
        _save_frame_simple("noncond", t)

    # summary
    summary = {
        "seq": seq,
        "export_fidx": int(target_fidx),
        "target_pred_frame": int(target_pred_frame),
        "cached_noncond_frames_exported": [int(x) for x in mem_frames],
        "cached_noncond_len": int(len(ncfo)) if isinstance(ncfo, dict) else None,
        "obj_ids": [int(x) for x in (obj_ids if isinstance(obj_ids, list) else [])],
        "obj_id_to_idx": {str(int(k)): int(v) for k, v in dict(obj_id_to_idx).items()} if isinstance(obj_id_to_idx, dict) else {},
        "prompt_boxes_xyxy": {str(int(k)): [int(vv) for vv in v] for k,v in prompt_boxes_xyxy.items()},
        "debug_memory_attn_key_used": int(dbg_key) if dbg_key is not None else None,
        "attn_selected_cond_frames": selected_cond,
        "attn_selected_noncond_frames": selected_noncond,
        "attn_selected_total_frames": (len(selected_cond) + len(selected_noncond)) if (selected_cond or selected_noncond) else None,
        "attn_info": attn_info if isinstance(attn_info, dict) else None,
    }

    ptr = target_entry.get("obj_ptr", None)
    if torch.is_tensor(ptr):
        sim = ptr_sim(ptr)
        summary["target_ptr_sim"] = sim.round(4).tolist()

    summary["target_entry_keys"] = list(target_entry.keys())

    safe_mkdir(out_root)
    
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[done] exported case to:", out_root)
    print("  - 00_target_frame_overlay.png")
    print("  - 01_condition_frame_prompts.png")
    print("  - memory_frames/*.png (cached noncond)")
    print("  - attn_memory_frames/*.png (actually used in memory_attention; cond uses real cond slots)")
    print("  - summary.json")


if __name__ == "__main__":
    main()