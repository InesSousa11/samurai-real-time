#!/usr/bin/env python3
"""
ktp_export_all_id_switch_cases.py

Automatically:
1) Runs SAM2/SAMURAI on a KTP sequence with oracle GT seeding.
2) Detects PREDICTION-CENTRIC ID switches using GLOBAL one-to-one pred<->GT matching
   + hysteresis confirmation (so overlap/occlusion flickers don't count).
   Switch definition:
      "The same predicted ID becomes matched to a different GT person (confirmed)."
3) For EACH detected switch, exports an adviser-friendly "case folder" showing:
   - target frame overlay + event text
   - conditioning frames + prompt bboxes used to seed IDs
   - memory frames currently stored in non_cond_frame_outputs at that time
   - per-frame stored-mask IoU overlays (idx pairs) + score logits

KTP expected:
  KTP/images/<Seq>/rgb/*.jpg
  KTP/ground_truth/<Seq>_gt2D.txt

Outputs:
  <out_dir>/<seq>/cases/
     case_<n>__fidx<...>__pf<...>__PRED<pid>_GT<prev>toGT<cur>/
        00_target_overlay.png
        conditioning_frames/cond_00.png, cond_01.png, ...
        memory_frames/mem_<predFrame>_overlay.png ...
        summary.json
  plus a global CSV log:
     <out_dir>/<seq>/id_switch_events.csv
"""

import sys, re, json, csv, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


# ---------------- filesystem helpers ----------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------- KTP parsing helpers ----------------
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

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min()); x2 = int(xs.max()) + 1
    y1 = int(ys.min()); y2 = int(ys.max()) + 1
    return (x1,y1,x2,y2)

def logits_to_mask_bbox(logits: torch.Tensor) -> Optional[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """
    logits: (1,H,W) or (H,W) torch
    returns (mask_bool, bbox_xyxy) with a few thresholds to avoid empty bbox
    """
    if logits is None or (not torch.is_tensor(logits)):
        return None

    if logits.ndim == 3:
        lg = logits[0]
    elif logits.ndim == 2:
        lg = logits
    else:
        return None

    lg = lg.detach()
    for thr in (0.0, -2.0, -4.0):
        m = (lg > thr).cpu().numpy().astype(bool)
        bb = mask_to_bbox(m)
        if bb is not None:
            return (m, bb)
    return None


# ---------------- visualization helpers ----------------
def id_to_hue(obj_id: int) -> int:
    return int((37 * int(obj_id) + 61) % 180)

def overlay_masks_from_logits(rgb_frame: np.ndarray, obj_ids: List[int], logits_tensor: torch.Tensor, alpha=0.5):
    """
    logits_tensor: (N,1,H,W) torch logits
    obj_ids length N corresponds to rows.
    """
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

    n = min(len(obj_ids), int(lg.shape[0]))
    for i in range(n):
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

def masks_iou_from_pred_masks(pm_1hw: torch.Tensor, max_n: int = 3) -> List[Tuple[int,int,float]]:
    """
    pm_1hw: (N,1,H,W) logits (torch).
    Returns list of (i,j,iou) for pairs among first max_n objects.
    """
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

def ptr_sim(ptr: torch.Tensor) -> np.ndarray:
    P = F.normalize(ptr.detach().float().cpu(), dim=1)
    return (P @ P.t()).cpu().numpy()


# ---------------- predictor interaction helpers ----------------
def seed_bbox(predictor, obj_id: int, bbox_xyxy, rgb_frame: np.ndarray, late: bool):
    bbox = np.array([[bbox_xyxy[0], bbox_xyxy[1]], [bbox_xyxy[2], bbox_xyxy[3]]], dtype=np.float32)
    if not late:
        predictor.add_new_prompt(frame_idx=0, obj_id=int(obj_id), bbox=bbox)
    else:
        predictor.add_conditioning_frame(rgb_frame)
        predictor.add_new_prompt_during_track(
            bbox=bbox, if_new_target=True, obj_id=int(obj_id),
            labels=None, clear_old_points=True
        )

def get_latest_noncond_entry(predictor):
    cs = predictor.condition_state
    od = cs.get("output_dict", {})
    ncfo = od.get("non_cond_frame_outputs", {})
    if not isinstance(ncfo, dict) or len(ncfo) == 0:
        return None, None, []
    f_last = max(ncfo.keys())
    return int(f_last), ncfo[f_last], sorted(list(ncfo.keys()))


# ---------------- GLOBAL assignment (one-to-one) ----------------
def compute_gt_max_overlap(gt_bb_by_id: Dict[int, Tuple[int,int,int,int]]) -> Dict[int, float]:
    gt_ids = list(gt_bb_by_id.keys())
    out: Dict[int, float] = {}
    for gid in gt_ids:
        bb = gt_bb_by_id[gid]
        mx = 0.0
        for ogid in gt_ids:
            if ogid == gid:
                continue
            mx = max(mx, iou_xyxy(bb, gt_bb_by_id[ogid]))
        out[gid] = mx
    return out

def assign_preds_to_gt_global(
    pred_bbox_by_id: Dict[int, Tuple[int,int,int,int]],
    gt_bb_by_id: Dict[int, Tuple[int,int,int,int]],
    match_iou_thr: float,
    gt_max_overlap: Dict[int, float],
    gt_overlap_ignore_thr: float,
    best_second_margin: float,
) -> Tuple[Dict[int, Optional[int]], Dict[int, float]]:
    """
    Returns:
      pred_to_gt: {pred_id -> gt_id or None}
      pred_best_iou: {pred_id -> best_iou (even if unmatched)}
    Matching is:
      - compute per-pred best/second IoU
      - if best < thr or best-second < margin or GT too overlapped -> pred gets no assignment
      - then do global greedy one-to-one selection by descending IoU of the best pairs
    """
    pred_to_gt: Dict[int, Optional[int]] = {pid: None for pid in pred_bbox_by_id.keys()}
    pred_best_iou: Dict[int, float] = {pid: 0.0 for pid in pred_bbox_by_id.keys()}

    # precompute per-pred best and second best
    best_pair: Dict[int, Tuple[int, float, float]] = {}  # pid -> (best_gid, best_iou, second_iou)

    for pid, pb in pred_bbox_by_id.items():
        scores = []
        for gid, gb in gt_bb_by_id.items():
            scores.append((gid, iou_xyxy(pb, gb)))
        if not scores:
            continue
        scores.sort(key=lambda t: t[1], reverse=True)
        best_gid, best_iou = scores[0]
        second_iou = scores[1][1] if len(scores) > 1 else 0.0
        pred_best_iou[pid] = float(best_iou)
        best_pair[pid] = (int(best_gid), float(best_iou), float(second_iou))

    # build candidate list (pid can propose only its best gt if it passes filters)
    candidates = []
    for pid, (best_gid, best_iou, second_iou) in best_pair.items():
        if best_iou < match_iou_thr:
            continue
        if (best_iou - second_iou) < best_second_margin:
            continue
        if gt_max_overlap.get(best_gid, 0.0) > gt_overlap_ignore_thr:
            continue
        candidates.append((best_iou, pid, best_gid))

    # global greedy one-to-one
    candidates.sort(key=lambda t: t[0], reverse=True)
    used_preds = set()
    used_gts = set()

    for best_iou, pid, gid in candidates:
        if pid in used_preds or gid in used_gts:
            continue
        pred_to_gt[pid] = gid
        used_preds.add(pid)
        used_gts.add(gid)

    return pred_to_gt, pred_best_iou


# ---------------- export a case folder at the CURRENT predictor state ----------------
def export_case(
    out_case_dir: Path,
    seq: str,
    fidx: int,
    ts_s: str,
    pred_frame: int,
    event_str: str,
    predictor,
    rgb0: np.ndarray,
    conditioning_frames: List[Dict],
    predframe_to_rgb: Dict[int, np.ndarray],
    memory_limit: int,
    alpha: float,
):
    safe_mkdir(out_case_dir)
    mem_dir = out_case_dir / "memory_frames"
    safe_mkdir(mem_dir)
    cond_dir = out_case_dir / "conditioning_frames"
    safe_mkdir(cond_dir)

    cs = predictor.condition_state
    od = cs.get("output_dict", {})
    ncfo = od.get("non_cond_frame_outputs", {})
    if not isinstance(ncfo, dict) or len(ncfo) == 0:
        return

    mem_frames = sorted(list(ncfo.keys()))
    if memory_limit > 0 and len(mem_frames) > memory_limit:
        mem_frames = mem_frames[-memory_limit:]

    # conditioning panels
    for info in conditioning_frames:
        cond_idx = int(info["cond_idx"])
        rgb = info["rgb"]
        prompts = info["prompts"]
        bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        if prompts:
            boxes = [bb for (_, bb) in prompts]
            labels = [f"prompt id={oid}" for (oid, _) in prompts]
            bgr = draw_boxes_and_labels(bgr, boxes, labels, color=(255,255,255), thick=2)
        cv2.putText(bgr, f"CONDITIONING idx={cond_idx}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(str(cond_dir / f"cond_{cond_idx:02d}.png"), bgr)

    # target overlay
    target_entry = ncfo.get(int(pred_frame), None)
    if target_entry is None:
        pred_frame = max(ncfo.keys())
        target_entry = ncfo[int(pred_frame)]

    tgt_rgb = predframe_to_rgb.get(int(pred_frame), rgb0.copy())
    out_rgb = tgt_rgb.copy()

    pm = target_entry.get("pred_masks", None)
    sc = target_entry.get("object_score_logits", None)

    obj_ids = cs.get("obj_ids", [])
    idx2id = cs.get("obj_idx_to_id", {})

    if torch.is_tensor(pm):
        N = int(pm.shape[0])
        ids_by_idx = []
        for i in range(N):
            oid = idx2id.get(i, None)
            if oid is None and i < len(obj_ids):
                oid = obj_ids[i]
            ids_by_idx.append(int(oid) if oid is not None else int(i))

        pm_cpu = pm.detach().float().cpu()
        pm_up = torch.nn.functional.interpolate(pm_cpu, size=(out_rgb.shape[0], out_rgb.shape[1]),
                                                mode="bilinear", align_corners=False)
        out_rgb = overlay_masks_from_logits(out_rgb, ids_by_idx, pm_up, alpha=alpha)

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(out_bgr, f"TARGET {seq} fidx={fidx} ts={ts_s} pred_frame={int(pred_frame)}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out_bgr, f"EVENT: {event_str}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    if torch.is_tensor(sc):
        scv = sc.detach().reshape(-1).cpu().numpy().tolist()
        y = 75
        for i, s in enumerate(scv[:10]):
            oid = idx2id.get(i, None)
            if oid is None and i < len(obj_ids):
                oid = obj_ids[i]
            cv2.putText(out_bgr, f"id={oid} score_logit={float(s):.3f}", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            y += 20

    if torch.is_tensor(pm):
        pairs = masks_iou_from_pred_masks(pm, max_n=3)
        y2 = out_bgr.shape[0] - 70
        for (i, j, v) in pairs:
            cv2.putText(out_bgr, f"IoU(idx{i},idx{j})={v:.3f}", (10, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            y2 += 20

    cv2.imwrite(str(out_case_dir / "00_target_overlay.png"), out_bgr)

    # memory frames export
    for pf in mem_frames:
        entry = ncfo[int(pf)]
        rgb_mem = predframe_to_rgb.get(int(pf), None)
        if rgb_mem is None:
            continue

        out_rgb = rgb_mem.copy()
        pm = entry.get("pred_masks", None)
        sc = entry.get("object_score_logits", None)

        obj_ids = cs.get("obj_ids", [])
        idx2id = cs.get("obj_idx_to_id", {})

        if torch.is_tensor(pm):
            N = int(pm.shape[0])
            ids_by_idx = []
            for i in range(N):
                oid = idx2id.get(i, None)
                if oid is None and i < len(obj_ids):
                    oid = obj_ids[i]
                ids_by_idx.append(int(oid) if oid is not None else int(i))

            pm_cpu = pm.detach().float().cpu()
            pm_up = torch.nn.functional.interpolate(pm_cpu, size=(out_rgb.shape[0], out_rgb.shape[1]),
                                                    mode="bilinear", align_corners=False)
            out_rgb = overlay_masks_from_logits(out_rgb, ids_by_idx, pm_up, alpha=alpha)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(out_bgr, f"MEM pred_frame={int(pf)}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if torch.is_tensor(sc):
            scv = sc.detach().reshape(-1).cpu().numpy().tolist()
            y = 50
            for i, s in enumerate(scv[:10]):
                oid = idx2id.get(i, None)
                if oid is None and i < len(obj_ids):
                    oid = obj_ids[i]
                cv2.putText(out_bgr, f"id={oid} score_logit={float(s):.3f}", (10,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                y += 20

        if torch.is_tensor(pm):
            pairs = masks_iou_from_pred_masks(pm, max_n=3)
            y2 = out_bgr.shape[0] - 70
            for (i, j, v) in pairs:
                cv2.putText(out_bgr, f"IoU(idx{i},idx{j})={v:.3f}", (10, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                y2 += 20

        cv2.imwrite(str(mem_dir / f"mem_{int(pf):05d}_overlay.png"), out_bgr)

    summary = {
        "seq": seq,
        "fidx": int(fidx),
        "ts": ts_s,
        "pred_frame": int(pred_frame),
        "event": event_str,
        "memory_frames_exported": [int(x) for x in mem_frames],
        "target_entry_keys": list(target_entry.keys()),
        "obj_ids": [int(x) for x in (cs.get("obj_ids", []) if isinstance(cs.get("obj_ids", []), list) else [])],
        "obj_id_to_idx": {str(int(k)): int(v) for k, v in dict(cs.get("obj_id_to_idx", {})).items()} if isinstance(cs.get("obj_id_to_idx", {}), dict) else {},
        "obj_idx_to_id": {str(int(k)): int(v) for k, v in dict(cs.get("obj_idx_to_id", {})).items()} if isinstance(cs.get("obj_idx_to_id", {}), dict) else {},
    }

    ptr = target_entry.get("obj_ptr", None)
    if torch.is_tensor(ptr):
        summary["target_ptr_sim"] = np.round(ptr_sim(ptr), 4).tolist()

    with (out_case_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------- main ----------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", type=str, required=True)
    ap.add_argument("--seq", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=-1)

    # oracle seed guards
    ap.add_argument("--visible_area_frac", type=float, default=0.02)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.10)

    # pred->GT assignment (global)
    ap.add_argument("--match_iou_thr", type=float, default=0.30)
    ap.add_argument("--gt_overlap_ignore_thr", type=float, default=0.20)
    ap.add_argument("--best_second_margin", type=float, default=0.10)

    # hysteresis (IMPORTANT)
    ap.add_argument("--min_prev_streak", type=int, default=3,
                    help="Previous (old) pred->GT assignment must be stable for at least this many frames before we allow a switch.")
    ap.add_argument("--confirm_frames", type=int, default=2,
                    help="New pred->GT assignment must persist for this many consecutive frames to confirm a switch.")

    # export params
    ap.add_argument("--memory_limit", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--max_cases", type=int, default=50)
    ap.add_argument("--min_frames_between_cases", type=int, default=10)

    args = ap.parse_args()

    ktp_root = Path(args.ktp_root).resolve()
    seq = args.seq
    out_root = Path(args.out_dir).resolve() / seq
    cases_root = out_root / "cases"
    safe_mkdir(cases_root)
    safe_mkdir(out_root)

    img_dir = ktp_root / "images" / seq / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq}_gt2D.txt"
    if not img_dir.exists(): raise FileNotFoundError(img_dir)
    if not gt_path.exists(): raise FileNotFoundError(gt_path)
    if not CKPT_PATH.exists(): raise FileNotFoundError(CKPT_PATH)
    if not CFG_PATH.exists(): raise FileNotFoundError(CFG_PATH)

    print("[paths]")
    print("  KTP_ROOT:", ktp_root)
    print("  IMG_DIR :", img_dir)
    print("  GT      :", gt_path)
    print("  OUT     :", out_root)

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
    if len(items) == 0:
        raise RuntimeError("No frames found after filtering.")

    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
    print("SAMURAI mode:", getattr(predictor, "samurai_mode", None))

    # load first frame
    _, ts0_s, p0 = items[0]
    bgr0 = cv2.imread(str(p0))
    if bgr0 is None:
        raise RuntimeError("Failed to read first frame")
    H, W = bgr0.shape[:2]
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
    predictor.load_first_frame(rgb0)

    conditioning_frames: List[Dict] = [{
        "cond_idx": 0,
        "rgb": rgb0.copy(),
        "prompts": []
    }]

    seeded: set = set()

    # Cache RGB for recent pred_frame indices (so memory frames can be saved)
    predframe_to_rgb: Dict[int, np.ndarray] = {}
    max_cache = 500

    # Hysteresis tracking per predicted id
    confirmed_gt: Dict[int, Optional[int]] = {}      # pid -> confirmed GT
    confirmed_streak: Dict[int, int] = {}            # pid -> how long confirmed GT has held
    pending_gt: Dict[int, Optional[int]] = {}        # pid -> candidate new GT
    pending_count: Dict[int, int] = {}               # pid -> how long candidate held

    events_csv = out_root / "id_switch_events.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["seq","fidx","ts","pred_frame","pred_id","prev_gt","cur_gt","details"])

        exported = 0
        last_export_fidx = -10_000

        for fidx, (ts_f, ts_s, fp) in enumerate(items):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            gt_dets = gt_map.get(ts_s, [])
            gt_bb_by_id: Dict[int, Tuple[int,int,int,int]] = {}
            for (gid, x, y, w_, h_) in gt_dets:
                gt_bb_by_id[gid] = clamp_bbox_xyxy(bbox_xywh_to_xyxy(x,y,w_,h_), W, H)

            # ---- oracle seeding ----
            for (gid, x, y, w_, h_) in gt_dets:
                if gid in seeded:
                    continue

                bb = gt_bb_by_id[gid]
                bw = max(0, bb[2]-bb[0]); bh = max(0, bb[3]-bb[1])
                area_frac = (bw*bh) / float(W*H + 1e-9)
                if not (area_frac >= args.visible_area_frac and bh >= args.visible_min_h):
                    continue

                max_iou_other = 0.0
                for ogid, obb in gt_bb_by_id.items():
                    if ogid == gid:
                        continue
                    max_iou_other = max(max_iou_other, iou_xyxy(bb, obb))
                if max_iou_other > args.seed_overlap_iou_max:
                    continue

                late = (fidx != 0)
                try:
                    seed_bbox(predictor, gid, bb, rgb_frame=rgb, late=late)
                    seeded.add(gid)

                    if late:
                        cond_idx = len(conditioning_frames)
                        conditioning_frames.append({
                            "cond_idx": cond_idx,
                            "rgb": rgb.copy(),
                            "prompts": [(int(gid), bb)]
                        })
                    else:
                        conditioning_frames[0]["prompts"].append((int(gid), bb))
                except Exception:
                    pass

            # ---- track ----
            try:
                out_obj_ids, out_mask_logits = predictor.track(rgb)
            except Exception:
                out_obj_ids, out_mask_logits = [], None

            pred_frame, entry, mem_keys = get_latest_noncond_entry(predictor)
            if pred_frame is None:
                continue

            predframe_to_rgb[int(pred_frame)] = rgb.copy()
            if len(predframe_to_rgb) > max_cache:
                for k in sorted(predframe_to_rgb.keys())[: len(predframe_to_rgb) - max_cache]:
                    predframe_to_rgb.pop(k, None)

            # ---- predicted bboxes by pred_id (obj_id) ----
            cs = predictor.condition_state
            obj_id_to_idx = cs.get("obj_id_to_idx", {}) if isinstance(cs, dict) else {}

            pred_bbox_by_id: Dict[int, Tuple[int,int,int,int]] = {}
            if out_mask_logits is not None and isinstance(obj_id_to_idx, dict):
                if out_obj_ids is None:
                    ids_list = []
                elif torch.is_tensor(out_obj_ids):
                    ids_list = [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
                elif isinstance(out_obj_ids, (list, tuple)):
                    ids_list = [int(x) for x in out_obj_ids]
                else:
                    ids_list = [int(out_obj_ids)]

                def logits_for_obj_id(obj_id: int) -> Optional[torch.Tensor]:
                    if obj_id not in obj_id_to_idx:
                        return None
                    idx = int(obj_id_to_idx[obj_id])
                    if torch.is_tensor(out_mask_logits):
                        if out_mask_logits.ndim < 3:
                            return None
                        if not (0 <= idx < int(out_mask_logits.shape[0])):
                            return None
                        return out_mask_logits[idx]
                    if isinstance(out_mask_logits, (list, tuple)):
                        if not (0 <= idx < len(out_mask_logits)):
                            return None
                        return out_mask_logits[idx] if torch.is_tensor(out_mask_logits[idx]) else None
                    return None

                for pid in ids_list:
                    lg = logits_for_obj_id(int(pid))
                    if lg is None:
                        continue
                    res = logits_to_mask_bbox(lg)
                    if res is None:
                        continue
                    _, bb = res
                    pred_bbox_by_id[int(pid)] = clamp_bbox_xyxy(bb, W, H)

            # ---- global one-to-one assignment ----
            gt_max_overlap = compute_gt_max_overlap(gt_bb_by_id)
            pred_to_gt_cur, pred_best_iou = assign_preds_to_gt_global(
                pred_bbox_by_id=pred_bbox_by_id,
                gt_bb_by_id=gt_bb_by_id,
                match_iou_thr=float(args.match_iou_thr),
                gt_max_overlap=gt_max_overlap,
                gt_overlap_ignore_thr=float(args.gt_overlap_ignore_thr),
                best_second_margin=float(args.best_second_margin),
            )

            # ---- hysteresis switch detection ----
            switch_events = []
            for pid, gt_cur in pred_to_gt_cur.items():
                # if no confident assignment now -> do not advance streaks aggressively
                if gt_cur is None:
                    pending_gt.pop(pid, None)
                    pending_count.pop(pid, None)
                    # decay confirmed streak
                    if pid in confirmed_streak:
                        confirmed_streak[pid] = max(0, confirmed_streak[pid] - 1)
                    continue

                gt_prev = confirmed_gt.get(pid, None)

                if gt_prev is None:
                    confirmed_gt[pid] = gt_cur
                    confirmed_streak[pid] = 1
                    pending_gt.pop(pid, None)
                    pending_count.pop(pid, None)
                    continue

                if gt_cur == gt_prev:
                    confirmed_streak[pid] = confirmed_streak.get(pid, 0) + 1
                    pending_gt.pop(pid, None)
                    pending_count.pop(pid, None)
                    continue

                # different from confirmed: build/advance pending
                if pending_gt.get(pid, None) == gt_cur:
                    pending_count[pid] = pending_count.get(pid, 0) + 1
                else:
                    pending_gt[pid] = gt_cur
                    pending_count[pid] = 1

                # confirm switch only if old was stable enough AND new persists enough
                if confirmed_streak.get(pid, 0) >= int(args.min_prev_streak) and pending_count[pid] >= int(args.confirm_frames):
                    switch_events.append((pid, gt_prev, gt_cur, pred_best_iou.get(pid, 0.0)))
                    confirmed_gt[pid] = gt_cur
                    confirmed_streak[pid] = pending_count[pid]
                    pending_gt.pop(pid, None)
                    pending_count.pop(pid, None)

            # ---- export cases ----
            for (pid, gt_prev, gt_cur, biou) in switch_events:
                if exported >= int(args.max_cases):
                    break
                if fidx - last_export_fidx < int(args.min_frames_between_cases):
                    continue

                event_str = f"PRED{pid}: GT{gt_prev} -> GT{gt_cur} (IoU={biou:.3f})"
                w.writerow([seq, fidx, ts_s, int(pred_frame), pid, gt_prev, gt_cur, event_str])

                case_name = f"case_{exported:03d}__fidx{fidx:05d}__pf{int(pred_frame):05d}__PRED{pid}_GT{gt_prev}toGT{gt_cur}"
                case_dir = cases_root / case_name

                export_case(
                    out_case_dir=case_dir,
                    seq=seq,
                    fidx=fidx,
                    ts_s=ts_s,
                    pred_frame=int(pred_frame),
                    event_str=event_str,
                    predictor=predictor,
                    rgb0=rgb0,
                    conditioning_frames=conditioning_frames,
                    predframe_to_rgb=predframe_to_rgb,
                    memory_limit=int(args.memory_limit),
                    alpha=float(args.alpha),
                )

                exported += 1
                last_export_fidx = fidx
                print(f"[export] {case_name}")

            if exported >= int(args.max_cases):
                break

    print("\n[done]")
    print("  events csv:", events_csv)
    print("  cases dir :", cases_root)
    print("If needed, tune: --match_iou_thr, --best_second_margin, --gt_overlap_ignore_thr, --min_prev_streak, --confirm_frames")

if __name__ == "__main__":
    main()