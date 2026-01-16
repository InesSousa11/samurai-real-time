# demo_ktp_colab.py
import os
import re
import cv2
import math
import glob
import numpy as np
import torch
import gradio as gr
import traceback
import plotly.graph_objects as go
from ultralytics import YOLO

import warnings
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

# ---------------- Optional: mount Google Drive (Colab) ----------------
try:
    from google.colab import drive  # type: ignore
    if not os.path.isdir("/content/drive"):
        drive.mount("/content/drive")
except Exception:
    pass

# ---------------- Performance knobs ----------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---------------- Build predictor ----------------
from sam2.build_sam import build_sam2_camera_predictor

REPO = "/content/samurai-real-time"   # adjust if needed
CKPT = f"{REPO}/checkpoints/sam2.1_hiera_small.pt"
CFG  = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(CFG, CKPT)

# ---------------- Runtime score logger ----------------
from rt_scores import ScoresLogger

# ---------------- YOLO models ----------------
yolo_body_model = YOLO("yolov8s.pt")

YOLO_FACE_CKPT = f"{REPO}/checkpoints/yolov8n-face.pt"
if os.path.exists(YOLO_FACE_CKPT):
    try:
        yolo_face_model = YOLO(YOLO_FACE_CKPT)
        print(f"[face] Loaded YOLO face model from local file: {YOLO_FACE_CKPT}")
    except Exception as e:
        print("[face] Failed to load local YOLO face model:", repr(e))
        yolo_face_model = None
else:
    print(f"[face] Local face model not found at {YOLO_FACE_CKPT}. Face proposals will be OFF.")
    yolo_face_model = None


# =========================================================
# Utility helpers
# =========================================================

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def read_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def yolo_person_bboxes(rgb_frame, model, conf_thres=0.25):
    if rgb_frame is None:
        return []
    res = model(rgb_frame, verbose=False, conf=conf_thres)[0]
    out = []
    for det in res.boxes:
        if int(det.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def yolo_face_bboxes(rgb_frame, model, conf_thres=0.30):
    if rgb_frame is None or model is None:
        return []
    res = model(rgb_frame, verbose=False, conf=conf_thres)[0]
    out = []
    for det in res.boxes:
        if int(det.cls) == 0:  # face
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def iou_bbox(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    areaA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    areaB = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return float(inter) / float(areaA + areaB - inter + 1e-9)

def greedy_assignment(iou_mat, iou_th=0.0):
    P, G = iou_mat.shape
    flat = []
    for i in range(P):
        for j in range(G):
            flat.append((float(iou_mat[i, j]), i, j))
    flat.sort(reverse=True, key=lambda t: t[0])

    used_p, used_g = set(), set()
    pairs = []
    for v, i, j in flat:
        if v < iou_th:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, v))
    return pairs

def _to_id_list(out_obj_ids):
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
    return [int(out_obj_ids)]

def _id_to_hue(obj_id: int) -> int:
    return int((37 * int(obj_id) + 61) % 180)

def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits):
    if rgb_frame is None:
        return None

    ids = _to_id_list(out_obj_ids)

    if isinstance(out_mask_logits, (list, tuple)):
        M = len(out_mask_logits)
        get_logits = lambda i: out_mask_logits[i]
    elif torch.is_tensor(out_mask_logits):
        M = int(out_mask_logits.shape[0]) if out_mask_logits.ndim >= 1 else 0
        get_logits = lambda i: out_mask_logits[i]
    else:
        M = 0
        get_logits = lambda i: None

    n = max(0, min(len(ids), M))
    if n == 0:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 0

    for i in range(n):
        logits_i = get_logits(i)
        if logits_i is None or not isinstance(logits_i, torch.Tensor):
            continue
        if logits_i.ndim == 3:
            pm = logits_i[0]
        elif logits_i.ndim == 2:
            pm = logits_i
        else:
            continue

        m = (pm > 0).detach().cpu().numpy().astype(np.uint8)
        sel = m.astype(bool)
        hue = _id_to_hue(ids[i])
        hsv[sel, 0] = hue
        hsv[sel, 2] = 255

    overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, 0.5, 0.0)

def mask_logits_to_bbox_and_centroid(mask_logit: torch.Tensor):
    if mask_logit is None or not isinstance(mask_logit, torch.Tensor):
        return None, None
    if mask_logit.ndim == 3:
        m = mask_logit[0]
    elif mask_logit.ndim == 2:
        m = mask_logit
    else:
        return None, None

    binm = (m > 0).detach().cpu().numpy().astype(np.uint8)
    ys, xs = np.where(binm > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx, cy = float(xs.mean()), float(ys.mean())
    return (x1, y1, x2, y2), (cx, cy)

def dist2d(a, b):
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))


# =========================================================
# KTP dataset helpers (YOUR structure)
#   root/
#     images/<Seq>/{rgb,depth}
#     ground_truth/<Seq>_gt2D(.txt)
# =========================================================

def _is_image_file(p):
    p = p.lower()
    return p.endswith((".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".tif", ".tiff"))

def _extract_timestamp_float_from_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+\.\d+)", base)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+)", base)
    if m2:
        return float(m2.group(1))
    return None

def _gather_files_sorted_any(dir_path: str):
    if not dir_path or not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return []

    img_files = [f for f in files if _is_image_file(f)]
    if img_files:
        files = img_files

    ts = [(_extract_timestamp_float_from_name(f), f) for f in files]
    if len(ts) > 0 and all(t is not None for t,_ in ts):
        ts.sort(key=lambda x: x[0])
        return [f for _, f in ts]

    files.sort()
    return files

def _normalize_path(p: str) -> str:
    p = (p or "").strip()
    p = os.path.expanduser(p)
    return p.rstrip("/")

def _looks_like_ktp_root(root: str) -> bool:
    return (
        os.path.isdir(os.path.join(root, "images")) and
        os.path.isdir(os.path.join(root, "ground_truth"))
    )

def resolve_ktp_root(user_path: str):
    """
    Makes Reload robust:
    - If user points to .../KTP -> OK
    - If user points to .../thesis_datasets -> finds .../thesis_datasets/KTP
    - If user points to .../KTP/images -> climbs one level
    Returns (resolved_root, debug_msg)
    """
    p = _normalize_path(user_path)
    if not p:
        return None, "Empty path."

    candidates = []
    candidates.append(p)
    candidates.append(os.path.join(p, "KTP"))
    candidates.append(os.path.join(p, "ktp"))

    # if they pasted the images folder
    if os.path.basename(p) == "images":
        candidates.append(os.path.dirname(p))

    for c in candidates:
        if _looks_like_ktp_root(c):
            images_dir = os.path.join(c, "images")
            gt_dir = os.path.join(c, "ground_truth")
            return c, f"Resolved KTP root: `{c}` (images=`{images_dir}`, gt=`{gt_dir}`)"

    # last: show what exists
    exists = [c for c in candidates if os.path.isdir(c)]
    if exists:
        return None, "Could not find a folder containing both `images/` and `ground_truth/` under: " + ", ".join(exists)
    return None, f"Path does not exist: `{p}`"

def list_ktp_sequences(ktp_root: str):
    if not ktp_root or not os.path.isdir(ktp_root):
        return []
    images_dir = os.path.join(ktp_root, "images")
    if not os.path.isdir(images_dir):
        return []
    subs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    subs.sort()
    return subs

def _find_rgb_depth_dirs_in_seq(seq_path: str):
    rgb_dir = os.path.join(seq_path, "rgb")
    dep_dir = os.path.join(seq_path, "depth")
    if not os.path.isdir(rgb_dir):
        rgb_dir = os.path.join(seq_path, "RGB") if os.path.isdir(os.path.join(seq_path, "RGB")) else rgb_dir
    if not os.path.isdir(dep_dir):
        dep_dir = os.path.join(seq_path, "Depth") if os.path.isdir(os.path.join(seq_path, "Depth")) else dep_dir
    return (rgb_dir if os.path.isdir(rgb_dir) else None), (dep_dir if os.path.isdir(dep_dir) else None)

def _find_ktp_gt2d_file(gt_dir: str, seq: str):
    pats = [
        f"{seq}_gt2D.txt",
        f"{seq}_gt2D",
        f"{seq}*gt2D*.txt",
        f"{seq}*gt2D*",
        f"{seq}_GT2D*.txt",
        f"{seq}*GT2D*",
    ]
    for pat in pats:
        cands = glob.glob(os.path.join(gt_dir, pat))
        cands = [c for c in cands if os.path.isfile(c)]
        if cands:
            cands.sort()
            return cands[0]
    return None

def _parse_ktp_gt2d(gt_path: str):
    if not gt_path or not os.path.isfile(gt_path):
        return None, [], "no_gt2d"

    entries = []
    ids = set()

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue

            ln = ln.replace(",", " ")
            if ":" not in ln:
                parts = ln.split()
                if len(parts) < 6:
                    continue
                try:
                    ts = float(parts[0])
                except Exception:
                    continue
                rest = parts[1:]
            else:
                left, right = ln.split(":", 1)
                try:
                    ts = float(left.strip())
                except Exception:
                    left2 = re.sub(r"[^\d\.]", "", left)
                    if not left2:
                        continue
                    ts = float(left2)
                rest = right.strip().split()

            nums = []
            for tok in rest:
                tok2 = tok.strip()
                if not tok2:
                    continue
                try:
                    nums.append(float(tok2))
                except Exception:
                    pass

            if len(nums) < 5:
                continue

            boxes = []
            n = (len(nums) // 5) * 5
            for i in range(0, n, 5):
                tid = int(nums[i])
                x, y, w, h = nums[i+1], nums[i+2], nums[i+3], nums[i+4]
                x1, y1 = float(x), float(y)
                x2, y2 = float(x + w), float(y + h)
                boxes.append({"id": tid, "bbox": (x1, y1, x2, y2)})
                ids.add(tid)

            entries.append((float(ts), boxes))

    if not entries:
        return None, [], "gt2d_empty_or_unparsed"

    entries.sort(key=lambda t: t[0])
    return entries, sorted(list(ids)), "ktp_gt2d_timestamp:id_xywh"

def _align_gt_entries_to_rgb_frames(gt_entries, rgb_paths, max_dt=0.05):
    if gt_entries is None:
        return None

    rgb_ts = [_extract_timestamp_float_from_name(p) for p in rgb_paths]
    if any(t is None for t in rgb_ts):
        gt_by_frame = {}
        L = min(len(rgb_paths), len(gt_entries))
        for i in range(L):
            gt_by_frame[i] = gt_entries[i][1]
        for i in range(L, len(rgb_paths)):
            gt_by_frame[i] = []
        return gt_by_frame

    gt_ts = [t for t,_ in gt_entries]
    gt_boxes = [b for _,b in gt_entries]

    gt_by_frame = {}
    j = 0
    for i, t in enumerate(rgb_ts):
        while j + 1 < len(gt_ts) and abs(gt_ts[j+1] - t) <= abs(gt_ts[j] - t):
            j += 1
        if j < len(gt_ts) and abs(gt_ts[j] - t) <= max_dt:
            gt_by_frame[i] = gt_boxes[j]
        else:
            gt_by_frame[i] = []
    return gt_by_frame

def draw_gt_boxes(rgb, gt_list, color=(255,255,0)):
    if rgb is None or not gt_list:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for g in gt_list:
        tid = g["id"]
        x1,y1,x2,y2 = g["bbox"]
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(bgr, (x1,y1), (x2,y2), color, 2)
        cv2.putText(bgr, f"GT:{tid}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# CLEAR MOT computation
# =========================================================

def clear_mot(gt_frames, pred_frames, iou_th=0.5):
    FP = 0
    FN = 0
    IDsw = 0
    matches = 0
    sum_iou = 0.0
    gt_total = 0
    prev_match = {}  # gt_id -> pred_id

    all_frames = sorted(set(list(gt_frames.keys()) + list(pred_frames.keys())))
    for fr in all_frames:
        gt_list = gt_frames.get(fr, [])
        pr_list = pred_frames.get(fr, [])

        gt_total += len(gt_list)

        if len(gt_list) == 0:
            FP += len(pr_list)
            prev_match = {}
            continue
        if len(pr_list) == 0:
            FN += len(gt_list)
            prev_match = {}
            continue

        gt_ids = [g["id"] for g in gt_list]
        pr_ids = [p["id"] for p in pr_list]

        iou_mat = np.zeros((len(pr_list), len(gt_list)), dtype=np.float32)
        for i,p in enumerate(pr_list):
            pb = tuple(map(int, p["bbox"]))
            for j,g in enumerate(gt_list):
                gb = tuple(map(int, g["bbox"]))
                iou_mat[i,j] = iou_bbox(pb, gb)

        pairs = greedy_assignment(iou_mat, iou_th=iou_th)

        matched_pr = set()
        matched_gt = set()

        cur_match = {}
        for i,j,v in pairs:
            pid = pr_ids[i]
            gid = gt_ids[j]
            matched_pr.add(i)
            matched_gt.add(j)
            matches += 1
            sum_iou += float(v)
            cur_match[gid] = pid

            if gid in prev_match and prev_match[gid] != pid:
                IDsw += 1

        FP += (len(pr_list) - len(matched_pr))
        FN += (len(gt_list) - len(matched_gt))
        prev_match = cur_match

    mota = 1.0 - float(FN + FP + IDsw) / float(gt_total + 1e-9)
    motp = float(sum_iou) / float(matches + 1e-9)

    return {
        "MOTA": mota,
        "MOTP": motp,
        "FP": int(FP),
        "FN": int(FN),
        "IDsw": int(IDsw),
        "matches": int(matches),
        "gt_total": int(gt_total),
        "iou_th": float(iou_th),
    }


# =========================================================
# Confusion matrix + Pair consistency
# =========================================================

def make_confusion_matrix_from_frames(per_frame_assign, pred_ids, gt_ids, iou_min=0.0):
    pred_ids = [int(x) for x in pred_ids]
    gt_ids = [int(x) for x in gt_ids]
    pred_ids.sort()
    gt_ids.sort()

    row_index = {p:i for i,p in enumerate(pred_ids)}
    col_index = {g:j for j,g in enumerate(gt_ids)}
    mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int32)

    for rec in per_frame_assign:
        m = rec.get("pred_to_gt", {})
        ious = rec.get("ious", {})
        for pid, gid in m.items():
            pid = int(pid); gid = int(gid)
            if pid not in row_index or gid not in col_index:
                continue
            if float(ious.get(pid, 0.0)) < float(iou_min):
                continue
            mat[row_index[pid], col_index[gid]] += 1

    return pred_ids, gt_ids, mat

def confusion_fig(pred_labels, gt_labels, mat, title):
    vmax = int(mat.max()) if mat.size else 1
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=[str(g) for g in gt_labels],
            y=[str(p) for p in pred_labels],
            colorscale="Blues",
            zmin=0,
            zmax=max(1, vmax),
            colorbar=dict(title="frames"),
            text=mat,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="GT person ID",
        yaxis_title="Tracker ID",
        height=520,
    )
    return fig

def pair_consistency_summary(per_frame_assign, pairs_map, iou_min=0.0):
    out = {}
    for gid, mp in pairs_map.items():
        b = mp.get("body", None)
        f = mp.get("face", None)
        if b is None or f is None:
            continue

        same = 0
        total = 0
        diverge_frames = []
        dists = []
        for rec in per_frame_assign:
            m = rec.get("pred_to_gt", {})
            ious = rec.get("ious", {})
            cents = rec.get("centroids", {})

            if b not in m or f not in m:
                continue
            if float(ious.get(b, 0.0)) < iou_min or float(ious.get(f, 0.0)) < iou_min:
                continue

            total += 1
            gb = int(m[b])
            gf = int(m[f])
            if gb == gf:
                same += 1
            else:
                diverge_frames.append(int(rec.get("frame", -1)))

            cb = cents.get(b, None)
            cf = cents.get(f, None)
            if cb is not None and cf is not None:
                dists.append(dist2d(cb, cf))

        out[int(gid)] = {
            "body_tid": int(b),
            "face_tid": int(f),
            "frames_compared": int(total),
            "same_gt_frames": int(same),
            "same_gt_ratio": (float(same)/float(total) if total > 0 else None),
            "mean_face_body_dist_px": (float(np.mean(dists)) if len(dists) > 0 else None),
            "diverge_frames_sample": diverge_frames[:40],
        }
    return out

def pair_plot(per_frame_assign, gid, body_tid, face_tid):
    xs, dist, same_flag = [], [], []
    for rec in per_frame_assign:
        fr = int(rec.get("frame", -1))
        m = rec.get("pred_to_gt", {})
        cents = rec.get("centroids", {})
        if body_tid not in m or face_tid not in m:
            continue
        cb = cents.get(body_tid, None)
        cf = cents.get(face_tid, None)
        if cb is None or cf is None:
            continue
        xs.append(fr)
        dist.append(dist2d(cb, cf))
        same_flag.append(1 if int(m[body_tid]) == int(m[face_tid]) else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=dist, mode="lines", name="dist(face,body) px"))
    fig.add_trace(go.Scatter(x=xs, y=same_flag, mode="lines", name="same GT (0/1)", yaxis="y2"))
    fig.update_layout(
        title=f"Pair consistency (GT {gid})",
        xaxis_title="frame",
        yaxis=dict(title="pixel distance"),
        yaxis2=dict(title="same GT", overlaying="y", side="right", range=[-0.05, 1.05]),
        height=420
    )
    return fig


# =========================================================
# App state
# =========================================================

state = {
    "first_frame_loaded": False,
    "seeded_any": False,
    "tracking": False,

    "proposal_type": "Body",
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,

    "next_obj_id": 1,
    "added_obj_ids": [],
    "tracker_meta": {},

    "out_obj_ids": None,
    "out_mask_logits": None,

    "frame_idx": 0,
    "scores": ScoresLogger(),
    "selected_obj_for_plot": 1,

    "ktp_root": "",
    "ktp_seq": None,
    "ktp_rgb_paths": [],
    "ktp_depth_paths": [],
    "ktp_has_depth": False,

    "gt_entries_ts": None,
    "gt_by_frame": None,
    "gt_format": None,
    "gt_people_ids": [],
    "gt_to_tracker_pair": {},

    "per_frame_assign": [],
    "pred_boxes_by_frame_all": {},
    "pred_boxes_by_frame_body": {},
    "pred_boxes_by_frame_face": {},
}

def reset_predictor_and_state(keep_dataset=False):
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)

    dataset_keep = {}
    if keep_dataset:
        dataset_keep = {
            "ktp_root": state.get("ktp_root",""),
            "ktp_seq": state.get("ktp_seq",None),
            "ktp_rgb_paths": state.get("ktp_rgb_paths",[]),
            "ktp_depth_paths": state.get("ktp_depth_paths",[]),
            "ktp_has_depth": state.get("ktp_has_depth",False),
            "gt_entries_ts": state.get("gt_entries_ts",None),
            "gt_by_frame": state.get("gt_by_frame",None),
            "gt_format": state.get("gt_format",None),
            "gt_people_ids": state.get("gt_people_ids",[]),
        }

    state.clear()
    state.update({
        "first_frame_loaded": False,
        "seeded_any": False,
        "tracking": False,

        "proposal_type": "Body",
        "selected_idx": 0,
        "cands": [],
        "last_frame": None,

        "next_obj_id": 1,
        "added_obj_ids": [],
        "tracker_meta": {},

        "out_obj_ids": None,
        "out_mask_logits": None,

        "frame_idx": 0,
        "scores": ScoresLogger(),
        "selected_obj_for_plot": 1,

        "ktp_root": "",
        "ktp_seq": None,
        "ktp_rgb_paths": [],
        "ktp_depth_paths": [],
        "ktp_has_depth": False,

        "gt_entries_ts": None,
        "gt_by_frame": None,
        "gt_format": None,
        "gt_people_ids": [],
        "gt_to_tracker_pair": {},

        "per_frame_assign": [],
        "pred_boxes_by_frame_all": {},
        "pred_boxes_by_frame_body": {},
        "pred_boxes_by_frame_face": {},
    })
    state.update(dataset_keep)


# =========================================================
# Load sequence + seed
# =========================================================

def load_ktp_sequence(user_root: str, seq: str, gt_align_max_dt: float = 0.05):
    resolved_root, dbg = resolve_ktp_root(user_root)
    if resolved_root is None:
        return None, gr.update(choices=[], value=[]), f"❌ {dbg}"

    images_dir = os.path.join(resolved_root, "images")
    gt_dir = os.path.join(resolved_root, "ground_truth")

    if not seq:
        return None, gr.update(choices=[], value=[]), f"❌ No sequence selected. ({dbg})"

    seq_path = os.path.join(images_dir, seq)
    if not os.path.isdir(seq_path):
        return None, gr.update(choices=[], value=[]), f"❌ Sequence '{seq}' not found in {images_dir}"

    rgb_dir, dep_dir = _find_rgb_depth_dirs_in_seq(seq_path)
    if rgb_dir is None:
        rgb_dir = seq_path

    rgb_paths = _gather_files_sorted_any(rgb_dir)
    dep_paths = _gather_files_sorted_any(dep_dir) if dep_dir else []

    if len(rgb_paths) == 0:
        return None, gr.update(choices=[], value=[]), f"❌ No RGB images in {rgb_dir}"

    state["ktp_root"] = resolved_root
    state["ktp_seq"] = seq
    state["ktp_rgb_paths"] = rgb_paths
    state["ktp_depth_paths"] = dep_paths
    state["ktp_has_depth"] = (len(dep_paths) == len(rgb_paths) and len(dep_paths) > 0)

    # GT
    gt_path = _find_ktp_gt2d_file(gt_dir, seq)
    msg_gt = "GT2D: not found"
    if gt_path:
        gt_entries, gt_ids, gt_fmt = _parse_ktp_gt2d(gt_path)
        if gt_entries is not None:
            gt_by_frame = _align_gt_entries_to_rgb_frames(gt_entries, rgb_paths, max_dt=float(gt_align_max_dt))
            state["gt_entries_ts"] = gt_entries
            state["gt_by_frame"] = gt_by_frame
            state["gt_people_ids"] = gt_ids
            state["gt_format"] = gt_fmt
            msg_gt = f"GT2D: {os.path.basename(gt_path)} | ids={gt_ids[:8]}{'...' if len(gt_ids)>8 else ''} | align_dt≤{gt_align_max_dt:.2f}s"
        else:
            state["gt_entries_ts"] = None
            state["gt_by_frame"] = None
            state["gt_people_ids"] = []
            state["gt_format"] = gt_fmt
            msg_gt = f"GT2D found but failed parse: {os.path.basename(gt_path)}"
    else:
        state["gt_entries_ts"] = None
        state["gt_by_frame"] = None
        state["gt_people_ids"] = []
        state["gt_format"] = None

    rgb0 = read_rgb(rgb_paths[0])
    if rgb0 is None:
        return None, gr.update(choices=[], value=[]), "❌ Failed to read RGB frame 0."

    if state["gt_by_frame"] is not None:
        rgb0 = draw_gt_boxes(rgb0, state["gt_by_frame"].get(0, []))

    id_choices = [str(i) for i in state["gt_people_ids"]]
    return rgb0, gr.update(choices=id_choices, value=id_choices), f"✅ Loaded '{seq}' frames={len(rgb_paths)} | {msg_gt}<br>{dbg}"

def seed_from_gt(selected_gt_ids, prefer_face=True, face_conf=0.30):
    if not state["ktp_rgb_paths"]:
        return "❌ Load a sequence first."
    if state["gt_by_frame"] is None:
        return "❌ No GT2D aligned (cannot seed from GT)."

    rgb0 = read_rgb(state["ktp_rgb_paths"][0])
    if rgb0 is None:
        return "❌ Failed to read frame 0."

    reset_predictor_and_state(keep_dataset=True)

    predictor.load_first_frame(rgb0)
    state["first_frame_loaded"] = True

    gt0 = state["gt_by_frame"].get(0, [])
    gt0_map = {int(g["id"]): g["bbox"] for g in gt0}

    face_boxes = yolo_face_bboxes(rgb0, yolo_face_model, conf_thres=face_conf) if prefer_face else []

    gt_to_pair = {}
    added = []

    for gid in selected_gt_ids:
        gid = int(gid)
        if gid not in gt0_map:
            continue

        x1,y1,x2,y2 = gt0_map[gid]
        body_bbox = np.array([[x1,y1],[x2,y2]], dtype=np.float32)

        body_tid = state["next_obj_id"]
        predictor.add_new_prompt(frame_idx=0, obj_id=body_tid, bbox=body_bbox)
        state["next_obj_id"] += 1
        added.append(body_tid)
        state["tracker_meta"][body_tid] = {"type":"body","gt":gid}

        face_tid = None
        if face_boxes:
            best = None
            best_iou = 0.0
            bx = (int(x1),int(y1),int(x2),int(y2))
            for fx1,fy1,fx2,fy2,fconf in face_boxes:
                if fx1 >= bx[0] and fy1 >= bx[1] and fx2 <= bx[2] and fy2 <= bx[3]:
                    i = iou_bbox(bx, (fx1,fy1,fx2,fy2))
                    if i > best_iou:
                        best_iou = i
                        best = (fx1,fy1,fx2,fy2)
            if best is not None:
                fx1,fy1,fx2,fy2 = best
                face_bbox = np.array([[fx1,fy1],[fx2,fy2]], dtype=np.float32)
                face_tid = state["next_obj_id"]
                predictor.add_new_prompt(frame_idx=0, obj_id=face_tid, bbox=face_bbox)
                state["next_obj_id"] += 1
                added.append(face_tid)
                state["tracker_meta"][face_tid] = {"type":"face","gt":gid}

        gt_to_pair[gid] = {"body": body_tid, "face": face_tid}

    state["seeded_any"] = (len(added) > 0)
    state["added_obj_ids"] = added
    state["scores"].register_ids(added)
    state["selected_obj_for_plot"] = added[0] if added else 1
    state["gt_to_tracker_pair"] = gt_to_pair

    return f"✅ Seed OK: people={len(gt_to_pair)} | trackers={len(added)} (body+face)."


# =========================================================
# Run KTP sequence + metrics
# =========================================================

def _choices_refresh():
    if state["added_obj_ids"]:
        ch = [int(x) for x in state["added_obj_ids"]]
        return gr.update(choices=ch, value=ch[0])
    return gr.update(choices=[1], value=1)

def _refresh_plot(obj_id:int):
    state["selected_obj_for_plot"] = int(obj_id)
    return state["scores"].make_plot(int(obj_id))

def _refresh_latest_scores(obj_id:int):
    state["selected_obj_for_plot"] = int(obj_id)
    row = state["scores"].latest_row(int(obj_id))
    if not row:
        return "—"
    cells = "".join(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>" for k,v in row.items())
    return f"<table>{cells}</table>"

def _export_csv(obj_id:int):
    path = f"/tmp/scores_obj_{int(obj_id)}.csv"
    state["scores"].export_csv(int(obj_id), path)
    return path

def _export_summary_csv():
    outp = "/tmp/ktp_per_frame_assign.csv"
    with open(outp, "w") as f:
        f.write("frame,tracker_id,tracker_type,assigned_gt,iou,cx,cy\n")
        for rec in state["per_frame_assign"]:
            fr = rec.get("frame", -1)
            m = rec.get("pred_to_gt", {})
            ious = rec.get("ious", {})
            cents = rec.get("centroids", {})
            for tid, gid in m.items():
                meta = state["tracker_meta"].get(int(tid), {})
                ttype = meta.get("type", "unk")
                iou = float(ious.get(int(tid), 0.0))
                c = cents.get(int(tid), (None,None))
                f.write(f"{fr},{int(tid)},{ttype},{int(gid)},{iou:.4f},{c[0]},{c[1]}\n")
    return outp

def _build_pred_boxes(out_obj_ids, out_mask_logits):
    pred_ids = _to_id_list(out_obj_ids)
    if not torch.is_tensor(out_mask_logits):
        return [], {}

    P = min(len(pred_ids), int(out_mask_logits.shape[0]))
    preds = []
    cents = {}
    for i in range(P):
        tid = int(pred_ids[i])
        bbox, cent = mask_logits_to_bbox_and_centroid(out_mask_logits[i])
        if bbox is None:
            continue
        meta = state["tracker_meta"].get(tid, {})
        ttype = meta.get("type", "unk")
        preds.append({"id": tid, "bbox": tuple(map(float,bbox)), "type": ttype})
        if cent is not None:
            cents[tid] = cent
    return preds, cents

@torch.inference_mode()
def run_ktp_sequence(iou_match_body=0.5, iou_match_face=0.2, cm_iou_min=0.0, clear_iou_th=0.5, show_gt_overlay=True):
    if not state["ktp_rgb_paths"]:
        yield None, gr.update(), gr.update(), gr.update(), gr.update(), "❌ Load a sequence first."
        return

    if not state["seeded_any"]:
        yield None, gr.update(), gr.update(), gr.update(), gr.update(), "❌ Seed first (preferably from GT)."
        return

    state["tracking"] = True
    state["frame_idx"] = 0
    state["per_frame_assign"] = []
    state["pred_boxes_by_frame_all"] = {}
    state["pred_boxes_by_frame_body"] = {}
    state["pred_boxes_by_frame_face"] = {}

    has_gt = (state["gt_by_frame"] is not None)
    gt_ids_all = state["gt_people_ids"] if has_gt else []

    for k, rgb_path in enumerate(state["ktp_rgb_paths"]):
        rgb = read_rgb(rgb_path)
        if rgb is None:
            continue
        state["last_frame"] = rgb

        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb)
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits

            state["scores"].log_from_predictor(
                predictor=predictor, obj_ids=out_obj_ids, frame_idx=state["frame_idx"]
            )
            state["frame_idx"] += 1

            vis = draw_mask_overlay(rgb, out_obj_ids, out_mask_logits)

            preds, cents = _build_pred_boxes(out_obj_ids, out_mask_logits)

            all_list = [{"id":p["id"], "bbox":p["bbox"]} for p in preds]
            body_list = [{"id":p["id"], "bbox":p["bbox"]} for p in preds if p["type"] == "body"]
            face_list = [{"id":p["id"], "bbox":p["bbox"]} for p in preds if p["type"] == "face"]

            state["pred_boxes_by_frame_all"][k] = all_list
            state["pred_boxes_by_frame_body"][k] = body_list
            state["pred_boxes_by_frame_face"][k] = face_list

            rec = {"frame": k, "pred_to_gt": {}, "ious": {}, "centroids": cents}

            if has_gt:
                gt_list = state["gt_by_frame"].get(k, [])
                if show_gt_overlay and gt_list:
                    vis = draw_gt_boxes(vis, gt_list, color=(255,255,0))

                if gt_list:
                    gt_ids = [int(g["id"]) for g in gt_list]
                    pred_ids = [int(p["id"]) for p in all_list]

                    if len(pred_ids) > 0 and len(gt_ids) > 0:
                        iou_mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
                        for i,p in enumerate(all_list):
                            pb = tuple(map(int, p["bbox"]))
                            for j,g in enumerate(gt_list):
                                gb = tuple(map(int, g["bbox"]))
                                iou_mat[i,j] = iou_bbox(pb, gb)

                        pairs = greedy_assignment(iou_mat, iou_th=min(float(iou_match_face), float(iou_match_body)))
                        for i,j,v in pairs:
                            tid = pred_ids[i]
                            gid = gt_ids[j]
                            ttype = state["tracker_meta"].get(tid, {}).get("type", "unk")
                            th = float(iou_match_body) if ttype == "body" else float(iou_match_face)
                            if float(v) < th:
                                continue
                            rec["pred_to_gt"][tid] = gid
                            rec["ious"][tid] = float(v)

            state["per_frame_assign"].append(rec)

            if k % 5 == 0:
                plot = state["scores"].make_plot(state["selected_obj_for_plot"])
                info_html = _refresh_latest_scores(state["selected_obj_for_plot"])
            else:
                plot = gr.update()
                info_html = gr.update()

            yield vis, plot, info_html, gr.update(), gr.update(), f"Running frame {k+1}/{len(state['ktp_rgb_paths'])}..."

        except Exception as e:
            print("[error] track() failed:", repr(e))
            print(traceback.format_exc())
            yield rgb, gr.update(), gr.update(), gr.update(), gr.update(), f"❌ Error on frame {k}: {repr(e)}"

    # ---- After run: metrics ----
    summary_lines = []

    if has_gt:
        pred_ids_seen = sorted({int(pid) for rec in state["per_frame_assign"] for pid in rec.get("pred_to_gt", {}).keys()})
        if len(pred_ids_seen) > 0 and len(gt_ids_all) > 0:
            pred_labels = []
            for tid in pred_ids_seen:
                ttype = state["tracker_meta"].get(tid, {}).get("type", "unk")
                pref = "B" if ttype == "body" else ("F" if ttype == "face" else "T")
                pred_labels.append(f"{pref}{tid}")

            _, _, mat = make_confusion_matrix_from_frames(
                state["per_frame_assign"],
                pred_ids=pred_ids_seen,
                gt_ids=gt_ids_all,
                iou_min=float(cm_iou_min),
            )
            cm_fig = confusion_fig(
                pred_labels=pred_labels,
                gt_labels=gt_ids_all,
                mat=mat,
                title=f"Confusion matrix (tracker → GT) | frames (IoU≥{float(cm_iou_min):.2f})"
            )
        else:
            cm_fig = go.Figure()
    else:
        cm_fig = go.Figure()

    if has_gt:
        mot_all  = clear_mot(state["gt_by_frame"], state["pred_boxes_by_frame_all"],  iou_th=float(clear_iou_th))
        mot_body = clear_mot(state["gt_by_frame"], state["pred_boxes_by_frame_body"], iou_th=float(clear_iou_th))
        mot_face = clear_mot(state["gt_by_frame"], state["pred_boxes_by_frame_face"], iou_th=float(clear_iou_th))

        def _mot_html(name, d):
            return (f"<b>{name}</b>: "
                    f"MOTA={d['MOTA']:.3f}, MOTP={d['MOTP']:.3f}, "
                    f"FP={d['FP']}, FN={d['FN']}, IDsw={d['IDsw']}, "
                    f"matches={d['matches']}, GT={d['gt_total']} (IoU_th={d['iou_th']:.2f})")

        summary_lines.append(_mot_html("CLEAR MOT (ALL trackers)", mot_all))
        summary_lines.append(_mot_html("CLEAR MOT (BODY only)", mot_body))
        summary_lines.append(_mot_html("CLEAR MOT (FACE only)", mot_face))
    else:
        summary_lines.append("<b>CLEAR MOT</b>: no GT (cannot compute MOTA/MOTP).")

    if has_gt and state["gt_to_tracker_pair"]:
        pair_stats = pair_consistency_summary(
            state["per_frame_assign"],
            state["gt_to_tracker_pair"],
            iou_min=float(cm_iou_min),
        )
        if pair_stats:
            summary_lines.append("<br><b>FACE↔BODY consistency (same GT?)</b>:")
            summary_lines.append("<table><tr><th>GT</th><th>body_tid</th><th>face_tid</th><th>frames</th><th>same_GT_ratio</th><th>mean_dist_px</th></tr>")
            for gid in sorted(pair_stats.keys()):
                st = pair_stats[gid]
                r = st["same_gt_ratio"]
                r_str = "" if r is None else f"{r:.3f}"
                d = st["mean_face_body_dist_px"]
                d_str = "" if d is None else f"{d:.1f}"
                summary_lines.append(
                    f"<tr><td>{gid}</td><td>{st['body_tid']}</td><td>{st['face_tid']}</td>"
                    f"<td>{st['frames_compared']}</td><td>{r_str}</td><td>{d_str}</td></tr>"
                )
            summary_lines.append("</table>")
        else:
            summary_lines.append("<br><b>FACE↔BODY consistency</b>: no valid face+body pairs.")
    else:
        summary_lines.append("<br><b>FACE↔BODY consistency</b>: no GT or no face/body pairs.")

    summary_html = "<br>".join(summary_lines)

    pair_fig = go.Figure()
    if state["gt_to_tracker_pair"]:
        for gid, mp in state["gt_to_tracker_pair"].items():
            if mp.get("body") is not None and mp.get("face") is not None:
                pair_fig = pair_plot(state["per_frame_assign"], gid, mp["body"], mp["face"])
                break

    yield None, gr.update(), gr.update(), cm_fig, pair_fig, summary_html


# =========================================================
# Manual proposals (optional)
# =========================================================

def update_proposals_on_frame(rgb_frame, proposal_type, conf_body=0.25, conf_face=0.30):
    if rgb_frame is None:
        return [], rgb_frame
    if proposal_type == "Body":
        cands = yolo_person_bboxes(rgb_frame, yolo_body_model, conf_thres=float(conf_body))
        label = "BODY"
        color_sel = (0,255,0)
        color_oth = (0,200,255)
    else:
        cands = yolo_face_bboxes(rgb_frame, yolo_face_model, conf_thres=float(conf_face))
        label = "FACE"
        color_sel = (255,0,255)
        color_oth = (200,100,255)

    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR).copy()
    if cands:
        state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
        for j,(x1,y1,x2,y2,conf) in enumerate(cands):
            color = color_sel if j == state["selected_idx"] else color_oth
            thick = 3 if j == state["selected_idx"] else 1
            cv2.rectangle(bgr, (x1,y1), (x2,y2), color, thick)
            cv2.putText(bgr, f"{label}:{conf:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        hint = "[Accept]=add  [Next]/[Prev]=cycle"
    else:
        hint = "No proposals."
    cv2.putText(bgr, hint, (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return cands, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def on_next():
    if state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_accept_manual(proposal_type):
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."
    n = len(state["cands"])
    state["selected_idx"] = max(0, min(state["selected_idx"], n-1))
    x1,y1,x2,y2,conf = state["cands"][state["selected_idx"]]
    bbox = np.array([[x1,y1],[x2,y2]], dtype=np.float32)

    if not state["first_frame_loaded"]:
        predictor.load_first_frame(state["last_frame"])
        state["first_frame_loaded"] = True

    tid = state["next_obj_id"]
    predictor.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox)
    state["next_obj_id"] += 1
    state["added_obj_ids"].append(tid)
    state["tracker_meta"][tid] = {"type": ("body" if proposal_type=="Body" else "face"), "gt": None}
    state["seeded_any"] = True
    state["scores"].register_ids([tid])
    if len(state["added_obj_ids"]) == 1:
        state["selected_obj_for_plot"] = tid
    return f"Added {proposal_type} tracker #{tid} (conf={conf:.2f})."


# =========================================================
# Gradio UI
# =========================================================

DEFAULT_ROOT = "/content/drive/MyDrive/thesis_datasets/KTP"

with gr.Blocks() as demo:
    gr.Markdown("## KTP Evaluation — SAMURAI (Body+Face prompts) + CLEAR MOT + Confusion matrix + Pair consistency")

    with gr.Accordion("KTP setup", open=True):
        ktp_root = gr.Textbox(
            label="KTP root (folder that contains 'images' and 'ground_truth')",
            value=DEFAULT_ROOT
        )
        seq_dd   = gr.Dropdown(label="Sequence", choices=[], value=None, interactive=True)
        btn_reload = gr.Button("Reload sequences")
        btn_load   = gr.Button("Load sequence")

        gt_align_dt = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="GT↔RGB max timestamp diff (s)")
        gt_ids_box = gr.CheckboxGroup(label="GT person IDs (select people)", choices=[], value=[])
        face_seed = gr.Checkbox(label="Seed face tracker too (YOLO face inside body GT)", value=True)
        face_conf = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="YOLO face conf for seeding")
        btn_seed_gt = gr.Button("Seed from GT (Body + optional Face)")

    out = gr.Image(label="Preview / Output", type="numpy")
    status = gr.Markdown("Status: —")

    with gr.Accordion("Manual seeding (optional)", open=False):
        proposal_type = gr.Radio(["Body","Face"], value="Body", label="Proposal type")
        conf_body = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="YOLO body conf")
        conf_face2 = gr.Slider(0.05, 0.9, value=0.30, step=0.05, label="YOLO face conf")
        with gr.Row():
            btn_prev = gr.Button("Prev")
            btn_accept = gr.Button("Accept (add tracker)")
            btn_next = gr.Button("Next")
        gr.Markdown(
            "Use only if you have no GT or want manual seed. "
            "For face+body pairs you must add two trackers and then check pair consistency."
        )

    with gr.Accordion("Metrics", open=True):
        with gr.Row():
            iou_match_body = gr.Slider(0.05, 0.9, value=0.50, step=0.05, label="IoU match threshold (BODY tracker → GT box)")
            iou_match_face = gr.Slider(0.01, 0.9, value=0.20, step=0.05, label="IoU match threshold (FACE tracker → GT box)")
        with gr.Row():
            clear_iou_th = gr.Slider(0.05, 0.9, value=0.50, step=0.05, label="CLEAR MOT IoU threshold")
            cm_iou_min   = gr.Slider(0.0, 0.9, value=0.00, step=0.05, label="Confusion matrix IoU-min (count only if IoU>=)")
        show_gt_overlay = gr.Checkbox(label="Overlay GT boxes during run", value=True)
        btn_run = gr.Button("Run KTP sequence (compute metrics)")

        with gr.Row():
            obj_select = gr.Dropdown(label="Object to plot (SAMURAI scores)", choices=[1], value=1, interactive=True)
            score_info = gr.HTML(label="Latest scores", value="—")
        plot_scores = gr.Plot(label="Scores over time (selected tracker)")
        cm_plot = gr.Plot(label="Confusion matrix (tracker → GT)")
        pair_plot_ui = gr.Plot(label="Pair plot (face↔body)")
        summary_html = gr.HTML(label="Summary", value="—")

        with gr.Row():
            btn_csv = gr.Button("Export scores CSV (selected tracker)")
            dl_csv = gr.File(label="Download scores CSV")
            btn_assign_csv = gr.Button("Export assignments CSV")
            dl_assign_csv = gr.File(label="Download assignments CSV")

    # ---------------- Reload sequences (FIX: show status) ----------------
    def _ui_reload(root):
        resolved_root, dbg = resolve_ktp_root(root)
        if resolved_root is None:
            return gr.update(choices=[], value=None), f"❌ {dbg}"
        seqs = list_ktp_sequences(resolved_root)
        if not seqs:
            return gr.update(choices=[], value=None), f"⚠️ No sequences found under `{resolved_root}/images`.<br>{dbg}"
        return gr.update(choices=seqs, value=seqs[0]), f"✅ Found {len(seqs)} sequences: {seqs}<br>{dbg}"

    btn_reload.click(fn=_ui_reload, inputs=ktp_root, outputs=[seq_dd, status])

    # Auto-populate sequences on launch
    demo.load(fn=_ui_reload, inputs=ktp_root, outputs=[seq_dd, status])

    # ---------------- Load sequence (FIX: pass numeric confs) ----------------
    def _ui_load(root, seq, align_dt, ptype, cb, cf):
        reset_predictor_and_state(keep_dataset=False)
        img0, ids_update, msg = load_ktp_sequence(root, seq, gt_align_max_dt=float(align_dt))
        state["last_frame"] = img0 if img0 is not None else None
        if img0 is not None:
            state["proposal_type"] = ptype
            cands, preview = update_proposals_on_frame(img0, ptype, float(cb), float(cf))
            state["cands"] = cands
            return preview, ids_update, msg, _choices_refresh()
        return img0, ids_update, msg, _choices_refresh()

    btn_load.click(
        fn=_ui_load,
        inputs=[ktp_root, seq_dd, gt_align_dt, proposal_type, conf_body, conf_face2],
        outputs=[out, gt_ids_box, status, obj_select]
    )

    # Seed from GT
    def _ui_seed_gt(gt_ids, do_face, fconf):
        msg = seed_from_gt([int(x) for x in gt_ids], prefer_face=bool(do_face), face_conf=float(fconf))
        return msg, _choices_refresh(), _refresh_plot(state["selected_obj_for_plot"]), _refresh_latest_scores(state["selected_obj_for_plot"])

    btn_seed_gt.click(fn=_ui_seed_gt, inputs=[gt_ids_box, face_seed, face_conf], outputs=[status, obj_select, plot_scores, score_info])

    # Manual proposals update
    def _ui_update_proposals(ptype, cb, cf):
        state["proposal_type"] = ptype
        if state["last_frame"] is None:
            return gr.update(), "—"
        cands, preview = update_proposals_on_frame(state["last_frame"], ptype, float(cb), float(cf))
        state["cands"] = cands
        return preview, f"Proposals: {ptype} ({len(cands)} found)"

    proposal_type.change(fn=_ui_update_proposals, inputs=[proposal_type, conf_body, conf_face2], outputs=[out, status])
    conf_body.change(fn=_ui_update_proposals, inputs=[proposal_type, conf_body, conf_face2], outputs=[out, status])
    conf_face2.change(fn=_ui_update_proposals, inputs=[proposal_type, conf_body, conf_face2], outputs=[out, status])

    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)

    def _ui_accept(ptype):
        msg = on_accept_manual(ptype)
        return msg, _choices_refresh(), _refresh_plot(state["selected_obj_for_plot"]), _refresh_latest_scores(state["selected_obj_for_plot"])

    btn_accept.click(fn=_ui_accept, inputs=proposal_type, outputs=[status, obj_select, plot_scores, score_info])

    # Run sequence
    btn_run.click(
        fn=run_ktp_sequence,
        inputs=[iou_match_body, iou_match_face, cm_iou_min, clear_iou_th, show_gt_overlay],
        outputs=[out, plot_scores, score_info, cm_plot, pair_plot_ui, summary_html],
    )

    # Export CSV
    btn_csv.click(fn=_export_csv, inputs=obj_select, outputs=dl_csv)
    btn_assign_csv.click(fn=_export_summary_csv, inputs=None, outputs=dl_assign_csv)

    # Timer refresh (scores)
    timer = gr.Timer(0.7)
    timer.tick(fn=_refresh_plot, inputs=obj_select, outputs=plot_scores)
    timer.tick(fn=_refresh_latest_scores, inputs=obj_select, outputs=score_info)

demo.launch(share=True)