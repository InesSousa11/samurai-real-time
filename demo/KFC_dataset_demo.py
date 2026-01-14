# demo_ktp_colab.py
import os
import re
import cv2
import time
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

# ---------------- Performance knobs ----------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---------------- Build predictor ----------------
from sam2.build_sam import build_sam2_camera_predictor

REPO = "/content/samurai-real-time"   # ajusta se necessário
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

def get_thresholds_from_predictor():
    stable_frames = int(getattr(predictor, "stable_frames_threshold", 15))
    stable_iou_th = float(getattr(predictor, "stable_ious_threshold", 0.3))
    min_obj_logit = float(getattr(predictor, "min_obj_score_logits", -1))
    obj_prob_th   = _sigmoid(min_obj_logit)
    return stable_frames, stable_iou_th, min_obj_logit, obj_prob_th

def read_rgb(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def read_depth(path: str):
    # KTP depth pode vir em PNG 16-bit. Nós só carregamos (opcional)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return d

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
    """
    Greedy max matching (simples, sem scipy):
    retorna lista de (pred_i, gt_j, iou) para pares com iou >= iou_th
    """
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
        # logits_i expected like (1,H,W) or (H,W) or (C,H,W)
        if logits_i.ndim == 3:
            pm = logits_i[0] if logits_i.shape[0] == 1 else logits_i[0]
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
    """
    Recebe um logit (H,W) ou (1,H,W) e devolve bbox (x1,y1,x2,y2) e centroid (cx,cy)
    """
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

def bbox_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def dist2d(a, b):
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))


# =========================================================
# KTP dataset helpers (robustos / tolerantes)
# =========================================================

def _is_image_file(p):
    return p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

def _extract_number(s):
    m = re.findall(r"\d+", os.path.basename(s))
    return int(m[-1]) if m else None

def _extract_timestamp_float(s):
    # tenta apanhar algo tipo 1234567890.123
    base = os.path.splitext(os.path.basename(s))[0]
    m = re.search(r"(\d+\.\d+)", base)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+)", base)
    if m2:
        return float(m2.group(1))
    return None

def list_ktp_sequences(ktp_root: str):
    if not ktp_root or not os.path.isdir(ktp_root):
        return []
    subs = []
    for d in os.listdir(ktp_root):
        p = os.path.join(ktp_root, d)
        if os.path.isdir(p):
            subs.append(d)
    subs.sort()
    return subs

def _find_rgb_depth_dirs(seq_path: str):
    """
    Procura dirs típicos: rgb/, RGB/, color/, Color/, depth/, Depth/
    Se não existir, tenta encontrar imagens diretamente.
    """
    cand_rgb = ["rgb", "RGB", "color", "Color", "image", "images", "Images"]
    cand_dep = ["depth", "Depth", "dep", "DEPTH"]

    rgb_dir = None
    dep_dir = None

    for c in cand_rgb:
        p = os.path.join(seq_path, c)
        if os.path.isdir(p):
            rgb_dir = p
            break

    for c in cand_dep:
        p = os.path.join(seq_path, c)
        if os.path.isdir(p):
            dep_dir = p
            break

    return rgb_dir, dep_dir

def _gather_images_sorted(img_dir_or_seq_path: str):
    """
    Junta imagens e ordena por timestamp/numero/nome.
    """
    if img_dir_or_seq_path is None:
        return []

    if os.path.isdir(img_dir_or_seq_path):
        files = [os.path.join(img_dir_or_seq_path, f) for f in os.listdir(img_dir_or_seq_path) if _is_image_file(f)]
    else:
        files = []

    # tenta ordenar por timestamp float, se existir
    ts = [(_extract_timestamp_float(f), f) for f in files]
    if all(t is not None for t,_ in ts) and len(ts) > 0:
        ts.sort(key=lambda x: x[0])
        return [f for _,f in ts]

    # senão por número
    nn = [(_extract_number(f), f) for f in files]
    if all(n is not None for n,_ in nn) and len(nn) > 0:
        nn.sort(key=lambda x: x[0])
        return [f for _,f in nn]

    # fallback: lexicográfico
    files.sort()
    return files

def _find_gt_file(seq_path: str):
    """
    Procura ficheiros de GT em formatos comuns.
    Exemplos:
      - gt.txt / groundtruth.txt / annotations.txt
      - algo com 'gt' no nome
    """
    patterns = [
        "gt.txt", "GT.txt", "groundtruth.txt", "GroundTruth.txt",
        "annotations.txt", "Annotations.txt", "label.txt", "labels.txt",
        "*gt*.txt", "*GT*.txt", "*ground*.txt", "*annot*.txt",
        "*.csv"
    ]
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(os.path.join(seq_path, pat)))
    # remove duplicados mantendo ordem
    seen = set()
    out = []
    for p in cands:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            out.append(p)
    return out[0] if out else None

def _parse_gt_generic(gt_path: str):
    """
    Lê GT e tenta interpretar como caixas 2D por frame.

    Suporta (heurístico):
    - MOTChallenge: frame, id, x, y, w, h, ...
    - frame id x y w h
    - timestamp id x y w h  (neste caso converte para frames por ordenação)
    - frame,id,x1,y1,x2,y2
    """
    if not gt_path or not os.path.isfile(gt_path):
        return None, "no_gt_file"

    lines = []
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            # split por comma/space/tab
            parts = re.split(r"[,\s]+", ln)
            parts = [p for p in parts if p != ""]
            if len(parts) < 6:
                continue
            # tenta converter tudo a float
            try:
                vals = [float(p) for p in parts]
                lines.append(vals)
            except Exception:
                continue

    if not lines:
        return None, "gt_empty_or_unparsed"

    # Decide formato
    # Heurística: primeira coluna (frame ou timestamp). Se for grande (>=1e9) talvez timestamp.
    firsts = [row[0] for row in lines[:50]]
    is_timestamp = (np.median(firsts) > 1e6)  # heurístico

    gt_by_frame = {}

    if is_timestamp:
        # map timestamps únicos para índice de frame por ordem
        ts_unique = sorted(set([row[0] for row in lines]))
        ts_to_frame = {ts:i for i,ts in enumerate(ts_unique)}
        for row in lines:
            fr = ts_to_frame[row[0]]
            tid = int(row[1])
            x, y, w, h = row[2], row[3], row[4], row[5]
            # se vier x2,y2 em vez de w,h
            if w > 0 and h > 0 and (row[4] > row[2]) and (row[5] > row[3]) and len(row) >= 6:
                # pode ser x1,y1,x2,y2
                # mas isto é ambíguo; mantemos MOT-like por default
                pass
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w), float(y + h)
            gt_by_frame.setdefault(fr, []).append({"id": tid, "bbox": (x1,y1,x2,y2)})
        return gt_by_frame, "timestamp_id_xywh"

    # não timestamp
    # Se colunas 3..6 parecem x2,y2 (maior que x1,y1) e não parecem w/h
    # tentamos distinguir:
    # - se (col4 > col2 e col5 > col3) e valores não parecem larguras pequenas?
    row0 = lines[0]
    # assume: [f, id, a, b, c, d, ...]
    maybe_x2y2 = (row0[4] > row0[2] and row0[5] > row0[3])

    for row in lines:
        fr = int(row[0])
        tid = int(row[1])
        a,b,c,d = row[2], row[3], row[4], row[5]

        if maybe_x2y2:
            # frame,id,x1,y1,x2,y2
            x1,y1,x2,y2 = float(a), float(b), float(c), float(d)
        else:
            # frame,id,x,y,w,h
            x1,y1 = float(a), float(b)
            x2,y2 = float(a + c), float(b + d)

        gt_by_frame.setdefault(fr, []).append({"id": tid, "bbox": (x1,y1,x2,y2)})

    return gt_by_frame, "frame_id_box"

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
    """
    gt_frames: dict frame_idx -> list of {"id":int, "bbox":(x1,y1,x2,y2)}
    pred_frames: dict frame_idx -> list of {"id":int, "bbox":(x1,y1,x2,y2)}
    Returns dict with MOTA/MOTP + FP/FN/IDs/matches/gt_total
    """
    FP = 0
    FN = 0
    IDsw = 0
    matches = 0
    sum_iou = 0.0
    gt_total = 0

    # Mapping do frame anterior: gt_id -> pred_id
    prev_match = {}

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
            for j,g in enumerate(gt_list):
                iou_mat[i,j] = iou_bbox(tuple(map(int,p["bbox"])), tuple(map(int,g["bbox"])))

        pairs = greedy_assignment(iou_mat, iou_th=iou_th)

        matched_pr = set()
        matched_gt = set()

        # ID switches
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
# Confusion matrix (tracker -> GT) + Pair consistency
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
    """
    pairs_map: gt_id -> {"body": body_tid, "face": face_tid or None}
    per_frame_assign: list of {"frame":k, "pred_to_gt":{tid:gid}, "ious":{tid:iou}, "centroids":{tid:(cx,cy)}}
    Retorna dict gt_id -> stats
    """
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
    """
    Plot da distância face-body e indicador same/diff GT ao longo do tempo.
    """
    xs = []
    dist = []
    same_flag = []
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
    # predictor tracking state
    "first_frame_loaded": False,
    "seeded_any": False,
    "tracking": False,

    # manual proposals (se quiseres usar fora do GT)
    "proposal_type": "Body",    # Body / Face
    "proposals_on": True,
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,

    "next_obj_id": 1,
    "added_obj_ids": [],
    "tracker_meta": {},         # tid -> {"type":"body"/"face", "gt": optional int}

    "out_obj_ids": None,
    "out_mask_logits": None,

    "frame_idx": 0,
    "scores": ScoresLogger(),
    "selected_obj_for_plot": 1,

    # KTP
    "ktp_root": "",
    "ktp_seq": None,
    "ktp_rgb_paths": [],
    "ktp_depth_paths": [],
    "ktp_has_depth": False,

    "gt_by_frame": None,        # dict frame -> list({id,bbox})
    "gt_format": None,
    "gt_people_ids": [],        # ids disponíveis (do GT)
    "gt_to_tracker_pair": {},   # gt_id -> {"body":tid, "face":tid or None}

    # per-frame tracking evaluation logs
    "per_frame_assign": [],     # dict per frame
    "pred_boxes_by_frame_all": {},   # frame -> list({"id":tid,"bbox":...})
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
        "proposals_on": True,
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
# KTP load + seed (GT)
# =========================================================

def load_ktp_sequence(ktp_root: str, seq: str):
    if not ktp_root or not os.path.isdir(ktp_root):
        return None, gr.update(choices=[], value=None), "KTP root inválido."

    seq_path = os.path.join(ktp_root, seq)
    if not os.path.isdir(seq_path):
        return None, gr.update(choices=[], value=None), f"Sequência '{seq}' não existe."

    rgb_dir, dep_dir = _find_rgb_depth_dirs(seq_path)

    if rgb_dir is None:
        # fallback: imagens diretamente na pasta
        rgb_dir = seq_path

    rgb_paths = _gather_images_sorted(rgb_dir)
    dep_paths = _gather_images_sorted(dep_dir) if dep_dir else []

    if len(rgb_paths) == 0:
        return None, gr.update(choices=[], value=None), "Não encontrei imagens RGB."

    state["ktp_root"] = ktp_root
    state["ktp_seq"] = seq
    state["ktp_rgb_paths"] = rgb_paths
    state["ktp_depth_paths"] = dep_paths
    state["ktp_has_depth"] = (len(dep_paths) == len(rgb_paths) and len(dep_paths) > 0)

    # GT
    gt_path = _find_gt_file(seq_path)
    gt_by_frame = None
    gt_fmt = None
    gt_ids = []
    msg_gt = "GT: not found"
    if gt_path:
        gt_by_frame, gt_fmt = _parse_gt_generic(gt_path)
        if gt_by_frame is not None:
            # ids totais
            all_ids = set()
            for fr, lst in gt_by_frame.items():
                for g in lst:
                    all_ids.add(int(g["id"]))
            gt_ids = sorted(list(all_ids))
            msg_gt = f"GT: {os.path.basename(gt_path)} ({gt_fmt}), IDs={gt_ids[:8]}{'...' if len(gt_ids)>8 else ''}"
        else:
            msg_gt = f"GT file found but failed parse: {os.path.basename(gt_path)}"

    state["gt_by_frame"] = gt_by_frame
    state["gt_format"] = gt_fmt
    state["gt_people_ids"] = gt_ids

    # mostra frame 0 (com GT boxes se existirem)
    rgb0 = read_rgb(rgb_paths[0])
    if rgb0 is None:
        return None, gr.update(choices=[], value=None), "Falha a ler RGB frame 0."

    if gt_by_frame is not None:
        gt0 = gt_by_frame.get(0, gt_by_frame.get(1, []))  # alguns GT começam em 1
        rgb0 = draw_gt_boxes(rgb0, gt0)

    # checkbox IDs (strings)
    id_choices = [str(i) for i in gt_ids]
    return rgb0, gr.update(choices=id_choices, value=id_choices), f"Loaded '{seq}' frames={len(rgb_paths)} | {msg_gt}"

def seed_from_gt(selected_gt_ids, prefer_face=True, face_conf=0.30):
    """
    Cria trackers (body + face opcional) para cada GT id selecionado no frame 0.
    """
    if not state["ktp_rgb_paths"]:
        return "Carrega uma sequência primeiro."
    if state["gt_by_frame"] is None:
        return "Não há GT parseado. (Sem GT não dá para seed automático.)"

    rgb0 = read_rgb(state["ktp_rgb_paths"][0])
    if rgb0 is None:
        return "Falha a ler frame 0."

    # reset predictor para seed limpo, mas mantém dataset carregado
    reset_predictor_and_state(keep_dataset=True)

    predictor.load_first_frame(rgb0)
    state["first_frame_loaded"] = True

    # GT frame 0 (tenta 0 senão 1)
    gt0 = state["gt_by_frame"].get(0, state["gt_by_frame"].get(1, []))
    gt0_map = {int(g["id"]): g["bbox"] for g in gt0}

    # faces no frame0
    face_boxes = yolo_face_bboxes(rgb0, yolo_face_model, conf_thres=face_conf) if prefer_face else []

    gt_to_pair = {}
    added = []

    for gid in selected_gt_ids:
        gid = int(gid)
        if gid not in gt0_map:
            continue

        x1,y1,x2,y2 = gt0_map[gid]
        # bbox de corpo (GT)
        body_bbox = np.array([[x1,y1],[x2,y2]], dtype=np.float32)

        body_tid = state["next_obj_id"]
        predictor.add_new_prompt(frame_idx=0, obj_id=body_tid, bbox=body_bbox)
        state["next_obj_id"] += 1
        added.append(body_tid)
        state["tracker_meta"][body_tid] = {"type":"body","gt":gid}

        face_tid = None
        if face_boxes:
            # escolhe melhor face dentro do bbox do corpo
            best = None
            best_iou = 0.0
            bx = (int(x1),int(y1),int(x2),int(y2))
            for fx1,fy1,fx2,fy2,fconf in face_boxes:
                # face dentro do corpo
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

    return f"Seed OK: people={len(gt_to_pair)} | trackers={len(added)} (body+face)."


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
    """
    Exporta per-frame assignments para CSV (para debug/slide).
    """
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
    """
    Devolve lista de {id,bbox,centroid,type}
    """
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
def run_ktp_sequence(iou_match_body=0.5, iou_match_face=0.2, cm_iou_min=0.0, clear_iou_th=0.5):
    """
    Corre todos os frames RGB do KTP e calcula:
      - confusion matrix tracker->GT (por-frame)
      - CLEAR MOT (body-only, face-only, all)
      - pair consistency face↔body
    """
    if not state["ktp_rgb_paths"]:
        yield None, gr.update(), gr.update(), gr.update(), gr.update(), "Carrega uma sequência."
        return

    if not state["seeded_any"]:
        yield None, gr.update(), gr.update(), gr.update(), gr.update(), "Faz seed primeiro (idealmente por GT)."
        return

    state["tracking"] = True
    state["frame_idx"] = 0
    state["per_frame_assign"] = []
    state["pred_boxes_by_frame_all"] = {}
    state["pred_boxes_by_frame_body"] = {}
    state["pred_boxes_by_frame_face"] = {}

    has_gt = (state["gt_by_frame"] is not None)

    # IDs GT disponíveis (se existirem)
    gt_ids_all = state["gt_people_ids"] if has_gt else []

    # stream
    for k, rgb_path in enumerate(state["ktp_rgb_paths"]):
        rgb = read_rgb(rgb_path)
        if rgb is None:
            continue
        state["last_frame"] = rgb

        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb)
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits

            # log scores SAMURAI
            state["scores"].log_from_predictor(
                predictor=predictor, obj_ids=out_obj_ids, frame_idx=state["frame_idx"]
            )
            state["frame_idx"] += 1

            # overlay
            vis = draw_mask_overlay(rgb, out_obj_ids, out_mask_logits)

            # pred bboxes
            preds, cents = _build_pred_boxes(out_obj_ids, out_mask_logits)

            all_list = [{"id":p["id"], "bbox":p["bbox"]} for p in preds]
            body_list = [{"id":p["id"], "bbox":p["bbox"]} for p in preds if p["type"] == "body"]
            face_list = [{"id":p["id"], "bbox":p["bbox"]} for p in preds if p["type"] == "face"]

            state["pred_boxes_by_frame_all"][k] = all_list
            state["pred_boxes_by_frame_body"][k] = body_list
            state["pred_boxes_by_frame_face"][k] = face_list

            # assignment pred->GT por frame (para confusion matrix e pair consistency)
            rec = {"frame": k, "pred_to_gt": {}, "ious": {}, "centroids": cents}

            if has_gt:
                gt_list = state["gt_by_frame"].get(k, state["gt_by_frame"].get(k+1, []))  # tolera GT 1-index
                if gt_list:
                    gt_ids = [int(g["id"]) for g in gt_list]

                    # faz match para TODOS os trackers, mas com thresholds diferentes (body vs face)
                    pred_ids = [int(p["id"]) for p in all_list]
                    if len(pred_ids) > 0 and len(gt_ids) > 0:
                        iou_mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
                        for i,p in enumerate(all_list):
                            pb = tuple(map(int, p["bbox"]))
                            for j,g in enumerate(gt_list):
                                gb = tuple(map(int, g["bbox"]))
                                iou_mat[i,j] = iou_bbox(pb, gb)

                        # greedy, mas filtrando por threshold por tipo
                        # (fazemos greedy global com threshold mínimo baixo e depois filtramos tipo-a-tipo)
                        pairs = greedy_assignment(iou_mat, iou_th=min(iou_match_face, iou_match_body))

                        for i,j,v in pairs:
                            tid = pred_ids[i]
                            gid = gt_ids[j]
                            ttype = state["tracker_meta"].get(tid, {}).get("type", "unk")
                            th = iou_match_body if ttype == "body" else iou_match_face
                            if float(v) < float(th):
                                continue
                            rec["pred_to_gt"][tid] = gid
                            rec["ious"][tid] = float(v)

            state["per_frame_assign"].append(rec)

            # refresh UI
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
            yield rgb, gr.update(), gr.update(), gr.update(), gr.update(), f"Error on frame {k}: {repr(e)}"

    # ==============================
    # AFTER RUN: compute metrics
    # ==============================
    summary_lines = []

    # Confusion matrix (por-frame)
    if has_gt:
        # pred ids que apareceram em assignments
        pred_ids_seen = sorted({int(pid) for rec in state["per_frame_assign"] for pid in rec.get("pred_to_gt", {}).keys()})
        if len(pred_ids_seen) > 0 and len(gt_ids_all) > 0:
            # labels com prefixo B/F para ficar claro na matriz
            pred_labels = []
            for tid in pred_ids_seen:
                ttype = state["tracker_meta"].get(tid, {}).get("type", "unk")
                pref = "B" if ttype == "body" else ("F" if ttype == "face" else "T")
                pred_labels.append(f"{pref}{tid}")

            # mas a matriz precisa dos ids "crus" para indexar
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
                title=f"Confusion matrix (tracker → GT) | counts over frames (IoU≥{cm_iou_min:.2f})"
            )
        else:
            cm_fig = go.Figure()
    else:
        cm_fig = go.Figure()

    # CLEAR MOT
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
        summary_lines.append("<b>CLEAR MOT</b>: sem GT (não foi possível calcular MOTA/MOTP).")

    # Pair consistency
    if has_gt and state["gt_to_tracker_pair"]:
        _, stable_iou_th, _, _ = get_thresholds_from_predictor()
        pair_stats = pair_consistency_summary(
            state["per_frame_assign"],
            state["gt_to_tracker_pair"],
            iou_min=float(max(cm_iou_min, stable_iou_th*0.0))  # não forces; podes mudar aqui
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
            summary_lines.append("<br><b>FACE↔BODY consistency</b>: sem pares (não houve trackers com face+body).")
    else:
        summary_lines.append("<br><b>FACE↔BODY consistency</b>: sem GT ou sem pares face/body.")

    summary_html = "<br>".join(summary_lines)

    # cria plot default de um par (se existir)
    pair_fig = go.Figure()
    if state["gt_to_tracker_pair"]:
        # escolhe primeiro GT com face+body
        for gid, mp in state["gt_to_tracker_pair"].items():
            if mp.get("body") is not None and mp.get("face") is not None:
                pair_fig = pair_plot(state["per_frame_assign"], gid, mp["body"], mp["face"])
                break

    yield None, gr.update(), gr.update(), cm_fig, pair_fig, summary_html


# =========================================================
# Manual seeding (opcional) para o frame que está no ecrã
# (se quiseres comparar com/sem GT)
# =========================================================

def update_proposals_on_frame(rgb_frame, proposal_type, conf_body=0.25, conf_face=0.30):
    if rgb_frame is None:
        return [], rgb_frame
    if proposal_type == "Body":
        cands = yolo_person_bboxes(rgb_frame, yolo_body_model, conf_thres=conf_body)
        label = "BODY"
        color_sel = (0,255,0)
        color_oth = (0,200,255)
    else:
        cands = yolo_face_bboxes(rgb_frame, yolo_face_model, conf_thres=conf_face)
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

with gr.Blocks() as demo:
    gr.Markdown("## KTP Evaluation — SAMURAI (Body+Face prompts) + CLEAR MOT + Confusion matrix + Pair consistency")

    with gr.Accordion("KTP setup", open=True):
        ktp_root = gr.Textbox(label="KTP root (ex: /content/drive/MyDrive/...)", value="/content/drive/MyDrive/thesis_datasets/KTP")
        seq_dd   = gr.Dropdown(label="Sequence", choices=[], value=None, interactive=True)
        btn_reload = gr.Button("Reload sequences")
        btn_load   = gr.Button("Load sequence")

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
            "Usa isto **só se não tiveres GT** ou se quiseres testar seed manual. "
            "Dica: para pares face+body tens de adicionar 2 trackers e depois analisar se batem no mesmo GT."
        )

    with gr.Accordion("Metrics", open=True):
        with gr.Row():
            iou_match_body = gr.Slider(0.05, 0.9, value=0.50, step=0.05, label="IoU match threshold (BODY tracker → GT box)")
            iou_match_face = gr.Slider(0.01, 0.9, value=0.20, step=0.05, label="IoU match threshold (FACE tracker → GT box)")
        with gr.Row():
            clear_iou_th = gr.Slider(0.05, 0.9, value=0.50, step=0.05, label="CLEAR MOT IoU threshold")
            cm_iou_min   = gr.Slider(0.0, 0.9, value=0.00, step=0.05, label="Confusion matrix IoU-min (count only if IoU>=)")
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

    # Reload sequences
    def _ui_reload(root):
        seqs = list_ktp_sequences(root)
        return gr.update(choices=seqs, value=(seqs[0] if seqs else None))
    btn_reload.click(fn=_ui_reload, inputs=ktp_root, outputs=seq_dd)

    # Load sequence
    def _ui_load(root, seq):
        # mantém predictor limpo mas não apaga dataset que vai ser preenchido agora
        reset_predictor_and_state(keep_dataset=False)
        img0, ids_update, msg = load_ktp_sequence(root, seq)
        state["last_frame"] = img0 if img0 is not None else None
        # também atualiza proposals preview se quiseres
        if img0 is not None:
            cands, preview = update_proposals_on_frame(img0, state["proposal_type"], conf_body, conf_face2)
            state["cands"] = cands
            return preview, ids_update, msg, _choices_refresh()
        return img0, ids_update, msg, _choices_refresh()

    btn_load.click(fn=_ui_load, inputs=[ktp_root, seq_dd], outputs=[out, gt_ids_box, status, obj_select])

    # Seed from GT
    def _ui_seed_gt(gt_ids, do_face, fconf):
        msg = seed_from_gt([int(x) for x in gt_ids], prefer_face=bool(do_face), face_conf=float(fconf))
        return msg, _choices_refresh(), _refresh_plot(state["selected_obj_for_plot"]), _refresh_latest_scores(state["selected_obj_for_plot"])
    btn_seed_gt.click(fn=_ui_seed_gt, inputs=[gt_ids_box, face_seed, face_conf], outputs=[status, obj_select, plot_scores, score_info])

    # Manual proposals update when changing type/conf
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
        # refresh plots
        return msg, _choices_refresh(), _refresh_plot(state["selected_obj_for_plot"]), _refresh_latest_scores(state["selected_obj_for_plot"])
    btn_accept.click(fn=_ui_accept, inputs=proposal_type, outputs=[status, obj_select, plot_scores, score_info])

    # Run sequence
    btn_run.click(
        fn=run_ktp_sequence,
        inputs=[iou_match_body, iou_match_face, cm_iou_min, clear_iou_th],
        outputs=[out, plot_scores, score_info, cm_plot, pair_plot_ui, summary_html],
    )

    # Export CSV
    btn_csv.click(fn=_export_csv, inputs=obj_select, outputs=dl_csv)
    btn_assign_csv.click(fn=_export_summary_csv, inputs=None, outputs=dl_assign_csv)

    # Timer refresh (scores)
    timer = gr.Timer(0.7)
    timer.tick(fn=_refresh_plot, inputs=obj_select, outputs=plot_scores)
    timer.tick(fn=_refresh_latest_scores, inputs=obj_select, outputs=score_info)

    gr.Markdown(f"""
### Notas importantes (para a tua pergunta “body GT vs face GT”)
- **O GT do KTP é corpo (caixa/posição)**. Mesmo assim, o tracker criado a partir de **face prompt** é só um **tracker ID** como os outros.
- A avaliação que interessa é: **o tracker ID da face “fica colado” ao mesmo GT person ID do corpo?**
- Por isso mostramos:
  - **Confusion matrix**: quantos frames cada tracker (Bxxx ou Fxxx) esteve associado a cada GT ID.
  - **Pair consistency**: para cada pessoa, % de frames em que BODY e FACE foram atribuídos ao mesmo GT.
  - **CLEAR MOT**: MOTA/MOTP e ID switches (para ALL/BODY/FACE).

### Onde meter a dataset (Drive)
- Monta Drive e usa um caminho tipo:  
  `/content/drive/MyDrive/thesis_datasets/KTP`
- O “KTP root” deve ter subpastas (sequências). O script tenta detectar `rgb/` e `depth/` automaticamente.

### Ajustes que vais querer testar
- **IoU match face→GT** costuma precisar ser menor (ex: 0.15–0.30), porque a face é pequena.
- **CLEAR MOT IoU** depende do GT: começa em 0.5 e ajusta se necessário.
- Se o GT do KTP no teu download não tiver caixas 2D, diz-me qual é o formato do ficheiro de GT que tens e eu ajusto o parser.
""")

demo.launch(share=True)