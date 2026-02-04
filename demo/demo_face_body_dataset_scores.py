# demo_colab_with_vid.py
import os
import cv2
import time
import math
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

# -------- Performance knobs --------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------- Build predictor --------
from sam2.build_sam import build_sam2_camera_predictor

REPO = "/content/samurai-real-time"
CKPT = f"{REPO}/checkpoints/sam2.1_hiera_small.pt"
CFG  = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(CFG, CKPT)

# --- runtime score logger ---
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

# ---------------- DAVIS helpers ----------------
def _find_davis_res_dir(davis_root: str):
    """
    DAVIS usually: JPEGImages/480p or JPEGImages/1080p.
    Returns chosen resolution dir name (e.g. '480p') or None.
    """
    jpeg_dir = os.path.join(davis_root, "JPEGImages")
    if not os.path.isdir(jpeg_dir):
        return None
    for cand in ["480p", "1080p", "Full-Resolution"]:
        if os.path.isdir(os.path.join(jpeg_dir, cand)):
            return cand
    subs = [d for d in os.listdir(jpeg_dir) if os.path.isdir(os.path.join(jpeg_dir, d))]
    return subs[0] if subs else None

def list_davis_sequences(davis_root: str):
    res = _find_davis_res_dir(davis_root)
    if res is None:
        return []
    seq_root = os.path.join(davis_root, "JPEGImages", res)
    seqs = [d for d in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, d))]
    seqs.sort()
    return seqs

def list_frame_paths(davis_root: str, seq: str):
    res = _find_davis_res_dir(davis_root)
    if res is None:
        return [], None
    seq_dir = os.path.join(davis_root, "JPEGImages", res, seq)
    if not os.path.isdir(seq_dir):
        return [], res
    files = [f for f in os.listdir(seq_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return [os.path.join(seq_dir, f) for f in files], res

def list_gt_paths(davis_root: str, res: str, seq: str):
    ann_dir = os.path.join(davis_root, "Annotations", res, seq)
    if not os.path.isdir(ann_dir):
        return []
    files = [f for f in os.listdir(ann_dir) if f.lower().endswith((".png",))]
    files.sort()
    return [os.path.join(ann_dir, f) for f in files]

def read_rgb(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def read_gt_mask(path: str):
    # DAVIS GT is indexed PNG
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.int32)

def unique_ids(mask: np.ndarray):
    """
    Show GT object ids present on the mask.
    We exclude 0 (background). We also exclude 255 (void/ignore) by default,
    because it tends to break confusion matrices (and is not a real object id).
    """
    if mask is None:
        return []
    u = np.unique(mask)
    ids = []
    for x in u.tolist():
        xi = int(x)
        if xi in (0, 255):
            continue
        ids.append(xi)
    return sorted(ids)

def bbox_from_mask(mask: np.ndarray, label: int):
    ys, xs = np.where(mask == int(label))
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def mask_iou(pred_bin: np.ndarray, gt_bin: np.ndarray):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter) / float(union + 1e-9)

def greedy_assignment(iou_mat):
    """
    Greedy max matching: returns list of (pred_i, gt_j, iou).
    iou_mat: [P, G]
    """
    P, G = iou_mat.shape
    pairs = []
    used_p = set()
    used_g = set()
    flat = []
    for i in range(P):
        for j in range(G):
            flat.append((iou_mat[i, j], i, j))
    flat.sort(reverse=True, key=lambda t: t[0])
    for v, i, j in flat:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, float(v)))
    return pairs

# ---------------- YOLO proposal helpers ----------------
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
    return inter / (areaA + areaB - inter + 1e-9)

# ---------------- ID color helpers ----------------
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
        if logits_i is None:
            continue
        if isinstance(logits_i, torch.Tensor):
            if logits_i.ndim == 3:
                m = (logits_i > 0).permute(1, 2, 0)
            elif logits_i.ndim == 2:
                m = (logits_i > 0).unsqueeze(-1)
            else:
                continue
            m = m.detach().cpu().numpy().astype(np.uint8) * 255
        else:
            continue

        sel = m[..., 0] == 255
        hue = _id_to_hue(ids[i])
        hsv[sel, 0] = hue
        hsv[sel, 2] = 255

    overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, 0.5, 0.0)

# ---------------- Meeting sequences (your list) ----------------
DAVIS_PERSON_SEQS = [
    "bmx-bumps","bmx-trees","breakdance","breakdance-flare","dance-jump","dance-twirl",
    "hike","hockey","horsebump-high","horsebump-low","kite-surf","kite-walk","lucia",
    "motocross-bumps","motocross-hump","motorbike","paragliding-launch","parkour",
    "rollerblade","scooter-black","scooter-gray","soapbox","stroller","swing","tennis"
]

# ---------------- App state ----------------
state = {
    "mode": "Webcam",
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
    "out_obj_ids": None,
    "out_mask_logits": None,

    "frame_idx": 0,
    "scores": ScoresLogger(),
    "selected_obj_for_plot": 1,

    "injecting": False,

    # DAVIS
    "davis_root": "/content/DAVIS",
    "davis_seq": None,
    "davis_frames": [],
    "davis_gts": [],
    "davis_res": None,
    "davis_people_ids": [],      # GT labels shown for selection
    "gt_to_tracker_ids": {},     # gt_label -> {"body": id, "face": id or None}

    # metrics accumulators (per run)
    "per_frame_assign": [],      # list: {"frame":k, "pred_to_gt":{pid:gt}, "ious":{pid:iou}}
    "per_pred_iou": {},          # pid -> [iou,...]
}

# ---------------- Thresholds (use model attrs if exist) ----------------
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def get_thresholds():
    stable_frames = int(getattr(predictor, "stable_frames_threshold", 15))
    stable_iou_th = float(getattr(predictor, "stable_ious_threshold", 0.3))
    min_obj_logit = float(getattr(predictor, "min_obj_score_logits", -1))
    obj_prob_th   = _sigmoid(min_obj_logit)
    return stable_frames, stable_iou_th, min_obj_logit, obj_prob_th

# ---------------- Seeding helpers (DAVIS) ----------------
def reset_predictor_and_state():
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)
    state.update({
        "first_frame_loaded": False,
        "seeded_any": False,
        "tracking": False,
        "selected_idx": 0,
        "cands": [],
        "last_frame": None,
        "next_obj_id": 1,
        "added_obj_ids": [],
        "out_obj_ids": None,
        "out_mask_logits": None,
        "frame_idx": 0,
        "scores": ScoresLogger(),
        "selected_obj_for_plot": 1,
        "injecting": False,
        "gt_to_tracker_ids": {},
        "per_frame_assign": [],
        "per_pred_iou": {},
    })

def load_davis_sequence(davis_root: str, seq: str):
    state["davis_root"] = davis_root
    state["davis_seq"] = seq
    frames, res = list_frame_paths(davis_root, seq)
    if not frames:
        state["davis_frames"] = []
        state["davis_gts"] = []
        state["davis_res"] = None
        state["davis_people_ids"] = []
        return None, [], "Could not find frames. Check DAVIS root/resolution folders."
    gts = list_gt_paths(davis_root, res, seq)
    state["davis_frames"] = frames
    state["davis_gts"] = gts
    state["davis_res"] = res

    rgb0 = read_rgb(frames[0])
    gt0 = read_gt_mask(gts[0]) if gts else None
    ids = unique_ids(gt0) if gt0 is not None else []
    state["davis_people_ids"] = ids

    overlay = rgb0.copy() if rgb0 is not None else None
    if overlay is not None and gt0 is not None:
        edges = cv2.Canny((gt0 > 0).astype(np.uint8) * 255, 50, 150)
        overlay[edges > 0] = (255, 255, 0)
    return overlay, ids, f"Loaded DAVIS seq '{seq}' with {len(frames)} frames, GT={'yes' if bool(gts) else 'no'}."

def seed_from_davis_gt(selected_gt_ids):
    if not state["davis_frames"]:
        return "Load a DAVIS sequence first."
    rgb0 = read_rgb(state["davis_frames"][0])
    if rgb0 is None:
        return "Failed to read first frame."
    if not state["davis_gts"]:
        return "No GT masks found for this sequence (Annotations missing)."

    gt0 = read_gt_mask(state["davis_gts"][0])
    if gt0 is None:
        return "Failed to read GT mask 0."

    reset_predictor_and_state()

    predictor.load_first_frame(rgb0)
    state["first_frame_loaded"] = True

    face_boxes = yolo_face_bboxes(rgb0, yolo_face_model, conf_thres=0.30)

    gt_to_ids = {}
    added = []
    for gt_label in selected_gt_ids:
        gt_label = int(gt_label)
        if gt_label in (0, 255):
            continue
        bb = bbox_from_mask(gt0, gt_label)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        body_bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        body_id = state["next_obj_id"]
        predictor.add_new_prompt(frame_idx=0, obj_id=body_id, bbox=body_bbox)
        state["next_obj_id"] += 1
        added.append(body_id)

        face_id = None
        if face_boxes:
            best = None
            best_iou = 0.0
            for fx1, fy1, fx2, fy2, fconf in face_boxes:
                if fx1 >= x1 and fy1 >= y1 and fx2 <= x2 and fy2 <= y2:
                    i = iou_bbox((x1, y1, x2, y2), (fx1, fy1, fx2, fy2))
                    if i > best_iou:
                        best_iou = i
                        best = (fx1, fy1, fx2, fy2)
            if best is not None:
                fx1, fy1, fx2, fy2 = best
                face_bbox = np.array([[fx1, fy1], [fx2, fy2]], dtype=np.float32)
                face_id = state["next_obj_id"]
                predictor.add_new_prompt(frame_idx=0, obj_id=face_id, bbox=face_bbox)
                state["next_obj_id"] += 1
                added.append(face_id)

        gt_to_ids[gt_label] = {"body": body_id, "face": face_id}

    state["seeded_any"] = len(added) > 0
    state["added_obj_ids"] = added
    state["scores"].register_ids(added)
    state["selected_obj_for_plot"] = added[0] if added else 1
    state["gt_to_tracker_ids"] = gt_to_ids

    return f"Seeded {len(gt_to_ids)} GT people. Created {len(added)} tracker IDs (body + optional face)."

# ---------------- Metrics ----------------
def compute_decision_delays():
    """
    Decision delay per tracker ID:
    first frame index where BOTH:
      - affinity(iou) >= stable_ious_threshold
      - object >= sigmoid(min_obj_score_logits)
    for stable_frames_threshold consecutive frames.
    """
    stable_frames, stable_iou_th, _, obj_prob_th = get_thresholds()
    delays = {}
    for oid, ss in state["scores"].per_obj.items():
        frames = list(ss.frames)
        aff = list(ss.values["affinity"])
        obj = list(ss.values["object"])
        ok_run = 0
        first_ok_frame = None
        for f, a, o in zip(frames, aff, obj):
            a_ok = isinstance(a, float) and not math.isnan(a) and a >= stable_iou_th
            o_ok = isinstance(o, float) and not math.isnan(o) and o >= obj_prob_th
            if a_ok and o_ok:
                ok_run += 1
                if first_ok_frame is None:
                    first_ok_frame = f
                if ok_run >= stable_frames:
                    delays[int(oid)] = int(first_ok_frame)
                    break
            else:
                ok_run = 0
                first_ok_frame = None
        if int(oid) not in delays:
            delays[int(oid)] = None
    return delays

def majority_gt_assignment():
    """
    For each predicted tracker ID, choose GT label that it matched most often.
    Returns pid -> gt_label or None
    """
    votes = {}
    for rec in state["per_frame_assign"]:
        p2g = rec.get("pred_to_gt", {})
        for pid, gt in p2g.items():
            pid = int(pid)
            gt = int(gt)
            votes.setdefault(pid, {})
            votes[pid][gt] = votes[pid].get(gt, 0) + 1
    maj = {}
    for pid, hist in votes.items():
        maj[pid] = max(hist.items(), key=lambda kv: kv[1])[0] if hist else None
    return maj

# ---- FIXED CONFUSION MATRIX: count per-frame assignments (optionally IoU-gated) ----
def make_confusion_matrix_from_frames(per_frame_assign, pred_ids, gt_labels, iou_min=0.0):
    """
    Confusion matrix rows=pred IDs, cols=GT labels.
    Counts how many FRAMES each pred_id was matched to each GT label.
    Optionally only count matches with IoU >= iou_min.
    """
    pred_ids = sorted([int(x) for x in pred_ids])
    gt_labels = sorted([int(x) for x in gt_labels])

    col_index = {g: j for j, g in enumerate(gt_labels)}
    row_index = {p: i for i, p in enumerate(pred_ids)}

    mat = np.zeros((len(pred_ids), len(gt_labels)), dtype=np.int32)

    for rec in per_frame_assign:
        m = rec.get("pred_to_gt", {})
        ious = rec.get("ious", {})
        for pid, glab in m.items():
            pid = int(pid)
            glab = int(glab)
            if pid not in row_index:
                continue
            if glab not in col_index:
                continue
            if float(ious.get(pid, 0.0)) < float(iou_min):
                continue
            mat[row_index[pid], col_index[glab]] += 1

    return pred_ids, gt_labels, mat

def confusion_fig(pred_ids, gt_labels, mat):
    # Force a non-pink, readable colorscale + show numbers
    vmax = int(mat.max()) if mat.size else 1
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=[str(g) for g in gt_labels],
            y=[str(p) for p in pred_ids],
            colorscale="Blues",
            zmin=0,
            zmax=max(1, vmax),
            colorbar=dict(title="frames"),
            text=mat,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Confusion matrix (tracker ID → GT person label) [counts over frames]",
        xaxis_title="GT label (person)",
        yaxis_title="Tracker ID",
        height=450,
    )
    return fig

# ---------------- Run DAVIS sequence ----------------
@torch.inference_mode()
def run_davis_sequence():
    if not state["davis_frames"]:
        yield None, None, None, None, None
        return
    if not state["seeded_any"]:
        yield None, None, None, None, "Seed first (GT or manual)."
        return

    state["tracking"] = True
    state["frame_idx"] = 0
    state["per_frame_assign"] = []
    state["per_pred_iou"] = {}

    # These are the GT person labels you selected (from the first-frame GT ids you seeded)
    gt_labels = sorted([int(k) for k in state["gt_to_tracker_ids"].keys() if int(k) not in (0, 255)])

    for k, fpath in enumerate(state["davis_frames"]):
        rgb = read_rgb(fpath)
        if rgb is None:
            continue

        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb)
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits

            # log scores
            state["scores"].log_from_predictor(predictor=predictor, obj_ids=out_obj_ids, frame_idx=state["frame_idx"])
            state["frame_idx"] += 1

            # visualization
            vis = draw_mask_overlay(rgb, out_obj_ids, out_mask_logits)

            # metrics vs GT
            rec = {"frame": k, "pred_to_gt": {}, "ious": {}}
            if state["davis_gts"] and k < len(state["davis_gts"]):
                gt = read_gt_mask(state["davis_gts"][k])
                if gt is not None and torch.is_tensor(out_mask_logits):
                    pred_ids = _to_id_list(out_obj_ids)
                    P = min(len(pred_ids), int(out_mask_logits.shape[0]))
                    G = len(gt_labels)
                    if P > 0 and G > 0:
                        iou_mat = np.zeros((P, G), dtype=np.float32)
                        for i in range(P):
                            pm = out_mask_logits[i]
                            if pm.ndim == 3:
                                pm = pm[0]
                            pm_bin = (pm.detach().cpu().numpy() > 0)
                            for j, glab in enumerate(gt_labels):
                                gt_bin = (gt == int(glab))
                                iou_mat[i, j] = mask_iou(pm_bin, gt_bin)

                        pairs = greedy_assignment(iou_mat)
                        for i, j, iou_v in pairs:
                            pid = int(pred_ids[i])
                            glab = int(gt_labels[j])
                            rec["pred_to_gt"][pid] = glab
                            rec["ious"][pid] = float(iou_v)
                            state["per_pred_iou"].setdefault(pid, []).append(float(iou_v))

            state["per_frame_assign"].append(rec)

            if k % 5 == 0:
                plot = state["scores"].make_plot(state["selected_obj_for_plot"])
                info = state["scores"].latest_row(state["selected_obj_for_plot"])
                info_html = "<table>" + "".join(
                    f"<tr><td><b>{kk}</b></td><td>{vv:.4f}</td></tr>" for kk, vv in info.items()
                ) + "</table>" if info else "—"
            else:
                plot = gr.update()
                info_html = gr.update()

            yield vis, plot, info_html, gr.update(), f"Running frame {k+1}/{len(state['davis_frames'])}..."

        except Exception as e:
            print("[error] predictor.track failed:", repr(e))
            print(traceback.format_exc())
            yield rgb, gr.update(), gr.update(), gr.update(), f"Error on frame {k}: {repr(e)}"

    # ---- sequence finished -> compute summary figs ----
    pred_major = majority_gt_assignment()

    # Collect all predicted tracker ids that ever got an assignment
    pred_ids_all = sorted({int(pid) for rec in state["per_frame_assign"] for pid in rec.get("pred_to_gt", {}).keys()})

    # Count per-frame assignments; gate by stable IoU threshold (you can set to 0.0 if you want)
    _, stable_iou_th, _, _ = get_thresholds()
    IOU_MIN_FOR_CM = float(stable_iou_th)

    pred_ids, gt_cols, cm = make_confusion_matrix_from_frames(
        state["per_frame_assign"],
        pred_ids=pred_ids_all,
        gt_labels=gt_labels,
        iou_min=IOU_MIN_FOR_CM,
    )
    cm_fig = confusion_fig(pred_ids, gt_cols, cm)

    delays = compute_decision_delays()
    mean_ious = {pid: (sum(v)/len(v) if v else 0.0) for pid, v in state["per_pred_iou"].items()}

    stable_frames, stable_iou_th, min_obj_logit, obj_prob_th = get_thresholds()
    lines = []
    lines.append(
        f"<b>Thresholds used</b>: stable_frames={stable_frames}, stable_iou_th={stable_iou_th:.2f}, "
        f"min_obj_logit={min_obj_logit:.2f} (obj_prob_th≈{obj_prob_th:.2f}), "
        f"confusion_iou_min={IOU_MIN_FOR_CM:.2f}"
    )
    lines.append("<br><b>Per-tracker summary</b>:")
    lines.append("<table><tr><th>ID</th><th>maj GT</th><th>mean IoU</th><th>decision frame</th></tr>")
    for pid in sorted(pred_major.keys()):
        lines.append(
            f"<tr><td>{pid}</td><td>{pred_major.get(pid)}</td>"
            f"<td>{mean_ious.get(pid, 0.0):.3f}</td><td>{delays.get(pid)}</td></tr>"
        )
    lines.append("</table>")
    summary_html = "\n".join(lines)

    yield None, gr.update(), gr.update(), cm_fig, summary_html

# ---------------- Batch evaluate (best/worst) ----------------
@torch.inference_mode()
def batch_eval(davis_root: str, seq_list):
    """
    Runs multiple sequences headlessly and returns a table + best/worst.
    NOTE: this auto-seeds ALL GT ids present on frame 0 (except 0/255).
    """
    seqs = [s for s in seq_list if s]
    rows = []

    for seq in seqs:
        overlay, ids, msg = load_davis_sequence(davis_root, seq)
        if not ids:
            rows.append({"seq": seq, "status": "no GT ids", "meanIoU": None})
            continue

        seed_from_davis_gt(ids)

        state["tracking"] = True
        state["frame_idx"] = 0
        state["per_frame_assign"] = []
        state["per_pred_iou"] = {}
        gt_labels = sorted([int(k) for k in state["gt_to_tracker_ids"].keys() if int(k) not in (0, 255)])

        for k, fpath in enumerate(state["davis_frames"]):
            rgb = read_rgb(fpath)
            if rgb is None:
                continue
            out_obj_ids, out_mask_logits = predictor.track(rgb)
            state["scores"].log_from_predictor(predictor=predictor, obj_ids=out_obj_ids, frame_idx=state["frame_idx"])
            state["frame_idx"] += 1

            if state["davis_gts"] and k < len(state["davis_gts"]):
                gt = read_gt_mask(state["davis_gts"][k])
                if gt is not None and torch.is_tensor(out_mask_logits):
                    pred_ids = _to_id_list(out_obj_ids)
                    P = min(len(pred_ids), int(out_mask_logits.shape[0]))
                    G = len(gt_labels)
                    if P > 0 and G > 0:
                        iou_mat = np.zeros((P, G), dtype=np.float32)
                        for i in range(P):
                            pm = out_mask_logits[i]
                            if pm.ndim == 3:
                                pm = pm[0]
                            pm_bin = (pm.detach().cpu().numpy() > 0)
                            for j, glab in enumerate(gt_labels):
                                gt_bin = (gt == int(glab))
                                iou_mat[i, j] = mask_iou(pm_bin, gt_bin)
                        pairs = greedy_assignment(iou_mat)
                        for i, j, iou_v in pairs:
                            pid = int(pred_ids[i])
                            state["per_pred_iou"].setdefault(pid, []).append(float(iou_v))

        mean_ious = [sum(v) / len(v) for v in state["per_pred_iou"].values() if v]
        meanIoU = float(np.mean(mean_ious)) if mean_ious else 0.0
        rows.append({"seq": seq, "status": "ok", "meanIoU": meanIoU})

    ok_rows = [r for r in rows if r["status"] == "ok" and r["meanIoU"] is not None]
    ok_rows.sort(key=lambda r: r["meanIoU"])
    worst = ok_rows[0] if ok_rows else None
    best  = ok_rows[-1] if ok_rows else None

    header = "<tr><th>seq</th><th>status</th><th>meanIoU</th></tr>"
    body_lines = []
    for r in rows:
        if r["meanIoU"] is None:
            miou_str = ""
        else:
            miou_str = f"{float(r['meanIoU']):.3f}"
        body_lines.append(f"<tr><td>{r['seq']}</td><td>{r['status']}</td><td>{miou_str}</td></tr>")
    summary = f"<table>{header}{''.join(body_lines)}</table>"
    if best and worst:
        summary += f"<br><b>Best</b>: {best['seq']} (meanIoU={best['meanIoU']:.3f})"
        summary += f"<br><b>Worst</b>: {worst['seq']} (meanIoU={worst['meanIoU']:.3f})"
    return summary

# ---------------- UI glue ----------------
def _choices_refresh():
    if state["added_obj_ids"]:
        ch = [int(x) for x in state["added_obj_ids"]]
        default = ch[0]
    else:
        ch, default = [1], 1
    return gr.update(choices=ch, value=default)

def _refresh_plot(obj_id:int):
    state["selected_obj_for_plot"] = int(obj_id)
    return state["scores"].make_plot(int(obj_id))

def _refresh_latest_scores(obj_id:int):
    state["selected_obj_for_plot"] = int(obj_id)
    row = state["scores"].latest_row(int(obj_id))
    if not row:
        return "—"
    cells = "".join(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>" for k, v in row.items())
    return f"<table>{cells}</table>"

def on_reset_ui():
    reset_predictor_and_state()
    return "Reset done.", gr.update(choices=[1], value=1), go.Figure(), "—", go.Figure(), "—"

# ---------------- Gradio ----------------
with gr.Blocks() as demo:
    gr.Markdown("## SAMURAI real-time — **DAVIS frames** + meeting metrics (scores + confusion matrix)")

    mode = gr.Radio(["DAVIS Frames"], value="DAVIS Frames", label="Source")

    with gr.Accordion("DAVIS setup", open=True):
        davis_root = gr.Textbox(label="DAVIS root", value="/content/DAVIS")
        davis_seq = gr.Dropdown(label="Sequence", choices=[], value=None, interactive=True)
        btn_reload = gr.Button("Reload sequences")
        btn_load = gr.Button("Load sequence")
        gt_ids_box = gr.CheckboxGroup(label="GT object IDs (select the ones that are PEOPLE)", choices=[], value=[])
        btn_seed = gr.Button("Seed from selected GT IDs (Body + optional Face)")
        btn_run = gr.Button("Run sequence (compute metrics)")

        batch_list = gr.CheckboxGroup(
            label="Batch evaluate these sequences",
            choices=DAVIS_PERSON_SEQS,
            value=DAVIS_PERSON_SEQS
        )
        btn_batch = gr.Button("Batch evaluate (best/worst by mean IoU)")

    out = gr.Image(label="Output", type="numpy")
    status = gr.Markdown("Status: —")

    with gr.Accordion("Scores & Diagnostics", open=True):
        with gr.Row():
            obj_select = gr.Dropdown(label="Object to plot", choices=[1], value=1, interactive=True)
            score_info = gr.HTML(label="Latest scores", value="—")

        plot = gr.Plot(label="Scores over time (selected object)")
        cm_plot = gr.Plot(label="Confusion matrix (tracker ID → GT person)")
        summary_html = gr.HTML(label="Summary", value="—")

    # Reload sequences
    def _reload(root):
        seqs = list_davis_sequences(root)
        return gr.update(choices=seqs, value=(seqs[0] if seqs else None))

    btn_reload.click(fn=_reload, inputs=davis_root, outputs=davis_seq)

    # Load a sequence
    def _load(root, seq):
        overlay, ids, msg = load_davis_sequence(root, seq)
        # ids are ints; checkbox wants strings
        ids_str = [str(i) for i in ids]
        return overlay, gr.update(choices=ids_str, value=ids_str), msg

    btn_load.click(fn=_load, inputs=[davis_root, davis_seq], outputs=[out, gt_ids_box, status])

    # Seed from GT
    def _seed(gt_ids):
        ids = [int(x) for x in gt_ids] if gt_ids else []
        msg = seed_from_davis_gt(ids)
        return (
            msg,
            _choices_refresh(),
            state["scores"].make_plot(state["selected_obj_for_plot"]),
            _refresh_latest_scores(state["selected_obj_for_plot"])
        )

    btn_seed.click(fn=_seed, inputs=gt_ids_box, outputs=[status, obj_select, plot, score_info])

    # Run DAVIS (stream)
    btn_run.click(
        fn=run_davis_sequence,
        inputs=None,
        outputs=[out, plot, score_info, cm_plot, summary_html],
    )

    # Batch eval
    btn_batch.click(fn=batch_eval, inputs=[davis_root, batch_list], outputs=summary_html)

    # Plot refresh timer
    timer = gr.Timer(0.7)
    timer.tick(fn=_refresh_plot, inputs=obj_select, outputs=plot)
    timer.tick(fn=_refresh_latest_scores, inputs=obj_select, outputs=score_info)

    # Reset
    btn_reset = gr.Button("Reset (predictor + state)")
    btn_reset.click(fn=on_reset_ui, inputs=None, outputs=[status, obj_select, plot, score_info, cm_plot, summary_html])

    gr.Markdown(f"""
**How to use (DAVIS):**
1) Set **DAVIS root** (folder that contains `JPEGImages/` and `Annotations/`).  
2) Click **Reload sequences** → pick a sequence → **Load sequence**.  
3) In **GT object IDs**, select the IDs that correspond to people.  
4) Click **Seed from selected GT IDs** (creates body + optional face trackers).  
5) Click **Run sequence** to generate:
   - overlay stream
   - score plots (affinity/object/motion/combined)
   - **fixed confusion matrix** counting per-frame assignments (IoU-gated)
   - decision delays + mean IoU summary

**Confusion matrix notes:**
- This now counts how many frames each tracker ID matched each GT person label.
- It ignores GT labels 0 and 255 by default (background/void).
- It uses the **stable IoU threshold** as the minimum IoU to count a match.

**Face seeding:** depends on `{YOLO_FACE_CKPT}` being present. If missing, only body IDs are created.
""")

demo.launch(share=True)