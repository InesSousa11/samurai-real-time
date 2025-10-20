# rt_scores.py
import math
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional

try:
    import plotly.graph_objects as go
except Exception:
    go = None  # Gradio's gr.Plot requires plotly; you already have it on Colab

SCORE_KEYS = ("affinity", "object", "motion", "iou", "combined")

class ScoreSeries:
    """
    Per-object rolling timeseries of scores and frames.
    Keeps a bounded history (maxlen) to avoid memory bloat in long streams.
    """
    def __init__(self, maxlen: int = 2000):
        self.frames = deque(maxlen=maxlen)
        self.values: Dict[str, deque] = {k: deque(maxlen=maxlen) for k in SCORE_KEYS}

    def log(self, frame_idx: int, scores: Dict[str, Optional[float]]):
        """Append one sample for this frame. Missing keys → NaN."""
        self.frames.append(frame_idx)
        for k in SCORE_KEYS:
            v = scores.get(k) if scores else None
            self.values[k].append(float(v) if (v is not None and not isinstance(v, bool)) else float("nan"))

    # Diagnostics helpers (not used by plotting now, but handy)
    def deltas(self, key: str) -> List[float]:
        vals = list(self.values[key])
        return [float("nan")] + [
            vals[i] - vals[i - 1]
            if (not math.isnan(vals[i]) and not math.isnan(vals[i - 1]))
            else float("nan")
            for i in range(1, len(vals))
        ]

    def drastic_change_indices(self, key: str, thr: float = 0.3) -> List[int]:
        ds = self.deltas(key)
        return [i for i, d in enumerate(ds) if not math.isnan(d) and abs(d) >= thr]


class ScoresLogger:
    def __init__(self, maxlen: int = 2000, change_thr: float = 0.3):
        self.per_obj: Dict[int, ScoreSeries] = defaultdict(lambda: ScoreSeries(maxlen=maxlen))
        self.change_thr = change_thr
        self.known_ids: set[int] = set()  # keep all ids we've seen so x-axes stay aligned

    # Optional – call this when you Accept a new object id
    def register_ids(self, ids):
        for oid in ids:
            oid = int(oid)
            self.known_ids.add(oid)
            _ = self.per_obj[oid]  # ensure series exists

    # ---- predictor integration ----
    def _extract_scores_from_predictor(self, predictor) -> Dict[str, Any]:
        """
        Probe several common places/names used across SAM-2/SAMURAI forks.
        Returns a flat dict with possible entries:
          affinity, object, motion, iou, combined
        Missing ones come back as None.
        """
        def _may_get(obj, names):
            for n in names:
                try:
                    if obj is not None and hasattr(obj, n):
                        v = getattr(obj, n)
                        # unwrap 0-dim tensors if any
                        try:
                            import torch
                            if isinstance(v, torch.Tensor):
                                v = v.detach().float().item()
                        except Exception:
                            pass
                        return v
                except Exception:
                    pass
            return None

        # search across predictor, predictor.model, predictor.module, nested .model
        cands = [predictor]
        for base in (getattr(predictor, "model", None), getattr(predictor, "module", None)):
            if base is not None:
                cands.append(base)
            if base is not None and hasattr(base, "model"):
                cands.append(getattr(base, "model"))

        vals = {k: None for k in SCORE_KEYS}

        name_map = {
            "affinity": ("last_affinity_score", "affinity_score", "s_mask", "mask_affinity", "last_s_mask"),
            "object":   ("last_object_score", "object_score", "s_obj", "obj_score", "last_s_obj"),
            "motion":   ("last_motion_score", "kf_score", "kalman_score", "last_kf_score"),
            "iou":      ("last_iou_score", "iou", "iou_score", "mask_iou", "iou_prediction", "iou_predictions"),
            "combined": ("last_combined_score", "final_score", "selection_score"),
        }

        for k, names in name_map.items():
            for c in cands:
                v = _may_get(c, names)
                if v is not None:
                    vals[k] = v
                    break

        # Some builds stash everything in a dict like .last_scores or .debug_scores
        for c in cands:
            d = _may_get(c, ("last_scores", "debug_scores", "scores"))
            if isinstance(d, dict):
                for k in SCORE_KEYS:
                    if vals[k] is None and k in d:
                        try:
                            v = d[k]
                            import torch
                            if isinstance(v, torch.Tensor):
                                v = v.detach().float().item()
                            vals[k] = v
                        except Exception:
                            pass

        return vals

    def _normalize_ids(self, obj_ids) -> List[int]:
        ids: List[int] = []
        try:
            import torch
            if isinstance(obj_ids, torch.Tensor):
                ids = list(map(int, obj_ids.detach().cpu().reshape(-1).tolist()))
        except Exception:
            pass
        if isinstance(obj_ids, (list, tuple)):
            ids = [int(x) for x in obj_ids]
        elif not ids:
            try:
                ids = [int(obj_ids)]
            except Exception:
                ids = []
        return ids

    def log_from_predictor(self, predictor, obj_ids, frame_idx: int):
        """
        Log ONE SAMPLE FOR EVERY KNOWN OBJECT on this frame.
        Prefer per-object dict in predictor.per_obj_last_scores.
        Fallback to predictor.last_scores (global), else attribute probing.
        Objects missing scores on this frame get NaNs, keeping x-axes aligned.
        """
        # normalize ids present this frame and update known set
        ids_this_frame = self._normalize_ids(obj_ids)
        for oid in ids_this_frame:
            self.register_ids([oid])

        # prefer per-object map
        per_obj = getattr(predictor, "per_obj_last_scores", None)
        global_scores = getattr(predictor, "last_scores", None)

        # Build a full map for ALL known ids this frame
        frame_scores: Dict[int, Dict[str, Optional[float]]] = {}

        if isinstance(per_obj, dict) and per_obj:
            for oid in self.known_ids:
                s = per_obj.get(int(oid))
                frame_scores[int(oid)] = s if isinstance(s, dict) else {}
        else:
            # fallback: single dict → assign to ids present this frame; others empty
            base = global_scores if isinstance(global_scores, dict) else {}
            for oid in self.known_ids:
                frame_scores[int(oid)] = base if (int(oid) in ids_this_frame and base) else {}

        # If still empty (e.g., very first frames), probe attributes once
        if not frame_scores and self.known_ids:
            probed = self._extract_scores_from_predictor(predictor)
            for oid in self.known_ids:
                frame_scores[int(oid)] = probed

        # Finally, append this frame for every known object
        for oid, s in frame_scores.items():
            self.per_obj[int(oid)].log(frame_idx, s)

        # Return something sensible for callers (not used by UI logic)
        if ids_this_frame:
            return frame_scores.get(ids_this_frame[0], {})
        return global_scores or {}

    # ---- Queries & export ----
    def frames_where(self, obj_id: int, key: str, mode: str, t1: float, t2: Optional[float] = None) -> List[int]:
        """
        Return frames where score[key] satisfies:
          - mode='<'  : v <  t1
          - mode='>'  : v >  t1
          - mode='<=  : v <= t1
          - mode='>=' : v >= t1
          - mode='between': t1 <= v <= t2
          - mode='nan':    is NaN
          - mode='notnan': is finite
        """
        if obj_id not in self.per_obj:
            return []
        ss = self.per_obj[obj_id]
        frames = list(ss.frames)
        values = list(ss.values[key])

        out = []
        for f, v in zip(frames, values):
            isnum = isinstance(v, float) and not math.isnan(v)
            if mode == "<" and isnum and v < t1: out.append(f)
            elif mode == ">" and isnum and v > t1: out.append(f)
            elif mode == "<=" and isnum and v <= t1: out.append(f)
            elif mode == ">=" and isnum and v >= t1: out.append(f)
            elif mode == "between" and isnum and t2 is not None and t1 <= v <= t2: out.append(f)
            elif mode == "nan" and (not isnum): out.append(f)
            elif mode == "notnan" and isnum: out.append(f)
        return out

    def export_csv(self, obj_id: int, path: str) -> str:
        """Write frame, affinity, object, motion, iou, combined to CSV for this object."""
        import csv
        if obj_id not in self.per_obj:
            with open(path, "w") as _:
                pass
            return path
        ss = self.per_obj[obj_id]
        frames = list(ss.frames)
        with open(path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["frame", *SCORE_KEYS])
            writer.writeheader()
            for i, f in enumerate(frames):
                row = {"frame": f}
                for k in SCORE_KEYS:
                    v = ss.values[k][i]
                    row[k] = None if (not isinstance(v, float) or math.isnan(v)) else v
                writer.writerow(row)
        return path

    # ---- plotting ----
    def make_plot(self, obj_id: int, keys: List[str] = list(SCORE_KEYS)):
        if go is None:
            return None
        if obj_id not in self.per_obj:
            fig = go.Figure()
            fig.update_layout(title=f"No scores yet for object #{obj_id}")
            return fig

        ss = self.per_obj[obj_id]
        x = list(ss.frames)
        fig = go.Figure()
        for k in keys:
            y = list(ss.values[k])
            if any(isinstance(v, float) and not math.isnan(v) for v in y):
                # lines only — no spike markers
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=k))

        fig.update_layout(
            title=f"Scores over time (object #{obj_id})",
            xaxis_title="frame",
            yaxis_title="score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=40, r=10, b=40, t=50),
        )
        return fig

    def latest_row(self, obj_id: int) -> Dict[str, float]:
        out = {}
        if obj_id not in self.per_obj:
            return out
        ss = self.per_obj[obj_id]
        for k in SCORE_KEYS:
            if len(ss.values[k]):
                out[k] = ss.values[k][-1]
        return out