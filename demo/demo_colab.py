import numpy as np
import torch
import cv2
import gradio as gr
import traceback
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Config ---
CKPT = "checkpoints/sam2.1_hiera_small.pt"
CFG  = "configs/samurai/sam2.1_hiera_s.yaml"
MAX_W = 480  # baixa para 384/320 se notar lag

# --- Build models ---
def make_predictor():
    return build_sam2_camera_predictor(CFG, CKPT)

predictor = make_predictor()
yolo = YOLO("yolov8n.pt")  # nano para menor overhead

# --- Estado ---
tracker_ready = False     # só fica True depois da primeira seed
obj_counter = 1
last_output_rgb = None
proc_count = 0            # contador para heartbeat

def downscale(img, max_w=MAX_W):
    H, W = img.shape[:2]
    scale = min(1.0, max_w / max(1, W))
    if scale < 1.0:
        img_s = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_s = img
    return img_s, scale

@torch.inference_mode()
def process_frame(rgb_frame):
    global predictor, tracker_ready, obj_counter, last_output_rgb, proc_count

    # Gradio por vezes envia None → mantém último frame válido
    if rgb_frame is None:
        return last_output_rgb

    try:
        # Reduz resolução para acelerar
        rgb_small, scale = downscale(rgb_frame)
        h, w = rgb_small.shape[:2]

        if not tracker_ready:
            # 1º frame: inicializar SAM2
            predictor.load_first_frame(rgb_small)

            # Seed da 1ª pessoa encontrada (apenas uma vez)
            results = yolo(rgb_small, verbose=False)[0]
            for det in results.boxes:
                if int(det.cls) == 0:  # 'person'
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    predictor.add_new_prompt(frame_idx=0, obj_id=obj_counter, bbox=bbox)
                    obj_counter += 1
                    tracker_ready = True
                    print("[init] primeira pessoa adicionada ao SAM2")
                    break

            out_small = rgb_small  # mostra algo enquanto inicia

        else:
            # Seguimento normal
            out_obj_ids, out_mask_logits = predictor.track(rgb_small)

            if len(out_obj_ids) == 0:
                # Sem máscaras nesta frame → mostra RGB simples (memória interna mantém-se)
                out_small = rgb_small
            else:
                # Desenhar overlay de máscaras
                all_mask = np.zeros((h, w, 3), dtype=np.uint8)
                all_mask[..., 1] = 255
                for i in range(len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                    hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
                    sel = out_mask[..., 0] == 255
                    all_mask[sel, 0] = hue
                    all_mask[sel, 2] = 255
                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
                out_small = cv2.addWeighted(rgb_small, 1.0, all_mask, 0.5, 0.0)

        # Upscale para o tamanho original para visualização
        out_img = (cv2.resize(out_small, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                   if scale < 1.0 else out_small)

        # Garantir formato adequado
        out_img = np.ascontiguousarray(out_img.astype(np.uint8))
        if out_img.ndim == 2:
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)

        last_output_rgb = out_img

        # Heartbeat: imprime a cada 30 frames processadas
        proc_count += 1
        if proc_count % 30 == 0:
            print(f"[heartbeat] frames processados: {proc_count}")

        return out_img

    except Exception as e:
        print("process_frame error:", repr(e))
        traceback.print_exc()
        # Em erro, manter a última imagem para não "apagar" o painel
        return last_output_rgb if last_output_rgb is not None else rgb_frame


# ---------- Gradio 5.x UI ----------
webcam = gr.Image(
    sources=["webcam"],
    streaming=True,
    label="Webcam Input",
    type="numpy",
)

demo = gr.Interface(
    fn=process_frame,
    inputs=webcam,
    outputs=gr.Image(label="Processed Output", type="numpy"),
    live=True,
    flagging_mode="never",
)

demo.launch(share=True)
