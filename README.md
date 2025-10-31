# Samurai Real-Time

Real-time **multi-person** segmentation & tracking that lets you add new targets **before _and_ during tracking** (“late-join”).  
Uses **[YOLOv8](https://github.com/ultralytics/ultralytics)** for person proposals and **[SAM2 (SAMURAI)](https://github.com/facebookresearch/segment-anything-2)** for instance segmentation. Gradio UI included.

---

## ▶️ Try it in Colab

[![Open In Colab](https://colab.research.googleusercontent.com/assets/colab-badge.svg)](https://colab.research.google.com/github/InesSousa11/samurai-real-time/blob/main/demo/demo_colab.ipynb
)

The Colab:
- installs dependencies & sets up the environment,
- downloads YOLO weights automatically,
- launches a Gradio app for **Webcam** or **Video** input,
- supports adding **new persons during tracking** (late-join).

---

## How to use the demo (Gradio UI)

### Webcam mode
1. Keep **YOLO proposals ON**.
2. Use **Prev/Next** to choose a green box.
3. Click **Accept** to add that person (repeat for more).
4. Click **Start Tracking**.
5. You can still click **Accept** later to add a **new person during tracking**.

### Video mode
1. Upload a video → click **Start video**.
2. On the first frame, click **Accept** for each person you want.
3. Click **Start Tracking**.
4. When it finishes, the segmented result is available to download.

### Controls
- **Prev / Next** — cycle YOLO boxes  
- **Accept** — add selected box as a new tracked person  
- **Toggle YOLO** — show/hide proposals  
- **Start Tracking** — begin tracking the added people  
- **Reset** — clear state and start over

> ℹ️ If only one person is tracked, *SAMURAI mode* is **ON**; with ≥2 people it auto-disables.

---

## Features

- ✅ YOLOv8 person proposals  
- ✅ SAM2 (SAMURAI) segmentation & tracking  
- ✅ Add new targets **before or during** tracking (late-join)  
- ✅ Webcam or Video input, with optional saving of segmented output  
- ✅ Gradio UI with score plots & CSV export

---

## Local installation

To be completed...