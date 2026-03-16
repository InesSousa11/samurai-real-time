# Samurai Real-Time

Real-time multi-person segmentation & tracking that lets you add new targets both before and during tracking.
Built with **[YOLOv8](https://github.com/ultralytics/ultralytics)** for person proposals and **[SAMURAI](https://github.com/yangchris11/samurai.git)** for instance segmentation, and a **Gradio** UI.

> This project was adapted from **[segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)** (SAM 2 real-time) to work with SAMURAI.

---

## ▶️ Try it in Colab

[![Open In Colab](https://colab.research.googleusercontent.com/assets/colab-badge.svg)](https://colab.research.google.com/github/InesSousa11/samurai-real-time/blob/main/demo/demo_colab.ipynb
)

The Colab:
- installs dependencies & sets up the environment,
- downloads YOLO weights automatically,
- launches a Gradio app for **Webcam** or **Video** input,
- supports adding **new persons during tracking**.

---

## How to use the demo (Gradio UI)

### Webcam mode
1. Keep **YOLO proposals ON**.
2. Use **Prev/Next** to choose a green box.
3. Click **Accept** to add that person (repeat for more).
4. Click **Start Tracking**.
5. **Late-join:** you can still click **Accept** later to add a **new person during tracking**.

### Video mode
1. Upload a video → click **Start video**.
2. (Optional) On the first frame, click **Accept** for each person you want.
3. Click **Start Tracking**.
4. **Late-join:** while tracking, you can still click **Accept (add person)** to add new people at any time.
5. When it finishes, the segmented result is available to download.

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
- ✅ Add new targets before or during tracking  
- ✅ Webcam or Video input, with optional saving of segmented output  
- ✅ Gradio UI with score plots & CSV export

---

## Local installation

To be completed...

## ReID dependency (as a git submodule)

This repo uses **KaiyangZhou/deep-person-reid** (OSNet via `torchreid`) as a **git submodule** under:

- `external/reid/deep-person-reid`

### Clone (recommended)
Clone this repo **with submodules**:

```bash
git clone --recurse-submodules git@github.com:InesSousa11/samurai-real-time.git