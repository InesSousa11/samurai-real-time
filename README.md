# ReID-SAMURAI

**Identity-aware multi-person tracking for social robotics using SAMURAI and TransReID.**

ReID-SAMURAI extends the mask-based SAMURAI tracker with appearance-based person re-identification, online identity galleries, identity-aware memory selection, and ReID-guided reacquisition. The system is designed for online tracking from a moving robot camera, where identities must remain consistent through occlusions, disappearance, re-entry, close interactions, and similar appearance.

<p align="center">
  <a href="https://inessousa11.github.io/person-reid-website/"><strong>Project page and demo videos</strong></a>
</p>

This repository contains the implementation developed for the MSc thesis:

> **Robust Multi-Person Re-Identification and Tracking for Social Robotics**  
> Inês Gomes Crispim de Sousa  
> Instituto Superior Técnico, Universidade de Lisboa, 2026

---

## Overview

SAMURAI provides accurate mask propagation and motion-aware temporal memory, but it does not explicitly verify that a recovered mask still belongs to the same person after tracking continuity is broken.

ReID-SAMURAI adds a person ReID branch based on TransReID. Each prompted identity maintains an online gallery of reliable appearance embeddings. Appearance similarity is then used to:

- verify that a tracked mask remains consistent with its assigned identity;
- prevent identity-inconsistent frames from entering the memory bank;
- update each identity gallery with reliable and sufficiently diverse appearances;
- validate candidate masks when a lost identity is reacquired;
- support online multi-person tracking and the addition of new people during operation.

The final system combines the complementary strengths of:

- **SAMURAI / SAM 2:** temporal mask propagation and pixel-level localization;
- **TransReID:** appearance-based identity verification;
- **Kalman motion modelling:** motion-aware candidate selection;
- **online identity galleries:** longer-term appearance memory.

---

## Demo

The project page contains the main real-robot teaser, architecture visualizations, benchmark comparisons, and additional qualitative videos:

### [Open the ReID-SAMURAI project page](https://inessousa11.github.io/person-reid-website/)

The main teaser shows the initialized target being followed by a mobile robot across different indoor areas, viewpoints, lighting conditions, partial occlusions, and interactions with other people.

---

## Architecture

### Normal identity-aware tracking

During normal tracking, SAMURAI propagates the target mask. The corresponding person crop is encoded by TransReID and compared with the identity gallery. Tracking confidence and appearance consistency determine whether the current observation may update the gallery and memory bank.

![Normal identity-aware tracking architecture](assets/readme/architecture_normal.png)

### ReID-guided reacquisition

When the target becomes uncertain or disappears, predicted masks are treated as candidates rather than confirmed outputs. A candidate is accepted only when the combined tracking, motion, mask-quality, and ReID evidence exceeds the reacquisition threshold.

![ReID-guided reacquisition architecture](assets/readme/architecture_reacquisition.png)

---

## Main features

- online RGB-frame processing;
- multi-person mask tracking with persistent identity labels;
- initialization from YOLO person proposals;
- addition of new identities after tracking has started;
- TransReID-based person embeddings;
- one online appearance gallery per tracked identity;
- protected prompt anchors in each gallery;
- identity-aware memory-bank filtering;
- ReID-guided reacquisition after loss or occlusion;
- interchangeable ReID backends;
- video and webcam demos;
- detailed debugging of masks, memory frames, galleries, candidate selection, and internal scores.

---

## Quantitative results

The main comparison was performed on the Kinect Tracking Precision (KTP) dataset using the same initialization and evaluation procedure for all systems.

| System | Masks | HOTA ↑ | MOTA ↑ | IDF1 ↑ | IDR ↑ | IDSW ↓ |
|---|:---:|---:|---:|---:|---:|---:|
| SAMURAI subsystem | ✓ | 29.33 | 24.80 | 45.13 | 32.71 | 26 |
| TransReID subsystem | ✗ | 38.26 | 41.43 | 61.08 | 46.02 | 4 |
| **ReID-SAMURAI** | **✓** | **50.48** | **57.30** | **77.44** | **73.37** | **7** |

ReID-SAMURAI achieved the strongest overall balance between tracking coverage and identity preservation. Compared with SAMURAI alone, it substantially improved HOTA and IDF1 while reducing identity switches. Compared with appearance matching alone, it recovered a much larger fraction of each target trajectory through temporal mask propagation.

The system was also evaluated qualitatively on a TIAGo robot. Tests included:

- an approximately 20-minute long-duration person-following run;
- similar-looking people wearing the same team shirt;
- temporary disappearance and re-entry;
- severe lighting variation;
- large viewpoint and pose changes;
- abrupt clothing changes.

---

## Repository structure

```text
samurai-real-time/
├── assets/                         # README and project media
├── checkpoints/                    # SAM 2.1 and ReID checkpoints
├── demo/                           # Video, webcam, evaluation, and debug scripts
├── external/
│   └── reid/
│       ├── TransReID/              # Original TransReID Git submodule
│       └── deep-person-reid/       # OSNet Git submodule
├── experiments/                    # Experimental scripts and outputs
├── outputs/                        # Generated tracking/evaluation outputs
├── sam2/
│   ├── configs/                    # SAMURAI / SAM 2 configurations
│   ├── reid_backends/              # Interchangeable ReID backends
│   └── ...                         # Modified online tracking implementation
├── requirements.txt
├── setup.py
└── README.md
```

The main interactive entry points are:

```text
demo/video_deep_debug_reid.py
demo/webcam_deep_debug_reid.py
```

---

## Requirements

The verified development environment was:

- Windows 11;
- Python 3.11.9;
- PyTorch 2.10.0+cu128;
- torchvision 0.25.0+cu128;
- NVIDIA GeForce RTX 5060 Laptop GPU.

An NVIDIA GPU is strongly recommended.

The verified video and webcam workflows run without the optional compiled `sam2._C` CUDA extension. For this reason, the setup below installs the Python dependencies directly and does not require `pip install -e .`.

---

## Installation

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/InesSousa11/samurai-real-time.git
cd samurai-real-time
```

If the repository was already cloned without its submodules:

```bash
git submodule update --init --recursive
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch and torchvision

Install a CUDA-enabled PyTorch build compatible with your operating system and GPU by following the official PyTorch installation selector:

[PyTorch installation instructions](https://pytorch.org/get-started/locally/)

PyTorch and torchvision are intentionally not pinned in `requirements.txt`, because the appropriate build depends on the local CUDA and hardware configuration.

Verify the installation:

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Checkpoints

The demo scripts expect the following files.

### SAM 2.1 checkpoint

```text
checkpoints/sam2.1_hiera_small.pt
```

Download the matching SAM 2.1 Hiera Small checkpoint from the official SAM 2 release and place it at the path above.

The corresponding configuration used by the demos is:

```text
sam2/configs/samurai/sam2.1_hiera_s.yaml
```

### TransReID checkpoint

```text
checkpoints/reid/transreid/vit_transreid_msmt.pth
```

Download the **TransReID\*(ViT) model trained on MSMT17** from the trained-model table in the official TransReID repository:

[Official TransReID repository](https://github.com/damo-cv/TransReID)

Create the destination folder if necessary.

#### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force checkpoints\reid\transreid
```

#### Linux or macOS

```bash
mkdir -p checkpoints/reid/transreid
```

Then place the downloaded file at:

```text
checkpoints/reid/transreid/vit_transreid_msmt.pth
```

### YOLO weights

The interactive demos use:

```text
yolov8s.pt
```

Ultralytics can download this file automatically when it is first requested. It may also be placed manually in the repository root.

---

## Required TransReID compatibility change

The repository uses the original TransReID project as a Git submodule:

```text
external/reid/TransReID
```

Recent PyTorch versions no longer provide `torch._six`, so one manual compatibility change is required after cloning the repository.

Open:

```text
external/reid/TransReID/model/backbones/vit_pytorch.py
```

Replace:

```python
from torch._six import container_abcs
```

with:

```python
import collections.abc as container_abcs
```

This modification is intentionally not committed inside the submodule because the submodule remains pinned to the original upstream TransReID repository.

To verify the local submodule modification:

```bash
git -C external/reid/TransReID status --short
```

The expected output is:

```text
 M model/backbones/vit_pytorch.py
```

---

## Run on a video

From the repository root:

### Windows PowerShell

```powershell
python demo/video_deep_debug_reid.py `
  --video_path "C:\path\to\video.mp4"
```

### Linux or macOS

```bash
python demo/video_deep_debug_reid.py \
  --video_path "/path/to/video.mp4"
```

Optional arguments:

```text
--video_path PATH      input video path; required
--yolo_conf FLOAT      YOLO confidence threshold; default: 0.25
--out_root PATH        output root; default: debug_cases_video
--ring_size INT        number of recent frames retained for debug export
--alpha FLOAT          mask-overlay opacity; default: 0.5
--reid_thr FLOAT       optional override for the ReID acceptance threshold
--reid_print           print internal ReID information to the terminal
```

Example:

```powershell
python demo/video_deep_debug_reid.py `
  --video_path "C:\path\to\video.mp4" `
  --reid_thr 0.80
```

The program pauses on the first frame so that identities can be initialized before playback starts. New identities can also be added later while the video is paused.

---

## Run with a webcam

```powershell
python demo/webcam_deep_debug_reid.py `
  --camera 0 `
  --reid_backend transreid
```

To hide the debugging HUD and show only the video, detections, and masks:

```powershell
python demo/webcam_deep_debug_reid.py `
  --camera 0 `
  --reid_backend transreid `
  --hide_hud
```

Optional arguments include:

```text
--camera INT                  camera index; default: 0
--yolo_conf FLOAT            YOLO confidence threshold
--out_root PATH              debug-output directory
--ring_size INT              number of recent frames retained
--alpha FLOAT                mask-overlay opacity
--reid_thr FLOAT             optional ReID-threshold override
--reid_print                 print ReID debug information
--hide_hud                   hide on-screen debug text
--reid_backend NAME          transreid, osnet_x1_0, or osnet_ain_x1_0
```

---

## Interactive controls

### Video demo

| Key | Action |
|---|---|
| Left / Right arrows | Select a YOLO person candidate |
| `A` | Add the selected candidate as a tracked identity |
| `T` | Start or resume tracking |
| `Space` | Pause or resume playback |
| `P` | Pause and enter prompting mode |
| `D` | Export a detailed debug case |
| `Y` | Show or hide YOLO proposals |
| `+` / `-` | Increase or decrease the YOLO confidence threshold |
| `R` | Reset the tracker and return to the beginning |
| `Q` or `Esc` | Exit |

### Webcam demo

| Key | Action |
|---|---|
| Left / Right arrows | Select a YOLO person candidate |
| `A` | Add the selected candidate as a tracked identity |
| `T` | Start tracking |
| `D` | Export a detailed debug case |
| `Y` | Show or hide YOLO proposals |
| `+` / `-` | Increase or decrease the YOLO confidence threshold |
| `R` | Reset the tracker |
| `Q` or `Esc` | Exit |

---

## Outputs

### Video demo

Each run creates a timestamped directory under:

```text
debug_cases_video/
```

The directory contains:

- a debug video with masks, YOLO proposals, scores, and HUD information;
- a clean masks-only video;
- any detailed debug cases exported by pressing `D`.

### Detailed debug exports

A debug case may contain:

- the current RGB frame;
- the current mask overlay;
- one binary mask per identity;
- all SAM mask candidates and their scores;
- the selected candidate mask;
- Kalman-predicted and candidate bounding boxes;
- memory-attention frames selected for each identity;
- non-conditioning memory masks;
- ReID gallery crops and full-frame gallery visualizations;
- gallery metadata and anchor information;
- object-pointer comparisons;
- JSON summaries of the tracker state;
- reacquisition and memory-gating scores.

Generated outputs, checkpoints, and debug folders are ignored by Git.

---

## ReID backends

The backend factory supports:

```text
transreid
transreid_msmt
osnet
osnet_x1_0
osnet_ain
osnet_ain_x1_0
```

The final thesis system uses:

```text
transreid
```

The TransReID backend processes `256 × 128` person crops using the ViT-based MSMT17 checkpoint and returns L2-normalized appearance embeddings compared through cosine similarity.

---

## Current limitations

- appearance verification depends on the quality and visibility of the predicted person crop;
- severe backlighting, motion blur, truncation, or very small visible regions may delay reacquisition;
- abrupt clothing changes that have never been observed by the gallery remain difficult;
- visually similar people may still be ambiguous under limited visual evidence;
- the complete system is computationally demanding, especially as the number of active identities increases;
- the implementation contains several research and debugging scripts that are not all intended as stable public APIs.

---

## Citation

Please cite the corresponding thesis when using this work:

```bibtex
@mastersthesis{sousa2026robust,
  author  = {In{\^e}s Gomes Crispim de Sousa},
  title   = {Robust Multi-Person Re-Identification and Tracking for Social Robotics},
  school  = {Instituto Superior T{\'e}cnico, Universidade de Lisboa},
  year    = {2026}
}
```

---

## Acknowledgements

This work builds on:

- [SAM 2](https://github.com/facebookresearch/sam2)
- [SAMURAI](https://github.com/yangchris11/samurai)
- [TransReID](https://github.com/damo-cv/TransReID)
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

Please consult the original repositories, papers, and licenses when using or redistributing their code and checkpoints.

---

## Project links

- **Project page:** [https://inessousa11.github.io/person-reid-website/](https://inessousa11.github.io/person-reid-website/)
- **Source repository:** [https://github.com/InesSousa11/samurai-real-time](https://github.com/InesSousa11/samurai-real-time)
