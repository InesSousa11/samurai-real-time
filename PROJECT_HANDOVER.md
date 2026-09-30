# Project Handover

## Project

**ReID-SAMURAI: Robust Multi-Person Re-Identification and Tracking for Social Robotics**

This repository contains the implementation developed for the MSc thesis
"Robust Multi-Person Re-Identification and Tracking for Social Robotics".

The final system, ReID-SAMURAI, combines:

- SAMURAI / SAM 2 mask-based tracking
- TransReID appearance embeddings
- online per-identity appearance galleries
- identity-aware memory selection
- ReID-based gating and reacquisition
- Kalman-filter motion information
- verified mask updates to tracker memory

The system was developed for long-horizon, identity-aware multi-person
tracking from a moving social robot camera, with particular attention to
occlusions, re-entry, similar clothing, and identity continuity.

## Current Repository State

The repository has been cleaned for handover.

The current layout is:

```text
samurai-real-time/
├── assets/
├── checkpoints/
├── demo/
│   ├── realsense_deep_debug_reid.py
│   ├── video_deep_debug_reid.py
│   └── webcam_deep_debug_reid.py
├── evaluation/
├── external/
├── sam2/
├── tools/
├── README.md
└── PROJECT_HANDOVER.md
```

The `demo/` directory now contains only the main interactive runtime demos.

The `evaluation/` directory contains KTP evaluation, sweeps, analysis, and
TrackEval preparation scripts.

The `tools/` directory contains helper utilities for rosbag replay, KTP video
generation, and annotated video export.

Older Colab demos, face/body experiments, debugging prototypes, and early
webcam ReID prototypes were removed from the active repository. Their history
is still available through Git.

## Final System

The final thesis system uses:

- SAMURAI / SAM 2 for mask propagation and temporal tracking
- TransReID for person appearance embeddings
- per-ID online ReID galleries
- identity-aware gallery update rules
- ReID checks during normal tracking
- ReID-guided reacquisition after tracking failure
- motion and overlap information during reacquisition

The final thesis backend is:

```text
transreid
```

OSNet backends remain available for comparison or experimentation but are not
the final thesis configuration.

## Final Thesis Configuration

The final operating parameters used for the thesis were:

```text
reid_thr                          = 0.80
memory_bank_reid_threshold        = 0.75
stable_frames_threshold           = 15
stable_ious_threshold             = 0.30
min_obj_score_logits              = 1.0
kf_score_weight                   = 0.25
memory_bank_iou_threshold         = 0.50
memory_bank_obj_score_threshold   = 0.50
```

Some evaluation scripts expose different default values because they were also
used for parameter sweeps and exploratory experiments.

Do not assume that script defaults always reproduce the final thesis operating
point. When reproducing the final system, pass the intended configuration
explicitly or verify the defaults first.

## KTP Evaluation Setup

The KTP dataset was used for quantitative evaluation.

Evaluated sequences:

- Arc
- Rotation
- Still
- Translation

The original KTP data was recorded at 30 Hz.

Identity initialization and evaluation used the following settings:

```text
visible_area_frac       = 0.05
visible_min_h           = 120
visible_min_w           = 50
seed_overlap_iou_max    = 0.05
IoU match threshold     = 0.30
random seed             = 42
```

Temporal sampling experiments used strides:

```text
1, 2, 6, 10, 30
```

The stride-6 setting approximates the effective frequency observed during
robot deployment.

## Main Thesis Results

At stride 1, the reported main results were:

```text
ReID-SAMURAI
HOTA   = 50.80
MOTA   = 59.63
IDF1   = 78.16
IDSW   = 23

SAMURAI baseline
HOTA   = 29.33

TransReID baseline
HOTA   = 38.26
```

Temporal sampling results for ReID-SAMURAI included:

```text
stride 2   HOTA = 50.72
stride 30  HOTA = 47.88
```

For full tables, plots, discussion, and experimental interpretation, refer to
the thesis and extended abstract.

## Evaluation Workflow

The main evaluation entry point is:

```powershell
python evaluation/KTP_eval_run.py --help
```

Typical workflow:

1. Run `evaluation/KTP_eval_run.py` for a chosen configuration.
2. Export predictions in MOT format.
3. Convert outputs for TrackEval with
   `evaluation/prepare_ktp_for_trackeval.py`.
4. Run TrackEval.
5. Aggregate multiple runs with
   `evaluation/aggregate_ktp_eval_runs.py`.
6. Use the sweep, plotting, or ID-switch export scripts when deeper analysis
   is needed.

See:

```text
evaluation/README.md
```

for a description of the scripts in that directory.

## Main Demos

### Video

```powershell
python demo/video_deep_debug_reid.py `
  --video_path "C:\path\to\video.mp4"
```

Optional:

```powershell
python demo/video_deep_debug_reid.py `
  --video_path "C:\path\to\video.mp4" `
  --reid_thr 0.80
```

### Webcam

```powershell
python demo/webcam_deep_debug_reid.py `
  --camera 0 `
  --reid_backend transreid
```

### RealSense

Use:

```text
demo/realsense_deep_debug_reid.py
```

for RealSense-based operation.

The root `README.md` contains the user-facing setup and demo instructions.

## ROS 2 Integration

The ROS 2 wrapper is maintained separately in:

```text
samurai_realtime_ros2
```

The tracker was integrated with RGB camera streams and robot-side ROS 2
topics.

Relevant topics used during development/deployment included:

```text
/k4a/rgb/image_raw
/samurai/img_masks
/samurai/init_prompt
/samurai/masks
/samurai_tracker/measurement_marker
/samurai_tracker/tracked_persons
/tf
/tf_static
```

Robot visualization was performed in RViz, and tablet interaction was used for
task-level control and user prompts.

The core repository should remain independent of ROS 2 wherever possible.
Robot-specific integration should stay in the ROS 2 wrapper.

## Robot Deployment

The system was deployed on the SocRob@HOME robot during RoboCup@Home 2026.

A long-horizon run of approximately 20 minutes was completed across multiple
floors.

Observed successful behaviors included:

- maintaining identity over long sequences
- recovery after temporary occlusion
- recovery after people turned away from the camera
- handling re-entry
- preserving identity in the presence of similarly dressed distractors

The robot deployment was slower than offline desktop evaluation. End-to-end
tracking on the robot was approximately 5 FPS, while the visualization
pipeline could update faster.

## TransReID Compatibility Patch

The TransReID repository is included as a Git submodule:

```text
external/reid/TransReID
```

Recent PyTorch versions no longer provide:

```python
from torch._six import container_abcs
```

The required local compatibility change is:

```python
import collections.abc as container_abcs
```

in:

```text
external/reid/TransReID/model/backbones/vit_pytorch.py
```

This modification is intentionally not committed inside the upstream
TransReID submodule.

The root `README.md` documents this setup step.

## Checkpoints

Required main checkpoints:

```text
checkpoints/sam2.1_hiera_small.pt
checkpoints/reid/transreid/vit_transreid_msmt.pth
```

The demos use `yolov8s.pt`; Ultralytics downloads it automatically when
required.

Model checkpoints are intentionally excluded from Git.

## Known Limitations

The system is functional, but several limitations remain.

### ReID depends strongly on appearance quality

Re-identification becomes harder when:

- the person is very small
- the body is heavily occluded
- only partial clothing is visible
- illumination changes significantly
- two people have very similar appearance

### Gallery contamination remains important

Incorrect gallery updates can propagate identity mistakes.

The final system includes similarity and stability checks to reduce this risk,
but gallery-update logic remains an important area for improvement.

### Reacquisition score fusion is heuristic

Reacquisition combines appearance, object confidence, motion, and overlap
signals using hand-selected weights.

These were experimentally tuned rather than learned.

### Tracking speed is hardware dependent

The complete model is computationally demanding, especially with multiple
people and ReID enabled.

Robot-side throughput was lower than desktop/offline execution.

### Some scripts are research-oriented

Evaluation and analysis scripts were created during thesis experimentation.
They are functional, but they are not intended to be a polished public API.

## Recommended Future Work

Potential continuation directions include:

### 1. Improve gallery management

Investigate:

- better gallery sample selection
- diversity-aware gallery updates
- gallery pruning
- long-term vs short-term appearance memory
- confidence-weighted embeddings
- protection against contaminated updates

### 2. Improve reacquisition

Possible directions:

- learn the reacquisition fusion function instead of using fixed weights
- use adaptive thresholds
- include temporal confidence
- model uncertainty explicitly
- use stronger motion prediction during long occlusions

### 3. Improve appearance representation

Possible work:

- evaluate newer person ReID models
- domain-adapt ReID to robot-camera data
- exploit multi-frame appearance aggregation
- combine global and local body features
- investigate pose-aware appearance representations

### 4. Improve multi-person data association

Current behavior is largely identity-gated per tracked person.

A stronger global assignment strategy could reduce conflicts when multiple
people compete for similar detections or masks.

### 5. Improve runtime efficiency

Potential optimizations:

- reduce redundant ReID inference
- cache embeddings more aggressively
- batch person crops
- optimize SAM 2 memory usage
- profile GPU synchronization
- investigate lower-resolution or mixed-precision variants

### 6. Expand evaluation

Useful additions include:

- additional crowded datasets
- more robot-recorded sequences
- longer occlusion scenarios
- stronger similar-clothing cases
- explicit re-entry benchmarks
- ablations of each identity-memory component

### 7. Improve software structure

The final runtime demo scripts remain large.

A future refactor could extract shared code into reusable modules for:

- video input
- visualization
- prompting
- YOLO proposal handling
- output video writing
- debugging exports

This should only be done with regression testing so that known-working runtime
behavior is preserved.

## Files Worth Keeping Stable

The most important runtime files are:

```text
demo/video_deep_debug_reid.py
demo/webcam_deep_debug_reid.py
demo/realsense_deep_debug_reid.py
sam2/sam2_camera_predictor.py
sam2/modeling/sam2_base.py
sam2/gating/reid_gate.py
sam2/reid_backends/
sam2/reid_embedder.py
sam2/transreid_embedder.py
sam2/utils/kalman_filter.py
```

Changes to these files can affect tracker behavior and should be tested
carefully.

## Repository Hygiene

Generated files should not be committed.

The repository `.gitignore` already excludes common items such as:

```text
__pycache__/
*.pyc
.venv/
outputs/
runs/
results/
logs/
debug_cases_video/
debug_cases_webcam/
debug_cases_realsense/
debug_cases_frames/
*.pt
*.pth
*.ckpt
```

Do not add a blanket `*.mp4` rule because the repository intentionally tracks
the README comparison video.

## Preservation of Working State

Before major refactoring, preserve the known-working state using Git.

The cleaned handover branch was created from the final working repository
state after RoboCup and thesis documentation updates.

For future development:

1. tag a known-working version before large refactors;
2. keep behavioral changes separate from cleanup commits;
3. avoid rewriting history only for cosmetic reasons;
4. test interactive demos and evaluation scripts after structural changes.

## Suggested First Steps for the Next Developer

1. Clone the repository with submodules.
2. Install the environment following the root `README.md`.
3. Apply the documented TransReID compatibility patch.
4. Download the SAM 2 and TransReID checkpoints.
5. Run the custom-video demo.
6. Run a small KTP evaluation.
7. Read the thesis before modifying the identity-memory or reacquisition logic.
8. Make one change at a time and compare against the existing evaluation
   pipeline.

## Related Material

For complete technical background, experimental results, and design
motivation, consult:

- the MSc thesis
- the IEEE-style extended abstract
- the project website
- this repository
- the ROS 2 wrapper repository

These materials together provide the intended handover context for continued
development.
