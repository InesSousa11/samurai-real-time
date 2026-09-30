# Project Handover

## Project

**ReID-SAMURAI: Robust Multi-Person Re-Identification and Tracking for Social Robotics**

ReID-SAMURAI extends SAMURAI with person re-identification so that multiple
people can be tracked with persistent identities from a moving robot camera.

The final system combines:

- SAMURAI / SAM 2 mask tracking
- TransReID appearance embeddings
- one online ReID gallery per identity
- identity-aware gallery and memory updates
- reacquisition logic for uncertain/lost identities
- online multi-person tracking from an RGB stream

The thesis contains the full algorithmic description and results. This file is
meant to help someone continue the implementation and reproduce the main
experiments.

---

## Repository Structure

```text
samurai-real-time/
├── assets/
│   └── readme/
├── checkpoints/
├── demo/
│   ├── realsense_deep_debug_reid.py
│   ├── video_deep_debug_reid.py
│   └── webcam_deep_debug_reid.py
├── evaluation/
├── external/
│   └── reid/
├── sam2/
├── tools/
├── README.md
└── PROJECT_HANDOVER.md
```

Main directories:

- `demo/` — runtime demos
- `evaluation/` — KTP evaluation, sweeps, TrackEval preparation, plots, and analysis
- `tools/` — supporting utilities
- `sam2/` — SAM 2 / SAMURAI code plus ReID-SAMURAI changes
- `external/reid/` — external ReID implementations

Historical states are preserved as tags:

```text
pre-handover-cleanup
archive/clean-mask-components
archive/samurai-baseline
archive/yolo-reid-reacquisition
```

---

## Main Implementation

### Online multi-person SAMURAI

Vanilla SAMURAI was adapted from offline video tracking to frame-by-frame
operation.

The main predictor processes one RGB frame at a time and keeps persistent state
for:

- active identities
- object-index mappings
- SAMURAI conditioning/non-conditioning outputs
- cached image features
- original RGB frames
- ReID galleries
- per-identity tracking metadata

New people can be added through bounding-box prompts while the other identities
remain active.

### ReID integration

The ReID code is separated behind interchangeable backends:

```text
sam2/reid_backends/base.py
sam2/reid_backends/factory.py
sam2/reid_backends/transreid_backend.py
sam2/reid_backends/osnet_backend.py
sam2/reid_backends/osnet_ain_backend.py
sam2/reid_embedder.py
sam2/transreid_embedder.py
```

The final system uses:

```text
transreid
```

For each identity, a crop is obtained from the predicted mask and embedded by
the ReID backend. The current embedding is compared with that identity's gallery
using cosine similarity.

### Online gallery

Each identity has an online gallery.

Important behaviour:

- the first accepted embedding is protected as an anchor
- reliable observations can add new gallery entries
- repeated/near-identical appearances are avoided
- protected anchors are not replaced
- gallery updates are disabled while the identity is in reacquisition mode

### Identity-aware memory

SAMURAI memory updates are additionally gated by identity consistency.

A frame is only used as normal tracking memory when the tracking evidence is
sufficient and the appearance is compatible with the identity gallery.

### Reacquisition

An identity enters reacquisition mode when object-presence confidence becomes
too low.

Candidate masks are then evaluated using a weighted combination of:

- ReID similarity
- object-presence score
- Kalman motion consistency
- selected-mask quality

The candidate is only restored as a valid output after passing the
reacquisition decision.

---

## Important Runtime Files

```text
sam2/sam2_camera_predictor.py
sam2/modeling/sam2_base.py
sam2/gating/reid_gate.py
sam2/reid_backends/
sam2/reid_embedder.py
sam2/transreid_embedder.py
sam2/utils/kalman_filter.py

demo/video_deep_debug_reid.py
demo/webcam_deep_debug_reid.py
demo/realsense_deep_debug_reid.py
```

The three demo files contain similar runtime/debug logic. They can be refactored
later, but changes to the common tracking path should be checked against a known
working run.

---

## Final Operating Point

The final operating sweep varied four parameters:

| Parameter | Values tested |
|---|---|
| `τ_ReID` | 0.75, 0.80, 0.85 |
| `τ_mem_ReID` | 0.55, 0.65, 0.75 |
| `τ_obj` | 0.0, 0.5, 1.0 |
| `τ_gal` | 0.75, 0.80, 0.85 |

Selected configuration:

```text
τ_obj       = 1.00
τ_mem_ReID  = 0.75
τ_ReID      = 0.80
τ_gal       = 0.85
```

Several evaluation scripts were also used for exploratory runs, so their
defaults are not necessarily the final thesis configuration.

---

# Experiments and How to Reproduce Them

The main quantitative experiments use the Kinect Tracking Precision (KTP)
dataset.

Sequences:

```text
Still
Translation
Rotation
Arc
```

KTP is recorded at 640×480 and 30 Hz. The evaluation uses the RGB stream and
2D person bounding-box annotations.

The current evaluation scripts are documented in:

```text
evaluation/README.md
```

A useful starting point is:

```powershell
python evaluation/KTP_eval_run.py --help
```

---

## 1. ReID Model Comparison

Before choosing the final ReID backend, several appearance models were compared
using pairwise cosine similarity on KTP:

- TransReID
- OSNet-AIN
- OSNet-IBN
- OSNet
- ResNet50-FC512
- MLFN
- ResNet50-IBN-A

The comparison used same-identity and different-identity pairs and evaluated
discrimination metrics such as AUC and EER, plus inference time and GPU memory.

TransReID was selected as the final backend.

The final reported AUC for TransReID was:

```text
0.9743
```

If repeating this experiment, keep the KTP pair construction fixed across
models and evaluate all models on exactly the same identity pairs.

---

## 2. Operating-Point Sweep

The final operating point was selected through a sweep over:

```text
τ_ReID
τ_mem_ReID
τ_obj
τ_gal
```

Use the sweep scripts in `evaluation/` rather than editing the tracker manually
between runs.

The selected configuration was configuration 78:

```text
τ_obj       = 1.00
τ_mem_ReID  = 0.75
τ_ReID      = 0.80
τ_gal       = 0.85
```

Reported metrics for this sweep configuration:

```text
HOTA  = 50.429
MOTA  = 57.337
IDF1  = 77.483
IDSW  = 7
```

The operating point was chosen as a compromise between tracking quality and
identity switches rather than by maximizing one metric alone.

---

## 3. Main System Comparison

Three systems were compared using the same KTP input and initialization rules:

### ReID-SAMURAI

Full proposed system.

### SAMURAI subsystem

Mask tracking without appearance verification, online ReID galleries, or
ReID-guided reacquisition.

The historical SAMURAI-baseline evaluation state is preserved at:

```text
archive/samurai-baseline
```

This is useful because the baseline scripts depend on the tracker state that
existed when they were produced.

### TransReID subsystem

Appearance-only baseline:

```text
YOLO detections
→ TransReID embeddings
→ cosine similarity
→ Hungarian assignment
```

The current baseline evaluator is:

```text
evaluation/KTP_eval_transreid_baseline.py
```

### Final comparison

| System | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| SAMURAI subsystem | 29.33 | 24.80 | 45.13 | 26 |
| TransReID subsystem | 38.26 | 41.43 | 61.08 | 4 |
| **ReID-SAMURAI** | **50.48** | **57.30** | **77.44** | 7 |

For the mask-based systems, convert masks to bounding boxes before TrackEval.

Typical evaluation flow:

```text
run KTP evaluation
→ export MOT-format predictions
→ prepare TrackEval folders
→ run TrackEval
→ aggregate results
```

Useful scripts:

```text
evaluation/KTP_eval_run.py
evaluation/KTP_eval_transreid_baseline.py
evaluation/prepare_ktp_for_trackeval.py
evaluation/aggregate_ktp_eval_runs.py
```

---

## 4. Temporal-Sampling Experiment

The goal was to test whether the tracker remains usable when the robot cannot
process every 30 Hz camera frame.

Repeat the same ReID-SAMURAI configuration with temporal strides:

```text
1, 2, 3, 6, 10, 15, 30
```

Equivalent processed frame rates:

```text
30, 15, 10, 5, 3, 2, 1 FPS
```

Reported results:

| Stride | Effective FPS | HOTA | IDF1 | IDSW |
|---:|---:|---:|---:|---:|
| 1 | 30.0 | 50.80 | 78.16 | 23 |
| 2 | 15.0 | 50.72 | 77.76 | 7 |
| 3 | 10.0 | 52.05 | 78.88 | 5 |
| 6 | 5.0 | 50.50 | 77.37 | 7 |
| 10 | 3.0 | 49.70 | 76.88 | 1 |
| 15 | 2.0 | 50.96 | 79.06 | 5 |
| 30 | 1.0 | 49.88 | 77.75 | 2 |

When reproducing this experiment, sample both RGB frames and annotations with
the same stride.

---

## 5. Annotation-Limited KTP Review

KTP annotations are bounding boxes, while ReID-SAMURAI can sometimes continue
tracking a small visible body region with a mask even when the available
bounding box does not represent that region well.

These cases were manually reviewed instead of treating every excluded
prediction as an ordinary false positive.

Reported review:

| System | Valid unannotated target | Wrong identity | Background / bad mask |
|---|---:|---:|---:|
| TransReID subsystem | 5 | 0 | 0 |
| SAMURAI subsystem | 169 | 768 | 42 |
| ReID-SAMURAI | 277 | 26 | 1 |

If extending the evaluation, keep this annotation limitation in mind when
interpreting false positives from mask-based tracking.

---

## 6. Runtime Benchmark

Runtime was measured on:

```text
NVIDIA RTX 5060 Laptop GPU
```

Reported complete-pipeline timing:

```text
Average FPS       = 1.27
Mean ms/frame     = 789.38
Median ms/frame   = 785.15
P90 ms/frame      = 1145.31
```

The benchmark measures model-processing time and excludes:

- disk writing
- TrackEval export
- plotting
- visualization

When comparing optimizations, keep the same timing boundary.

---

## 7. Robot Experiments

The robot experiments used a TIAGo with a Microsoft Azure Kinect RGB-D camera.

The ROS 2 integration is maintained in the **SocRob GitHub repository**, not in
this core repository.

Five useful test scenarios were used:

### Test 1 — Long-duration person following

Follow one initialized target for approximately 20 minutes through different
indoor areas.

Include:

- robot motion
- background changes
- lighting changes
- viewpoint changes
- temporary occlusions
- interactions with other people

### Test 2 — Similar-looking people

Initialize one person and introduce another person wearing the same team
t-shirt.

Test both close crossings and side-by-side motion.

### Test 3 — Clothing change

Initialize one person, let them leave the camera view, and make them reappear
wearing a different shirt.

This is one of the clearest failure cases for appearance-only identity
reasoning.

### Test 4 — Lighting change

Track/reacquire a person before and after severe backlighting.

### Test 5 — Drastic pose change

Initialize two people while standing, then observe them later while seated.

This tests whether the stored identity appearance generalizes across a large
pose change.

The robot experiments are qualitative because the recordings do not have
frame-level ground truth.

---

## Setup and Runtime Usage

Installation, environment setup, model checkpoints, the TransReID compatibility
patch, and commands for the video/webcam/RealSense demos are documented in the
root:

```text
README.md
```

The main runtime entry points are:

```text
demo/video_deep_debug_reid.py
demo/webcam_deep_debug_reid.py
demo/realsense_deep_debug_reid.py
```

---

# Recommended Future Work

## 1. Add a complementary face identity cue

The main remaining ambiguity is when full-body appearance is unreliable or two
people wear similar clothing.

A useful extension would be:

```text
body ReID gallery
+
face embedding / face identity confidence
→ fused identity score
```

Only use the face cue when the detected face has sufficient size and quality.
Keep body and face galleries separate so a poor face crop cannot contaminate
the body representation.

The first place to use this cue should be reacquisition, where identity evidence
matters most.

---

## 2. Use depth and 3D position during reacquisition

The robot already has an RGB-D camera, but the final identity logic is primarily
image-based.

Depth could be used to:

- reject masks with implausible depth
- separate overlapping people
- maintain a rough 3D trajectory for each identity
- compare a reacquisition candidate with the last known 3D position
- provide a better spatial prior than image-plane motion alone

A simple first version would attach median mask depth and camera-frame 3D
position to each tracked identity and include a 3D distance term in the
reacquisition score.

---

## 3. Replace heuristic reacquisition fusion

The current reacquisition decision uses a hand-weighted combination of:

```text
ReID
object presence
Kalman consistency
mask quality
```

A useful next step is to collect accepted/rejected reacquisition candidates from
robot recordings and learn or calibrate the fusion instead of selecting the
weights manually.

Possible approaches:

- logistic regression on the existing scores
- a small MLP
- calibrated probabilities followed by a weighted fusion
- adaptive thresholds depending on target visibility or time since loss

This can be done without replacing SAMURAI or TransReID.

---

## 4. Improve gallery management

The gallery is central to long-term performance and is also a possible source
of identity contamination.

Useful experiments:

- separate short-term and long-term gallery entries
- keep multiple protected anchors representing genuinely different appearances
- use confidence-weighted gallery prototypes
- explicitly detect clothing/appearance changes before accepting them
- compare max similarity against top-k or prototype-based similarity
- prune entries that are redundant or consistently low quality
- require cross-frame confirmation before inserting a very different appearance

The clothing-change test is a good scenario for evaluating this.

---

## 5. Improve multi-person assignment

The current system reasons mainly per tracked identity.

When several identities are lost or close together, it would be useful to build
a global candidate-to-identity assignment step using:

```text
ReID similarity
+
motion / 3D position
+
mask quality
```

and solve the assignment jointly, for example with Hungarian matching.

This would prevent two identities from independently preferring the same
candidate.

---

## 6. Reduce runtime cost

The complete pipeline is expensive with several active identities.

Profile the system first, then consider:

- batch TransReID crops from all identities into one forward pass
- avoid running ReID on every stable frame
- trigger ReID more aggressively only around uncertainty/reacquisition
- cache embeddings when the crop changes very little
- reduce duplicate crop extraction / tensor transfers
- profile SAM 2 memory attention separately from ReID
- compare lighter ReID backends using the same KTP identity-pair benchmark

Keep the runtime benchmark boundary identical to the existing one so results are
comparable.

---

## 7. Record a dedicated robot dataset

KTP is useful but its bounding-box annotations are not ideal for evaluating an
identity-aware mask tracker.

A useful continuation would be a robot-recorded dataset containing:

- persistent person IDs
- segmentation masks
- occlusion and re-entry events
- partial visibility
- similar clothing
- clothing changes
- pose changes
- difficult lighting
- RGB-D frames
- robot/camera motion

The five robot test scenarios above are a good starting protocol.

This would make it possible to evaluate the exact problem ReID-SAMURAI is
designed for instead of relying on manual review of annotation-limited cases.

---

## 8. Re-evaluate the ReID backend

TransReID gave the strongest discrimination in the original comparison, but the
backend interface was intentionally kept replaceable.

Future comparisons should measure both:

```text
identity discrimination on robot/KTP data
+
runtime / memory cost
```

rather than selecting a model only from standard ReID benchmark performance.

A lighter model may be preferable if it allows more frequent identity checks on
the robot.

---

## 9. Broaden the operating-parameter optimization

Only a restricted subset of thresholds was swept for the final thesis
configuration.

A later optimization could include:

- gallery update thresholds
- reacquisition threshold/weights
- mask-quality thresholds
- motion-consistency thresholds
- gallery size / replacement behaviour

Use the existing KTP pipeline first, then validate the selected point on robot
recordings so the system is not over-tuned to KTP.

---

## References Inside the Project

For implementation:

```text
README.md
evaluation/README.md
sam2/
demo/
evaluation/
```

For the robot integration, use the **SocRob GitHub repository** containing the
ROS 2 wrapper.

For the reasoning behind the design and the full experimental analysis, see the
thesis.
