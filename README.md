# Samurai Real-Time

Real-time person re-identification using [YOLOv8](https://github.com/ultralytics/ultralytics) for person detection and [SAMURAI](https://github.com/facebookresearch/sam2) for instance segmentation.

## Features

- Detects people in a webcam stream using YOLO.
- Tracks and segments the first person detected using SAMURAI.
- Displays real-time segmentation overlaid on the video feed.
- Built to be efficient with GPU support.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/InesSousa11/samurai-real-time.git
   cd samurai-real-time