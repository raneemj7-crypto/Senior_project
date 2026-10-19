# Fall Detection Using YOLOv8-Pose and LSTM

## Overview
This project is part of our senior project, which aims to detect falls and collapse incidents using computer vision and AI.

Our approach combines **YOLOv8-Pose** for human pose estimation with **Long Short-Term Memory (LSTM)** to analyze body movements over time and distinguish falls from normal daily activities.

## Model Architecture

The proposed pipeline is:

```text
RGB Video
    ↓
YOLOv8-Pose
    ↓
Body Keypoint Extraction
    ↓
Keypoint Sequence Preparation
    ↓
LSTM
    ↓
Fall / Non-Fall Classification
```

## How It Works

1. **Pose Estimation:** YOLOv8-Pose extracts 17 body keypoints from each video frame.
2. **Sequence Preparation:** Keypoints from consecutive frames are organized into sequences to represent movement over time.
3. **Fall Classification:** The LSTM analyzes these sequences to identify patterns associated with falling or normal activities.

Unlike rule-based detection, which relies on predefined conditions such as body angle or position, the LSTM approach is designed to learn movement patterns from training data.

## Datasets

The project uses RGB video data from fall detection datasets, including:

- **URFD (University of Rzeszow Fall Detection Dataset)**
- **OOPS (Oops! Dataset)**

Dataset preparation includes cleaning, organizing video sequences, and splitting the data into training, validation, and testing sets while avoiding overlap between source videos.

## Development

The implementation builds on the YOLOv8-Pose approach explored through the [j22k/falling](https://github.com/j22k/falling) repository, with the goal of using an LSTM for temporal fall classification.

## Project Goal

Develop a fall detection approach that considers body movement across multiple frames rather than relying only on a person's posture in a single frame.
