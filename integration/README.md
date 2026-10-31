# Model 1 + Model 2 Integration

## Overview

This folder contains the initial integration of Model 1
(person detection and tracking) with Raneem's Model 2
(pose estimation and temporal fall classification).

The purpose is to detect people in live RGB camera footage,
track each person, analyze their skeleton movements, and
identify possible fall events.

## Model Architecture

RGB Camera
    |
YOLOv5m (CrowdHuman person detector)
    |
ByteTrack (person tracking and track IDs)
    |
YOLOv8n Pose (17 COCO keypoints)
    |
ST-GCN++ (30-frame temporal classification)
    |
Fall / Non-Fall Prediction

## Implemented

- Connected YOLOv5m person detection with ByteTrack.
- Connected tracked persons to Raneem's pose model.
- Integrated ST-GCN++ for fall/non-fall prediction.
- Added 30-frame skeleton sequence buffers.
- Added skeleton visualization to live video.
- Added fall confidence display and event logging.
- Added optional video recording.

## Model Files Required

Model 1:
My_Work/weights/crowdhuman_yolov5m.pt

Model 2:
Raneem_Work/yolov8n-pose.pt
Raneem_Work/stgcn_project/trained_model/best_stgcn_model.pth

These weights are not bundled in this folder.

## Current Status

The integration prototype has been implemented but has
not been fully validated through end-to-end testing.

## Known Issues

- ByteTrack compatibility depends on the installed
  supervision library version.
- Person tracking can lose identity under occlusion.
- Pose estimation can miss keypoints.
- Each tracked person needs a complete 30-frame sequence.
- Live fall-alert accuracy and latency remain unverified.

## Remaining Work

1. Verify library versions and run the live pipeline.
2. Confirm consistent person IDs and skeleton association.
3. Test fall and non-fall videos.
4. Measure false positives, recall, and detection latency.
5. Test multiple people and occlusion.
6. Integrate validated fall events with the dashboard.

## Scope

This branch preserves the initial Model 1 + Model 2
integration only.

It does not include later fine-tuning experiments,
additional datasets, or replacement model checkpoints.
