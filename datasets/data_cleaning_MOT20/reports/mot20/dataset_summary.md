# MOT20 Dataset Summary

## Basic Information

- Dataset name: MOTChallenge/MOT20
- Short name: mot20
- Project owner: Raneem Abumoustafa
- Pipeline role: C - Tracking
- Purpose: Give each pedestrian a stable track ID across video frames
- Modality: RGB
- Data type: Sequential JPG image frames
- Raw dataset location: raw/mot20
- Raw dataset size: 4.8 GB

## Project Use

MOT20 is used only for multi-object pedestrian tracking in crowded
environments. It does not contain collapse or activity labels.

All usable MOT20 samples receive:

- project_label: person
- role: C
- modality: rgb

MOT20 must not be labelled as collapse, normal, postural, or on_ground.

## Dataset Structure

The dataset contains eight sequences:

### Training sequences

- MOT20-01
- MOT20-02
- MOT20-03
- MOT20-05

### Test sequences

- MOT20-04
- MOT20-06
- MOT20-07
- MOT20-08

Each training sequence contains:

- img1/: Sequential JPG frames
- gt/gt.txt: Ground-truth tracking annotations
- det/det.txt: Public pedestrian detections
- seqinfo.ini: Sequence metadata

Test sequences contain images and detections but do not provide local public
ground truth for normal evaluation.

## Annotation Format

Each row in gt/gt.txt contains:

frame_id, track_id, x, y, width, height, mark, class_id, visibility

The fields represent:

- frame_id: Frame number
- track_id: Identity of the tracked object
- x, y: Top-left bounding-box coordinates
- width, height: Bounding-box dimensions
- mark: Whether the annotation is active
- class_id: Original MOT20 object class
- visibility: Visible proportion from 0 to 1

## Original Classes Observed

- Class 1: Pedestrian
- Class 6: Non-motorized vehicle
- Class 7: Static person
- Class 11: Full occluder
- Class 13: Crowd region

The tracker-ready target annotations use:

mark == 1 and class_id == 1

The original class information remains unchanged in raw/mot20.

## Dataset Counts

- Total sequences: 8
- Training sequences: 4
- Test sequences: 4
- Total frames: 13,410
- Training frames: 8,931
- Test frames: 4,479
- Training ground-truth rows: 1,336,920
- Active pedestrian boxes: 1,134,614

## Existing Validation Results

The initial structural validation found:

- 8 valid sequences
- 0 invalid sequences
- 0 missing frames
- 0 invalid frame references
- 0 invalid track IDs
- 0 invalid bounding-box sizes
- 0 invalid visibility values
- 0 duplicate frame-ID pairs

File readability, exact duplicate detection, issue codes, and standardized
frame-level manifest generation will be completed in the revised cleaning
process.

## Raw Data Rule

The contents of raw/mot20 must remain exactly as downloaded.

Files inside the raw directory must never be:

- Renamed
- Edited
- Re-saved
- Manually deleted
- Replaced with processed copies