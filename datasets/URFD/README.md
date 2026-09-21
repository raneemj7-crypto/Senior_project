# UR Fall Detection Dataset Cleaning

## Dataset

The UR Fall Detection Dataset (URFD) contains fall sequences and Activities of Daily Living (ADL).

For this project, only the RGB camera data relevant to the planned camera-based fall detection system was selected.

## Selected Data

The cleaned dataset contains:

- Fall RGB sequences from cam0 and cam1.
- Selected ADL RGB sequences from cam0.
- Synchronization metadata for the selected sequences.
- Frame-level posture labels available for cam0.

Some ADL sequences from the original dataset were intentionally not included because they were not relevant to the target use case.

For example, activities involving a person intentionally lying on a bed were excluded because the target environment is a crowded public space such as a concert or event.

The original URFD files were not modified.

## Labels

The project-level classes were standardized as:

- `fall`
- `non_fall`

The original URFD sequence names such as `fall-01` and `adl-01` were preserved.

URFD also provides frame-level posture labels for cam0:

- `-1` = person is not lying
- `0` = person is in a falling/transition pose
- `1` = person is lying on the ground

Some downloaded ADL frames do not have posture annotations in the original URFD label file. These frames were kept and marked with `has_posture_label = False`.

## Cleaning Process

The cleaning script performs the following steps:

1. Finds all selected RGB frames.
2. Extracts the sequence, camera, and frame number from each filename.
3. Checks that every image can be opened.
4. Checks for missing frame numbers within each sequence.
5. Matches cam0 frames with synchronization timestamps.
6. Matches cam0 frames with available posture labels.
7. Excludes metadata belonging to original URFD sequences that were not selected for this project.
8. Creates a cleaned manifest containing the selected dataset.

## Cleaned Manifest

The output file is:

`cleaned/cleaned_manifest.csv`

It contains the following columns:

- `sequence_name`
- `class`
- `camera`
- `frame`
- `timestamp_ms`
- `posture_label`
- `has_posture_label`
- `image_path`
- `valid`

Paths are stored relative to the project folder so the manifest can be used on different machines.

## Cleaning Statistics

The cleaning statistics are stored in:

`cleaned/cleaning_statistics.txt`

Current results:

- Total RGB frames: 11,157
- Valid RGB frames: 11,157
- Invalid RGB frames: 0
- Sequence/camera groups checked: 87
- Groups with missing frames: 0
- Cam0 frames: 8,161
- Cam1 frames: 2,996
- Cam0 frames with timestamps: 8,161
- Cam0 frames without timestamps: 0
- Cam0 frames with posture labels: 7,948
- Cam0 frames without posture labels: 213
- Original label rows excluded because RGB was not selected: 3,596
- Final cleaned manifest rows: 11,157

## Running the Script

From the main project folder, run:

```bash
python3 scripts/clean_urfd.py