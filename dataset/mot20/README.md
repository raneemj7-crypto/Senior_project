# MOT20 Dataset Preparation and Validation

## Purpose

MOT20 is used to validate the multi-object tracking component of the
collapse-detection system in highly crowded pedestrian scenes.

The dataset does not contain collapse or fall labels. Its purpose is to test
whether the system can:

- Detect multiple pedestrians.
- Assign a unique track ID to each pedestrian.
- Maintain identities between frames.
- Handle partial visibility and occlusion.
- Avoid mixing identities in crowded scenes.

After tracking, the collapse-detection model can analyze the frame sequence
associated with each individual track ID.

## Dataset Source

- Dataset: MOTChallenge MOT20
- Official archive: https://motchallenge.net/data/MOT20/
- Benchmark website: https://motchallenge.net/
- Reference: Dendorfer et al., "MOT20: A Benchmark for Multi Object Tracking
  in Crowded Scenes," 2020.

## Storage Policy

The complete MOT20 sequences are stored in shared project storage:

```text
senior-project-storage/datasets/MOT20/
```

The full images and annotations are not committed to GitHub.

GitHub contains only:

- Validation scripts
- Sequence manifest
- Validation statistics
- Documentation
- A small sample only if redistribution is permitted

## Dataset Structure

```text
MOT20/
├── train/
│   ├── MOT20-01/
│   ├── MOT20-02/
│   ├── MOT20-03/
│   └── MOT20-05/
└── test/
    ├── MOT20-04/
    ├── MOT20-06/
    ├── MOT20-07/
    └── MOT20-08/
```

A training sequence contains:

```text
MOT20-01/
├── det/
│   └── det.txt
├── gt/
│   └── gt.txt
├── img1/
│   ├── 000001.jpg
│   └── ...
└── seqinfo.ini
```

## Ground-Truth Format

Each row in `gt/gt.txt` contains nine comma-separated fields:

```text
frame,id,x,y,width,height,mark,class,visibility
```

| Field | Description |
|---|---|
| `frame` | Frame number containing the annotation |
| `id` | Unique trajectory or track ID |
| `x` | Left coordinate of the bounding box |
| `y` | Top coordinate of the bounding box |
| `width` | Bounding-box width |
| `height` | Bounding-box height |
| `mark` | Whether the annotation is active |
| `class` | Annotated object class |
| `visibility` | Visible proportion of the object from 0 to 1 |

For tracker-ready pedestrian ground truth, this project selects:

```text
mark == 1 and class == 1
```

The original ground-truth files are never modified.

## Classes Found

The training data includes the following relevant class identifiers:

| ID | Class | Tracking decision |
|---:|---|---|
| 1 | Pedestrian | Included |
| 6 | Non-motorized vehicle | Excluded |
| 7 | Static person | Excluded |
| 11 | Full occluder | Excluded |
| 13 | Crowd region | Excluded |

Excluded annotations remain in the original files and are reported in the
validation statistics.

## Validation

`clean_mot20.py` validates:

- Sequence metadata
- Image directory existence
- Declared and actual frame counts
- Missing and extra frames
- Frame filename format
- Ground-truth field count
- Numeric ground-truth values
- Frame references
- Positive track IDs
- Positive bounding-box sizes
- Mark and class values
- Visibility range
- Duplicate frame-ID pairs
- Boxes completely outside the image

Bounding boxes that cross the image boundary are recorded but are not
automatically rejected because partially cropped people can be valid.

## Running the Validator

From the root of the `collapse-detection` project:

```bash
python3 dataset/mot20/clean_mot20.py \
  --dataset-root "$HOME/Desktop/senior-project-storage/datasets/MOT20" \
  --output-dir "dataset/mot20"
```

The script generates:

```text
dataset/mot20/sequence_manifest.csv
dataset/mot20/validation_statistics.json
```

## Validation Results

| Statistic | Result |
|---|---:|
| Sequences | 8 |
| Valid sequences | 8 |
| Invalid sequences | 0 |
| Total frames | 13,410 |
| Training ground-truth rows | 1,336,920 |
| Active pedestrian boxes | 1,134,614 |

## Selected Sequences

| Sequence | Intended use |
|---|---|
| `MOT20-01` | Initial smoke test |
| `MOT20-02` | Long indoor crowd test |
| `MOT20-03` | High-density nighttime test |
| `MOT20-05` | Maximum crowd stress test |
| Test sequences | Tracker testing without local ground truth |

Development should begin with `MOT20-01` because it contains only 429 frames.
After basic integration works, `MOT20-03` and `MOT20-05` should be used as
high-density stress tests.

## Tracker-Ready Paths

`sequence_manifest.csv` stores paths using the portable root:

```text
${MOT20_ROOT}
```

Set it before running tracking code:

```bash
export MOT20_ROOT="$HOME/Desktop/senior-project-storage/datasets/MOT20"
```

Example image path:

```text
${MOT20_ROOT}/train/MOT20-01/img1
```

This prevents personal absolute paths from being committed to GitHub.

## Limitations

- MOT20 evaluates pedestrian tracking, not collapse classification.
- Test-sequence ground truth is not used for local validation.
- Only class 1 pedestrians are selected as tracking targets.
- Dataset images should not be uploaded to GitHub unless redistribution
  permission has been confirmed.