import argparse
import configparser
import csv
import json
import hashlib
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError

MOT20_CLASS_NAMES = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_motorized_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
    13: "crowd",
}

TARGET_CLASS_ID = 1
PROJECT_LABEL = "person"

CLEAN_MANIFEST_FIELDS = [
    "sample_id",
    "dataset",
    "role",
    "modality",
    "path_rgb",
    "path_thermal",
    "sequence_id",
    "subject_id",
    "frame_index",
    "original_label",
    "project_label",
    "status",
    "issue_code",
    "split",
    "notes",
]

def map_mot20_annotation(mark, class_id):
    """Map one MOT20 annotation to the project label system."""
    original_label = MOT20_CLASS_NAMES.get(class_id)

    if original_label is None:
        return {
            "original_label": "unknown",
            "project_label": "none",
            "status": "exclude",
            "issue_code": "E-LABEL-UNMAPPABLE",
        }

    if mark == 1 and class_id == TARGET_CLASS_ID:
        return {
            "original_label": original_label,
            "project_label": PROJECT_LABEL,
            "status": "fix",
            "issue_code": "F-LABEL-NORM",
        }

    return {
        "original_label": original_label,
        "project_label": "none",
        "status": "exclude",
        "issue_code": "none",
    }

def format_mot_number(value):
    """Format a MOT numeric value without unnecessary trailing zeros."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_clean_ground_truth(sequence_path, destination_path, info):
    """Write target-pedestrian rows to a cleaned MOT ground-truth file."""
    source_path = sequence_path / "gt" / "gt.txt"

    counts = {
        "kept_rows": 0,
        "excluded_rows": 0,
        "clipped_boxes": 0,
    }

    if not source_path.exists():
        return counts

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source_file, destination_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination_file:
        reader = csv.reader(source_file)
        writer = csv.writer(
            destination_file,
            lineterminator="\n",
        )

        for row in reader:
            if len(row) != 9:
                counts["excluded_rows"] += 1
                continue

            try:
                x = float(row[2])
                y = float(row[3])
                box_width = float(row[4])
                box_height = float(row[5])
                mark = int(float(row[6]))
                class_id = int(float(row[7]))
            except ValueError:
                counts["excluded_rows"] += 1
                continue

            decision = map_mot20_annotation(mark, class_id)

            if decision["project_label"] == PROJECT_LABEL:
                right = x + box_width - 1
                bottom = y + box_height - 1
                clipped_left = max(1.0, x)
                clipped_top = max(1.0, y)
                clipped_right = min(float(info["width"]), right)
                clipped_bottom = min(float(info["height"]), bottom)
                clipped_width = clipped_right - clipped_left + 1
                clipped_height = clipped_bottom - clipped_top + 1

                if clipped_width <= 0 or clipped_height <= 0:
                    counts["excluded_rows"] += 1
                    continue

                if (
                    clipped_left != x
                    or clipped_top != y
                    or clipped_right != right
                    or clipped_bottom != bottom
                ):
                    row[2] = format_mot_number(clipped_left)
                    row[3] = format_mot_number(clipped_top)
                    row[4] = format_mot_number(clipped_width)
                    row[5] = format_mot_number(clipped_height)
                    counts["clipped_boxes"] += 1

                writer.writerow(row)
                counts["kept_rows"] += 1
            else:
                counts["excluded_rows"] += 1

    return counts

def copy_processed_images(
    sequence_path,
    split,
    info,
    processed_root,
):
    """Copy validated images into the standardized processed structure."""
    source_directory = sequence_path / info["image_directory"]

    destination_directory = (
        processed_root
        / "images"
        / split
        / sequence_path.name
    )
    destination_directory.mkdir(parents=True, exist_ok=True)

    image_count = 0
    copied_count = 0

    for source_path in sorted(source_directory.iterdir()):
        if (
            not source_path.is_file()
            or source_path.suffix.lower()
            != info["image_extension"].lower()
        ):
            continue

        destination_path = destination_directory / source_path.name
        image_count += 1

        if (
            not destination_path.exists()
            or destination_path.stat().st_size
            != source_path.stat().st_size
        ):
            shutil.copy2(source_path, destination_path)
            copied_count += 1

    return {
        "image_count": image_count,
        "copied_count": copied_count,
        "destination": destination_directory,
    }


def get_target_frame_info(sequence_path, info):
    """Return frames containing target pedestrians and boxes needing clips."""
    gt_path = sequence_path / "gt" / "gt.txt"
    target_frames = set()
    clipped_frames = set()

    if not gt_path.exists():
        return target_frames, clipped_frames

    with gt_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if len(row) != 9:
                continue

            try:
                frame = int(float(row[0]))
                x = float(row[2])
                y = float(row[3])
                width = float(row[4])
                height = float(row[5])
                mark = int(float(row[6]))
                class_id = int(float(row[7]))
            except ValueError:
                continue

            decision = map_mot20_annotation(mark, class_id)
            if decision["project_label"] != PROJECT_LABEL:
                continue

            target_frames.add(frame)
            right = x + width - 1
            bottom = y + height - 1
            if (
                x < 1
                or y < 1
                or right > info["width"]
                or bottom > info["height"]
            ):
                clipped_frames.add(frame)

    return target_frames, clipped_frames


def build_clean_manifest_rows(
    dataset_root,
    sequence_results,
    processed_root,
    exact_duplicates,
):
    """Create the standardized 15-column frame-level clean manifest."""
    duplicate_paths = {
        item["duplicate"] for item in exact_duplicates
    }
    rows = []

    for item in sequence_results:
        split = item["split"]
        sequence = item["sequence"]
        sequence_path = dataset_root / split / sequence
        info = read_sequence_info(sequence_path)
        image_directory = sequence_path / info["image_directory"]
        target_frames, clipped_frames = get_target_frame_info(
            sequence_path,
            info,
        )
        corrupt_images = set(item["corrupt_images"])

        image_files = sorted(
            path
            for path in image_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == info["image_extension"].lower()
            and path.stem.isdigit()
        )

        for image_file in image_files:
            frame = int(image_file.stem)
            raw_relative_path = (
                f"{split}/{sequence}/{info['image_directory']}/"
                f"{image_file.name}"
            )

            if processed_root is not None:
                path_rgb = (
                    f"${{MOT20_PROCESSED_ROOT}}/images/{split}/"
                    f"{sequence}/{image_file.name}"
                )
            else:
                path_rgb = f"${{MOT20_ROOT}}/{raw_relative_path}"

            if image_file.name in corrupt_images:
                status = "exclude"
                issue_code = "E-FILE-CORRUPT"
                notes = "Image could not be decoded."
            elif raw_relative_path in duplicate_paths:
                status = "exclude"
                issue_code = "E-DUP-EXACT"
                notes = "Exact duplicate image; original is recorded in statistics."
            elif split == "train" and frame not in target_frames:
                status = "exclude"
                issue_code = "E-NO-ANNOT"
                notes = "Frame has no active target-pedestrian annotation."
            elif split == "train" and frame in clipped_frames:
                status = "fix"
                issue_code = "F-BOX-CLIP"
                notes = "Pedestrian box clipped to the image boundary."
            elif split == "train":
                status = "fix"
                issue_code = "F-LABEL-NORM"
                notes = "MOT20 pedestrian label normalized to person."
            else:
                status = "keep"
                issue_code = "none"
                notes = "Official test frame; local ground truth is unavailable."

            rows.append({
                "sample_id": (
                    f"mot20_{split}_{sequence}_{frame:06d}"
                ),
                "dataset": "MOT20",
                "role": "C",
                "modality": "rgb",
                "path_rgb": path_rgb,
                "path_thermal": "none",
                "sequence_id": sequence,
                "subject_id": "none",
                "frame_index": frame,
                "original_label": (
                    "pedestrian"
                    if split == "train" and frame in target_frames
                    else "none"
                ),
                "project_label": PROJECT_LABEL,
                "status": status,
                "issue_code": issue_code,
                "split": split,
                "notes": notes,
            })

    return rows


def calculate_sha256(file_path):
    """Return a SHA-256 fingerprint for a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def read_sequence_info(sequence_path):
    """Read metadata such as FPS, resolution, and expected frame count."""
    seqinfo_path = sequence_path / "seqinfo.ini"

    if not seqinfo_path.exists():
        raise FileNotFoundError(f"Missing file: {seqinfo_path}")

    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    if "Sequence" not in config:
        raise ValueError(f"Missing [Sequence] section: {seqinfo_path}")

    info = config["Sequence"]

    return {
        "name": info.get("name", sequence_path.name),
        "image_directory": info.get("imDir", "img1"),
        "fps": info.getint("frameRate"),
        "declared_frames": info.getint("seqLength"),
        "width": info.getint("imWidth"),
        "height": info.getint("imHeight"),
        "image_extension": info.get("imExt", ".jpg"),
    }


def validate_images(sequence_path, info):
    """Check image count, names, missing frames, and extra frames."""
    image_path = sequence_path / info["image_directory"]
    extension = info["image_extension"].lower()

    results = {
        "image_directory_exists": image_path.exists(),
        "actual_frames": 0,
        "missing_frames": [],
        "extra_frames": [],
        "unexpected_image_names": [],
        "corrupt_images": [],
        "dimension_mismatches": [],
        "image_hashes": {},
    }

    if not image_path.exists():
        return results

    image_files = sorted(
        path
        for path in image_path.iterdir()
        if path.is_file() and path.suffix.lower() == extension
    )

    frame_numbers = set()

    for image_file in image_files:
        try:
            frame_numbers.add(int(image_file.stem))
        except ValueError:
            results["unexpected_image_names"].append(image_file.name)

        try:
            with Image.open(image_file) as image:
                image.load()
                actual_size = image.size

            expected_size = (info["width"], info["height"])

            if actual_size != expected_size:
                results["dimension_mismatches"].append({
                    "file": image_file.name,
                    "expected": list(expected_size),
                    "actual": list(actual_size),
                })

            results["image_hashes"][image_file.name] = (
                calculate_sha256(image_file)
            )

        except (UnidentifiedImageError, OSError, ValueError):
            results["corrupt_images"].append(image_file.name)
            
            

    expected_frames = set(range(1, info["declared_frames"] + 1))

    results["actual_frames"] = len(image_files)
    results["missing_frames"] = sorted(expected_frames - frame_numbers)
    results["extra_frames"] = sorted(frame_numbers - expected_frames)

    return results


def validate_ground_truth(sequence_path, info, split):
    """Validate ground-truth rows, track IDs, boxes, and visibility."""
    gt_path = sequence_path / "gt" / "gt.txt"

    results = {
        "ground_truth_exists": gt_path.exists(),
        "total_gt_rows": 0,
        "wrong_field_count": 0,
        "non_numeric_rows": 0,
        "invalid_frame_references": 0,
        "invalid_track_ids": 0,
        "invalid_box_sizes": 0,
        "invalid_visibility_values": 0,
        "invalid_mark_values": 0,
        "invalid_class_values": 0,
        "duplicate_frame_id_pairs": 0,
        "boxes_crossing_image_boundary": 0,
        "boxes_completely_outside_image": 0,
        "total_track_ids": 0,
        "active_pedestrian_track_ids": 0,
        "active_pedestrian_boxes": 0,
        "excluded_rows": 0,
        "class_counts": {},
        "annotation_references_missing_files": 0,
        "frames_without_any_annotation": [],
        "frames_without_target_pedestrians": [],
    }
    

    # Test ground truth may be unavailable, which is not an error.
    if not gt_path.exists():
        return results

    all_track_ids = set()
    active_pedestrian_ids = set()
    seen_frame_id_pairs = set()
    class_counts = {}
    annotated_frames = set()
    target_pedestrian_frames = set()

    image_directory = sequence_path / info["image_directory"]

    available_frames = {
        int(path.stem)
        for path in image_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == info["image_extension"].lower()
        and path.stem.isdigit()
    }

    with gt_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            results["total_gt_rows"] += 1

            if len(row) != 9:
                results["wrong_field_count"] += 1
                continue

            try:
                frame = int(float(row[0]))
                track_id = int(float(row[1]))
                x = float(row[2])
                y = float(row[3])
                box_width = float(row[4])
                box_height = float(row[5])
                mark = int(float(row[6]))
                class_id = int(float(row[7]))
                visibility = float(row[8])
            except ValueError:
                results["non_numeric_rows"] += 1
                continue
            
            annotated_frames.add(frame)

            if frame not in available_frames:
                results["annotation_references_missing_files"] += 1

            all_track_ids.add(track_id)

            class_counts[str(class_id)] = (
                class_counts.get(str(class_id), 0) + 1
            )

            if frame < 1 or frame > info["declared_frames"]:
                results["invalid_frame_references"] += 1

            if track_id <= 0:
                results["invalid_track_ids"] += 1

            if box_width <= 0 or box_height <= 0:
                results["invalid_box_sizes"] += 1

            if visibility < 0 or visibility > 1:
                results["invalid_visibility_values"] += 1

            if mark not in (0, 1):
                results["invalid_mark_values"] += 1

            if class_id < 1 or class_id > 13:
                results["invalid_class_values"] += 1

            frame_id_pair = (frame, track_id)

            if frame_id_pair in seen_frame_id_pairs:
                results["duplicate_frame_id_pairs"] += 1
            else:
                seen_frame_id_pairs.add(frame_id_pair)

            right = x + box_width - 1
            bottom = y + box_height - 1

            if x < 1 or y < 1 or right > info["width"] or bottom > info["height"]:
                results["boxes_crossing_image_boundary"] += 1

            if (
                right < 1
                or bottom < 1
                or x > info["width"]
                or y > info["height"]
            ):
                results["boxes_completely_outside_image"] += 1

            if mark == 1 and class_id == 1:
                results["active_pedestrian_boxes"] += 1
                active_pedestrian_ids.add(track_id)
                target_pedestrian_frames.add(frame)
            else:
                results["excluded_rows"] += 1

    results["total_track_ids"] = len(all_track_ids)
    results["active_pedestrian_track_ids"] = len(active_pedestrian_ids)
    results["class_counts"] = class_counts

    expected_frames = set(range(1, info["declared_frames"] + 1))

    results["frames_without_any_annotation"] = sorted(
        expected_frames - annotated_frames
    )

    results["frames_without_target_pedestrians"] = sorted(
        expected_frames - target_pedestrian_frames
    )

    return results


def intended_use(sequence_name, split):
    """Describe how each sequence will be used in the project."""
    if split == "test":
        return "tracker_testing_without_local_ground_truth"

    uses = {
        "MOT20-01": "initial_smoke_test",
        "MOT20-02": "long_indoor_crowd_test",
        "MOT20-03": "high_density_night_test",
        "MOT20-05": "maximum_crowd_stress_test",
    }

    return uses.get(sequence_name, "tracking_evaluation")


def determine_status(split, image_results, gt_results):
    """Return valid only when no critical structural errors are found."""
    critical_errors = [
        len(image_results["missing_frames"]),
        len(image_results["extra_frames"]),
        len(image_results["unexpected_image_names"]),
        len(image_results["corrupt_images"]),
        len(image_results["dimension_mismatches"]),
        gt_results["wrong_field_count"],
        gt_results["non_numeric_rows"],
        gt_results["invalid_frame_references"],
        gt_results["invalid_track_ids"],
        gt_results["invalid_box_sizes"],
        gt_results["invalid_visibility_values"],
        gt_results["invalid_mark_values"],
        gt_results["invalid_class_values"],
        gt_results["duplicate_frame_id_pairs"],
        gt_results["boxes_completely_outside_image"],
        gt_results["annotation_references_missing_files"],
    ]

    if not image_results["image_directory_exists"]:
        return "invalid"

    if split == "train" and not gt_results["ground_truth_exists"]:
        return "invalid"

    if any(value > 0 for value in critical_errors):
        return "invalid"

    return "valid"

def find_exact_duplicates(sequence_results):
    """Find byte-for-byte identical images across all sequences."""
    seen_hashes = {}
    duplicates = []

    for sequence_result in sequence_results:
        split = sequence_result["split"]
        sequence = sequence_result["sequence"]

        for filename, image_hash in sequence_result.get(
            "image_hashes", {}
        ).items():
            image_path = (
                f"{split}/{sequence}/img1/{filename}"
            )

            if image_hash in seen_hashes:
                duplicates.append({
                    "original": seen_hashes[image_hash],
                    "duplicate": image_path,
                    "sha256": image_hash,
                    "issue_code": "E-DUP-EXACT",
                })
            else:
                seen_hashes[image_hash] = image_path

    return duplicates

def build_integrity_rows(sequence_results, exact_duplicates):
    """Create one integrity-report row for each MOT20 sequence."""
    duplicate_counts = {}

    for duplicate in exact_duplicates:
        path_parts = Path(duplicate["duplicate"]).parts

        if len(path_parts) >= 2:
            key = (path_parts[0], path_parts[1])
            duplicate_counts[key] = duplicate_counts.get(key, 0) + 1

    rows = []

    for item in sequence_results:
        key = (item["split"], item["sequence"])

        rows.append({
            "dataset": "MOT20",
            "split": item["split"],
            "sequence": item["sequence"],
            "status": item["status"],
            "declared_frames": item["declared_frames"],
            "actual_frames": item["actual_frames"],
            "ground_truth_exists": item["ground_truth_exists"],
            "total_gt_rows": item["total_gt_rows"],
            "missing_frames": len(item["missing_frames"]),
            "extra_frames": len(item["extra_frames"]),
            "unexpected_image_names": len(
                item["unexpected_image_names"]
            ),
            "corrupt_images": len(item["corrupt_images"]),
            "dimension_mismatches": len(
                item["dimension_mismatches"]
            ),
            "annotation_references_missing_files": item[
                "annotation_references_missing_files"
            ],
            "frames_without_any_annotation": len(
                item["frames_without_any_annotation"]
            ),
            "frames_without_target_pedestrians": len(
                item["frames_without_target_pedestrians"]
            ),
            "wrong_field_count": item["wrong_field_count"],
            "non_numeric_rows": item["non_numeric_rows"],
            "invalid_frame_references": item[
                "invalid_frame_references"
            ],
            "invalid_track_ids": item["invalid_track_ids"],
            "invalid_box_sizes": item["invalid_box_sizes"],
            "invalid_visibility_values": item[
                "invalid_visibility_values"
            ],
            "invalid_mark_values": item["invalid_mark_values"],
            "invalid_class_values": item["invalid_class_values"],
            "duplicate_frame_id_pairs": item[
                "duplicate_frame_id_pairs"
            ],
            "boxes_crossing_image_boundary": item[
                "boxes_crossing_image_boundary"
            ],
            "boxes_completely_outside_image": item[
                "boxes_completely_outside_image"
            ],
            "exact_duplicate_images": duplicate_counts.get(key, 0),
        })

    return rows


def build_cleaning_statistics(
    clean_manifest_rows,
    sequence_results,
    exact_duplicates,
):
    """Create a compact summary of cleaning decisions and outputs."""
    status_counts = {"keep": 0, "fix": 0, "exclude": 0}
    for row in clean_manifest_rows:
        status_counts[row["status"]] += 1

    return [{
        "dataset": "MOT20",
        "sequence_count": len(sequence_results),
        "total_samples": len(clean_manifest_rows),
        "kept_samples": status_counts["keep"],
        "fixed_samples": status_counts["fix"],
        "excluded_samples": status_counts["exclude"],
        "raw_annotation_rows": sum(
            item["total_gt_rows"] for item in sequence_results
        ),
        "clean_annotation_rows": sum(
            item.get(
                "cleaned_annotation_rows",
                item["active_pedestrian_boxes"],
            )
            for item in sequence_results
        ),
        "removed_annotation_rows": sum(
            item.get("removed_annotation_rows", item["excluded_rows"])
            for item in sequence_results
        ),
        "clipped_boxes": sum(
            item.get("clipped_boxes", 0)
            for item in sequence_results
        ),
        "corrupt_images": sum(
            len(item["corrupt_images"])
            for item in sequence_results
        ),
        "exact_duplicate_images": len(exact_duplicates),
    }]

def validate_sequence(dataset_root, sequence_path, split):
    """Validate one complete MOT20 sequence."""
    info = read_sequence_info(sequence_path)
    image_results = validate_images(sequence_path, info)
    gt_results = validate_ground_truth(sequence_path, info, split)

    status = determine_status(split, image_results, gt_results)
    relative_sequence = f"{split}/{sequence_path.name}"

    statistics = {
        "split": split,
        "sequence": sequence_path.name,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "declared_frames": info["declared_frames"],
        **image_results,
        **gt_results,
        "status": status,
        "intended_use": intended_use(sequence_path.name, split),
    }

    manifest_row = {
        "split": split,
        "sequence": sequence_path.name,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "frame_count": image_results["actual_frames"],
        "sequence_path": f"${{MOT20_ROOT}}/{relative_sequence}",
        "image_path": (
            f"${{MOT20_ROOT}}/{relative_sequence}/"
            f"{info['image_directory']}"
        ),
        "gt_path": (
            f"${{MOT20_ROOT}}/{relative_sequence}/gt/gt.txt"
            if gt_results["ground_truth_exists"]
            else ""
        ),
        "det_path": f"${{MOT20_ROOT}}/{relative_sequence}/det/det.txt",
        "processed_image_path": (
            f"${{MOT20_PROCESSED_ROOT}}/images/{split}/"
            f"{sequence_path.name}"
        ),
        "clean_gt_path": (
            f"${{MOT20_PROCESSED_ROOT}}/annotations/{split}/"
            f"{sequence_path.name}/gt.txt"
            if gt_results["ground_truth_exists"]
            else "none"
        ),
        "has_ground_truth": gt_results["ground_truth_exists"],
        "total_gt_rows": gt_results["total_gt_rows"],
        "total_track_ids": gt_results["total_track_ids"],
        "active_pedestrian_track_ids": (
            gt_results["active_pedestrian_track_ids"]
        ),
        "active_pedestrian_boxes": gt_results["active_pedestrian_boxes"],
        "excluded_rows": gt_results["excluded_rows"],
        "status": status,
        "intended_use": intended_use(sequence_path.name, split),
    }

    return statistics, manifest_row


def main():
    parser = argparse.ArgumentParser(
        description="Validate MOT20 and generate tracker-ready files."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted MOT20 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the generated CSV and JSON files.",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Directory for cleaned tracker-ready MOT20 data.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    processed_root = None

    if args.processed_root:
        processed_root = (
            Path(args.processed_root).expanduser().resolve()
        )
        processed_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    all_statistics = []
    manifest_rows = []

    for split in ("train", "test"):
        split_path = dataset_root / split

        if not split_path.exists():
            print(f"Warning: {split_path} was not found.")
            continue

        for sequence_path in sorted(split_path.glob("MOT20-*")):
            if not sequence_path.is_dir():
                continue

            print(f"Validating {split}/{sequence_path.name}...")

            try:
                statistics, manifest_row = validate_sequence(
                    dataset_root, sequence_path, split
                )
                all_statistics.append(statistics)
                manifest_rows.append(manifest_row)

                if processed_root is not None:
                    info = read_sequence_info(sequence_path)

                    image_output = copy_processed_images(
                        sequence_path,
                        split,
                        info,
                        processed_root,
                    )

                    clean_gt_path = (
                        processed_root
                        / "annotations"
                        / split
                        / sequence_path.name
                        / "gt.txt"
                    )

                    annotation_counts = write_clean_ground_truth(
                        sequence_path,
                        clean_gt_path,
                        info,
                    )

                    statistics["processed_image_count"] = (
                        image_output["image_count"]
                    )
                    statistics["new_images_copied"] = (
                        image_output["copied_count"]
                    )
                    statistics["cleaned_annotation_rows"] = (
                        annotation_counts["kept_rows"]
                    )
                    statistics["removed_annotation_rows"] = (
                        annotation_counts["excluded_rows"]
                    )
                    statistics["clipped_boxes"] = (
                        annotation_counts["clipped_boxes"]
                    )

                print(f"  Status: {statistics['status']}")
            except Exception as error:
                print(f"  Error: {error}")

    manifest_path = output_directory / "sequence_manifest.csv"
    clean_manifest_path = output_directory / "clean_manifest.csv"
    integrity_path = output_directory / "integrity_report.csv"
    cleaning_statistics_path = (
        output_directory / "cleaning_statistics.csv"
    )
    statistics_path = output_directory / "validation_statistics.json"

    exact_duplicates = find_exact_duplicates(all_statistics)

    integrity_rows = build_integrity_rows(
        all_statistics,
        exact_duplicates,
    )

    clean_manifest_rows = build_clean_manifest_rows(
        dataset_root,
        all_statistics,
        processed_root,
        exact_duplicates,
    )

    cleaning_statistics_rows = build_cleaning_statistics(
        clean_manifest_rows,
        all_statistics,
        exact_duplicates,
    )

    if manifest_rows:
        with manifest_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(manifest_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    if integrity_rows:
        with integrity_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(integrity_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(integrity_rows)

    if clean_manifest_rows:
        with clean_manifest_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CLEAN_MANIFEST_FIELDS,
            )
            writer.writeheader()
            writer.writerows(clean_manifest_rows)

    with cleaning_statistics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(cleaning_statistics_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(cleaning_statistics_rows)

    summary = {
        "dataset": "MOT20",
        "sequence_count": len(all_statistics),
        "valid_sequences": sum(
            item["status"] == "valid" for item in all_statistics
        ),
        "invalid_sequences": sum(
            item["status"] == "invalid" for item in all_statistics
        ),
        "total_frames": sum(
            item["actual_frames"] for item in all_statistics
        ),
        "total_ground_truth_rows": sum(
            item["total_gt_rows"] for item in all_statistics
        ),
        "active_pedestrian_boxes": sum(
            item["active_pedestrian_boxes"] for item in all_statistics
        ),
        "exact_duplicate_images": len(exact_duplicates),
    }

    output_sequences = []
    for item in all_statistics:
        clean_item = dict(item)
        clean_item.pop("image_hashes", None)
        output_sequences.append(clean_item)

    output = {
        "summary": summary,
        "exact_duplicates": exact_duplicates,
        "sequences": output_sequences,
    }

    with statistics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print("Validation finished.")
    print(f"Sequence manifest: {manifest_path}")
    print(f"Clean manifest: {clean_manifest_path}")
    print(f"Integrity report: {integrity_path}")
    print(f"Cleaning statistics: {cleaning_statistics_path}")
    print(f"Statistics: {statistics_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
