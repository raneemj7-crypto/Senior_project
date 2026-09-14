import argparse
import configparser
import csv
import json
from pathlib import Path


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
    }

    # Test ground truth may be unavailable, which is not an error.
    if not gt_path.exists():
        return results

    all_track_ids = set()
    active_pedestrian_ids = set()
    seen_frame_id_pairs = set()
    class_counts = {}

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
            else:
                results["excluded_rows"] += 1

    results["total_track_ids"] = len(all_track_ids)
    results["active_pedestrian_track_ids"] = len(active_pedestrian_ids)
    results["class_counts"] = class_counts

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
    ]

    if not image_results["image_directory_exists"]:
        return "invalid"

    if split == "train" and not gt_results["ground_truth_exists"]:
        return "invalid"

    if any(value > 0 for value in critical_errors):
        return "invalid"

    return "valid"


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
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

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
                print(f"  Status: {statistics['status']}")
            except Exception as error:
                print(f"  Error: {error}")

    manifest_path = output_directory / "sequence_manifest.csv"
    statistics_path = output_directory / "validation_statistics.json"

    if manifest_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(manifest_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

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
    }

    output = {
        "summary": summary,
        "sequences": all_statistics,
    }

    with statistics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print("Validation finished.")
    print(f"Manifest: {manifest_path}")
    print(f"Statistics: {statistics_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()