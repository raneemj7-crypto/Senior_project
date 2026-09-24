import os
import re

# This is the only thing you should need to change -- point it at ARIM_v1.
ARIM_ROOT = r"C:\Users\USER\Desktop\SE499\darkact_datasets\ARIM_v1"

ANNOTATIONS_DIR = os.path.join(ARIM_ROOT, "annotations")
VIDEOS_DIR = os.path.join(ARIM_ROOT, "videos")  # this is what RGB/... and Thermal/... paths are relative to


def extract_id(path):
    """Pulls the timestamp_clipnumber ID out of a DarkAct filename,
    e.g. 'Thermal/pour/pour_250712122330_00063_inf.mp4' -> '250712122330_00063'
    """
    match = re.search(r"(\d{12}_\d{5})", path)
    return match.group(1) if match else None


def load_multimodal(path):
    """Each line: RGB_path Thermal_path class_id"""
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                if parts:
                    print(f"  WARNING: unexpected line format: {line!r}")
                continue
            rgb_path, thermal_path, class_id = parts
            entries.append((rgb_path, thermal_path, class_id))
    return entries


def load_single(path):
    """Each line: path class_id -- used for the RGB-only / Thermal-only files"""
    d = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, cid = line.rsplit(" ", 1)
            d[extract_id(p)] = cid.strip()
    return d


def check_split(split_name, multimodal_file, thermal_only_file, rgb_only_file):
    print(f"\n=== {split_name} ===")
    entries = load_multimodal(multimodal_file)
    print(f"Total pairs listed: {len(entries)}")

    id_mismatches = []
    missing_files = []
    label_mismatches = []

    thermal_lookup = load_single(thermal_only_file)
    rgb_lookup = load_single(rgb_only_file)

    for rgb_path, thermal_path, class_id in entries:
        rgb_id = extract_id(rgb_path)
        thermal_id = extract_id(thermal_path)

        if rgb_id != thermal_id:
            id_mismatches.append((rgb_path, thermal_path))

        for p in (rgb_path, thermal_path):
            full_path = os.path.join(VIDEOS_DIR, p)
            if not os.path.exists(full_path):
                missing_files.append(full_path)

        if thermal_lookup.get(thermal_id) != class_id:
            label_mismatches.append((thermal_path, class_id, thermal_lookup.get(thermal_id)))
        if rgb_lookup.get(rgb_id) != class_id:
            label_mismatches.append((rgb_path, class_id, rgb_lookup.get(rgb_id)))

    print(f"RGB/Thermal ID mismatches within a line: {len(id_mismatches)}")
    print(f"Files listed but missing on disk: {len(missing_files)}")
    print(f"class_id disagreements with the single-modality files: {len(label_mismatches)}")

    for label, items in [("id mismatch", id_mismatches), ("missing file", missing_files), ("label mismatch", label_mismatches)]:
        if items:
            print(f"  first few {label}s:")
            for x in items[:5]:
                print("    ", x)


if __name__ == "__main__":
    # Update these filenames if explore_annotations.py told you the "1" versions
    # aren't the right ones to use.
    check_split(
        "TRAIN",
        multimodal_file=os.path.join(ANNOTATIONS_DIR, "_multimodal_train1_annotations.txt"),
        thermal_only_file=os.path.join(ANNOTATIONS_DIR, "_thermal_train1_annotations.txt"),
        rgb_only_file=os.path.join(ANNOTATIONS_DIR, "_rgb_train1_annotations.txt"),
    )
    check_split(
        "TEST",
        multimodal_file=os.path.join(ANNOTATIONS_DIR, "_multimodal_test1_annotations.txt"),
        thermal_only_file=os.path.join(ANNOTATIONS_DIR, "_thermal_test1_annotations.txt"),
        rgb_only_file=os.path.join(ANNOTATIONS_DIR, "_rgb_test1_annotations.txt"),
    )