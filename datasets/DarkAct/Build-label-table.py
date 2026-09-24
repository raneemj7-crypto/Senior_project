import os
import re
import csv
from collections import defaultdict, Counter

# Only this one path should need changing.
ARIM_ROOT = r"C:\Users\USER\Desktop\SE499\darkact_datasets\ARIM_v1"

ANNOTATIONS_DIR = os.path.join(ARIM_ROOT, "annotations")
OUTPUT_CSV = os.path.join(ARIM_ROOT, "darkact_label_table.csv")

# The classes that could look like a fall/collapse in a single frame, but aren't.
HARD_NEGATIVE_CLASSES = {"sit", "squat", "crouch", "pick", "lift"}


def extract_id(path):
    """e.g. 'Thermal/pour/pour_250712122330_00063_inf.mp4' -> '250712122330_00063'"""
    match = re.search(r"(\d{12}_\d{5})", path)
    return match.group(1) if match else None


def load_multimodal(path):
    """Each line: RGB_path Thermal_path class_id"""
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            rgb_path, thermal_path, class_id = parts
            entries.append((rgb_path, thermal_path, class_id))
    return entries


def build_rows(entries, split_name, id_to_names):
    rows = []
    for rgb_path, thermal_path, class_id in entries:
        video_id = extract_id(rgb_path)
        class_name = rgb_path.split("/")[1]  # the action folder name, e.g. "sit"
        id_to_names[class_id].add(class_name)
        rows.append({
            "video_id": video_id,
            "rgb_path": rgb_path,
            "thermal_path": thermal_path,
            "class_id": class_id,
            "class_name": class_name,
            "split": split_name,
            "is_hard_negative": class_name in HARD_NEGATIVE_CLASSES,
        })
    return rows


def main():
    train_entries = load_multimodal(os.path.join(ANNOTATIONS_DIR, "_multimodal_train1_annotations.txt"))
    test_entries = load_multimodal(os.path.join(ANNOTATIONS_DIR, "_multimodal_test1_annotations.txt"))
    print(f"Loaded {len(train_entries)} train entries, {len(test_entries)} test entries")

    id_to_names = defaultdict(set)
    rows = build_rows(train_entries, "train", id_to_names) + build_rows(test_entries, "test", id_to_names)

    # sanity check: does every class_id consistently mean the same class_name?
    for cid, names in id_to_names.items():
        if len(names) > 1:
            print(f"  WARNING: class_id {cid} maps to multiple different names: {names}")

    fieldnames = ["video_id", "rgb_path", "thermal_path", "class_id", "class_name", "split", "is_hard_negative"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to:\n  {OUTPUT_CSV}")

    print("\nClass balance (train / test):")
    train_counts = Counter(r["class_name"] for r in rows if r["split"] == "train")
    test_counts = Counter(r["class_name"] for r in rows if r["split"] == "test")
    for cname in sorted(set(train_counts) | set(test_counts)):
        flag = "  <-- hard negative" if cname in HARD_NEGATIVE_CLASSES else ""
        print(f"  {cname:20s} train={train_counts[cname]:4d}  test={test_counts[cname]:4d}{flag}")

    n_hard_neg = sum(1 for r in rows if r["is_hard_negative"])
    print(f"\nHard-negative clips total: {n_hard_neg} out of {len(rows)}")


if __name__ == "__main__":
    main()