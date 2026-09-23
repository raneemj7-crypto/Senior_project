import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np


# ============================================================
# CrowdHuman Validation Dataset Inspection
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "CrowdHuman_val" / "Images"
ANNOTATION_FILE = PROJECT_ROOT / "data" / "raw" / "annotation_val.odgt"


def get_image_files():
    """Return all supported image files in the dataset."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    return [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]


def load_annotations():
    """Read CrowdHuman .odgt annotation file."""
    annotations = []

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                annotations.append(record)

            except json.JSONDecodeError:
                print(
                    f"WARNING: Invalid JSON on line {line_number}"
                )

    return annotations


def read_image(image_path):
    """
    Read an image using NumPy + OpenCV.

    np.fromfile() + cv2.imdecode() is used instead of
    cv2.imread() because the Windows user path contains
    non-ASCII characters.
    """

    try:
        image_data = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_data,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:
        return None


def main():

    print("=" * 60)
    print("CrowdHuman Validation Dataset Inspection")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Check paths
    # --------------------------------------------------------

    print("\n[1] Checking paths...")

    print(f"Image directory:     {IMAGE_DIR}")
    print(f"Annotation file:     {ANNOTATION_FILE}")

    if not IMAGE_DIR.exists():
        print("\nERROR: Image directory was not found.")
        return

    if not ANNOTATION_FILE.exists():
        print("\nERROR: Annotation file was not found.")
        return

    print("Paths: OK")

    # --------------------------------------------------------
    # 2. Find images
    # --------------------------------------------------------

    print("\n[2] Checking images...")

    image_files = get_image_files()

    print(f"Total image files found: {len(image_files)}")

    if len(image_files) == 0:
        print("ERROR: No images were found.")
        return

    # Create a dictionary:
    # image ID -> image path
    #
    # This makes image lookup much faster.

    image_map = {
        path.stem: path
        for path in image_files
    }

    # --------------------------------------------------------
    # 3. Load annotations
    # --------------------------------------------------------

    print("\n[3] Reading annotations...")

    annotations = load_annotations()

    print(
        f"Annotation records found: {len(annotations)}"
    )

    annotation_ids = {
        str(record["ID"])
        for record in annotations
        if "ID" in record
    }

    # --------------------------------------------------------
    # 4. Compare images and annotations
    # --------------------------------------------------------

    print("\n[4] Comparing images and annotations...")

    image_ids = set(image_map.keys())

    images_without_annotations = (
        image_ids - annotation_ids
    )

    annotations_without_images = (
        annotation_ids - image_ids
    )

    print(
        f"Images without matching annotations: "
        f"{len(images_without_annotations)}"
    )

    print(
        f"Annotations without matching images: "
        f"{len(annotations_without_images)}"
    )

    # --------------------------------------------------------
    # 5. Analyze bounding boxes
    # --------------------------------------------------------

    print("\n[5] Checking bounding boxes...")

    total_boxes = 0
    person_boxes = 0
    invalid_boxes = 0
    unreadable_images = 0

    tag_counter = Counter()

    for record in annotations:

        record_id = str(record.get("ID", ""))

        # Find corresponding image
        image_path = image_map.get(record_id)

        if image_path is None:
            continue

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = read_image(image_path)

        if image is None:
            unreadable_images += 1
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # Read ground-truth boxes
        # ----------------------------------------------------

        gtboxes = record.get("gtboxes", [])

        for gtbox in gtboxes:

            total_boxes += 1

            tag = gtbox.get(
                "tag",
                "unknown"
            )

            tag_counter[tag] += 1

            # Count person boxes
            if tag == "person":
                person_boxes += 1

            # Get full-body bounding box
            fbox = gtbox.get("fbox")

            if not fbox or len(fbox) != 4:
                invalid_boxes += 1
                continue

            x, y, width, height = fbox

            # ------------------------------------------------
            # Check dimensions
            # ------------------------------------------------

            if width <= 0 or height <= 0:
                invalid_boxes += 1
                continue

            # ------------------------------------------------
            # Check whether box is completely outside image
            # ------------------------------------------------

            if (
                x + width <= 0
                or y + height <= 0
                or x >= image_width
                or y >= image_height
            ):
                invalid_boxes += 1

    # --------------------------------------------------------
    # 6. Annotation categories
    # --------------------------------------------------------

    print("\n[6] Annotation categories:")

    for tag, count in tag_counter.items():
        print(f"  {tag}: {count}")

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INSPECTION SUMMARY")
    print("=" * 60)

    print(
        f"Images:                 {len(image_files)}"
    )

    print(
        f"Annotation records:     {len(annotations)}"
    )

    print(
        f"Person boxes:           {person_boxes}"
    )

    print(
        f"Total bounding boxes:   {total_boxes}"
    )

    print(
        f"Invalid boxes:          {invalid_boxes}"
    )

    print(
        f"Unreadable images:      {unreadable_images}"
    )

    print(
        f"Images without annotations: "
        f"{len(images_without_annotations)}"
    )

    print(
        f"Annotations without images: "
        f"{len(annotations_without_images)}"
    )

    print("\nInspection completed.")
    print("No files were modified or deleted.")


if __name__ == "__main__":
    main()