import json
import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CrowdHuman Ground-Truth Visualization
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CrowdHuman_val"
    / "Images"
)

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annotation_val.odgt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "images"
)


def read_image(image_path):
    """
    Read image using NumPy + OpenCV.

    This method works correctly with the Arabic
    characters in the Windows username/path.
    """

    image_data = np.fromfile(
        str(image_path),
        dtype=np.uint8
    )

    return cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR
    )


def load_annotations():
    """Load CrowdHuman annotations."""

    annotations = []

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                annotations.append(
                    json.loads(line)
                )

    return annotations


def main():

    print("=" * 60)
    print("CrowdHuman Ground-Truth Visualization")
    print("=" * 60)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    print("\nLoading annotations...")

    annotations = load_annotations()

    print(
        f"Loaded {len(annotations)} annotation records."
    )

    # --------------------------------------------------------
    # Select 5 random images
    # --------------------------------------------------------

    sample = random.sample(
        annotations,
        min(5, len(annotations))
    )

    print("\nVisualizing 5 random images...")

    # --------------------------------------------------------
    # Process selected images
    # --------------------------------------------------------

    for index, record in enumerate(sample, start=1):

        image_id = str(record["ID"])

        image_path = IMAGE_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            print(
                f"WARNING: Image not found: {image_id}"
            )
            continue

        image = read_image(image_path)

        if image is None:
            print(
                f"WARNING: Could not read: {image_id}"
            )
            continue

        person_count = 0

        # ----------------------------------------------------
        # Draw person bounding boxes
        # ----------------------------------------------------

        for gtbox in record.get("gtboxes", []):

            # Only visualize actual person annotations.
            if gtbox.get("tag") != "person":
                continue

            fbox = gtbox.get("fbox")

            if not fbox or len(fbox) != 4:
                continue

            x, y, width, height = map(
                int,
                fbox
            )

            # Draw bounding box
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            person_count += 1

        # ----------------------------------------------------
        # Add information to image
        # ----------------------------------------------------

        cv2.putText(
            image,
            f"Persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"crowdhuman_sample_{index}.jpg"
        )

        # Use imencode + tofile for Arabic-path compatibility.
        success, encoded_image = cv2.imencode(
            ".jpg",
            image
        )

        if success:
            encoded_image.tofile(
                str(output_path)
            )

            print(
                f"Saved: {output_path.name} "
                f"({person_count} persons)"
            )

    print("\n" + "=" * 60)
    print("Visualization completed.")
    print(f"Images saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()