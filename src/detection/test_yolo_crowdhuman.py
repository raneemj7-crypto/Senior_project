from pathlib import Path

from ultralytics import YOLO


# ============================================================
# YOLO Person Detection Baseline - CrowdHuman
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CrowdHuman_val"
    / "Images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "images"
    / "yolo_baseline"
)


def main():

    print("=" * 60)
    print("YOLO Person Detection Baseline")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load pretrained model
    # --------------------------------------------------------

    print("\nLoading YOLO model...")

    model = YOLO("yolo26n.pt")

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # 2. Select a small number of CrowdHuman images
    # --------------------------------------------------------

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )[:5]

    print(
        f"\nTesting on {len(image_files)} CrowdHuman images..."
    )

    if not image_files:
        print("ERROR: No images found.")
        return

    # --------------------------------------------------------
    # 3. Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 4. Run detection
    # --------------------------------------------------------

    results = model.predict(
        source=[str(path) for path in image_files],
        classes=[0],          # COCO class 0 = person
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        exist_ok=True,
        verbose=True
    )

    # --------------------------------------------------------
    # 5. Print detection summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)

    for image_path, result in zip(
        image_files,
        results
    ):

        if result.boxes is None:
            count = 0
        else:
            count = len(result.boxes)

        print(
            f"{image_path.name}: "
            f"{count} people detected"
        )

    print("\nBaseline test completed.")
    print(
        f"Results saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()