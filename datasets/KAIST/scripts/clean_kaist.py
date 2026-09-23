from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import csv
import shutil

# --------------------------------
# PATHS
# --------------------------------

ROOT = Path(__file__).resolve().parent.parent

THERMAL = ROOT / "images" / "set04" / "V001" / "lwir"
ANNOTATIONS = ROOT / "annotations-xml-new-sanitized" / "set04" / "V001"

OUTPUT = ROOT / "cleaned"
CLEAN_IMAGES = OUTPUT / "thermal_with_people"

OUTPUT.mkdir(exist_ok=True)
CLEAN_IMAGES.mkdir(exist_ok=True)

rows = []

thermal_files = sorted(THERMAL.glob("*.jpg"))

print("Cleaning KAIST - THERMAL ONLY")
print("Total thermal frames:", len(thermal_files))
print()


# --------------------------------
# CHECK EVERY THERMAL FRAME
# --------------------------------

for i, thermal_path in enumerate(thermal_files):

    name = thermal_path.stem
    annotation_path = ANNOTATIONS / f"{name}.xml"

    status = "keep"
    issue_code = "none"
    notes = "none"
    people_count = 0

    width = 0
    height = 0

    # Check thermal image
    try:
        with Image.open(thermal_path) as img:
            width, height = img.size
            img.verify()

    except Exception:
        status = "exclude"
        issue_code = "E-FILE-CORRUPT"
        notes = "Thermal image could not be opened"

    # Check annotation
    if not annotation_path.exists():

        status = "exclude"
        issue_code = "E-NO-ANNOT"
        notes = "Annotation file missing"

    else:

        try:
            tree = ET.parse(annotation_path)
            xml_root = tree.getroot()

            objects = xml_root.findall(".//object")

            # Count pedestrian/person annotations
            for obj in objects:

                label = obj.findtext("name", "").strip().lower()

                if label in ["person", "pedestrian"]:
                    people_count += 1

                    box = obj.find("bndbox")

                    if box is not None:

                        x = float(box.findtext("x"))
                        y = float(box.findtext("y"))
                        w = float(box.findtext("w"))
                        h = float(box.findtext("h"))

                        # Invalid bounding box
                        if w <= 0 or h <= 0:
                            status = "exclude"
                            issue_code = "E-LABEL-UNMAPPABLE"
                            notes = "Invalid pedestrian bounding box"

            # No people = remove from cleaned sample
            if people_count == 0 and status != "exclude":

                status = "exclude"
                issue_code = "E-NO-PERSON"
                notes = "No pedestrian in frame"

        except Exception as e:

            status = "exclude"
            issue_code = "E-NO-ANNOT"
            notes = f"Annotation error: {e}"

    # --------------------------------
    # COPY ONLY GOOD THERMAL FRAMES
    # --------------------------------

    if status == "keep":

        destination = CLEAN_IMAGES / thermal_path.name

        shutil.copy2(
            thermal_path,
            destination
        )

    # --------------------------------
    # MANIFEST
    # --------------------------------

    rows.append({

        "sample_id": f"kaist_set04v001_{i:06d}",
        "dataset": "kaist",
        "role": "B+D",
        "modality": "thermal",
        "path_rgb": "none",

        "path_thermal": (
            f"cleaned/thermal_with_people/{thermal_path.name}"
            if status == "keep"
            else str(thermal_path.relative_to(ROOT))
        ),

        "sequence_id": "set04v001",
        "subject_id": "none",
        "frame_index": i,

        "original_label": (
            "person"
            if people_count > 0
            else "none"
        ),

        "project_label": (
            "person"
            if people_count > 0
            else "none"
        ),

        "status": status,
        "issue_code": issue_code,
        "split": "none",
        "notes": notes
    })


# --------------------------------
# SAVE MANIFEST
# --------------------------------

columns = [
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
    "notes"
]

manifest_path = OUTPUT / "cleaned_manifest.csv"

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------
# FINAL COUNTS
# --------------------------------

total = len(rows)

kept = sum(
    r["status"] == "keep"
    for r in rows
)

removed_empty = sum(
    r["issue_code"] == "E-NO-PERSON"
    for r in rows
)

other_excluded = sum(
    r["status"] == "exclude"
    and r["issue_code"] != "E-NO-PERSON"
    for r in rows
)


# --------------------------------
# SAVE STATISTICS
# --------------------------------

statistics_path = OUTPUT / "cleaning_statistics.txt"

with open(
    statistics_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("KAIST THERMAL CLEANING\n")
    f.write("----------------------\n")
    f.write(f"Original thermal frames: {total}\n")
    f.write(f"Frames kept with people: {kept}\n")
    f.write(f"Empty frames removed: {removed_empty}\n")
    f.write(f"Other excluded frames: {other_excluded}\n")


# --------------------------------
# RESULTS
# --------------------------------

print("DONE!")
print()
print("Original thermal frames:", total)
print("Frames WITH people kept:", kept)
print("Empty frames removed:", removed_empty)
print("Other excluded:", other_excluded)
print()
print("Clean thermal images saved in:")
print(CLEAN_IMAGES)