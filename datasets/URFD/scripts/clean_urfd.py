from pathlib import Path
import pandas as pd
from PIL import Image

# =========================================================
# 1. Project folders
# =========================================================

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent

rgb_root = project_folder / "raw" / "RGB"
sync_root = project_folder / "raw" / "metadata" / "synchronization"
labels_root = project_folder / "raw" / "metadata" / "frame_labels" / "cam0"
cleaned_folder = project_folder / "cleaned"

cleaned_folder.mkdir(exist_ok=True)

print("Looking for RGB folder here:")
print(rgb_root)

# =========================================================
# 2. Find all RGB images
# =========================================================

image_files = sorted(rgb_root.rglob("*.png"))

print("\nTotal RGB images found:", len(image_files))

# =========================================================
# 3. Parse filenames and validate images
# =========================================================

rows = []

for image_path in image_files:
    filename = image_path.stem
    parts = filename.split("-")

    class_name = parts[0]
    sequence_number = parts[1]
    camera = parts[2]
    frame_number = int(parts[-1])

    try:
        with Image.open(image_path) as img:
            img.verify()
        valid = True
    except Exception:
        valid = False

    rows.append({
        "class": class_name,
        "sequence": sequence_number,
        "camera": camera,
        "frame": frame_number,
        "image_path": str(image_path.relative_to(project_folder)),
        "valid": valid
    })

df = pd.DataFrame(rows)

# Standardize ADL label for this project
df["class"] = df["class"].replace({
    "adl": "non_fall"
})

print("\nFirst 5 rows:")
print(df.head())

print("\nRows in table:", len(df))

# =========================================================
# 4. Image validation summary
# =========================================================

print("\nImage validation summary:")
print("Valid images:", df["valid"].sum())
print("Invalid images:", (~df["valid"]).sum())

# =========================================================
# 5. Check frame continuity
# =========================================================

print("\nChecking for missing frames...")

grouped = df.groupby(
    ["class", "sequence", "camera"]
)

total_groups = len(grouped)
groups_with_missing = 0

for (class_name, sequence, camera), group in grouped:
    frames = sorted(group["frame"].tolist())

    expected_frames = list(
        range(frames[0], frames[-1] + 1)
    )

    missing_frames = sorted(
        set(expected_frames) - set(frames)
    )

    if missing_frames:
        groups_with_missing += 1

        print(
            f"{class_name} {sequence} {camera} "
            f"has missing frames: {missing_frames}"
        )

print("\nFrame continuity summary:")
print("Sequence/camera groups checked:", total_groups)
print("Groups with missing frames:", groups_with_missing)
print("Complete groups:", total_groups - groups_with_missing)

# =========================================================
# 6. Build original URFD sequence names
# =========================================================

# Important:
# non_fall folders still use original URFD names like adl-01
# so sequence_name must preserve the original dataset naming

df["sequence_name"] = df.apply(
    lambda row:
        f"fall-{row['sequence']}"
        if row["class"] == "fall"
        else f"adl-{row['sequence']}",
    axis=1
)

# =========================================================
# 7. Load synchronization CSVs
# =========================================================

sync_files = sorted(sync_root.rglob("*.csv"))

print("\nSynchronization CSV files found:", len(sync_files))

all_sync_rows = []

for sync_file in sync_files:
    sync_df = pd.read_csv(
        sync_file,
        header=None,
        names=[
            "frame",
            "timestamp_ms",
            "accelerometer"
        ]
    )

    sequence_name = sync_file.stem.replace(
        "-data",
        ""
    )

    for _, row in sync_df.iterrows():
        all_sync_rows.append({
            "sequence_name": sequence_name,
            "frame": int(row["frame"]),
            "timestamp_ms": row["timestamp_ms"],
            "accelerometer": row["accelerometer"]
        })

sync_all_df = pd.DataFrame(all_sync_rows)

print("\nTotal synchronization rows:", len(sync_all_df))

# =========================================================
# 8. Match synchronization data with cam0
# =========================================================

cam0_df = df[df["camera"] == "cam0"].copy()

cam0_with_sync = cam0_df.merge(
    sync_all_df,
    on=["sequence_name", "frame"],
    how="left"
)

print("\nSynchronization matching summary:")
print("Cam0 frames:", len(cam0_df))

print(
    "Cam0 frames with timestamps:",
    cam0_with_sync["timestamp_ms"].notna().sum()
)

print(
    "Cam0 frames without timestamps:",
    cam0_with_sync["timestamp_ms"].isna().sum()
)

# =========================================================
# 9. Load frame-level posture label CSVs
# =========================================================

falls_labels_file = (
    labels_root / "urfall-cam0-falls.csv"
)

adls_labels_file = (
    labels_root / "urfall-cam0-adls.csv"
)

falls_labels_df = pd.read_csv(
    falls_labels_file,
    header=None
)

adls_labels_df = pd.read_csv(
    adls_labels_file,
    header=None
)

# Keep only:
# sequence name
# frame number
# posture label

falls_labels_clean = falls_labels_df.iloc[:, :3].copy()
falls_labels_clean.columns = [
    "sequence_name",
    "frame",
    "posture_label"
]

adls_labels_clean = adls_labels_df.iloc[:, :3].copy()
adls_labels_clean.columns = [
    "sequence_name",
    "frame",
    "posture_label"
]

labels_all_df = pd.concat(
    [
        falls_labels_clean,
        adls_labels_clean
    ],
    ignore_index=True
)

labels_all_df["frame"] = (
    labels_all_df["frame"].astype(int)
)

print(
    "\nTotal frame-label rows in original label files:",
    len(labels_all_df)
)

# =========================================================
# 10. Check duplicate posture labels
# =========================================================

duplicate_labels = labels_all_df.duplicated(
    subset=["sequence_name", "frame"],
    keep=False
)

print(
    "Duplicate sequence/frame label rows:",
    duplicate_labels.sum()
)

# =========================================================
# 11. Match posture labels to cam0 RGB frames
# =========================================================

cam0_complete = cam0_with_sync.merge(
    labels_all_df,
    on=["sequence_name", "frame"],
    how="left"
)

print("\nFrame-label matching summary:")

print(
    "Cam0 RGB frames:",
    len(cam0_complete)
)

print(
    "Cam0 frames with posture labels:",
    cam0_complete["posture_label"]
    .notna()
    .sum()
)

print(
    "Cam0 frames without posture labels:",
    cam0_complete["posture_label"]
    .isna()
    .sum()
)

# =========================================================
# 12. Count original labels not used in selected RGB subset
# =========================================================

selected_keys = cam0_df[
    ["sequence_name", "frame"]
].drop_duplicates()

labels_check = labels_all_df.merge(
    selected_keys,
    on=["sequence_name", "frame"],
    how="left",
    indicator=True
)

unused_labels = (
    labels_check["_merge"] == "left_only"
).sum()

print(
    "Original label rows not used in selected RGB subset:",
    unused_labels
)

# =========================================================
# 13. Investigate downloaded frames with missing posture labels
# =========================================================

missing_labels = cam0_complete[
    cam0_complete["posture_label"].isna()
].copy()

print("\nDownloaded cam0 frames without posture labels:")

print(
    missing_labels[
        [
            "sequence_name",
            "frame",
            "camera",
            "image_path"
        ]
    ].head(20)
)

print("\nMissing posture labels by sequence:")

missing_by_sequence = (
    missing_labels
    .groupby("sequence_name")
    .size()
    .sort_values(ascending=False)
)

print(missing_by_sequence)

print(
    "\nNumber of sequences affected:",
    missing_labels["sequence_name"].nunique()
)

# =========================================================
# 14. Prepare cleaned cam0 data
# =========================================================

cam0_complete["has_posture_label"] = (
    cam0_complete["posture_label"].notna()
)

cleaned_cam0 = cam0_complete[
    [
        "sequence_name",
        "class",
        "camera",
        "frame",
        "timestamp_ms",
        "posture_label",
        "has_posture_label",
        "image_path",
        "valid"
    ]
].copy()

# =========================================================
# 15. Prepare cam1 data
# =========================================================

cam1_df = df[
    df["camera"] == "cam1"
].copy()

# cam1 does not have the same verified
# frame-level posture annotation mapping yet

cam1_df["timestamp_ms"] = pd.NA
cam1_df["posture_label"] = pd.NA
cam1_df["has_posture_label"] = False

cleaned_cam1 = cam1_df[
    [
        "sequence_name",
        "class",
        "camera",
        "frame",
        "timestamp_ms",
        "posture_label",
        "has_posture_label",
        "image_path",
        "valid"
    ]
].copy()

# =========================================================
# 16. Combine cam0 + cam1
# =========================================================

cleaned_manifest = pd.concat(
    [
        cleaned_cam0,
        cleaned_cam1
    ],
    ignore_index=True
)

cleaned_manifest = cleaned_manifest.sort_values(
    by=[
        "class",
        "sequence_name",
        "camera",
        "frame"
    ]
)

# =========================================================
# 17. Save cleaned manifest
# =========================================================

manifest_path = (
    cleaned_folder / "cleaned_manifest.csv"
)

cleaned_manifest.to_csv(
    manifest_path,
    index=False
)

print("\nCleaned manifest saved to:")
print(manifest_path)

print(
    "\nCleaned manifest rows:",
    len(cleaned_manifest)
)

# =========================================================
# 18. Final cleaning statistics
# =========================================================

print("\n================================")
print("FINAL CLEANING SUMMARY")
print("================================")

print(
    "Total RGB frames:",
    len(df)
)

print(
    "Valid RGB frames:",
    df["valid"].sum()
)

print(
    "Invalid RGB frames:",
    (~df["valid"]).sum()
)

print(
    "Sequence/camera groups:",
    total_groups
)

print(
    "Groups with missing frames:",
    groups_with_missing
)

print(
    "Cam0 frames:",
    len(cleaned_cam0)
)

print(
    "Cam1 frames:",
    len(cleaned_cam1)
)

print(
    "Cam0 frames with posture labels:",
    cleaned_cam0["has_posture_label"].sum()
)

print(
    "Cam0 frames without posture labels:",
    (~cleaned_cam0["has_posture_label"]).sum()
)

print(
    "Original label rows excluded because RGB was not selected:",
    unused_labels
)

print(
    "Final manifest rows:",
    len(cleaned_manifest)
)

print("================================")

# =========================================================
# 19. Save cleaning statistics
# =========================================================

statistics_path = cleaned_folder / "cleaning_statistics.txt"

with open(statistics_path, "w") as file:
    file.write("URFD CLEANING STATISTICS\n")
    file.write("========================\n\n")

    file.write(f"Total RGB frames: {len(df)}\n")
    file.write(f"Valid RGB frames: {df['valid'].sum()}\n")
    file.write(f"Invalid RGB frames: {(~df['valid']).sum()}\n\n")

    file.write(f"Sequence/camera groups checked: {total_groups}\n")
    file.write(f"Groups with missing frames: {groups_with_missing}\n")
    file.write(
        f"Complete groups: {total_groups - groups_with_missing}\n\n"
    )

    file.write(f"Cam0 frames: {len(cleaned_cam0)}\n")
    file.write(f"Cam1 frames: {len(cleaned_cam1)}\n\n")

    file.write(
        f"Cam0 frames with timestamps: "
        f"{cam0_with_sync['timestamp_ms'].notna().sum()}\n"
    )

    file.write(
        f"Cam0 frames without timestamps: "
        f"{cam0_with_sync['timestamp_ms'].isna().sum()}\n\n"
    )

    file.write(
        f"Cam0 frames with posture labels: "
        f"{cleaned_cam0['has_posture_label'].sum()}\n"
    )

    file.write(
        f"Cam0 frames without posture labels: "
        f"{(~cleaned_cam0['has_posture_label']).sum()}\n\n"
    )

    file.write(
        f"Original label rows excluded because RGB was not selected: "
        f"{unused_labels}\n"
    )

    file.write(
        f"Final cleaned manifest rows: {len(cleaned_manifest)}\n"
    )

print("\nCleaning statistics saved to:")
print(statistics_path)