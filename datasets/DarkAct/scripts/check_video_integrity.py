"""
Checks every video referenced in darkact_label_table.csv (both RGB and
Thermal) to confirm it actually opens and has at least one readable frame.
This catches corrupted/truncated files that "exists on disk" alone can't
catch -- check_multimodal_pairs.py only checked existence, this checks the
file actually works.
"""

import os
import csv
import cv2

ARIM_ROOT = r"C:\Users\USER\Desktop\SE499\darkact_datasets\ARIM_v1"
CSV_PATH = os.path.join(ARIM_ROOT, "darkact_label_table.csv")
VIDEOS_DIR = os.path.join(ARIM_ROOT, "videos")
REPORT_PATH = os.path.join(ARIM_ROOT, "corrupted_videos_report.txt")


def check_video(full_path):
    """Returns None if the file is fine, or a short reason string if it's broken."""
    if not os.path.exists(full_path):
        return "file missing"
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        cap.release()
        return "could not open"
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return "opened but could not read a frame"
    return None


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_files = len(rows) * 2
    print(f"Checking {total_files} video files ({len(rows)} rows x RGB + Thermal)...")

    broken = []
    checked = 0
    for row in rows:
        for key in ("rgb_path", "thermal_path"):
            rel_path = row[key]
            full_path = os.path.join(VIDEOS_DIR, rel_path)
            problem = check_video(full_path)
            if problem:
                broken.append((rel_path, problem))
            checked += 1
            if checked % 2000 == 0:
                print(f"  checked {checked}/{total_files}...")

    print(f"\nDone. {len(broken)} broken file(s) out of {total_files} checked.")

    if broken:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            for path, reason in broken:
                line = f"{path}: {reason}"
                print("  ", line)
                f.write(line + "\n")
        print(f"\nFull list saved to:\n  {REPORT_PATH}")
    else:
        print("No corrupted or unreadable files found -- the dataset is clean.")


if __name__ == "__main__":
    main()