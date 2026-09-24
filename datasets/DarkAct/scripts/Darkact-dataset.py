"""
PyTorch Dataset for DarkAct. Reads darkact_label_table.csv (built by
build_label_table.py) and loads sampled frames from the matched RGB and
Thermal clips for each row.

Frame count and frame size below are reasonable starting defaults, not
final decisions -- once the team picks the actual fusion-backbone
architecture, these will likely need to change to match it.
"""

import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DarkActDataset(Dataset):
    def __init__(self, csv_path, videos_dir, split="train", num_frames=16, frame_size=(224, 224)):
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.frame_size = frame_size

        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {csv_path}")

        class_names = sorted(set(r["class_name"] for r in self.rows))
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.rows)

    def _load_clip(self, rel_path):
        full_path = os.path.join(self.videos_dir, rel_path)
        cap = cv2.VideoCapture(full_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Could not read frames from: {full_path}")

        target_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        frames = []
        current = 0
        ptr = 0
        while ptr < len(target_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if current == target_indices[ptr]:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                ptr += 1
            current += 1
        cap.release()

        # pad with the last frame if the clip was shorter than expected
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((*self.frame_size, 3), dtype=np.uint8))

        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0   # (T, H, W, 3)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)            # (3, T, H, W)
        return clip

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb_clip = self._load_clip(row["rgb_path"])
        thermal_clip = self._load_clip(row["thermal_path"])

        return {
            "rgb": rgb_clip,
            "thermal": thermal_clip,
            "class_name": row["class_name"],
            "label": self.class_to_idx[row["class_name"]],
            "is_hard_negative": row["is_hard_negative"] == "True",
            "video_id": row["video_id"],
        }


if __name__ == "__main__":
    ARIM_ROOT = r"C:\Users\USER\Desktop\SE499\darkact_datasets\ARIM_v1"
    CSV_PATH = os.path.join(ARIM_ROOT, "darkact_label_table.csv")
    VIDEOS_DIR = os.path.join(ARIM_ROOT, "videos")

    train_dataset = DarkActDataset(CSV_PATH, VIDEOS_DIR, split="train", num_frames=16, frame_size=(224, 224))
    print(f"Train dataset size: {len(train_dataset)}")

    sample = train_dataset[0]
    print(f"RGB clip shape:     {tuple(sample['rgb'].shape)}")
    print(f"Thermal clip shape: {tuple(sample['thermal'].shape)}")
    print(f"class_name: {sample['class_name']}  label idx: {sample['label']}  hard_negative: {sample['is_hard_negative']}")

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch RGB shape: {tuple(batch['rgb'].shape)}")
    print(f"Batch labels: {batch['label'].tolist()}")