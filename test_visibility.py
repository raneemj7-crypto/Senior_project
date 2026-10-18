from pathlib import Path
import numpy as np

from visibility_detection import detect_visibility

BASE = Path(__file__).resolve().parent

# Load real skeletons extracted from our dataset
DATA = BASE / "results" / "training_data" / "X_train_sample.npy"

X = np.load(DATA)

print("\nTESTING REAL SKELETON DATA")
print("Loaded sequences:", len(X))

# Find a frame with a well-detected skeleton
best_frame = None
best_count = -1

for sequence in X:
    for frame in sequence:
        visible_count = int(np.sum(frame[:, 2] >= 0.35))

        if visible_count > best_count:
            best_count = visible_count
            best_frame = frame.copy()

print("\nBest detected frame:")
print("Visible keypoints:", best_count)
print("Visibility:", detect_visibility(best_frame))

# Test 1: Original skeleton
print("\nTEST 1 — Original skeleton")
print("Result:", detect_visibility(best_frame))

# Test 2: Hide legs
upper_body = best_frame.copy()
upper_body[13:17, :] = 0

print("\nTEST 2 — Legs hidden")
print("Result:", detect_visibility(upper_body))

# Test 3: Keep head and shoulders only
head_shoulders = best_frame.copy()
head_shoulders[7:17, :] = 0

print("\nTEST 3 — Head and shoulders only")
print("Result:", detect_visibility(head_shoulders))

# Test 4: Hide almost everything
uncertain = np.zeros_like(best_frame)

print("\nTEST 4 — Insufficient visibility")
print("Result:", detect_visibility(uncertain))

print("\nTEST COMPLETE!")