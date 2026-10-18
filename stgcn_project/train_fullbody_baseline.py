from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stgcn_model import STGCNPlusPlus


# --------------------------------
# 1. SETTINGS
# --------------------------------

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUT = BASE / "trained_model"

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------
# 2. LOAD FULL-BODY TRAINING DATA
# --------------------------------

# The existing training data contains three consecutive
# versions of each sequence:
# full body, upper body, head and shoulders.
#
# Keep only the first version of each sequence.

X_augmented = np.load(DATA / "X_train.npy")
y_augmented = np.load(DATA / "y_train.npy")

X_train = X_augmented[::3].copy()
y_train = y_augmented[::3].copy()

X_val = np.load(DATA / "X_val.npy")
y_val = np.load(DATA / "y_val.npy")

print("\nFull-body training data:", X_train.shape)
print("Validation data:", X_val.shape)

print("Training falls:", int(np.sum(y_train == 1)))
print("Training non-falls:", int(np.sum(y_train == 0)))

train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).long()
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(y_val).long()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------
# 3. CREATE MODEL
# --------------------------------

model = STGCNPlusPlus().to(device)

class_counts = np.bincount(y_train, minlength=2)

class_weights = (
    len(y_train) /
    (2.0 * np.maximum(class_counts, 1))
)

weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

print("Class weights:", class_weights)


# --------------------------------
# 4. TRAIN MODEL A
# --------------------------------

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(labels)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += len(labels)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * len(labels)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += len(labels)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Acc: {val_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f}",
        flush=True
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "training_type": "full_body_only"
            },
            OUTPUT / "fullbody_baseline.pth"
        )

print("\nFULL-BODY BASELINE TRAINING COMPLETE!")
print("Best validation loss:", round(best_val_loss, 4))
print("Model saved:", OUTPUT / "fullbody_baseline.pth")