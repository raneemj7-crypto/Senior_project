from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
# 2. LOAD DATA
# --------------------------------

X_train = np.load(DATA / "X_train.npy")
y_train = np.load(DATA / "y_train.npy")

X_val = np.load(DATA / "X_val.npy")
y_val = np.load(DATA / "y_val.npy")

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)

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

# Give additional importance to fall examples
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

print("\nClass weights:", class_weights)


# --------------------------------
# 4. TRAINING
# --------------------------------

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item() * len(labels)

        predictions = outputs.argmax(dim=1)

        correct_train += (
            predictions == labels
        ).sum().item()

        total_train += len(labels)

    train_loss = total_train_loss / total_train
    train_accuracy = correct_train / total_train

    # ----------------------------
    # VALIDATION
    # ----------------------------

    model.eval()

    total_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * len(labels)

            predictions = outputs.argmax(dim=1)

            correct_val += (
                predictions == labels
            ).sum().item()

            total_val += len(labels)

    val_loss = total_val_loss / total_val
    val_accuracy = correct_val / total_val

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2%}",
        flush=True
    )

    # Save the model with lowest validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "class_weights": class_weights.tolist()
            },
            OUTPUT / "best_stgcn_model.pth"
        )

print("\nTRAINING COMPLETE!")
print("Best validation loss:", round(best_val_loss, 4))
print("Model saved to:", OUTPUT / "best_stgcn_model.pth")