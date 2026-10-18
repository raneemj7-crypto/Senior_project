from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------
# 1. PATHS
# --------------------------------

BASE = Path(__file__).resolve().parent

DATA = BASE / "results" / "crowd_training"
ORIGINAL_DATA = BASE / "results" / "training_data"
OUTPUT = BASE / "results" / "trained_model"

OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. LOAD DATA
# --------------------------------

X = np.load(DATA / "X_crowd.npy")
y = np.load(DATA / "y_crowd.npy")

event_ids = np.load(ORIGINAL_DATA / "event_ids.npy")

# Each original sequence has 3 visibility versions
event_ids = np.repeat(event_ids, 3)

print("Total examples:", len(X))
print("Fall examples:", int(sum(y == 1)))
print("Non-fall examples:", int(sum(y == 0)))

# --------------------------------
# 3. SEPARATE TRAINING AND TESTING
# --------------------------------

# Keep Fall 04 completely unseen during training
test_mask = event_ids == "fall-04"

train_mask = ~test_mask

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))

print("Training falls:", int(sum(y_train == 1)))
print("Testing falls:", int(sum(y_test == 1)))

# --------------------------------
# 4. PREPARE TENSORS
# --------------------------------

# Convert 30 x 17 x 3 into 30 x 51
X_train = X_train.reshape(-1, 30, 51)
X_test = X_test.reshape(-1, 30, 51)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# --------------------------------
# 5. CREATE LSTM MODEL
# --------------------------------

class FallDetectionLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)

# --------------------------------
# 6. SET UP TRAINING
# --------------------------------

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("\nUsing device:", device)

model = FallDetectionLSTM().to(device)

# Balance fall and non-fall classes
class_counts = np.bincount(y_train, minlength=2)

class_weights = (
    len(y_train) /
    (2 * np.maximum(class_counts, 1))
)

weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# --------------------------------
# 7. TRAIN MODEL
# --------------------------------

EPOCHS = 30

print("\nSTARTING TRAINING...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )

# --------------------------------
# 8. EVALUATE ON UNSEEN FALL EVENT
# --------------------------------

model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu()

        correct += (
            predictions == labels
        ).sum().item()

        total += len(labels)

        all_predictions.extend(
            predictions.numpy().tolist()
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

accuracy = 100 * correct / total if total else 0

print("\nTEST RESULTS")
print("Accuracy:", round(accuracy, 2), "%")

for actual in [0, 1]:

    indices = [
        i for i, label in enumerate(all_labels)
        if label == actual
    ]

    if indices:

        correct_class = sum(
            all_predictions[i] == actual
            for i in indices
        )

        print(
            f"Class {actual} accuracy:",
            round(100 * correct_class / len(indices), 2),
            "%"
        )

# --------------------------------
# 9. SAVE MODEL
# --------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": 51,
        "hidden_size": 64,
        "num_layers": 2,
        "window_size": 30,
        "classes": ["non_fall", "fall"]
    },
    OUTPUT / "fall_detection_lstm.pth"
)

print("\nMODEL SAVED!")
print("Location:", OUTPUT / "fall_detection_lstm.pth")