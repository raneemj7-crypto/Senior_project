import numpy as np
import torch
import torch.nn as nn

# ============================================================
# LOAD THE PREPARED LSTM DATA
# ============================================================

train_data = np.load("train_sequences.npz")
val_data = np.load("val_sequences.npz")
test_data = np.load("test_sequences.npz")

X_train = train_data["X"]
y_train = train_data["y"]

X_val = val_data["X"]
y_val = val_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]


# ============================================================
# CHECK THE DATA
# ============================================================

print("Train X:", X_train.shape)
print("Train y:", y_train.shape)

print("Validation X:", X_val.shape)
print("Validation y:", y_val.shape)

print("Test X:", X_test.shape)
print("Test y:", y_test.shape)


# ============================================================
# PREPARE DATA SHAPE FOR LSTM
# ============================================================

# Current shape:
# (samples, 30 frames, 17 keypoints, 3 values)
#
# LSTM expects:
# (samples, time steps, features)
#
# 17 keypoints × 3 values = 51 features per frame

X_train_lstm = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    -1
)

X_val_lstm = X_val.reshape(
    X_val.shape[0],
    X_val.shape[1],
    -1
)

X_test_lstm = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    -1
)

print()
print("LSTM input shapes:")
print("Train:", X_train_lstm.shape)
print("Validation:", X_val_lstm.shape)
print("Test:", X_test_lstm.shape)


# ============================================================
# DEFINE LSTM MODEL
# ============================================================

class FallDetectionLSTM(nn.Module):

    def __init__(
        self,
        input_size=51,
        hidden_size=64,
        num_layers=1
    ):
        super().__init__()

        # The LSTM learns how the pose changes over time
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final classification layer
        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # Output across all 30 frames
        lstm_output, _ = self.lstm(x)

        # Use the final time step representation
        final_output = lstm_output[:, -1, :]

        # One score for fall vs non-fall
        prediction = self.classifier(final_output)

        return prediction


# ============================================================
# CREATE MODEL
# ============================================================

model = FallDetectionLSTM()

print()
print(model)


# ============================================================
# CONVERT NUMPY DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_lstm, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Binary classification:
# 0 = Non-fall
# 1 = Fall
criterion = nn.BCEWithLogitsLoss()

# Adam updates the model weights during training
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Maximum number of epochs
NUM_EPOCHS = 30

# === ADDED: EARLY STOPPING SETTINGS ===
# Stop training if validation loss does not improve
# for this many consecutive epochs.
PATIENCE = 5

best_val_loss = float("inf")
epochs_without_improvement = 0


# ============================================================
# TRAIN THE MODEL
# ============================================================

for epoch in range(NUM_EPOCHS):

    # -------------------------
    # TRAINING
    # -------------------------

    model.train()

    optimizer.zero_grad()

    train_outputs = model(X_train_tensor)

    train_loss = criterion(
        train_outputs,
        y_train_tensor
    )

    train_loss.backward()

    optimizer.step()


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val_tensor)

        val_loss = criterion(
            val_outputs,
            y_val_tensor
        )

        # Convert model scores into probabilities
        val_probabilities = torch.sigmoid(
            val_outputs
        )

        # Probability >= 0.5 means Fall
        val_predictions = (
            val_probabilities >= 0.5
        ).float()

        # Calculate validation accuracy
        val_accuracy = (
            val_predictions
            == y_val_tensor
        ).float().mean()


    # -------------------------
    # PRINT PROGRESS
    # -------------------------

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss.item():.4f} | "
        f"Val Loss: {val_loss.item():.4f} | "
        f"Val Accuracy: {val_accuracy.item():.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "best_fall_detection_lstm.pth"
        )

        print(
            f"  -> New best model saved "
            f"(Val Loss: {best_val_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  -> No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("Training finished.")

print(
    "Best model saved as: "
    "best_fall_detection_lstm.pth"
)