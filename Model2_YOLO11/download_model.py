from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pathlib import Path
import shutil

# Download the pretrained YOLO11 fall-detection model
weights = hf_hub_download(
    repo_id="melihuzunoglu/human-fall-detection",
    filename="best.pt"
)

# Save the model in the models folder
destination = Path("models/yolo11_fall_pretrained.pt")
destination.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(weights, destination)

# Verify the downloaded model
model = YOLO(str(destination))

print("Model downloaded successfully!")
print("Model task:", model.task)
print("Model classes:", model.names)
print("Saved at:", destination)
