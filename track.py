import cv2
import csv
import numpy as np
import torch
import supervision as sv

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device


# =========================
# SETTINGS
# =========================

WEIGHTS = "weights/crowdhuman_yolov5m.pt"
SOURCE = "test_videos/crowd.mp4"

OUTPUT_VIDEO = "tracked_crowd.mp4"
OUTPUT_CSV = "tracking_history.csv"

CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 640


# =========================
# LOAD MODEL
# =========================

device = select_device("")

model = attempt_load(
    WEIGHTS,
    map_location=device
)

stride = int(model.stride.max())

names = (
    model.module.names
    if hasattr(model, "module")
    else model.names
)


# =========================
# CREATE BYTETRACK
# =========================

tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# =========================
# OPEN VIDEO
# =========================

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise Exception(f"Could not open video: {SOURCE}")


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# =========================
# OUTPUT VIDEO
# =========================

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)


# =========================
# OUTPUT CSV
# =========================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
    "confidence"
])


# =========================
# FRAME COUNTER
# =========================

frame_number = 0


# =========================
# PROCESS VIDEO
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ---------------------------------
    # PREPROCESS FRAME FOR YOLO
    # ---------------------------------

    img = letterbox(
        frame,
        IMG_SIZE,
        stride=stride
    )[0]

    # BGR -> RGB
    # HWC -> CHW
    img = img[:, :, ::-1].transpose(
        2,
        0,
        1
    )

    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)

    img = img.float()

    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    # ---------------------------------
    # YOLO INFERENCE
    # ---------------------------------

    pred = model(
        img,
        augment=False
    )[0]


    # ---------------------------------
    # NON-MAX SUPPRESSION
    # ---------------------------------

    pred = non_max_suppression(
        pred,
        CONF_THRES,
        IOU_THRES,
        classes=None,
        agnostic=False,
    )


    # ---------------------------------
    # STORE PERSON DETECTIONS
    # ---------------------------------

    xyxy_list = []
    confidence_list = []


    for det in pred:

        if len(det):

            det[:, :4] = scale_coords(
                img.shape[2:],
                det[:, :4],
                frame.shape
            ).round()


            for *xyxy, conf, cls in det:

                class_id = int(cls)

                class_name = names[class_id]


                # -------------------------
                # ONLY TRACK PERSON CLASS
                # -------------------------

                if class_name != "person":
                    continue


                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )


                xyxy_list.append([
                    x1,
                    y1,
                    x2,
                    y2
                ])

                confidence_list.append(
                    float(conf)
                )


    # ---------------------------------
    # BYTE TRACK
    # ---------------------------------

    if len(xyxy_list) > 0:

        detections = sv.Detections(
            xyxy=np.array(
                xyxy_list,
                dtype=np.float32
            ),
            confidence=np.array(
                confidence_list,
                dtype=np.float32
            ),
            class_id=np.zeros(
                len(xyxy_list),
                dtype=int
            ),
        )


        detections = tracker.update_with_detections(
            detections
        )


        # ---------------------------------
        # SAVE TRACKING HISTORY TO CSV
        # ---------------------------------

        for xyxy, conf, tracker_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.tracker_id
        ):

            x1, y1, x2, y2 = map(
                float,
                xyxy
            )


            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2


            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )


            csv_writer.writerow([
                frame_number,
                int(tracker_id),
                x1,
                y1,
                x2,
                y2,
                center_x,
                center_y,
                box_width,
                box_height,
                float(conf)
            ])


        # ---------------------------------
        # CREATE LABELS
        # ---------------------------------

        labels = [
            f"ID {tracker_id}"
            for tracker_id
            in detections.tracker_id
        ]


        # ---------------------------------
        # DRAW BOXES
        # ---------------------------------

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
        )


        # ---------------------------------
        # DRAW IDs
        # ---------------------------------

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )


    else:

        annotated_frame = frame


    # ---------------------------------
    # WRITE FRAME TO OUTPUT VIDEO
    # ---------------------------------

    writer.write(
        annotated_frame
    )


    # ---------------------------------
    # OPTIONAL TERMINAL PROGRESS
    # ---------------------------------

    if frame_number % 50 == 0:
        print(
            f"Processed frame: {frame_number}"
        )


# =========================
# CLEAN UP
# =========================

cap.release()

writer.release()

csv_file.close()


# =========================
# FINISHED
# =========================

print("")
print("Tracking finished successfully.")
print(f"Tracked video saved to: {OUTPUT_VIDEO}")
print(f"Tracking history saved to: {OUTPUT_CSV}")