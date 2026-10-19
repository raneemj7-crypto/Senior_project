import numpy as np

# YOLOv8 Pose uses 17 COCO keypoints
HEAD = [0, 1, 2, 3, 4]
SHOULDERS = [5, 6]
ARMS = [7, 8, 9, 10]
HIPS = [11, 12]
LEGS = [13, 14, 15, 16]

CONF_THRESHOLD = 0.35


def detect_visibility(keypoints):
    """
    Input:
        keypoints: numpy array (17, 3)
        Each point contains [x, y, confidence]

    Output:
        visibility category
    """

    visible = keypoints[:, 2] >= CONF_THRESHOLD

    head_count = sum(visible[HEAD])
    shoulder_count = sum(visible[SHOULDERS])
    arm_count = sum(visible[ARMS])
    hip_count = sum(visible[HIPS])
    leg_count = sum(visible[LEGS])

    total_visible = sum(visible)

    # Full body is mostly visible
    if total_visible >= 13 and leg_count >= 3:
        return "full_body"

    # Upper body visible
    if shoulder_count >= 1 and (
        arm_count >= 2 or hip_count >= 1
    ):
        return "upper_body"

    # Head and shoulders visible
    if head_count >= 2 and shoulder_count >= 1:
        return "head_shoulders"

    return "uncertain"


def detect_sequence_visibility(sequence):
    """
    Input:
        sequence: (30, 17, 3)

    Output:
        dominant visibility category
    """

    categories = []

    for frame in sequence:
        categories.append(detect_visibility(frame))

    counts = {
        category: categories.count(category)
        for category in set(categories)
    }

    return max(counts, key=counts.get)


# --------------------------------
# QUICK TEST
# --------------------------------

if __name__ == "__main__":

    # Simulate full visibility
    full = np.ones((17, 3), dtype=np.float32)
    print("Full body:", detect_visibility(full))

    # Simulate upper-body visibility
    upper = full.copy()
    upper[13:17, 2] = 0
    print("Upper body:", detect_visibility(upper))

    # Simulate head and shoulders
    head = np.zeros((17, 3), dtype=np.float32)
    head[0:7, 2] = 0.9
    print("Head and shoulders:", detect_visibility(head))

    # Simulate insufficient visibility
    uncertain = np.zeros((17, 3), dtype=np.float32)
    print("Uncertain:", detect_visibility(uncertain))