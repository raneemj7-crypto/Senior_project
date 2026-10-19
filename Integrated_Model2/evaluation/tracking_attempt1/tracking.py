"""Conservative identity association; never interpolates classifier keypoints.

Distances are fractions of body-box size, not camera-specific pixels. Ambiguous
matches are deliberately left unmatched. This is geometry, not biometric ReID.
"""
import numpy as np

HANDOVER_MAX_GAP = 1.0
AMBIGUITY_MARGIN = 0.15


def centre_size(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2)/2, (y1+y2)/2]), max(x2-x1, y2-y1, 1e-6)


def continuity_score(t, buf, box, xy, kc):
    """Return (lower-is-better score or None, diagnostic evidence)."""
    gap = t - buf[-1][0]
    evidence = {'gap_seconds': round(float(gap), 4)}
    if not 0 < gap <= HANDOVER_MAX_GAP:
        return None, dict(evidence, reason='gap_too_long')
    old_t, old_xy, old_kc, old_box = buf[-1]
    old_c, old_size = centre_size(old_box)
    new_c, new_size = centre_size(box)
    scale = max(old_size, new_size)
    velocity = np.zeros(2)
    # Short recent motion estimate; cap speed and extrapolation horizon.
    recent = [b for b in buf[:-1] if 0.1 <= old_t-b[0] <= 0.4]
    if recent:
        prev = recent[0]
        velocity = (old_c-centre_size(prev[3])[0]) / (old_t-prev[0])
        speed = np.linalg.norm(velocity)
        if speed > 2 * old_size:
            velocity *= 2 * old_size / speed
    displacement = velocity * min(gap, .5)
    distance = float(np.linalg.norm(new_c-old_c-displacement) / scale)
    evidence['motion_distance'] = round(distance, 4)
    if distance > .75 or max(old_size, new_size) / min(old_size, new_size) > 2.5:
        return None, dict(evidence, reason='spatial_discontinuity')
    xy, kc = np.asarray(xy), np.asarray(kc)
    common = ((np.asarray(old_kc) >= .3) & (kc >= .3)
              & np.isfinite(old_xy).all(axis=1) & np.isfinite(xy).all(axis=1)
              & (np.asarray(old_xy) > 0).all(axis=1) & (xy > 0).all(axis=1))
    evidence['common_joints'] = int(common.sum())
    # No weak-pose fallback: proximity alone cannot establish identity.
    if common.sum() < 4 or common[5:].sum() < 2:
        return None, dict(evidence, reason='insufficient_common_joints')
    a, b = np.asarray(old_xy)[common], xy[common]
    world = float(np.median(np.linalg.norm(b-a-displacement, axis=1)) / scale)
    shape = float(np.median(np.linalg.norm((b-b.mean(0))-(a-a.mean(0)), axis=1)) / scale)
    evidence.update(keypoint_distance=round(world, 4), shape_distance=round(shape, 4))
    if world > .65 or shape > .40:
        return None, dict(evidence, reason='keypoint_discontinuity')
    return distance + world + shape + .1 * gap, dict(evidence, reason='compatible')


def unique_handover(candidates):
    """Mutual best matches with margins on BOTH sides, independent of input order.

    candidates: (score, new_id, old_id). A donor can never be used twice.
    """
    chosen = {}
    for score, new, old in sorted(candidates):
        receiver = sorted(p for p in candidates if p[1] == new)
        donor = sorted(p for p in candidates if p[2] == old)
        if receiver[0] != (score, new, old) or donor[0] != (score, new, old):
            continue
        if any(len(group) > 1 and group[1][0]-score < AMBIGUITY_MARGIN for group in (receiver, donor)):
            continue
        chosen[new] = old
    return chosen
