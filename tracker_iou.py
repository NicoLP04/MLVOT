"""IoU-based multi-object tracker (Hungarian assignment)."""
from typing import Iterator

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]  # (left, top, width, height)

IOU_THRESHOLD = 0.3   # below this, a match is rejected
MAX_MISSED_FRAMES = 5  # a track is deleted after this many missed frames in a row


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (left, top, width, height)."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])

    inter = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union


class Track:
    next_id: int = 1

    def __init__(self, box: Box) -> None:
        self.id = Track.next_id
        Track.next_id += 1
        self.box = box
        self.missed = 0


def match(
    tracks: list[Track], boxes: list[Box]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on IoU.

    Returns (matches, unmatched_tracks, unmatched_detections),
    matches being a list of (track_index, detection_index).
    """
    if len(tracks) == 0 or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))

    # similarity matrix: rows = tracks, columns = detections
    similarity = np.zeros((len(tracks), len(boxes)))
    for i, track in enumerate(tracks):
        for j, box in enumerate(boxes):
            similarity[i, j] = iou(track.box, box)

    rows, cols = linear_sum_assignment(similarity, maximize=True)

    matches = []
    for i, j in zip(rows, cols):
        if similarity[i, j] >= IOU_THRESHOLD:
            matches.append((i, j))

    matched_tracks = [i for i, _ in matches]
    matched_dets = [j for _, j in matches]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(boxes)) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def run_tracker(
    detections: dict[int, list[Box]], n_frames: int
) -> Iterator[tuple[int, list[Track]]]:
    """For each frame, yield (frame, tracks seen in that frame)."""
    tracks = []
    for frame in range(1, n_frames + 1):
        boxes = detections.get(frame, [])
        matches, unmatched_tracks, unmatched_dets = match(tracks, boxes)

        for i, j in matches:
            tracks[i].box = boxes[j]
            tracks[i].missed = 0
        for i in unmatched_tracks:
            tracks[i].missed += 1
        for j in unmatched_dets:
            tracks.append(Track(boxes[j]))

        yield frame, [t for t in tracks if t.missed == 0]

        tracks = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]


def draw(image: np.ndarray, tracks: list[Track]) -> None:
    for track in tracks:
        left, top, width, height = [int(v) for v in track.box]
        # same ID -> same color
        color = tuple(int(c) for c in np.random.RandomState(track.id).randint(60, 255, 3))
        cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
        cv2.putText(image, str(track.id), (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
