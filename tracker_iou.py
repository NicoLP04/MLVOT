"""Part I: IoU-based multi-object tracker (Hungarian assignment).

Usage: python tracker_iou.py [Yolov5s|Yolov5l|public-dataset] [--show] [--video out.mp4]
"""
import argparse
import os
from typing import Iterator

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]  # (left, top, width, height)

SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ADL-Rundle-6")
IOU_THRESHOLD = 0.3   # below this, a match is rejected
MAX_MISSED_FRAMES = 5  # a track is deleted after this many missed frames in a row


def load_detections(path: str) -> dict[int, list[Box]]:
    """Return {frame: [box, box, ...]} where box = (left, top, width, height)."""
    detections = {}
    for line in open(path):
        values = line.replace(",", " ").split()
        frame = int(values[0])
        box = tuple(float(v) for v in values[2:6])
        detections.setdefault(frame, []).append(box)
    return detections


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


def main(run=run_tracker, draw_tracks=draw, title: str = "IoU tracker") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("det", nargs="?", default="Yolov5l")
    parser.add_argument("--show", action="store_true", help="display frames (q to quit)")
    parser.add_argument("--video", help="save the tracking video to this file")
    parser.add_argument("--out", default="ADL-Rundle-6.txt")
    args = parser.parse_args()

    detections = load_detections(os.path.join(SEQ_DIR, "det", args.det, "det.txt"))
    n_frames = max(detections)

    video = None
    rows = []
    for frame, tracks in run(detections, n_frames):
        for t in tracks:
            left, top, width, height = t.box
            rows.append(f"{frame},{t.id},{left},{top},{width},{height},1,-1,-1,-1")

        if args.show or args.video:
            image = cv2.imread(os.path.join(SEQ_DIR, "img1", f"{frame:06d}.jpg"))
            draw_tracks(image, tracks)

            if args.video:
                if video is None:
                    h, w = image.shape[:2]
                    video = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                video.write(image)

            if args.show:
                cv2.imshow(title, cv2.resize(image, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) == ord("q"):
                    break

    if video:
        video.release()
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()
