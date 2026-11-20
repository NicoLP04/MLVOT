"""Multi-object tracker: IoU + Hungarian (Part I), optionally guided by a Kalman filter (Part II).

Usage: python main.py [Yolov5s|Yolov5l|public-dataset] [--tracker iou|kalman]
                     [--show] [--video out.mp4] [--out file.txt]
"""
import argparse
import os
from typing import Iterator

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from KalmanFilter import KalmanFilter

Box = tuple[float, float, float, float]  # (left, top, width, height)

SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ADL-Rundle-6")
IOU_THRESHOLD = 0.3    # below this, a match is rejected
MAX_MISSED_FRAMES = 5  # a track is deleted after this many missed frames in a row

# Kalman filter parameters
DT = 1.0           # one frame = one time step
U_X = 0.0          # constant-velocity model: zero acceleration
U_Y = 0.0
STD_ACC = 1.0      # process noise (how much the speed can change)
X_STD_MEAS = 5.0   # measurement noise in pixels, independently on each axis
Y_STD_MEAS = 5.0


# ---------------------------------------------------------------- detections

def load_detections(path: str) -> dict[int, list[Box]]:
    """Return {frame: [box, box, ...]} where box = (left, top, width, height)."""
    detections = {}
    for line in open(path):
        values = line.replace(",", " ").split()
        frame = int(values[0])
        box = tuple(float(v) for v in values[2:6])
        detections.setdefault(frame, []).append(box)
    return detections


# ------------------------------------------------------------------- tracks

class Track:
    next_id: int = 1

    def __init__(self, box: Box) -> None:
        self.id = Track.next_id
        Track.next_id += 1
        self.box = box
        self.missed = 0

    def predict(self) -> None:
        pass  # nothing to predict without a Kalman filter

    def update(self, box: Box) -> None:
        self.box = box


class KalmanTrack(Track):
    def __init__(self, box: Box) -> None:
        super().__init__(box)
        left, top, self.width, self.height = box
        cx, cy = left + self.width / 2, top + self.height / 2
        self.kf = KalmanFilter(
            dt=DT, u_x=U_X, u_y=U_Y, std_acc=STD_ACC,
            x_std_meas=X_STD_MEAS, y_std_meas=Y_STD_MEAS,
        )
        # Part II, 1.2: [x0, y0, vx0, vy0], with the detection centroid
        # and zero initial velocity. Each track owns its independent filter.
        self.kf.x[:, 0] = [cx, cy, 0.0, 0.0]
        self.predicted = box

    def predict(self) -> None:
        """Predict the new position; the box keeps the last known width and height."""
        self.kf.predict()
        cx, cy = self.kf.x[0, 0], self.kf.x[1, 0]
        self.box = (cx - self.width / 2, cy - self.height / 2, self.width, self.height)
        self.predicted = self.box

    def update(self, box: Box) -> None:
        """Correct the filter with the detection and keep its width and height."""
        left, top, self.width, self.height = box
        self.kf.update((left + self.width / 2, top + self.height / 2))
        self.box = box


# ------------------------------------------------------------------ tracking

def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (left, top, width, height)."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])

    inter = max(0, right - left) * max(0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union


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
    detections: dict[int, list[Box]], n_frames: int, use_kalman: bool
) -> Iterator[tuple[int, list[Track]]]:
    """For each frame, yield (frame, tracks seen in that frame)."""
    new_track = KalmanTrack if use_kalman else Track
    tracks = []
    for frame in range(1, n_frames + 1):
        boxes = detections.get(frame, [])

        for track in tracks:
            track.predict()
        matches, unmatched_tracks, unmatched_dets = match(tracks, boxes)

        for i, j in matches:
            tracks[i].update(boxes[j])
            tracks[i].missed = 0
        for i in unmatched_tracks:
            tracks[i].missed += 1   # no update: with Kalman we keep the predicted state
        for j in unmatched_dets:
            tracks.append(new_track(boxes[j]))

        yield frame, [t for t in tracks if t.missed == 0]

        tracks = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]


# ------------------------------------------------------------- visualization

def draw(image: np.ndarray, tracks: list[Track]) -> None:
    """Tracked boxes with their IDs."""
    for track in tracks:
        left, top, width, height = [int(v) for v in track.box]
        # same ID -> same color
        color = tuple(int(c) for c in np.random.RandomState(track.id).randint(60, 255, 3))
        cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
        cv2.putText(image, str(track.id), (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_predictions(image: np.ndarray, tracks: list[KalmanTrack]) -> None:
    """Kalman-predicted boxes, in red and thinner than the observed ones."""
    for track in tracks:
        left, top, width, height = [int(v) for v in track.predicted]
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 1)


# ---------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("iou", "kalman"), default="iou",
                        help="tracking method: iou or iou with Kalman prediction")
    parser.add_argument("det", nargs="?", default="Yolov5l")
    parser.add_argument("--show", action="store_true", help="display frames (q to quit)")
    parser.add_argument("--video", help="save the tracking video to this file")
    parser.add_argument("--out", default="ADL-Rundle-6.txt")
    args = parser.parse_args()
    use_kalman = args.tracker == "kalman"

    detections = load_detections(os.path.join(SEQ_DIR, "det", args.det, "det.txt"))
    n_frames = max(detections)

    video = None
    rows = []
    for frame, tracks in run_tracker(detections, n_frames, use_kalman):
        for t in tracks:
            left, top, width, height = t.box
            rows.append(f"{frame},{t.id},{left},{top},{width},{height},1,-1,-1,-1")

        if args.show or args.video:
            image = cv2.imread(os.path.join(SEQ_DIR, "img1", f"{frame:06d}.jpg"))
            draw(image, tracks)
            if use_kalman:
                draw_predictions(image, tracks)

            if args.video:
                if video is None:
                    h, w = image.shape[:2]
                    video = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                video.write(image)

            if args.show:
                cv2.imshow(f"{args.tracker} tracker", cv2.resize(image, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) == ord("q"):
                    break

    if video:
        video.release()
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()
