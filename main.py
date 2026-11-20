"""Multi-object tracker: IoU + Hungarian (Part I), optionally guided by a Kalman filter (Part II).

Part IV adds an appearance-aware tracker (Kalman + appearance descriptors).

Usage: python main.py [Yolov5s|Yolov5l|public-dataset] [--tracker iou|kalman|appearance]
                     [--descriptor hist|cnn] [--show] [--video out.mp4] [--out file.txt]
"""
import argparse
import os
from typing import Iterator

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from Appearance import get_features
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

# Part IV (appearance-aware tracker): similarity S = ALPHA * IoU + BETA * cosine(appearance)
# Values tuned for YOLO26x detections + ReID descriptor (HOTA 49.34). They are worse than
# beta=0.5, threshold=0.3, memory=30 on the provided detections (Yolov5s/l, public-dataset).
APP_ALPHA = 0.75
APP_BETA = 0.25
APP_THRESHOLD = 0.15   # minimum S to accept a match
APP_MAX_MISSED = 60    # lost tracks are kept longer: appearance helps to find them again
APP_EMA = 0.8          # track descriptor = EMA * old descriptor + (1 - EMA) * new one
APP_STD_ACC = 0.25     # Kalman noise used by this tracker
APP_STD_MEAS = 10.0


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


class AppearanceTrack(KalmanTrack):
    """Part IV: a Kalman track that also remembers what the person looks like."""

    def __init__(self, box: Box, feature: np.ndarray) -> None:
        super().__init__(box)
        self.feature = feature

    def update(self, box: Box, feature: np.ndarray) -> None:
        super().update(box)
        mixed = APP_EMA * self.feature + (1 - APP_EMA) * feature
        self.feature = mixed / max(np.linalg.norm(mixed), 1e-9)


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
    tracks: list[Track], boxes: list[Box], features: np.ndarray | None = None
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on IoU (or on IoU + appearance if `features` is given).

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

    threshold = IOU_THRESHOLD
    if features is not None:
        # Part IV: S = alpha * IoU + beta * appearance similarity (cosine, clipped to [0, 1])
        appearance = np.array([t.feature for t in tracks]) @ features.T
        similarity = APP_ALPHA * similarity + APP_BETA * np.clip(appearance, 0, 1)
        threshold = APP_THRESHOLD

    rows, cols = linear_sum_assignment(similarity, maximize=True)

    matches = []
    for i, j in zip(rows, cols):
        if similarity[i, j] >= threshold:
            matches.append((i, j))

    matched_tracks = [i for i, _ in matches]
    matched_dets = [j for _, j in matches]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(boxes)) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def run_tracker(
    detections: dict[int, list[Box]], n_frames: int, use_kalman: bool,
    features: dict[int, np.ndarray] | None = None,
) -> Iterator[tuple[int, list[Track]]]:
    """For each frame, yield (frame, tracks seen in that frame).

    If `features` (appearance descriptors of the detections) is given, the Part IV
    appearance-aware tracker is used (it implies Kalman).
    """
    use_appearance = features is not None
    max_missed = APP_MAX_MISSED if use_appearance else MAX_MISSED_FRAMES
    tracks = []
    for frame in range(1, n_frames + 1):
        boxes = detections.get(frame, [])
        frame_features = features.get(frame) if use_appearance else None

        for track in tracks:
            track.predict()
        matches, unmatched_tracks, unmatched_dets = match(tracks, boxes, frame_features)

        for i, j in matches:
            if use_appearance:
                tracks[i].update(boxes[j], frame_features[j])
            else:
                tracks[i].update(boxes[j])
            tracks[i].missed = 0
        for i in unmatched_tracks:
            tracks[i].missed += 1   # no update: with Kalman we keep the predicted state
        for j in unmatched_dets:
            if use_appearance:
                tracks.append(AppearanceTrack(boxes[j], frame_features[j]))
            elif use_kalman:
                tracks.append(KalmanTrack(boxes[j]))
            else:
                tracks.append(Track(boxes[j]))

        yield frame, [t for t in tracks if t.missed == 0]

        tracks = [t for t in tracks if t.missed <= max_missed]


def interpolate(rows: list[tuple[int, int, Box]]) -> list[tuple[int, int, Box]]:
    """Part IV post-processing: fill the frames where a track was not seen (hidden person).

    rows = (frame, track_id, box). Between two appearances of the same track, the
    missing boxes are interpolated linearly. Offline: it needs the box after the gap.
    """
    last_seen = {}   # track id -> (frame, box)
    filled = []
    for frame, track_id, box in sorted(rows, key=lambda r: (r[1], r[0])):
        if track_id in last_seen:
            prev_frame, prev_box = last_seen[track_id]
            gap = frame - prev_frame
            for k in range(1, gap):
                t = k / gap
                new_box = tuple(a + t * (b - a) for a, b in zip(prev_box, box))
                filled.append((prev_frame + k, track_id, new_box))
        last_seen[track_id] = (frame, box)
    return sorted(rows + filled)


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
    global STD_ACC, X_STD_MEAS, Y_STD_MEAS
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("iou", "kalman", "appearance"), default="iou",
                        help="iou, iou with Kalman prediction, or Kalman + appearance (Part IV)")
    parser.add_argument("--descriptor", choices=("hist", "cnn", "reid"), default="reid",
                        help="appearance descriptor, only used with --tracker appearance")
    parser.add_argument("det", nargs="?", default="Yolov5l")
    parser.add_argument("--show", action="store_true", help="display frames (q to quit)")
    parser.add_argument("--video", help="save the tracking video to this file")
    parser.add_argument("--out", default="ADL-Rundle-6.txt")
    parser.add_argument("--interpolate", action="store_true",
                        help="fill the frames where a track was hidden (Part IV, not drawn in the video)")
    args = parser.parse_args()
    use_kalman = args.tracker != "iou"

    detections = load_detections(os.path.join(SEQ_DIR, "det", args.det, "det.txt"))
    n_frames = max(detections)
    features = None
    if args.tracker == "appearance":
        STD_ACC, X_STD_MEAS, Y_STD_MEAS = APP_STD_ACC, APP_STD_MEAS, APP_STD_MEAS
        features = get_features(args.det, args.descriptor, detections)

    video = None
    rows = []
    for frame, tracks in run_tracker(detections, n_frames, use_kalman, features):
        for t in tracks:
            rows.append((frame, t.id, t.box))

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
    if args.interpolate:
        rows = interpolate(rows)
    with open(args.out, "w") as f:
        for frame, track_id, (left, top, width, height) in rows:
            f.write(f"{frame},{track_id},{left},{top},{width},{height},1,-1,-1,-1\n")
    print(f"{len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()
