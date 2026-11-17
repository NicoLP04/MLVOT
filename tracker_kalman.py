"""Part II: Kalman-guided multi-object tracker.

Same as Part I, but each track has a Kalman filter that predicts where the
object will be. The Hungarian matching uses the predicted boxes.

Usage: python tracker_kalman.py [Yolov5s|Yolov5l|public-dataset] [--show] [--video out.mp4] [--out file.txt]
"""
from typing import Iterator

import cv2
import numpy as np

from KalmanFilter import KalmanFilter
from tracker_iou import MAX_MISSED_FRAMES, Box, Track, draw, main, match

DT = 1.0           # one frame = one time step
STD_ACC = 1.0      # process noise (how much the speed can change)
STD_MEAS = 5.0     # measurement noise in pixels


class KalmanTrack(Track):
    """A Track whose `box` is the position predicted by its Kalman filter."""

    def __init__(self, box: Box) -> None:
        super().__init__(box)
        left, top, self.width, self.height = box
        cx, cy = left + self.width / 2, top + self.height / 2
        self.kf = KalmanFilter(DT, 0, 0, STD_ACC, STD_MEAS, STD_MEAS, cx, cy)
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


def run_kalman_tracker(
    detections: dict[int, list[Box]], n_frames: int
) -> Iterator[tuple[int, list[KalmanTrack]]]:
    """For each frame, yield (frame, tracks seen in that frame)."""
    tracks: list[KalmanTrack] = []
    for frame in range(1, n_frames + 1):
        boxes = detections.get(frame, [])

        for track in tracks:
            track.predict()
        matches, unmatched_tracks, unmatched_dets = match(tracks, boxes)

        for i, j in matches:
            tracks[i].update(boxes[j])
            tracks[i].missed = 0
        for i in unmatched_tracks:
            tracks[i].missed += 1   # no update: we keep the predicted state
        for j in unmatched_dets:
            tracks.append(KalmanTrack(boxes[j]))

        yield frame, [t for t in tracks if t.missed == 0]

        tracks = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]


def draw_kalman(image: np.ndarray, tracks: list[KalmanTrack]) -> None:
    """Observed boxes with IDs (colored by ID, as in Part I) + predicted boxes in red."""
    draw(image, tracks)
    for track in tracks:
        left, top, width, height = [int(v) for v in track.predicted]
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 1)


if __name__ == "__main__":
    main(run_kalman_tracker, draw_kalman, "Kalman tracker")
