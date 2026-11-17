"""Launch the IoU tracker, optionally guided by a Kalman filter.

Usage: python main.py [Yolov5s|Yolov5l|public-dataset] [--tracker iou|kalman]
                     [--show] [--video out.mp4] [--out file.txt]
"""
import argparse
import os

import cv2

from tracker_iou import Box, draw, run_tracker
from tracker_kalman import draw_kalman, run_kalman_tracker

SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ADL-Rundle-6")


def load_detections(path: str) -> dict[int, list[Box]]:
    """Return {frame: [box, box, ...]} where box = (left, top, width, height)."""
    detections = {}
    for line in open(path):
        values = line.replace(",", " ").split()
        frame = int(values[0])
        box = tuple(float(v) for v in values[2:6])
        detections.setdefault(frame, []).append(box)
    return detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("iou", "kalman"), default="iou",
                        help="tracking method: iou or iou with Kalman prediction")
    parser.add_argument("det", nargs="?", default="Yolov5l")
    parser.add_argument("--show", action="store_true", help="display frames (q to quit)")
    parser.add_argument("--video", help="save the tracking video to this file")
    parser.add_argument("--out", default="ADL-Rundle-6.txt")
    args = parser.parse_args()

    if args.tracker == "kalman":
        run, draw_tracks, title = run_kalman_tracker, draw_kalman, "Kalman tracker"
    else:
        run, draw_tracks, title = run_tracker, draw, "IoU tracker"

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
