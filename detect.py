"""Part IV (improved detection): run a recent YOLO offline and save its detections in MOT format.

Usage: uv run detect.py [model] [--imgsz 1280] [--conf 0.3] [--name folder]
Writes ADL-Rundle-6/det/<name>/det.txt, then: uv run main.py <name> --tracker kalman
The model weights (COCO-pretrained) are downloaded by ultralytics on first use.
"""
import argparse
import os

from ultralytics import YOLO

from main import SEQ_DIR

PERSON = 0  # COCO class id of "person"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="yolo26x.pt")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference size (frames are 1920 wide)")
    parser.add_argument("--conf", type=float, default=0.7, help="minimum detection confidence (0.7 was best on ADL-Rundle-6)")
    parser.add_argument("--name", help="output folder in det/ (default: model name)")
    args = parser.parse_args()
    name = args.name or os.path.splitext(args.model)[0]

    model = YOLO(args.model)
    images = sorted(os.listdir(os.path.join(SEQ_DIR, "img1")))
    rows = []
    for frame, image in enumerate(images, start=1):
        result = model.predict(os.path.join(SEQ_DIR, "img1", image), imgsz=args.imgsz,
                               conf=args.conf, classes=[PERSON], verbose=False)[0]
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            rows.append(f"{frame},-1,{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{conf:.4f},-1,-1,-1")
        if frame % 50 == 0:
            print(f"{frame}/{len(images)} frames")

    folder = os.path.join(SEQ_DIR, "det", name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "det.txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} detections written to {folder}/det.txt")


if __name__ == "__main__":
    main()
