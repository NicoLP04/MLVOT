"""Part III: compare the IoU and Kalman trackers with TrackEval (HOTA, IDF1, IDSW, FPS).

Usage: uv run evaluate.py
Needs TrackEval cloned next to this file:
    git clone https://github.com/JonathonLuiten/TrackEval.git
"""
import os
import sys
import time

import numpy as np

# TrackEval was written for old NumPy: restore the aliases removed in NumPy 1.24
# (only the missing ones: NumPy 2 has np.bool again and uses it internally)
for name, python_type in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if name not in vars(np):
        setattr(np, name, python_type)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "TrackEval"))
import trackeval  # noqa: E402

from main import SEQ_DIR, Track, load_detections, run_tracker  # noqa: E402

SEQ = "ADL-Rundle-6"
DETECTORS = ["Yolov5s", "Yolov5l", "public-dataset"]
TRACKERS = {"IoU": False, "Kalman": True}   # name -> use_kalman
TIMING_REPEATS = 3                          # FPS = best of 3 runs, to reduce noise
RESULTS_DIR = os.path.join(ROOT, "results")


def run_and_save(detector: str, tracker: str, use_kalman: bool) -> float:
    """Run a tracker, save its MOT file for TrackEval, return its speed in FPS.

    Only the tracking is timed (no image reading, no drawing).
    """
    detections = load_detections(os.path.join(SEQ_DIR, "det", detector, "det.txt"))
    n_frames = max(detections)

    best_time = float("inf")
    for _ in range(TIMING_REPEATS):
        Track.next_id = 1   # same IDs at every run
        start = time.perf_counter()
        for _ in run_tracker(detections, n_frames, use_kalman):
            pass
        best_time = min(best_time, time.perf_counter() - start)

    # Format the rows during the run: the tracks are mutated from one frame to the next
    Track.next_id = 1
    rows = []
    for frame, tracks in run_tracker(detections, n_frames, use_kalman):
        for t in tracks:
            left, top, width, height = t.box
            rows.append(f"{frame},{t.id},{left},{top},{width},{height},1,-1,-1,-1")

    folder = os.path.join(RESULTS_DIR, "trackers", f"{tracker}_{detector}", "data")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{SEQ}.txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return n_frames / best_time


def evaluate(names: list[str]) -> dict:
    """Run TrackEval on the saved results, return its raw output."""
    seqmap = os.path.join(RESULTS_DIR, "seqmap.txt")
    with open(seqmap, "w") as f:
        f.write(f"name\n{SEQ}\n")

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": ROOT,                      # <ROOT>/ADL-Rundle-6/gt/gt.txt
        "TRACKERS_FOLDER": os.path.join(RESULTS_DIR, "trackers"),
        "SEQMAP_FILE": seqmap,
        "BENCHMARK": "MOT15",                   # gt has no class column (-1)
        "SKIP_SPLIT_FOL": True,
        "TRACKERS_TO_EVAL": names,
        "PRINT_CONFIG": False,
    })
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False, "PRINT_CONFIG": False, "PRINT_RESULTS": False,
        "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
        "TIME_PROGRESS": False,
    })
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]

    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    output, _ = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)
    return output["MotChallenge2DBox"]


def main() -> None:
    fps = {}
    for detector in DETECTORS:
        for tracker, use_kalman in TRACKERS.items():
            fps[f"{tracker}_{detector}"] = run_and_save(detector, tracker, use_kalman)

    output = evaluate(list(fps))

    lines = ["| Detections | Tracker | HOTA | IDF1 | IDSW | FPS |",
             "|---|---|---|---|---|---|"]
    for detector in DETECTORS:
        for tracker in TRACKERS:
            name = f"{tracker}_{detector}"
            res = output[name]["COMBINED_SEQ"]["pedestrian"]
            hota = 100 * np.mean(res["HOTA"]["HOTA"])   # mean over the alpha thresholds
            idf1 = 100 * res["Identity"]["IDF1"]
            idsw = res["CLEAR"]["IDSW"]
            lines.append(f"| {detector} | {tracker} | {hota:.2f} | {idf1:.2f} | {idsw} | {fps[name]:.0f} |")

    table = "\n".join(lines)
    print(table)
    with open(os.path.join(RESULTS_DIR, "results.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
