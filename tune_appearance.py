"""Part IV: tune and evaluate the appearance-aware tracker (uses evaluate.py + TrackEval).

Usage: uv run tune_appearance.py hist|cnn
Stage 1: grid search on Yolov5l, ranked by HOTA.
Stage 2: compare with Part II trackers on all detections (best settings chosen on Yolov5l only).
"""
import itertools
import os
import sys

import evaluate
import main
from Appearance import get_features

DESCRIPTOR = sys.argv[1] if len(sys.argv) > 1 else "hist"
TUNE_ON = "Yolov5l"
CHECK_ON = ["Yolov5s", "public-dataset"]

evaluate.TIMING_REPEATS = 1
np = evaluate.np

# Kalman parameters found by tune.py (Part III); the appearance tracker starts from them
KALMAN_TUNED = dict(iou=0.2, max_missed=30, std_acc=0.5, std_meas=10.0)
KALMAN_DEFAULT = dict(iou=0.3, max_missed=5, std_acc=1.0, std_meas=5.0)

BETAS = [0.0, 0.25, 0.5, 0.75]            # weight of appearance (alpha = 1 - beta); 0 = no appearance
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]    # minimum similarity S to accept a match
MAX_MISSED = [10, 30]
GRID = list(itertools.product(BETAS, THRESHOLDS, MAX_MISSED))


def apply(kalman: dict, beta: float = 0.5, threshold: float = 0.3, max_missed: int = 30) -> None:
    main.IOU_THRESHOLD = kalman["iou"]
    main.MAX_MISSED_FRAMES = kalman["max_missed"]
    main.STD_ACC = kalman["std_acc"]
    main.X_STD_MEAS = main.Y_STD_MEAS = kalman["std_meas"]
    main.APP_ALPHA, main.APP_BETA = 1 - beta, beta
    main.APP_THRESHOLD, main.APP_MAX_MISSED = threshold, max_missed


def score(output: dict, name: str) -> tuple:
    res = output[name]["COMBINED_SEQ"]["pedestrian"]
    return (100 * np.mean(res["HOTA"]["HOTA"]), 100 * np.mean(res["HOTA"]["DetA"]),
            100 * np.mean(res["HOTA"]["AssA"]), 100 * res["Identity"]["IDF1"],
            res["CLEAR"]["IDSW"])


def run_all(detector: str, configs: dict) -> dict:
    """configs: {label: (use_kalman, use_appearance, apply_kwargs)} -> {label: (HOTA, DetA, AssA, IDF1, IDSW, FPS)}"""
    detections = main.load_detections(os.path.join(main.SEQ_DIR, "det", detector, "det.txt"))
    features = get_features(detector, DESCRIPTOR, detections)
    fps = {}
    for label, (use_kalman, use_app, kalman, params) in configs.items():
        apply(kalman, **params)
        fps[label] = evaluate.run_and_save(detector, label, use_kalman, features if use_app else None)
    output = evaluate.evaluate([f"{label}_{detector}" for label in configs])
    return {label: score(output, f"{label}_{detector}") + (fps[label],) for label in configs}


def row(label: str, s: tuple) -> str:
    return f"| {label} | {s[0]:.2f} | {s[1]:.2f} | {s[2]:.2f} | {s[3]:.2f} | {s[4]} | {s[5]:.0f} |"


HEADER = ["| Tracker | HOTA | DetA | AssA | IDF1 | IDSW | FPS |", "|---|---|---|---|---|---|---|"]


def main_tune() -> None:
    lines = []

    # Stage 1
    configs = {f"app_b{b}_t{t}_m{m}": (True, True, KALMAN_TUNED, dict(beta=b, threshold=t, max_missed=m))
               for b, t, m in GRID}
    results = run_all(TUNE_ON, configs)
    ranked = sorted(results, key=lambda k: -results[k][0])
    lines += [f"### Appearance ({DESCRIPTOR}) - top 8 on {TUNE_ON}", *HEADER]
    lines += [row(k, results[k]) for k in ranked[:8]]
    best = ranked[0]
    lines += [f"\nBest: {best}"]

    # Stage 2: same detections for every tracker
    b, t, m = (best.split("_")[i][1:] for i in (1, 2, 3))
    best_params = dict(beta=float(b), threshold=float(t), max_missed=int(m))
    final = {
        "IoU (Part I)": (False, False, KALMAN_DEFAULT, {}),
        "Kalman (Part II)": (True, False, KALMAN_DEFAULT, {}),
        "Kalman tuned": (True, False, KALMAN_TUNED, {}),
        f"Appearance ({DESCRIPTOR})": (True, True, KALMAN_TUNED, best_params),
    }
    for detector in [TUNE_ON] + CHECK_ON:
        res = run_all(detector, final)
        lines += [f"\n### {detector}" + ("" if detector == TUNE_ON else " (not used for tuning)"), *HEADER]
        lines += [row(k, v) for k, v in res.items()]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(evaluate.RESULTS_DIR, f"appearance_{DESCRIPTOR}.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main_tune()
