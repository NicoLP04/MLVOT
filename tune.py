"""Hyperparameter search for the IoU and Kalman trackers (uses evaluate.py + TrackEval).

Usage: uv run tune.py
Stage 1: grid search on Yolov5l, ranked by HOTA.
Stage 2: best settings checked on the other detections (not used for the choice).
"""
import itertools
import os

import evaluate
import main

evaluate.TIMING_REPEATS = 1   # speed is not the point here

TUNE_ON = "Yolov5l"
CHECK_ON = ["Yolov5s", "public-dataset"]

# Grids: (iou_threshold, max_missed_frames) and, for Kalman, (std_acc, std_meas)
IOU_GRID = list(itertools.product([0.1, 0.2, 0.3, 0.4, 0.5], [1, 5, 10, 20, 30]))
KALMAN_GRID = list(itertools.product([0.1, 0.2, 0.3], [5, 10, 20, 30], [0.5, 1, 3], [1, 5, 10]))
DEFAULT = (0.3, 5, 1.0, 5.0)   # current values of main.py


def set_params(iou_thr, max_missed, std_acc, std_meas) -> None:
    main.IOU_THRESHOLD = iou_thr
    main.MAX_MISSED_FRAMES = max_missed
    main.STD_ACC = std_acc
    main.X_STD_MEAS = main.Y_STD_MEAS = std_meas


def score(output: dict, name: str) -> tuple[float, float, int]:
    res = output[name]["COMBINED_SEQ"]["pedestrian"]
    return (100 * evaluate.np.mean(res["HOTA"]["HOTA"]),
            100 * res["Identity"]["IDF1"],
            res["CLEAR"]["IDSW"])


def run_configs(detector: str, configs: list[tuple]) -> dict[tuple, tuple]:
    """configs: list of (use_kalman, iou_thr, max_missed, std_acc, std_meas) -> (HOTA, IDF1, IDSW)."""
    names = {}
    for cfg in configs:
        use_kalman, *params = cfg
        set_params(*params)
        name = "_".join(str(p) for p in cfg)
        evaluate.run_and_save(detector, name, use_kalman)
        names[cfg] = f"{name}_{detector}"
    output = evaluate.evaluate(list(names.values()))
    return {cfg: score(output, name) for cfg, name in names.items()}


def fmt(cfg: tuple, s: tuple) -> str:
    kalman, thr, miss, acc, meas = cfg
    params = f"iou>={thr}, max_missed={miss}" + (f", std_acc={acc}, std_meas={meas}" if kalman else "")
    return f"| {'Kalman' if kalman else 'IoU'} | {params} | {s[0]:.2f} | {s[1]:.2f} | {s[2]} |"


def main_tune() -> None:
    lines = []

    configs = [(False, *p, 1.0, 5.0) for p in IOU_GRID] + [(True, *p) for p in KALMAN_GRID]
    configs.append((False, *DEFAULT))
    configs.append((True, *DEFAULT))
    results = run_configs(TUNE_ON, configs)

    best = {}
    header = ["| Tracker | Parameters | HOTA | IDF1 | IDSW |", "|---|---|---|---|---|"]
    for kalman in (False, True):
        ranked = sorted((c for c in results if c[0] == kalman), key=lambda c: -results[c][0])
        best[kalman] = ranked[0]
        lines += [f"\n### {'Kalman' if kalman else 'IoU'} - top 5 on {TUNE_ON}", *header]
        lines += [fmt(c, results[c]) for c in ranked[:5]]
        lines += ["Default:", *header, fmt((kalman, *DEFAULT), results[(kalman, *DEFAULT)])]

    # Stage 2: same settings on detections that did not take part in the choice
    for detector in CHECK_ON:
        check = [(k, *DEFAULT) for k in (False, True)] + list(best.values())
        res = run_configs(detector, check)
        lines += [f"\n### Check on {detector} (not used for tuning)", *header]
        lines += [fmt(c, res[c]) + (" default" if c[1:] == DEFAULT else " tuned") for c in check]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(evaluate.RESULTS_DIR, "tuning.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main_tune()
